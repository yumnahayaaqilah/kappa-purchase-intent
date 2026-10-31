"""
Driver-side handler for the session stream (foreachBatch): online model + PostgreSQL.

Each micro-batch holds "pred" rows (a session reached its K-th event) and "close" rows
(a predicted session ended, with its label). Rows are applied in event-time order,
prediction before learning at equal times. Model state, predictions, evaluation rows and
the batch id are committed in one transaction, so a repeated batch is skipped safely.
"""
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from river import linear_model, optim, preprocessing
from sklearn.metrics import average_precision_score, roc_auc_score

from pipeline.sessions import FEATURES

QUERY = "sessions"
WEIGHTS_EVERY = 10


def new_model():
    return preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=optim.AdaGrad(0.5))


def ts(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def day_metrics(prob, y):
    prob, y = np.asarray(prob, float), np.asarray(y, int)
    n = len(y)
    out = {"n": n, "pos_rate": float(y.mean()) if n else None, "auc": None, "pr_auc": None, "p_at_5": None}
    if n >= 50 and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, prob))
        out["pr_auc"] = float(average_precision_score(y, prob))
        top = max(int(n * 0.05), 1)
        out["p_at_5"] = float(y[np.argsort(-prob)[:top]].mean())
    return out


class Learner:
    def __init__(self, db):
        self.db = db
        conn = psycopg2.connect(**db)
        with conn, conn.cursor() as cur:
            cur.execute("SELECT model, n_learned FROM model_state WHERE id = 1")
            r = cur.fetchone()
            self.model, self.n_learned = (pickle.loads(bytes(r[0])), r[1]) if r else (new_model(), 0)
            cur.execute("SELECT last_batch_id FROM stream_progress WHERE query = %s", (QUERY,))
            r = cur.fetchone()
            self.last_batch = r[0] if r else -1
        conn.close()
        print(f"Learner ready: {self.n_learned:,} sessions learned so far, last batch {self.last_batch}", flush=True)

    def __call__(self, batch_df, batch_id):
        if batch_id <= self.last_batch:
            print(f"Batch {batch_id}: already committed, skipped", flush=True)
            return
        pdf = batch_df.toPandas()
        if pdf.empty:
            self._commit_progress_only(batch_id)
            return
        pdf["order"] = (pdf["kind"] == "close").astype(int)       # predict before learn at equal time
        pdf = pdf.sort_values(["t_event", "order", "session_id"], kind="stable")
        closing = pdf.loc[pdf["kind"] == "close", "session_id"].tolist()

        conn = psycopg2.connect(**self.db)
        try:
            with conn, conn.cursor() as cur:
                probs = {}
                if closing:
                    cur.execute("SELECT session_id, prob FROM active_predictions WHERE session_id = ANY(%s)", (closing,))
                    probs = dict(cur.fetchall())
                preds, evals, missing = [], [], 0
                for r in pdf.itertuples(index=False):
                    x = dict(zip(FEATURES, (r.f_c, r.f_avg_price, r.f_max_price, r.f_dur)))
                    if r.kind == "pred":
                        p = float(self.model.predict_proba_one(x).get(True, 0.5))
                        probs[r.session_id] = p
                        preds.append((r.session_id, ts(r.t_pred), int(r.n_events), int(r.f_c), p, ts(r.t_pred)))
                    else:
                        p = probs.get(r.session_id)
                        if p is None:            # should not happen with transactional commits
                            missing += 1
                            continue
                        self.model.learn_one(x, bool(r.y))
                        self.n_learned += 1
                        evals.append((r.session_id, ts(r.t_pred), ts(r.t_event), r.f_c, r.f_avg_price,
                                      r.f_max_price, r.f_dur, p, int(r.y)))

                if preds:
                    execute_values(cur, """
                        INSERT INTO active_predictions (session_id, t_pred, n_events, n_cart, prob, last_time)
                        VALUES %s ON CONFLICT (session_id) DO NOTHING""", preds, page_size=5000)
                if evals:
                    execute_values(cur, """
                        INSERT INTO eval_log (session_id, t_pred, t_close, f_c, f_avg_price, f_max_price, f_dur, prob, y)
                        VALUES %s ON CONFLICT (session_id) DO NOTHING""", evals, page_size=5000)
                    cur.execute("DELETE FROM active_predictions WHERE session_id = ANY(%s)", ([e[0] for e in evals],))
                    self._update_daily(cur, {e[1].date() for e in evals})

                cur.execute("""
                    INSERT INTO model_state (id, batch_id, model, n_learned, updated_at) VALUES (1, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET batch_id = EXCLUDED.batch_id, model = EXCLUDED.model,
                        n_learned = EXCLUDED.n_learned, updated_at = now()""",
                            (batch_id, psycopg2.Binary(pickle.dumps(self.model)), self.n_learned))
                if batch_id % WEIGHTS_EVERY == 0:
                    w = self.model["LogisticRegression"].weights
                    execute_values(cur, """INSERT INTO model_weights (batch, feature, weight) VALUES %s
                        ON CONFLICT (batch, feature) DO UPDATE SET weight = EXCLUDED.weight""",
                                   [(batch_id, f, float(w.get(f, 0.0))) for f in FEATURES])
                cur.execute("""
                    INSERT INTO pipeline_stats (id, batches, sessions_learned, updated_at) VALUES (1, 1, %s, now())
                    ON CONFLICT (id) DO UPDATE SET batches = coalesce(pipeline_stats.batches, 0) + 1,
                        sessions_learned = coalesce(pipeline_stats.sessions_learned, 0) + EXCLUDED.sessions_learned,
                        updated_at = now()""", (len(evals),))
                self._set_progress(cur, batch_id)
        except Exception:
            self._reload()                       # the transaction rolled back: restore the committed model
            raise
        finally:
            conn.close()
        self.last_batch = batch_id
        print(f"Batch {batch_id}: {len(preds):,} predictions | {len(evals):,} sessions learned"
              + (f" | {missing} close rows without prediction" if missing else ""), flush=True)

    def _update_daily(self, cur, days):
        for d in sorted(days):
            cur.execute("SELECT prob, y FROM eval_log WHERE t_pred >= %s AND t_pred < %s",
                        (d, pd.Timestamp(d) + pd.Timedelta(days=1)))
            r = cur.fetchall()
            m = day_metrics([a for a, _ in r], [b for _, b in r])
            cur.execute("""
                INSERT INTO metrics_daily (day, n, pos_rate, auc, pr_auc, p_at_5, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (day) DO UPDATE SET n = EXCLUDED.n, pos_rate = EXCLUDED.pos_rate, auc = EXCLUDED.auc,
                    pr_auc = EXCLUDED.pr_auc, p_at_5 = EXCLUDED.p_at_5, updated_at = now()""",
                        (d, m["n"], m["pos_rate"], m["auc"], m["pr_auc"], m["p_at_5"]))

    def _set_progress(self, cur, batch_id):
        cur.execute("""INSERT INTO stream_progress (query, last_batch_id, updated_at) VALUES (%s, %s, now())
                       ON CONFLICT (query) DO UPDATE SET last_batch_id = EXCLUDED.last_batch_id, updated_at = now()""",
                    (QUERY, batch_id))

    def _commit_progress_only(self, batch_id):
        conn = psycopg2.connect(**self.db)
        try:
            with conn, conn.cursor() as cur:
                self._set_progress(cur, batch_id)
        finally:
            conn.close()
        self.last_batch = batch_id

    def _reload(self):
        conn = psycopg2.connect(**self.db)
        with conn, conn.cursor() as cur:
            cur.execute("SELECT model, n_learned FROM model_state WHERE id = 1")
            r = cur.fetchone()
            self.model, self.n_learned = (pickle.loads(bytes(r[0])), r[1]) if r else (new_model(), 0)
        conn.close()
