"""
Pipeline streaming end-to-end:  Kafka -> Spark Structured Streaming -> model online (River) -> PostgreSQL

Definisi tugas (SAMA dengan yang dievaluasi):
  - Prediksi dibuat tepat saat sebuah session mencapai event ke-K (K = 3).
  - Hanya session yang BELUM membeli dalam K event pertama.
  - Fitur dari K event pertama: jumlah cart, harga rata-rata, harga maksimum, durasi (detik).
  - Label = 1 jika ada pembelian pada event setelah ke-K. Label diketahui setelah session selesai
    (tidak ada event baru selama TIMEOUT_MIN menit menurut WAKTU DATA).
  - Evaluasi prequential: prediksi dibuat dulu, model baru belajar dari label setelah session selesai.

Menjalankan:   python spark_stream_v2.py
"""
import math
import os
import pickle
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from river import linear_model, optim, preprocessing
from sklearn.metrics import average_precision_score, roc_auc_score

from db_config import DB

K = 3
TIMEOUT_MIN = 30
TOPIC = "ecommerce-v2"
MAX_OFFSETS_PER_TRIGGER = 20000
MODEL_FILE = "model_v2.pkl"
CHECKPOINT = "checkpoint/v2"
SAVE_EVERY = 10  # batch


def new_model():
    return preprocessing.StandardScaler() | linear_model.LogisticRegression(
        optimizer=optim.AdaGrad(0.5))


def make_features(c, ps, pc, pmax, dur):
    return {
        "c": float(c),
        "avg_price": math.log1p(ps / pc) if pc > 0 else 0.0,
        "max_price": math.log1p(pmax),
        "dur": math.log1p(max(dur, 0.0)),
    }


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


class Pipeline:
    def __init__(self, db=DB, k=K, timeout_min=TIMEOUT_MIN, model_file=MODEL_FILE):
        self.db, self.k, self.timeout = db, k, timedelta(minutes=timeout_min)
        self.model_file = model_file
        if model_file and os.path.exists(model_file):
            with open(model_file, "rb") as f:
                self.model = pickle.load(f)
            print(f"Model dimuat dari {model_file}")
        else:
            self.model = new_model()
        self.watermark = None
        self.batches = 0
        self.last_offset = -1
        conn = self._conn()
        with conn, conn.cursor() as cur:
            cur.execute("SELECT last_event_time, batches, last_offset FROM pipeline_stats WHERE id = 1")
            r = cur.fetchone()
            if r and r[0] is not None:
                self.watermark = pd.Timestamp(r[0])
                self.batches = int(r[1] or 0)
                self.last_offset = int(r[2]) if r[2] is not None else -1
        conn.close()

    def _conn(self):
        return psycopg2.connect(**self.db)

    # ------------------------------------------------------------------ batch
    def process_batch(self, pdf, batch_id, cat_counts=None):
        """pdf: kolom offset, event_time, event_type, price, user_session."""
        t_start = time.time()
        pdf = pdf[pdf["user_session"].notna() & (pdf["user_session"] != "")]
        if pdf.empty:
            return {}
        pdf = pdf[pdf["offset"] > self.last_offset]      # lewati offset yang sudah diproses (idempoten)
        if pdf.empty:
            return {}
        pdf = pdf.sort_values("offset", kind="stable")
        max_offset = int(pdf["offset"].max())
        ts = pd.to_datetime(
            pdf["event_time"].str.replace(" UTC", "", regex=False), utc=True, format="%Y-%m-%d %H:%M:%S"
        ).tolist()
        sids = pdf["user_session"].tolist()
        ets = pdf["event_type"].tolist()
        prices = pdf["price"].astype(float).tolist()

        conn = self._conn()
        stats = {"events": len(sids), "new_pred": 0, "closed": 0}
        try:
            with conn, conn.cursor() as cur:
                ids = list(dict.fromkeys(sids))
                cur.execute(
                    """SELECT session_id, n_events, first_time, last_time, k_time, n_cart, price_sum,
                              price_cnt, price_max, purchase_before_k, purchase_after, predicted, prob, closed
                       FROM session_state WHERE session_id = ANY(%s)""",
                    (ids,),
                )
                st = {}
                for r in cur.fetchall():
                    st[r[0]] = dict(
                        n=r[1], first=pd.Timestamp(r[2]) if r[2] else None,
                        last=pd.Timestamp(r[3]) if r[3] else None,
                        kt=pd.Timestamp(r[4]) if r[4] else None, c=r[5], ps=r[6], pc=r[7], pmax=r[8],
                        pbk=r[9], pa=r[10], pred=r[11], prob=r[12], closed=r[13],
                    )
                touched = set()
                for sid, t, et, price in zip(sids, ts, ets, prices):
                    s = st.get(sid)
                    if s is None:
                        s = st[sid] = dict(n=0, first=None, last=None, kt=None, c=0, ps=0.0, pc=0,
                                           pmax=0.0, pbk=False, pa=False, pred=False, prob=None, closed=False)
                    if s["closed"]:
                        continue
                    touched.add(sid)
                    s["n"] += 1
                    s["last"] = t
                    if s["n"] == 1:
                        s["first"] = t
                    if s["n"] <= self.k:
                        if et == "purchase":
                            s["pbk"] = True
                        else:
                            if et == "cart":
                                s["c"] += 1
                            if price == price and price > 0:  # bukan NaN dan > 0
                                s["ps"] += price
                                s["pc"] += 1
                                if price > s["pmax"]:
                                    s["pmax"] = price
                        if s["n"] == self.k:
                            s["kt"] = t
                            if not s["pbk"]:
                                x = make_features(s["c"], s["ps"], s["pc"], s["pmax"],
                                                  (t - s["first"]).total_seconds())
                                s["prob"] = float(self.model.predict_proba_one(x).get(True, 0.5))
                                s["pred"] = True
                                stats["new_pred"] += 1
                    elif et == "purchase":
                        s["pa"] = True

                # simpan state session yang tersentuh
                rows = []
                for sid in touched:
                    s = st[sid]
                    rows.append((sid, s["n"], s["first"].to_pydatetime(), s["last"].to_pydatetime(),
                                 s["kt"].to_pydatetime() if s["kt"] is not None else None,
                                 s["c"], s["ps"], s["pc"], s["pmax"], s["pbk"], s["pa"], s["pred"],
                                 s["prob"], False))
                execute_values(cur, """
                    INSERT INTO session_state (session_id, n_events, first_time, last_time, k_time, n_cart,
                        price_sum, price_cnt, price_max, purchase_before_k, purchase_after, predicted, prob, closed)
                    VALUES %s
                    ON CONFLICT (session_id) DO UPDATE SET
                        n_events=EXCLUDED.n_events, first_time=EXCLUDED.first_time, last_time=EXCLUDED.last_time,
                        k_time=EXCLUDED.k_time, n_cart=EXCLUDED.n_cart, price_sum=EXCLUDED.price_sum,
                        price_cnt=EXCLUDED.price_cnt, price_max=EXCLUDED.price_max,
                        purchase_before_k=EXCLUDED.purchase_before_k, purchase_after=EXCLUDED.purchase_after,
                        predicted=EXCLUDED.predicted, prob=EXCLUDED.prob
                """, rows, page_size=5000)

                wm = max(ts)
                self.watermark = wm if self.watermark is None else max(self.watermark, wm)
                stats["closed"] = self._close_sessions(cur, self.watermark)

                if cat_counts:
                    agg = {}
                    for cat, et, n in cat_counts:
                        idx = {"view": 0, "cart": 1, "purchase": 2}.get(et)
                        if idx is not None:
                            agg.setdefault(cat, [0, 0, 0])[idx] += n
                    execute_values(cur, """
                        INSERT INTO category_insights (category, views, carts, purchases) VALUES %s
                        ON CONFLICT (category) DO UPDATE SET
                          views = category_insights.views + EXCLUDED.views,
                          carts = category_insights.carts + EXCLUDED.carts,
                          purchases = category_insights.purchases + EXCLUDED.purchases
                    """, [(c, v[0], v[1], v[2]) for c, v in agg.items()])

                self.batches += 1
                cur.execute("""
                    INSERT INTO pipeline_stats (id, events_total, last_event_time, batches, sessions_learned,
                                                last_offset, updated_at)
                    VALUES (1, %s, %s, 1, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET
                      events_total = pipeline_stats.events_total + EXCLUDED.events_total,
                      last_event_time = EXCLUDED.last_event_time,
                      batches = pipeline_stats.batches + 1,
                      sessions_learned = pipeline_stats.sessions_learned + EXCLUDED.sessions_learned,
                      last_offset = EXCLUDED.last_offset,
                      updated_at = now()
                """, (stats["events"], self.watermark.to_pydatetime(), stats["closed"], max_offset))
                self.last_offset = max_offset

                if self.batches % SAVE_EVERY == 0:
                    self._save_model(cur)
        finally:
            conn.close()
        stats["sec"] = time.time() - t_start
        return stats

    # ------------------------------------------------------------ menutup session
    def _close_sessions(self, cur, watermark):
        limit = (watermark - self.timeout).to_pydatetime()
        cur.execute(
            """SELECT session_id, n_cart, price_sum, price_cnt, price_max, first_time, k_time,
                      last_time, purchase_after, prob
               FROM session_state
               WHERE predicted AND NOT closed AND last_time < %s ORDER BY last_time""",
            (limit,),
        )
        closing = cur.fetchall()
        if not closing:
            return 0
        log, ids, days = [], [], set()
        for sid, c, ps, pc, pmax, first, kt, last, pa, prob in closing:
            x = make_features(c, ps, pc, pmax, (kt - first).total_seconds())
            y = bool(pa)
            self.model.learn_one(x, y)          # belajar SETELAH prediksi (prequential)
            log.append((sid, kt, watermark.to_pydatetime(), x["c"], x["avg_price"], x["max_price"],
                        x["dur"], prob, int(y)))
            ids.append(sid)
            days.add(pd.Timestamp(kt).tz_convert("UTC").normalize())
        execute_values(cur, """
            INSERT INTO eval_log (session_id, t_pred, t_close, f_c, f_avg_price, f_max_price, f_dur, prob, y)
            VALUES %s ON CONFLICT (session_id) DO NOTHING
        """, log, page_size=5000)
        cur.execute("UPDATE session_state SET closed = TRUE WHERE session_id = ANY(%s)", (ids,))
        for d in days:
            cur.execute("SELECT prob, y FROM eval_log WHERE t_pred >= %s AND t_pred < %s",
                        (d.to_pydatetime(), (d + pd.Timedelta(days=1)).to_pydatetime()))
            r = cur.fetchall()
            if not r:
                continue
            m = day_metrics([a for a, _ in r], [b for _, b in r])
            cur.execute("""
                INSERT INTO metrics_daily (day, n, pos_rate, auc, pr_auc, p_at_5, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (day) DO UPDATE SET n=EXCLUDED.n, pos_rate=EXCLUDED.pos_rate, auc=EXCLUDED.auc,
                    pr_auc=EXCLUDED.pr_auc, p_at_5=EXCLUDED.p_at_5, updated_at=now()
            """, (d.date(), m["n"], m["pos_rate"], m["auc"], m["pr_auc"], m["p_at_5"]))
        return len(ids)

    def close_all(self, far_future):
        """Hanya untuk pengujian/akhir aliran: tutup semua session yang masih terbuka."""
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                self.watermark = pd.Timestamp(far_future)
                return self._close_sessions(cur, self.watermark)
        finally:
            conn.close()

    def _save_model(self, cur):
        if self.model_file:
            tmp = self.model_file + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self.model, f)
            os.replace(tmp, self.model_file)
        try:
            w = self.model["LogisticRegression"].weights
            rows = [(self.batches, f, float(w.get(f, 0.0))) for f in ("c", "avg_price", "max_price", "dur")]
            execute_values(cur, """INSERT INTO model_weights (batch, feature, weight) VALUES %s
                                   ON CONFLICT (batch, feature) DO UPDATE SET weight = EXCLUDED.weight""", rows)
        except Exception as e:  # bobot hanya untuk tampilan, jangan hentikan pipeline
            print("Peringatan: bobot model tidak tersimpan:", e)


# ===================================================================== Spark
def main():
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    os.environ["PATH"] += r";C:\hadoop\bin"

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

    spark = (
        SparkSession.builder
        .appName("EcommerceKappaPipeline")
        .master("local[4]")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    schema = StructType([
        StructField("event_time", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("product_id", LongType(), True),
        StructField("category_id", LongType(), True),
        StructField("category_code", StringType(), True),
        StructField("brand", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("user_id", LongType(), True),
        StructField("user_session", StringType(), True),
    ])

    kafka_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", MAX_OFFSETS_PER_TRIGGER)
        .option("failOnDataLoss", "false")
        .load()
    )
    events = (
        kafka_df.select(F.col("offset"), F.from_json(F.col("value").cast("string"), schema).alias("d"))
        .select("offset", "d.*")
    )

    cat_expr = F.when(
        F.col("category_code").isNull() | (F.col("category_code") == ""), F.lit("unknown")
    ).otherwise(F.split(F.col("category_code"), r"\.").getItem(0))

    pipe = Pipeline()
    cols = ["offset", "event_time", "event_type", "price", "user_session"]

    def foreach(batch_df, batch_id):
        batch_df.persist()
        try:
            rows = batch_df.select(*cols).collect()
            pdf = pd.DataFrame.from_records(rows, columns=cols)
            cat = (batch_df.withColumn("cat", cat_expr).groupBy("cat", "event_type").count().collect())
            cat_counts = [(r["cat"], r["event_type"], int(r["count"])) for r in cat]
            s = pipe.process_batch(pdf, batch_id, cat_counts)
            if s:
                print(f"Batch {batch_id}: {s['events']:,} event | prediksi baru {s['new_pred']:,} | "
                      f"session selesai & dipelajari {s['closed']:,} | {s['events'] / max(s['sec'], 1e-9):,.0f} event/detik | "
                      f"data sampai {pipe.watermark}", flush=True)
        finally:
            batch_df.unpersist()

    query = (
        events.writeStream
        .foreachBatch(foreach)
        .option("checkpointLocation", CHECKPOINT)
        .trigger(processingTime="1 second")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
