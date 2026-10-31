"""
Pembanding model batch (sklearn) yang dihitung dari CATATAN PIPELINE (eval_log) yang sama.
  - sklearn dilatih pada 70% session paling awal, diuji pada 30% paling akhir.
  - Model online dinilai dari probabilitas yang dicatat pipeline saat prediksi (prequential)
    pada sesi uji yang SAMA.

    python -m evaluation.baseline_from_log
"""
import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from db_config import DB

MAX_TRAIN, MAX_TEST = 1_000_000, 500_000
FEATS = ["f_c", "f_avg_price", "f_max_price", "f_dur"]

conn = psycopg2.connect(**DB)
cur = conn.cursor()
cur.execute("SELECT t_pred, f_c, f_avg_price, f_max_price, f_dur, prob, y FROM eval_log ORDER BY t_pred")
df = pd.DataFrame(cur.fetchall(), columns=["t_pred"] + FEATS + ["prob", "y"])
conn.close()
df["t_pred"] = pd.to_datetime(df["t_pred"], utc=True)
if len(df) < 1000 or df["y"].nunique() < 2:
    raise SystemExit("eval_log belum cukup berisi. Jalankan pipeline lebih lama.")

cut = int(len(df) * 0.7)
train, test = df.iloc[:cut], df.iloc[cut:]
trn = train.iloc[:: max(len(train) // MAX_TRAIN, 1)]
tst = test.sample(min(len(test), MAX_TEST), random_state=0).sort_values("t_pred")
y = tst["y"].values
days = tst["t_pred"].dt.date.astype(str).values
print(f"Session tercatat: {len(df):,} | latih {len(trn):,} (sampai {train.t_pred.max()}) | "
      f"uji {len(tst):,} (mulai {test.t_pred.min()}) | proporsi beli uji {y.mean():.4f}")


def daily_auc(d, yy, s):
    v = []
    for day in np.unique(d):
        m = d == day
        if m.sum() >= 200 and len(np.unique(yy[m])) == 2:
            v.append(roc_auc_score(yy[m], s[m]))
    return float(np.mean(v)) if v else float("nan")


def p_at_5(yy, s):
    return float(yy[np.argsort(-s)[: max(int(len(yy) * 0.05), 1)]].mean())


def row(name, s):
    return {"model": name, "ROC-AUC": round(roc_auc_score(y, s), 4), "AUC-harian": round(daily_auc(days, y, s), 4),
            "PR-AUC": round(average_precision_score(y, s), 4), "Precision@5%": round(p_at_5(y, s), 4)}


sc = StandardScaler().fit(trn[FEATS])
lr = LogisticRegression(max_iter=300, class_weight="balanced").fit(sc.transform(trn[FEATS]), trn["y"])
rows = [
    {"model": "Acak (rujukan)", "ROC-AUC": 0.5, "AUC-harian": 0.5, "PR-AUC": round(float(y.mean()), 4),
     "Precision@5%": round(float(y.mean()), 4)},
    row("Batch: LogReg sklearn (dilatih sekali)", lr.predict_proba(sc.transform(tst[FEATS]))[:, 1]),
    row("Online: River dalam pipeline (prequential)", tst["prob"].values),
]
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("results/pipeline_vs_batch.csv", index=False)
print("Disimpan di results/pipeline_vs_batch.csv")
