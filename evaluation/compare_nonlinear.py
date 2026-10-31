"""
Cek batas kemampuan fitur: apakah model non-linear jauh lebih baik dari logistic regression
pada EMPAT FITUR YANG SAMA (jumlah keranjang, harga rata-rata, harga tertinggi, durasi)?

    python -m evaluation.compare_nonlinear
    python -m evaluation.compare_nonlinear --train-frac 0.7 --max-train 1000000

Hanya perlu PostgreSQL aktif. Latih sekali pada periode awal, uji pada periode uji (sama dengan full_evaluation.py).
Keluaran: results/nonlinear.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from db_config import DB

ap = argparse.ArgumentParser()
ap.add_argument("--train-frac", type=float, default=0.7)
ap.add_argument("--max-train", type=int, default=1_000_000)
ap.add_argument("--top-pct", type=float, default=5.0)
args = ap.parse_args()

FEATS = ["f_c", "f_avg_price", "f_max_price", "f_dur"]
conn = psycopg2.connect(**DB)
with conn.cursor() as cur:
    cur.execute("SELECT t_pred, f_c, f_avg_price, f_max_price, f_dur, prob, y FROM eval_log ORDER BY t_pred")
    df = pd.DataFrame(cur.fetchall(), columns=["t_pred", *FEATS, "prob", "y"])
conn.close()
for c in FEATS + ["prob"]:
    df[c] = df[c].astype(float)
df["y"] = df["y"].astype(int)

cut = int(len(df) * args.train_frac)
awal, uji = df.iloc[:cut], df.iloc[cut:]
tr = awal.iloc[:: max(1, len(awal) // args.max_train)]
y = uji["y"].to_numpy()
print(f"Latih {len(tr):,} sesi (periode awal) | uji {len(uji):,} sesi | proporsi beli uji {y.mean():.4f}\n")


def topk(y, p, pct):
    k = max(int(len(y) * pct / 100), 1)
    idx = np.argsort(-p, kind="stable")[:k]
    return float(y[idx].mean()), float(y[idx].sum() / max(y.sum(), 1))


def nilai(nama, p, detik=None):
    pr, rc = topk(y, p, args.top_pct)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"model": nama, "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            f"Precision@{args.top_pct:g}%": pr, f"Recall@{args.top_pct:g}%": rc,
            "Brier": brier_score_loss(y, pc), "detik_latih": detik}


rows = [nilai("River (skor pipeline, prequential)", uji["prob"].to_numpy())]

t = time.time()
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[FEATS], tr["y"])
rows.append(nilai("Logistic regression (sklearn)", lr.predict_proba(uji[FEATS])[:, 1], time.time() - t))

for nama, kw in [("Gradient boosting (dangkal)", dict(max_depth=3, max_iter=200, learning_rate=0.1)),
                 ("Gradient boosting (lebih dalam)", dict(max_depth=6, max_iter=300, learning_rate=0.05))]:
    t = time.time()
    gb = HistGradientBoostingClassifier(random_state=0, early_stopping=True, validation_fraction=0.1, **kw)
    gb.fit(tr[FEATS], tr["y"])
    rows.append(nilai(nama, gb.predict_proba(uji[FEATS])[:, 1], time.time() - t))

tab = pd.DataFrame(rows)
tab.to_csv("results/nonlinear.csv", index=False)
pd.set_option("display.width", 200)
print(tab.round(4).to_string(index=False))

best_lin = tab.loc[1, "ROC-AUC"]
best_gb = tab.loc[2:, "ROC-AUC"].max()
sel = best_gb - best_lin
print(f"\nSelisih ROC-AUC gradient boosting terbaik dibanding logistic regression: {sel:+.4f}")
if sel < 0.01:
    print("-> Hampir sama: batas ada pada informasi fitur, bukan pada jenis model.")
elif sel < 0.03:
    print("-> Ada keuntungan kecil dari non-linearitas; patut dicoba pohon online di River atau fitur interaksi.")
else:
    print("-> Keuntungan jelas: model non-linear layak dicoba di pipeline.")
