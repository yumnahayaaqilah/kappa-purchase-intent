"""
Pemilihan optimizer dan learning rate model online (River) TANPA menyentuh periode uji.

Data: kunci jawaban DuckDB (sama persis dengan sesi pipeline), hanya 70% sesi pertama menurut waktu
prediksi. Di dalamnya, sesi ke-50% sampai ke-70% menjadi periode VALIDASI. Model diputar ulang
seperti pipeline: prediksi pada t_pred, belajar pada t_close (prequential), lalu metrik dihitung
pada skor periode validasi. Periode uji (30% terakhir) tidak dibaca sama sekali.

    python -m evaluation.tune_validation
    python -m evaluation.tune_validation --ref ref/session_reference_7m.parquet

Keluaran: results/tuning_validation.csv
"""
import argparse
import os
import time

import duckdb
import numpy as np
import pandas as pd
from river import linear_model, optim, preprocessing
from sklearn.metrics import average_precision_score, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="ref/session_reference_7m.parquet")
ap.add_argument("--train-frac", type=float, default=0.7, help="batas awal periode uji (tidak dipakai)")
ap.add_argument("--val-start", type=float, default=0.5, help="awal periode validasi")
ap.add_argument("--outdir", default="results")
args = ap.parse_args()

FEATS = ["f_c", "f_avg_price", "f_max_price", "f_dur"]
NAMES = ["c", "avg_price", "max_price", "dur"]

df = duckdb.sql(f"SELECT t_pred, t_close, {', '.join(FEATS)}, y FROM '{args.ref}' "
                f"ORDER BY t_pred, session_id").df()
n_all = len(df)
cut = int(n_all * args.train_frac)
v0 = int(n_all * args.val_start)
df = df.iloc[:cut].reset_index(drop=True)              # periode uji dibuang di sini
n = len(df)
print(f"Sesi total {n_all:,} | dipakai {n:,} (periode awal, sampai {df.t_pred.iloc[-1]})")
print(f"Validasi: sesi ke-{v0:,} sampai ke-{cut:,} (mulai {df.t_pred.iloc[v0]})\n")

xs = [dict(zip(NAMES, r)) for r in df[FEATS].astype(float).values.tolist()]
ys = df["y"].astype(bool).values
tp = pd.to_datetime(df["t_pred"]).astype("int64").values
tc = pd.to_datetime(df["t_close"]).astype("int64").values
times = np.concatenate([tp, tc])
kind = np.concatenate([np.zeros(n, int), np.ones(n, int)])   # 0 = prediksi, 1 = belajar
idx = np.concatenate([np.arange(n), np.arange(n)])
order = np.lexsort((kind, times))                              # waktu, prediksi sebelum belajar
y_val = df["y"].values[v0:]


def replay(opt):
    model = preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=opt)
    prob = np.full(n, 0.5)
    for e in order:
        i = idx[e]
        if kind[e] == 0:
            prob[i] = model.predict_proba_one(xs[i]).get(True, 0.5)
        else:
            model.learn_one(xs[i], ys[i])
    return prob[v0:]


variants = [("AdaGrad", lr) for lr in (0.1, 0.2, 0.5, 1.0, 2.0)] + [("SGD", lr) for lr in (0.01, 0.05, 0.1)]
rows = []
for opt_name, lr in variants:
    t0 = time.time()
    s = replay(getattr(optim, opt_name)(lr))
    top = max(int(len(y_val) * 0.05), 1)
    rows.append({"optimizer": opt_name, "learning_rate": lr,
                 "roc_auc_val": round(roc_auc_score(y_val, s), 4),
                 "pr_auc_val": round(average_precision_score(y_val, s), 4),
                 "precision_at_5pct_val": round(float(y_val[np.argsort(-s, kind="stable")[:top]].mean()), 4)})
    print(f"  {opt_name} {lr}: ROC-AUC {rows[-1]['roc_auc_val']} | {time.time() - t0:.0f} detik", flush=True)

out = pd.DataFrame(rows).sort_values("roc_auc_val", ascending=False)
pd.set_option("display.width", 200)
print("\n" + out.to_string(index=False))
best = out.iloc[0]
print(f"\nTerbaik di validasi: {best.optimizer} {best.learning_rate} (ROC-AUC {best.roc_auc_val})")
os.makedirs(args.outdir, exist_ok=True)
out.to_csv(os.path.join(args.outdir, "tuning_validation.csv"), index=False)
print(f"Tersimpan: {args.outdir}/tuning_validation.csv")
