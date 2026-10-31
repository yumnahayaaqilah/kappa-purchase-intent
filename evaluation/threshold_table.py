"""
Tabel trade-off ambang dari eval_log (hanya perlu PostgreSQL aktif, pipeline tidak dijalankan ulang).

    python -m evaluation.threshold_table
    python -m evaluation.threshold_table --train-frac 0.7

Protokol sama dengan full_evaluation.py: periode uji = bagian akhir menurut waktu prediksi,
skor = skor River yang dibuat pipeline saat itu. Setiap baris = "tandai X% sesi dengan skor tertinggi".
Keluaran: results/thresholds.csv
"""
import argparse

import numpy as np
import pandas as pd
import psycopg2

from db_config import DB

ap = argparse.ArgumentParser()
ap.add_argument("--train-frac", type=float, default=0.7)
ap.add_argument("--tandai", type=float, nargs="+", default=[1, 2, 5, 10, 15, 20, 30, 40],
                help="persen sesi teratas yang ditandai")
args = ap.parse_args()

conn = psycopg2.connect(**DB)
with conn.cursor() as cur:
    cur.execute("SELECT t_pred, prob, y FROM eval_log ORDER BY t_pred")
    df = pd.DataFrame(cur.fetchall(), columns=["t_pred", "prob", "y"])
conn.close()
if len(df) < 1000:
    raise SystemExit(f"Baris eval_log terlalu sedikit ({len(df)}).")

df["prob"] = df["prob"].astype(float)
df["y"] = df["y"].astype(int)
cut = int(len(df) * args.train_frac)
awal, uji = df.iloc[:cut], df.iloc[cut:]
y, p = uji["y"].to_numpy(), uji["prob"].to_numpy()
n, pos = len(y), int(y.sum())
base = y.mean()
print(f"Periode uji: {n:,} sesi, {pos:,} beli ({base:.4f})\n")

rows = []
for pct in args.tandai:
    # ambang ditetapkan dari periode awal (tidak melihat periode uji)
    thr = float(np.percentile(awal["prob"], 100 - pct))
    yhat = p >= thr
    tp = int((yhat & (y == 1)).sum())
    fp = int((yhat & (y == 0)).sum())
    fn = int((~yhat & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(pos, 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    rows.append({"ditandai_%_target": pct, "ambang": round(thr, 4), "ditandai_%_nyata": 100 * yhat.mean(),
                 "precision": prec, "recall": rec, "F1": f1, "lift": prec / base, "TP": tp, "FP": fp, "FN": fn})

tab = pd.DataFrame(rows)
tab.to_csv("results/thresholds.csv", index=False)
pd.set_option("display.width", 200)
print(tab.round(4).to_string(index=False))
best = tab.loc[tab["F1"].idxmax()]
print(f"\nF1 tertinggi pada ambang {best['ambang']} (menandai {best['ditandai_%_nyata']:.1f}% sesi): "
      f"precision {best['precision']:.3f}, recall {best['recall']:.3f}, F1 {best['F1']:.3f}")
print("Catatan: 'ditandai_%_nyata' bisa berbeda dari target karena ambang diambil dari periode awal "
      "dan proporsi beli periode uji sedikit lebih tinggi.")
