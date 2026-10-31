"""
Uji offline: apakah fitur turunan (interaksi, kelompok harga, indikator) menaikkan kualitas logistic regression
mendekati gradient boosting? Tidak ada informasi baru, hanya kombinasi dari empat fitur yang sudah ada.

    python -m evaluation.compare_interactions
    python -m evaluation.compare_interactions --train-frac 0.7 --max-train 1000000

Hanya perlu PostgreSQL aktif. Keluaran: results/interactions.csv
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

BASE = ["f_c", "f_avg_price", "f_max_price", "f_dur"]
conn = psycopg2.connect(**DB)
with conn.cursor() as cur:
    cur.execute("SELECT t_pred, f_c, f_avg_price, f_max_price, f_dur, prob, y FROM eval_log ORDER BY t_pred")
    df = pd.DataFrame(cur.fetchall(), columns=["t_pred", *BASE, "prob", "y"])
conn.close()
for c in BASE + ["prob"]:
    df[c] = df[c].astype(float)
df["y"] = df["y"].astype(int)

cut = int(len(df) * args.train_frac)
awal, uji = df.iloc[:cut], df.iloc[cut:]
tr = awal.iloc[:: max(1, len(awal) // args.max_train)].copy()
uji = uji.copy()

# batas kelompok harga ditetapkan dari data latih saja (harga rata-rata yang valid, > 0)
valid = tr.loc[tr["f_avg_price"] > 0, "f_avg_price"]
EDGES = np.quantile(valid, [1 / 3, 2 / 3])


def turunan(d):
    o = pd.DataFrame(index=d.index)
    c = d["f_c"]
    o["ada_keranjang"] = (c > 0).astype(float)
    o["keranjang_1"] = (c == 1).astype(float)
    o["keranjang_2plus"] = (c >= 2).astype(float)
    o["harga_kosong"] = (d["f_avg_price"] <= 0).astype(float)
    o["c_x_avg"] = c * d["f_avg_price"]
    o["c_x_max"] = c * d["f_max_price"]
    o["c_x_dur"] = c * d["f_dur"]
    o["avg_x_dur"] = d["f_avg_price"] * d["f_dur"]
    b = np.digitize(d["f_avg_price"], EDGES)  # 0 rendah, 1 sedang, 2 tinggi
    ada = d["f_avg_price"] > 0
    o["harga_rendah"] = ((b == 0) & ada).astype(float)
    o["harga_sedang"] = ((b == 1) & ada).astype(float)
    o["harga_tinggi"] = ((b == 2) & ada).astype(float)
    o["kranjang_x_tinggi"] = o["ada_keranjang"] * o["harga_tinggi"]
    return o


Ttr, Tuj = turunan(tr), turunan(uji)
SETS = {
    "A. dasar (4 fitur)": [],
    "B. + ada keranjang & kelompok jumlah keranjang": ["ada_keranjang", "keranjang_1", "keranjang_2plus"],
    "C. + indikator harga kosong": ["harga_kosong"],
    "D. + interaksi keranjang x (harga, durasi)": ["c_x_avg", "c_x_max", "c_x_dur", "avg_x_dur"],
    "E. + kelompok harga": ["harga_rendah", "harga_sedang", "harga_tinggi", "kranjang_x_tinggi"],
    "F. semua turunan": list(Ttr.columns),
}

y = uji["y"].to_numpy()


def topk(p, pct):
    k = max(int(len(y) * pct / 100), 1)
    idx = np.argsort(-p, kind="stable")[:k]
    return float(y[idx].mean()), float(y[idx].sum() / max(y.sum(), 1))


def nilai(nama, p, detik=None):
    pr, rc = topk(p, args.top_pct)
    return {"model": nama, "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            f"Precision@{args.top_pct:g}%": pr, f"Recall@{args.top_pct:g}%": rc,
            "Brier": brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "detik": detik}


rows = []
for nama, extra in SETS.items():
    t = time.time()
    Xtr = pd.concat([tr[BASE], Ttr[extra]], axis=1)
    Xuj = pd.concat([uji[BASE], Tuj[extra]], axis=1)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, tr["y"])
    rows.append(nilai("Logistic " + nama, m.predict_proba(Xuj)[:, 1], time.time() - t))

# acuan atas: gradient boosting pada empat fitur dasar
t = time.time()
gb = HistGradientBoostingClassifier(random_state=0, max_depth=6, max_iter=300, learning_rate=0.05,
                                    early_stopping=True, validation_fraction=0.1).fit(tr[BASE], tr["y"])
rows.append(nilai("Acuan atas: gradient boosting (4 fitur dasar)", gb.predict_proba(uji[BASE])[:, 1], time.time() - t))

tab = pd.DataFrame(rows)
tab.to_csv("results/interactions.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
print(f"Latih {len(tr):,} | uji {len(uji):,} | proporsi beli uji {y.mean():.4f}\n")
print(tab.round(4).to_string(index=False))

a = tab.iloc[0]
f = tab.iloc[len(SETS) - 1]
g = tab.iloc[-1]
print(f"\nDari dasar ke semua turunan: PR-AUC {a['PR-AUC']:.4f} -> {f['PR-AUC']:.4f} "
      f"(gradient boosting {g['PR-AUC']:.4f}); Precision@{args.top_pct:g}% {a.iloc[3]:.4f} -> {f.iloc[3]:.4f} "
      f"(gradient boosting {g.iloc[3]:.4f})")
gap_total = g["PR-AUC"] - a["PR-AUC"]
if gap_total > 0:
    print(f"Porsi selisih PR-AUC terhadap gradient boosting yang tertutup oleh fitur turunan: "
          f"{(f['PR-AUC'] - a['PR-AUC']) / gap_total:.0%}")
else:
    print("Gradient boosting tidak lebih baik dari dasar pada PR-AUC, porsi selisih tidak dihitung.")
