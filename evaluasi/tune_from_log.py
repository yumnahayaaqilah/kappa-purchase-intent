"""
Menguji model online (River) dengan MEMUTAR ULANG catatan pipeline (eval_log).
Tanpa Kafka dan Spark, jadi cepat. Protokolnya meniru pipeline:
  - prediksi dibuat pada t_pred (memakai model yang saat itu sudah dilatih),
  - model belajar dari label pada t_close (setelah session selesai).
Pembanding: sklearn dilatih sekali pada 70% awal, diuji pada 30% akhir (sesi uji yang sama).

Yang dibandingkan sekarang:
  - empat fitur dasar (c, avg_price, max_price, dur)  = yang dipakai pipeline saat ini
  - fitur dasar + fitur turunan (kombinasi dari empat fitur yang sama, tanpa informasi baru)
Fitur turunan dihitung dari empat fitur mentah yang SUDAH tersimpan di eval_log, jadi tabel tidak berubah.
Batas kelompok harga (EDGES) dihitung dari 70% awal dan dicetak; angka itulah yang nanti dipakai tetap di pipeline.

    python tune_from_log.py
Hasil: tabel di layar dan hasil_tuning.csv
"""
import time

import numpy as np
import pandas as pd
import psycopg2
from river import linear_model, optim, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from db_config import DB

FEATS = ["f_c", "f_avg_price", "f_max_price", "f_dur"]
NAMES = ["c", "avg_price", "max_price", "dur"]


def fitur_turunan(x, edges):
    """x: dict {c, avg_price, max_price, dur}. Kembalikan x ditambah fitur turunan.
    Fungsi yang SAMA nanti dipakai di spark_stream_v2.py (make_features)."""
    c, avg, mx, dur = x["c"], x["avg_price"], x["max_price"], x["dur"]
    ada = avg > 0
    rendah = ada and avg < edges[0]
    sedang = ada and edges[0] <= avg < edges[1]
    tinggi = ada and avg >= edges[1]
    y = dict(x)
    y["ada_keranjang"] = 1.0 if c > 0 else 0.0
    y["keranjang_1"] = 1.0 if c == 1 else 0.0
    y["keranjang_2plus"] = 1.0 if c >= 2 else 0.0
    y["harga_kosong"] = 0.0 if ada else 1.0
    y["c_x_avg"] = c * avg
    y["c_x_max"] = c * mx
    y["c_x_dur"] = c * dur
    y["avg_x_dur"] = avg * dur
    y["harga_rendah"] = 1.0 if rendah else 0.0
    y["harga_sedang"] = 1.0 if sedang else 0.0
    y["harga_tinggi"] = 1.0 if tinggi else 0.0
    y["keranjang_x_tinggi"] = y["ada_keranjang"] * y["harga_tinggi"]
    return y


conn = psycopg2.connect(**DB)
cur = conn.cursor()
cur.execute("SELECT t_pred, t_close, f_c, f_avg_price, f_max_price, f_dur, y FROM eval_log ORDER BY t_pred")
df = pd.DataFrame(cur.fetchall(), columns=["t_pred", "t_close"] + FEATS + ["y"])
conn.close()
if len(df) < 2000 or df["y"].nunique() < 2:
    raise SystemExit("eval_log belum cukup berisi.")
df["t_pred"] = pd.to_datetime(df["t_pred"], utc=True)
df["t_close"] = pd.to_datetime(df["t_close"], utc=True)
for c in FEATS:
    df[c] = df[c].astype(float)
n = len(df)
cut = int(n * 0.7)
print(f"Session: {n:,} | uji = {n - cut:,} session terakhir (mulai {df.t_pred.iloc[cut]})")

# batas kelompok harga dari 70% awal saja (harga rata-rata valid > 0)
valid = df["f_avg_price"].iloc[:cut]
valid = valid[valid > 0]
EDGES = (float(np.quantile(valid, 1 / 3)), float(np.quantile(valid, 2 / 3)))
print(f"EDGES (batas kelompok harga, dari 70% awal) = ({EDGES[0]!r}, {EDGES[1]!r})\n")

xs_dasar = [dict(zip(NAMES, r)) for r in df[FEATS].values.tolist()]
xs_turun = [fitur_turunan(x, EDGES) for x in xs_dasar]
ys = df["y"].astype(bool).values
tp = df["t_pred"].astype("int64").values
tc = df["t_close"].astype("int64").values
# urutan kejadian: waktu, lalu prediksi (0) sebelum belajar (1) bila bersamaan
times = np.concatenate([tp, tc])
kind = np.concatenate([np.zeros(n, int), np.ones(n, int)])
idx = np.concatenate([np.arange(n), np.arange(n)])
order = np.lexsort((kind, times))

y_test = df["y"].values[cut:]
days = df["t_pred"].dt.date.astype(str).values[cut:]


def daily_auc(yy, s):
    v = []
    for d in np.unique(days):
        m = days == d
        if m.sum() >= 200 and len(np.unique(yy[m])) == 2:
            v.append(roc_auc_score(yy[m], s[m]))
    return float(np.mean(v)) if v else float("nan")


def metrics(name, s):
    top = max(int(len(y_test) * 0.05), 1)
    sel = y_test[np.argsort(-s)[:top]]
    return {
        "model": name,
        "ROC-AUC": round(roc_auc_score(y_test, s), 4),
        "AUC-harian": round(daily_auc(y_test, s), 4),
        "PR-AUC": round(average_precision_score(y_test, s), 4),
        "Precision@5%": round(float(sel.mean()), 4),
        "Recall@5%": round(float(sel.sum() / max(y_test.sum(), 1)), 4),
    }


def replay(make_model, xs):
    model = make_model()
    prob = np.full(n, 0.5)
    for e in order:
        i = idx[e]
        if kind[e] == 0:
            prob[i] = model.predict_proba_one(xs[i]).get(True, 0.5)
        else:
            model.learn_one(xs[i], ys[i])
    return prob[cut:]


def river(opt):
    return lambda: preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=opt())


variants = [
    ("SAAT INI: 4 fitur dasar, AdaGrad 0.5", river(lambda: optim.AdaGrad(0.5)), xs_dasar),
    ("Fitur dasar + turunan, AdaGrad 0.5", river(lambda: optim.AdaGrad(0.5)), xs_turun),
    ("Fitur dasar + turunan, AdaGrad 0.2", river(lambda: optim.AdaGrad(0.2)), xs_turun),
    ("Fitur dasar + turunan, AdaGrad 1.0", river(lambda: optim.AdaGrad(1.0)), xs_turun),
]

rows = []
tr = df.iloc[:cut]
sc = StandardScaler().fit(tr[FEATS])
lr = LogisticRegression(max_iter=300).fit(sc.transform(tr[FEATS]), tr["y"])
rows.append(metrics("Rujukan: sklearn, 4 fitur dasar (dilatih sekali)", lr.predict_proba(sc.transform(df[FEATS].iloc[cut:]))[:, 1]))

cols_t = list(xs_turun[0].keys())
Xt = pd.DataFrame(xs_turun)[cols_t]
sct = StandardScaler().fit(Xt.iloc[:cut])
lrt = LogisticRegression(max_iter=1000).fit(sct.transform(Xt.iloc[:cut]), tr["y"])
rows.append(metrics("Rujukan: sklearn, dasar + turunan (dilatih sekali)", lrt.predict_proba(sct.transform(Xt.iloc[cut:]))[:, 1]))

for name, mk, xs in variants:
    t0 = time.time()
    rows.append(metrics(name, replay(mk, xs)))
    print(f"  {name}: selesai {time.time() - t0:.0f} detik", flush=True)

out = pd.DataFrame(rows)
print()
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 70)
print(out.to_string(index=False))
out.to_csv("hasil_tuning.csv", index=False)
print("\nDisimpan di hasil_tuning.csv")
print(f"Catat angka ini untuk pipeline: EDGES = ({EDGES[0]!r}, {EDGES[1]!r})")