"""
Evaluasi offline model prediksi pembelian pada SELURUH data (semua CSV di folder data/).

Cara pakai:
    python -m evaluation.evaluate_offline                 # K = 3, 5, 10 (snapshot awal session)
    python -m evaluation.evaluate_offline --rebuild       # jalankan ini setelah menambah bulan baru ke data/
    python -m evaluation.evaluate_offline --k 5           # satu nilai K saja
    python -m evaluation.evaluate_offline --with-time     # tambah fitur jam (sin/cos) dan akhir pekan
    python -m evaluation.evaluate_offline --k 3 --tune    # bandingkan beberapa kecepatan belajar River + cetak bobot

River memakai kecepatan belajar menurun bertahap (hasil --tune); harga <= 0 dianggap kosong.
Semua model memakai DATA LATIH YANG SAMA dan DATA UJI YANG SAMA (sampel acak yang sama),
dan AUC dihitung per hari lalu dirata-ratakan agar tidak terpengaruh tren waktu.

Definisi evaluasi:
  - Prediksi dibuat SETELAH K event pertama sebuah session (bukan saat session selesai).
  - Hanya session dengan >= K event dan belum ada pembelian dalam K event pertama.
  - Label = 1 jika ada pembelian pada event setelah ke-K.
  - Split berdasarkan WAKTU: 70% session paling awal untuk belajar, 30% paling akhir untuk uji.
"""
import argparse
import os
import time

import duckdb
import numpy as np
import pandas as pd
from river import linear_model, metrics, optim, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA_GLOB = "data/20*.csv"
DB_FILE = "eval.duckdb"
MAX_TRAIN = 1_000_000   # sampel latih (tersebar merata di seluruh periode latih), dipakai SEMUA model
MAX_TEST = 500_000      # sampel uji acak, dipakai SEMUA model

ap = argparse.ArgumentParser()
ap.add_argument("--rebuild", action="store_true")
ap.add_argument("--k", type=int, nargs="*", default=[3, 5, 10])
ap.add_argument("--with-time", action="store_true")
ap.add_argument("--tune", action="store_true")
args = ap.parse_args()

os.makedirs("duck_tmp", exist_ok=True)
con = duckdb.connect(DB_FILE)
con.execute("PRAGMA memory_limit='8GB'")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA temp_directory='duck_tmp'")

have = con.execute(
    "SELECT count(*) FROM information_schema.tables WHERE table_name='evr'"
).fetchone()[0]

if args.rebuild or not have:
    t0 = time.time()
    print("Membaca semua CSV dan menyusun tabel event (sekali saja, bisa beberapa menit)...")
    con.execute(f"""
        CREATE OR REPLACE TABLE evr AS
        SELECT sid, t, et, price, cat, brand, pid,
               row_number() OVER (PARTITION BY sid ORDER BY t, pid) AS rn
        FROM (
            SELECT user_session AS sid,
                   CAST(replace(event_time, ' UTC', '') AS TIMESTAMP) AS t,
                   event_type AS et,
                   CASE WHEN price > 0 THEN price END AS price,
                   COALESCE(NULLIF(split_part(category_code, '.', 1), ''), 'unknown') AS cat,
                   COALESCE(brand, 'unknown') AS brand,
                   product_id AS pid
            FROM read_csv('{DATA_GLOB}', header=true, columns={{
                'event_time':'VARCHAR','event_type':'VARCHAR','product_id':'BIGINT',
                'category_id':'BIGINT','category_code':'VARCHAR','brand':'VARCHAR',
                'price':'DOUBLE','user_id':'BIGINT','user_session':'VARCHAR'}})
            WHERE user_session IS NOT NULL
        )
    """)
    print(f"Selesai dalam {time.time() - t0:.0f} detik.")

tot, tmin, tmax = con.execute("SELECT count(*), min(t), max(t) FROM evr").fetchone()
print(f"Total event: {tot:,} | rentang waktu: {tmin} sampai {tmax}\n")

FEATS = ["c", "avg_price", "max_price", "n_cat", "n_brand", "n_prod", "dur"]
if args.with_time:
    FEATS += ["hr_sin", "hr_cos", "wkend"]
LOGF = ["avg_price", "max_price", "dur"]


def load(k):
    df = con.execute(f"""
        WITH first_k AS (
            SELECT sid,
                   count(*) FILTER (WHERE et='view') AS v,
                   count(*) FILTER (WHERE et='cart') AS c,
                   count(*) FILTER (WHERE et='purchase') AS p0,
                   coalesce(avg(price) FILTER (WHERE et<>'purchase'), 0) AS avg_price,
                   coalesce(max(price) FILTER (WHERE et<>'purchase'), 0) AS max_price,
                   count(DISTINCT cat) AS n_cat,
                   count(DISTINCT brand) AS n_brand,
                   count(DISTINCT pid) AS n_prod,
                   date_diff('second', min(t), max(t)) AS dur,
                   min(t) AS t0
            FROM evr WHERE rn <= {k}
            GROUP BY sid HAVING count(*) = {k}
        ),
        rest AS (
            SELECT sid, max(CASE WHEN et='purchase' THEN 1 ELSE 0 END) AS y
            FROM evr WHERE rn > {k} GROUP BY sid
        )
        SELECT f.v, f.c, f.avg_price, f.max_price, f.n_cat, f.n_brand, f.n_prod, f.dur, f.t0,
               hour(f.t0) AS hr, dayofweek(f.t0) AS dow, coalesce(r.y, 0) AS y
        FROM first_k f LEFT JOIN rest r USING (sid)
        WHERE f.p0 = 0
        ORDER BY f.t0
    """).df()
    for c in LOGF:
        df[c] = np.log1p(df[c].clip(lower=0))
    df["hr_sin"] = np.sin(2 * np.pi * df.hr / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df.hr / 24)
    df["wkend"] = (df.dow.isin([0, 6])).astype(int)
    return df


def p_at_top(y, s, frac=0.05):
    n = max(int(len(y) * frac), 1)
    idx = np.argsort(-s)[:n]
    return float(np.mean(np.asarray(y)[idx]))


def daily_auc(days, y, s):
    vals = []
    for d in np.unique(days):
        m = days == d
        if m.sum() >= 200 and len(np.unique(y[m])) == 2:
            vals.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(vals)) if vals else float("nan")


def score(name, y, s, days):
    return {
        "model": name,
        "ROC-AUC": round(roc_auc_score(y, s), 4),
        "AUC-harian": round(daily_auc(days, y, s), 4),
        "PR-AUC": round(average_precision_score(y, s), 4),
        "Precision@5%": round(p_at_top(y, s), 4),
    }


rows = []
for k in args.k:
    df = load(k)
    if len(df) < 1000 or df.y.nunique() < 2:
        print(f"K={k}: data terlalu sedikit, dilewati.")
        continue
    cut = int(len(df) * 0.7)
    train, test = df.iloc[:cut], df.iloc[cut:]
    print(f"=== K={k} event pertama ===")
    print(f"Session: {len(df):,} | train {len(train):,} (sampai {train.t0.max()}) | "
          f"test {len(test):,} (mulai {test.t0.min()})")
    print(f"Proporsi beli: train {train.y.mean():.4f} | test {test.y.mean():.4f}")

    # sampel latih tersebar merata menurut waktu; sampel uji acak; keduanya sama untuk semua model
    step = max(len(train) // MAX_TRAIN, 1)
    trn = train.iloc[::step]
    tst = test.sample(min(len(test), MAX_TEST), random_state=0).sort_values("t0")
    ys = tst.y.values
    days = tst.t0.dt.date.astype(str).values
    prev = float(ys.mean())
    print(f"Latih: {len(trn):,} session | Uji: {len(tst):,} session | proporsi beli uji {prev:.4f}")
    rows.append({"K": k, "model": "Acak (rujukan)", "ROC-AUC": 0.5, "AUC-harian": 0.5,
                 "PR-AUC": round(prev, 4), "Precision@5%": round(prev, 4)})

    # --- Model offline (sklearn), pembanding ---
    sc = StandardScaler().fit(trn[FEATS])
    lr = LogisticRegression(max_iter=300, class_weight="balanced")
    lr.fit(sc.transform(trn[FEATS]), trn.y)
    s_off = lr.predict_proba(sc.transform(tst[FEATS]))[:, 1]
    rows.append({"K": k, **score("Offline LogReg (sklearn)", ys, s_off, days)})

    # --- Model online (River), belajar berurutan menurut waktu ---
    if args.tune:
        variants = [
            ("lr=0.01 tetap", optim.SGD(0.01)),
            ("lr=0.001 tetap", optim.SGD(0.001)),
            ("lr menurun bertahap", optim.SGD(optim.schedulers.InverseScaling(0.1, 0.5))),
        ]
        print("Koefisien sklearn (fitur terstandar):",
              {f: round(float(w), 3) for f, w in zip(FEATS, lr.coef_[0])})
    else:
        variants = [("", optim.SGD(optim.schedulers.InverseScaling(0.1, 0.5)))]

    xs = tst[FEATS].to_dict("records")
    xtr = trn[FEATS].to_dict("records")
    ytr = trn.y.values
    for label, opt in variants:
        tag = f" [{label}]" if label else ""
        model = preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=opt)
        for x, y in zip(xtr, ytr):
            model.learn_one(x, bool(y))
        if args.tune:
            w = model["LogisticRegression"].weights
            print(f"Bobot River{tag}:", {f: round(float(w.get(f, 0.0)), 3) for f in FEATS})
        s_frozen = np.array([model.predict_proba_one(x).get(True, 0.5) for x in xs])
        rows.append({"K": k, **score(f"River beku{tag}", ys, s_frozen, days)})

        s_preq = []
        for x, y in zip(xs, ys):
            s_preq.append(model.predict_proba_one(x).get(True, 0.5))
            model.learn_one(x, bool(y))
        rows.append({"K": k, **score(f"River terus belajar{tag}", ys, np.array(s_preq), days)})
    print(pd.DataFrame([r for r in rows if r["K"] == k]).drop(columns="K").to_string(index=False))
    print()

if rows:
    out = pd.DataFrame(rows)
    out.to_csv("results/offline_evaluation.csv", index=False)
    print("Ringkasan semua K disimpan di results/offline_evaluation.csv")
