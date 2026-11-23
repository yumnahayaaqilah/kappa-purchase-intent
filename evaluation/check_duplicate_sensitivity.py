"""
Uji sensitivitas: apakah baris duplikat persis mengubah hasil model?

Membandingkan dua kunci jawaban DuckDB:
  - asli  : ref/session_reference_7m.parquet        (duplikat dipertahankan, sama dengan pipeline)
  - dedup : ref/session_reference_7m_dedup.parquet  (duplikat persis dibuang dulu)

Untuk masing-masing: pembagian waktu yang sama (titik potong 70% dari versi asli), logistic regression
scikit-learn dilatih sekali pada periode awal (sama seperti full_evaluation), lalu diuji pada periode uji.
Tidak perlu Spark, Kafka, atau PostgreSQL.

    python -m evaluation.check_duplicate_sensitivity

Keluaran: results/duplicate_sensitivity.csv
"""
import argparse
import os

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="ref/session_reference_7m.parquet")
ap.add_argument("--ref-dedup", default="ref/session_reference_7m_dedup.parquet")
ap.add_argument("--train-frac", type=float, default=0.7)
ap.add_argument("--max-train", type=int, default=1_000_000)
ap.add_argument("--top-pct", type=float, default=5.0)
ap.add_argument("--outdir", default="results")
args = ap.parse_args()

FEATS = ["f_c", "f_avg_price", "f_max_price", "f_dur"]
for p in (args.ref, args.ref_dedup):
    if not os.path.exists(p):
        raise SystemExit(f"File tidak ada: {p}. Jalankan evaluation.build_session_reference dulu.")


def load(path):
    df = duckdb.sql(f"SELECT session_id, t_pred, {', '.join(FEATS)}, y FROM '{path}' "
                    f"ORDER BY t_pred, session_id").df()
    df["t_pred"] = pd.to_datetime(df["t_pred"])
    return df


asli, dedup = load(args.ref), load(args.ref_dedup)
cut = asli["t_pred"].iloc[int(len(asli) * args.train_frac)]        # titik potong yang sama untuk keduanya
print(f"Titik potong periode uji: {cut}\n")

# ------------------------------------------------------------ sesi yang berubah
m = asli.merge(dedup, on="session_id", how="outer", suffixes=("_a", "_d"), indicator=True)
both = m[m["_merge"] == "both"]
beda_fitur = (~np.isclose(both[[f + "_a" for f in FEATS]].to_numpy(),
                          both[[f + "_d" for f in FEATS]].to_numpy())).any(axis=1)
beda_label = (both["y_a"] != both["y_d"]).to_numpy()
print("Perbandingan sesi:")
print(f"  sesi dinilai, asli : {len(asli):,}")
print(f"  sesi dinilai, dedup: {len(dedup):,}")
print(f"  hanya di asli      : {(m['_merge'] == 'left_only').sum():,}")
print(f"  hanya di dedup     : {(m['_merge'] == 'right_only').sum():,}")
print(f"  fitur berbeda      : {beda_fitur.sum():,}")
print(f"  label berbeda      : {beda_label.sum():,}\n")


# ------------------------------------------------------------ kinerja model
def evaluate(df, name):
    awal, uji = df[df["t_pred"] < cut], df[df["t_pred"] >= cut]
    tr = awal.iloc[:: max(1, len(awal) // args.max_train)]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[FEATS], tr["y"])
    y = uji["y"].to_numpy()
    p = model.predict_proba(uji[FEATS])[:, 1]
    k = max(int(len(y) * args.top_pct / 100), 1)
    top = y[np.argsort(-p, kind="stable")[:k]].mean()
    return {
        "versi": name,
        "sesi_uji": len(uji),
        "proporsi_beli_uji": round(float(y.mean()), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        f"precision_at_{args.top_pct:g}pct": round(float(top), 4),
        "roc_auc_aturan_keranjang": round(roc_auc_score(y, uji["f_c"].to_numpy()), 4),
    }


res = pd.DataFrame([evaluate(asli, "asli (duplikat dipertahankan)"),
                    evaluate(dedup, "dedup (duplikat dibuang)")])
selisih = {"versi": "selisih (dedup - asli)"}
for c in res.columns[1:]:
    selisih[c] = round(float(res[c].iloc[1] - res[c].iloc[0]), 4)
res = pd.concat([res, pd.DataFrame([selisih])], ignore_index=True)
res["sesi_uji"] = res["sesi_uji"].astype(int)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print(res.to_string(index=False))
os.makedirs(args.outdir, exist_ok=True)
res.to_csv(os.path.join(args.outdir, "duplicate_sensitivity.csv"), index=False)
print(f"\nTersimpan: {args.outdir}/duplicate_sensitivity.csv")
