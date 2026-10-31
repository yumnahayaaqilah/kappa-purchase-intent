"""
Evaluasi lengkap dari eval_log (hasil pipeline). Tidak perlu Spark/Kafka, hanya PostgreSQL yang aktif.

    python -m evaluation.full_evaluation
    python -m evaluation.full_evaluation --train-frac 0.7 --thr-pct 95 --top-pct 5

Protokol:
  - Urutkan sesi menurut waktu prediksi. Bagian awal (train-frac) = "periode awal", sisanya = "periode uji".
  - Ambang keputusan biner = persentil thr-pct dari skor pada periode awal (ditetapkan SEBELUM melihat periode uji).
  - Skor River adalah skor yang dibuat pipeline saat itu (prequential), bukan dihitung ulang.
  - Pembanding sklearn dilatih sekali pada periode awal lalu dibekukan.
Keluaran: results/metrics.csv, results/calibration.csv, results/ablation.csv, results/monthly.csv
"""
import argparse

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from db_config import DB

ap = argparse.ArgumentParser()
ap.add_argument("--train-frac", type=float, default=0.7, help="porsi awal (menurut waktu) sebagai periode awal")
ap.add_argument("--thr-pct", type=float, default=95.0, help="persentil skor periode awal untuk ambang")
ap.add_argument("--top-pct", type=float, default=5.0, help="persen sesi teratas untuk Precision/Recall@k")
ap.add_argument("--max-train", type=int, default=1_000_000, help="maks baris latih sklearn")
args = ap.parse_args()

FEATS = ["f_c", "f_avg_price", "f_max_price", "f_dur"]
NAMA = {"f_c": "jumlah keranjang", "f_avg_price": "harga rata-rata", "f_max_price": "harga tertinggi", "f_dur": "durasi"}
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# ------------------------------------------------------------------ data
conn = psycopg2.connect(**DB)
with conn.cursor() as cur:
    cur.execute("SELECT t_pred, f_c, f_avg_price, f_max_price, f_dur, prob, y FROM eval_log ORDER BY t_pred")
    df = pd.DataFrame(cur.fetchall(), columns=["t_pred", *FEATS, "prob", "y"])
conn.close()
if len(df) < 1000:
    raise SystemExit(f"Baris eval_log terlalu sedikit ({len(df)}). Jalankan pipeline dulu.")
df["t_pred"] = pd.to_datetime(df["t_pred"], utc=True)
for c in FEATS + ["prob"]:
    df[c] = df[c].astype(float)
df["y"] = df["y"].astype(int)
df["hari"] = df["t_pred"].dt.strftime("%Y-%m-%d")
df["bulan"] = df["t_pred"].dt.strftime("%Y-%m")

cut = int(len(df) * args.train_frac)
awal, uji = df.iloc[:cut].copy(), df.iloc[cut:].copy()
base = float(awal["y"].mean())
print(f"Sesi: {len(df):,} | periode awal {len(awal):,} (sampai {awal.t_pred.iloc[-1]}) | "
      f"periode uji {len(uji):,} (mulai {uji.t_pred.iloc[0]}) | proporsi beli awal {base:.4f}, uji {uji.y.mean():.4f}")

thr = float(np.percentile(awal["prob"], args.thr_pct))
print(f"Ambang keputusan = persentil ke-{args.thr_pct:g} skor periode awal = {thr:.4f}\n")


# ------------------------------------------------------------------ fungsi
def topk_stats(y, p, pct):
    n = len(y)
    k = max(int(n * pct / 100), 1)
    idx = np.argsort(-p, kind="stable")[:k]
    prec = float(y[idx].mean())
    rec = float(y[idx].sum() / max(y.sum(), 1))
    return prec, rec


def daily(d, scorecol):
    """AUC, PR-AUC, Precision@k rata-rata harian berbobot (hari dengan >=50 sesi dan dua kelas)."""
    rows = []
    for _, g in d.groupby("hari"):
        if len(g) >= 50 and g["y"].nunique() == 2:
            y, p = g["y"].to_numpy(), g[scorecol].to_numpy()
            rows.append((len(g), roc_auc_score(y, p), average_precision_score(y, p), topk_stats(y, p, args.top_pct)[0]))
    if not rows:
        return (np.nan,) * 3
    a = np.array(rows, float)
    w = a[:, 0] / a[:, 0].sum()
    return tuple((a[:, i] * w).sum() for i in (1, 2, 3))


def ringkas(nama, d, scorecol, thr_value, prob_ok=True):
    y, p = d["y"].to_numpy(), d[scorecol].to_numpy()
    auc, pr = roc_auc_score(y, p), average_precision_score(y, p)
    dauc, dpr, dp5 = daily(d, scorecol)
    prec_k, rec_k = topk_stats(y, p, args.top_pct)
    yhat = (p >= thr_value).astype(int)
    out = {
        "model": nama,
        "ROC-AUC": auc, "AUC-harian": dauc, "PR-AUC": pr,
        f"Precision@{args.top_pct:g}%": prec_k, f"Recall@{args.top_pct:g}%": rec_k,
        f"Lift@{args.top_pct:g}%": prec_k / y.mean(),
        "Precision(ambang)": precision_score(y, yhat, zero_division=0),
        "Recall(ambang)": recall_score(y, yhat, zero_division=0),
        "F1(ambang)": f1_score(y, yhat, zero_division=0),
        "disaring(%)": 100 * yhat.mean(),
    }
    if prob_ok:
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        out["Brier"] = brier_score_loss(y, pc)
        out["LogLoss"] = log_loss(y, pc)
    else:
        out["Brier"] = out["LogLoss"] = np.nan
    return out, yhat


# ------------------------------------------------------------------ pembanding
tr = awal.iloc[:: max(1, len(awal) // args.max_train)]
sk = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
sk.fit(tr[FEATS], tr["y"])
uji["p_sk"] = sk.predict_proba(uji[FEATS])[:, 1]
awal["p_sk"] = sk.predict_proba(awal[FEATS])[:, 1]
thr_sk = float(np.percentile(awal["p_sk"], args.thr_pct))
uji["p_acak"] = np.random.default_rng(0).random(len(uji))
uji["p_cart"] = uji["f_c"].clip(upper=1.0) + 1e-6 * uji["f_avg_price"]  # aturan: ada keranjang atau tidak

res, yh_river = [], None
r, yh_river = ringkas("River (pipeline, prequential)", uji, "prob", thr)
res.append(r)
r, yh_sk = ringkas("sklearn (dilatih sekali pada periode awal)", uji, "p_sk", thr_sk)
res.append(r)
r, _ = ringkas("Aturan sederhana: ada keranjang", uji, "p_cart", 1.0, prob_ok=False)
res.append(r)
r, _ = ringkas("Acak", uji, "p_acak", float(np.percentile(uji["p_acak"], args.thr_pct)), prob_ok=False)
res.append(r)
# acuan Brier untuk "selalu menebak proporsi beli"
brier_const = brier_score_loss(uji["y"], np.full(len(uji), base))
tab = pd.DataFrame(res)
tab.loc[tab.model == "Acak", ["Brier", "LogLoss"]] = [brier_const, log_loss(uji["y"], np.full(len(uji), base))]
tab.to_csv("results/metrics.csv", index=False)

print("=== 1. METRIK PADA PERIODE UJI ===")
print(tab.round(4).to_string(index=False))
print(f"(Baris Acak: Brier/LogLoss = menebak proporsi beli periode awal {base:.4f} untuk semua sesi)\n")

cm = confusion_matrix(uji["y"], yh_river)
print(f"=== 2. CONFUSION MATRIX River pada ambang {thr:.4f} (periode uji) ===")
print(pd.DataFrame(cm, index=["aktual tidak beli", "aktual beli"], columns=["tebak tidak beli", "tebak beli"]).to_string(), "\n")

# ------------------------------------------------------------------ kalibrasi
uji["bin"] = pd.qcut(uji["prob"].rank(method="first"), 10, labels=False)
kal = uji.groupby("bin").agg(sesi=("y", "size"), rata_skor=("prob", "mean"), proporsi_beli_nyata=("y", "mean")).reset_index()
kal["selisih"] = kal["rata_skor"] - kal["proporsi_beli_nyata"]
ece = float((kal["sesi"] / kal["sesi"].sum() * kal["selisih"].abs()).sum())
kal.to_csv("results/calibration.csv", index=False)
print("=== 3. KALIBRASI River (10 kelompok sama besar, periode uji) ===")
print(kal.round(4).to_string(index=False))
print(f"ECE (rata-rata selisih berbobot) = {ece:.4f}  | selisih>0: skor terlalu tinggi, <0: terlalu rendah\n")

# ------------------------------------------------------------------ ablasi
print("=== 4. ABLASI FITUR (sklearn, dilatih pada periode awal, diuji pada periode uji) ===")
subsets = {"semua fitur": FEATS}
for f in FEATS:
    subsets[f"hanya {NAMA[f]}"] = [f]
    subsets[f"tanpa {NAMA[f]}"] = [x for x in FEATS if x != f]
subsets["keranjang + harga"] = ["f_c", "f_avg_price", "f_max_price"]
rows = []
for nama, cols in subsets.items():
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[cols], tr["y"])
    p = m.predict_proba(uji[cols])[:, 1]
    y = uji["y"].to_numpy()
    pk, rk = topk_stats(y, p, args.top_pct)
    rows.append({"fitur": nama, "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
                 f"Precision@{args.top_pct:g}%": pk, f"Recall@{args.top_pct:g}%": rk})
abl = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
abl.to_csv("results/ablation.csv", index=False)
print(abl.round(4).to_string(index=False), "\n")

# ------------------------------------------------------------------ bulanan
print("=== 5. PER BULAN: River dibanding sklearn yang dibekukan pada periode awal ===")
sub_mask = df["t_pred"] > awal["t_pred"].iloc[-1]
df["p_sk"] = sk.predict_proba(df[FEATS])[:, 1]
rows = []
for b, g in df.groupby("bulan"):
    row = {"bulan": b, "sesi": len(g), "proporsi_beli": g["y"].mean(),
           "River_AUC_semua": roc_auc_score(g["y"], g["prob"]) if g["y"].nunique() == 2 else np.nan}
    gs = g[sub_mask.loc[g.index]]
    row["sesi_periode_uji"] = len(gs)
    if len(gs) >= 200 and gs["y"].nunique() == 2:
        row["River_AUC_uji"] = roc_auc_score(gs["y"], gs["prob"])
        row["sklearn_AUC_uji"] = roc_auc_score(gs["y"], gs["p_sk"])
    rows.append(row)
bul = pd.DataFrame(rows)
bul.to_csv("results/monthly.csv", index=False)
print(bul.round(4).to_string(index=False))
print("\nSelesai. Berkas: results/metrics.csv, results/calibration.csv, results/ablation.csv, results/monthly.csv")
