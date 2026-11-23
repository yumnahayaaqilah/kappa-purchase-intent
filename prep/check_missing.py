"""
Hitung nilai kosong per kolom pada SEMUA file bulanan di data/ (CSV biasa dan .csv.gz),
total dan per bulan.

    python -m prep.check_missing
    python -m prep.check_missing --glob "data/20*.csv" "data/20*.csv.gz"

Keluaran: results/missing_values.csv (total), results/missing_values_monthly.csv (per bulan)
"""
import argparse
import glob
import os
import re
import sys

import duckdb
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--glob", nargs="+", default=["data/20*.csv", "data/20*.csv.gz"])
ap.add_argument("--outdir", default="results")
args = ap.parse_args()

files = sorted({f for g in args.glob for f in glob.glob(g)})
if not files:
    sys.exit(f"Tidak ada file yang cocok dengan {args.glob}")

# satu bulan tidak boleh terbaca dua kali (misalnya 2020-Mar.csv dan 2020-Mar.csv.gz)
base = [re.sub(r"\.csv(\.gz)?$", "", os.path.basename(f)) for f in files]
dobel = sorted({b for b in base if base.count(b) > 1})
if dobel:
    sys.exit(f"Bulan yang sama ada dalam dua file: {dobel}. Hapus salah satu atau atur --glob.")

print("File yang dibaca:")
for f in files:
    print("  ", f)

os.makedirs(args.outdir, exist_ok=True)
os.makedirs("duck_tmp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='8GB'")
con.execute("PRAGMA temp_directory='duck_tmp'")

lst = ", ".join(f"'{f}'" for f in files)
SRC = f"""read_csv([{lst}], header=true, columns={{
    'event_time':'VARCHAR','event_type':'VARCHAR','product_id':'BIGINT',
    'category_id':'BIGINT','category_code':'VARCHAR','brand':'VARCHAR',
    'price':'DOUBLE','user_id':'BIGINT','user_session':'VARCHAR'}})"""

CHECKS = {
    "event_time_kosong": "event_time IS NULL OR event_time = ''",
    "event_type_kosong": "event_type IS NULL OR event_type = ''",
    "product_id_kosong": "product_id IS NULL",
    "category_id_kosong": "category_id IS NULL",
    "category_code_kosong": "category_code IS NULL OR category_code = ''",
    "brand_kosong": "brand IS NULL OR brand = ''",
    "price_kosong": "price IS NULL",
    "price_nol_atau_negatif": "price <= 0",
    "user_id_kosong": "user_id IS NULL",
    "user_session_kosong": "user_session IS NULL OR user_session = ''",
}
agg = ",\n           ".join(f"count(*) FILTER (WHERE {cond}) AS {name}" for name, cond in CHECKS.items())

monthly = con.execute(f"""
    SELECT left(event_time, 7) AS bulan,
           count(*) AS total_baris,
           {agg}
    FROM {SRC}
    GROUP BY bulan ORDER BY bulan
""").df()

# total = jumlah dari semua bulan
total = int(monthly["total_baris"].sum())
res = pd.DataFrame([{"kolom": c, "jumlah": int(monthly[c].sum()),
                     "persen": round(100.0 * monthly[c].sum() / total, 3)} for c in CHECKS])

pct = monthly.copy()
for c in CHECKS:
    pct[c] = (100.0 * pct[c] / pct["total_baris"]).round(3)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print(f"\nTotal baris: {total:,} dari {len(monthly)} bulan\n")
print(res.to_string(index=False))
print("\nPersen kosong per bulan:")
print(pct.to_string(index=False))

res.to_csv(os.path.join(args.outdir, "missing_values.csv"), index=False)
monthly.to_csv(os.path.join(args.outdir, "missing_values_monthly.csv"), index=False)
print(f"\nTersimpan: {args.outdir}/missing_values.csv, {args.outdir}/missing_values_monthly.csv")
