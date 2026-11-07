"""
Check whether cart events were recorded completely: daily views, carts and purchases in the sample.
A day with fewer carts than purchases (cart/purchase < 1) suggests carts were not logged in full.

    python -m evaluation.check_cart_logging                  # Oct to Nov 2019
    python -m evaluation.check_cart_logging --all            # every day of the sample

Output: results/check_cart_logging_daily.csv
"""
import argparse
import os

import duckdb
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="sample/sample_10pct_7bulan.csv")
ap.add_argument("--all", action="store_true")
ap.add_argument("--outdir", default="results")
args = ap.parse_args()

os.makedirs("duck_tmp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA memory_limit='4GB'")
con.execute("PRAGMA threads=2")
con.execute("PRAGMA temp_directory='duck_tmp'")
where = "" if args.all else "WHERE left(event_time, 7) IN ('2019-10', '2019-11')"
d = con.execute(f"""
    SELECT left(event_time, 10) AS tanggal,
           count(*) FILTER (WHERE event_type = 'view') AS lihat,
           count(*) FILTER (WHERE event_type = 'cart') AS keranjang,
           count(*) FILTER (WHERE event_type = 'purchase') AS beli
    FROM read_csv('{args.sample}', header=true, all_varchar=true) {where}
    GROUP BY 1 ORDER BY 1
""").df()
d["keranjang_per_beli"] = (d.keranjang / d.beli.where(d.beli > 0)).round(2)
d["persen_keranjang_dari_lihat"] = (100 * d.keranjang / d.lihat).round(2)
os.makedirs(args.outdir, exist_ok=True)
path = os.path.join(args.outdir, "check_cart_logging_daily.csv")
d.to_csv(path, index=False)
pd.set_option("display.max_rows", 400)
print(d.to_string(index=False))
low = d[d.keranjang_per_beli < 1]
print(f"\nHari dengan keranjang lebih sedikit dari pembelian: {len(low)} dari {len(d)}"
      + (f" ({low.tanggal.min()} sampai {low.tanggal.max()})" if len(low) else ""))
print(f"Saved {path}")
