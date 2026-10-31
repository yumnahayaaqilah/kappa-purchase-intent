"""
Check whether the same product_id carries a different category_code before and after
December 2019 (most frequent code per product in each period).

    python -m prep.check_category_shift

Output: hasil/report_category_shift.csv (top code transitions)
"""
import argparse
import os

import duckdb
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="sample/sample_10pct_7bulan.csv")
ap.add_argument("--split", default="2019-12", help="first month of the second period")
ap.add_argument("--outdir", default="hasil")
args = ap.parse_args()

con = duckdb.connect()
con.execute("PRAGMA threads=4")
SRC = f"""read_csv('{args.sample}', header=true, all_varchar=true)"""
per = f"""
    SELECT product_id, mode(category_code) AS code, mode(category_id) AS cid, count(*) AS n
    FROM {SRC}
    WHERE category_code IS NOT NULL AND category_code <> '' AND left(event_time, 7) {{op}} '{args.split}'
    GROUP BY product_id"""
con.execute(f"CREATE TEMP TABLE a AS {per.format(op='<')}")
con.execute(f"CREATE TEMP TABLE b AS {per.format(op='>=')}")

s = con.execute("""
    SELECT count(*) AS products_in_both,
           count(*) FILTER (WHERE a.code <> b.code) AS code_changed,
           count(*) FILTER (WHERE a.cid <> b.cid) AS category_id_changed,
           sum(b.n) FILTER (WHERE a.code <> b.code) AS later_events_with_changed_code,
           sum(b.n) AS later_events_total
    FROM a JOIN b USING (product_id)
""").df()
print(s.T.to_string(header=False))

tr = con.execute("""
    SELECT a.code AS code_before, b.code AS code_after, count(*) AS products, sum(b.n) AS events_after
    FROM a JOIN b USING (product_id)
    WHERE a.code <> b.code
    GROUP BY ALL ORDER BY events_after DESC LIMIT 15
""").df()
os.makedirs(args.outdir, exist_ok=True)
tr.to_csv(os.path.join(args.outdir, "report_category_shift.csv"), index=False)
pd.set_option("display.width", 200)
print("\nTop code transitions (before -> after):")
print(tr.to_string(index=False))