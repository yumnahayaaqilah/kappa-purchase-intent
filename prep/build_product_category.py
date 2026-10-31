"""
Build a product_id -> category reference table from October-November 2019, the months whose
category_code is consistent, then measure how much of the sample it can label.

    python -m prep.build_product_category

Output: ref/product_category.parquet (product_id, category_code, category_top)
        results/report_category_remap.csv (coverage per month)
"""
import argparse
import os

import duckdb
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--source", nargs="+", default=["data/2019-Oct.csv", "data/2019-Nov.csv"])
ap.add_argument("--sample", default="sample/sample_10pct_7bulan.csv")
ap.add_argument("--out", default="ref/product_category.parquet")
ap.add_argument("--outdir", default="results")
args = ap.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
os.makedirs(args.outdir, exist_ok=True)
os.makedirs("duck_tmp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA memory_limit='8GB'")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA temp_directory='duck_tmp'")

src = ", ".join(f"'{f}'" for f in args.source)
con.execute(f"""
    CREATE TEMP TABLE codes AS
    SELECT CAST(product_id AS BIGINT) AS product_id, category_code AS code, count(*) AS n
    FROM read_csv([{src}], header=true, all_varchar=true)
    WHERE category_code IS NOT NULL AND category_code <> ''
    GROUP BY ALL
""")
con.execute("""
    CREATE TEMP TABLE ref AS
    SELECT product_id,
           arg_max(code, n) AS category_code,
           split_part(arg_max(code, n), '.', 1) AS category_top,
           count(*) AS n_codes
    FROM codes GROUP BY product_id
""")
s = con.execute("""
    SELECT count(*) AS products_mapped,
           count(*) FILTER (WHERE n_codes > 1) AS products_with_several_codes
    FROM ref
""").df()
print(s.T.to_string(header=False))
con.execute(f"COPY (SELECT product_id, category_code, category_top FROM ref) TO '{args.out}' (FORMAT PARQUET)")
print(f"Saved {args.out}")

cov = con.execute(f"""
    SELECT left(e.event_time, 7) AS month,
           count(*) AS events,
           avg((e.category_code IS NOT NULL AND e.category_code <> '')::INT) AS share_with_original_code,
           avg((r.product_id IS NOT NULL)::INT) AS share_with_remapped_code
    FROM read_csv('{args.sample}', header=true, all_varchar=true) e
    LEFT JOIN ref r ON r.product_id = CAST(e.product_id AS BIGINT)
    GROUP BY month ORDER BY month
""").df()
cov.to_csv(os.path.join(args.outdir, "report_category_remap.csv"), index=False)
pd.set_option("display.width", 200)
print("\nShare of sample events with a category, original vs remapped:")
print(cov.round(4).to_string(index=False))