"""
Check the descriptive query: counts per category and pipeline totals in PostgreSQL must equal
a batch recount with DuckDB over the same events (first --limit rows of the sample).

    python -m evaluation.check_descriptive --limit 300000
"""
import argparse
import sys

import duckdb
import pandas as pd
import psycopg2

from db_config import DB

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="sample/sample_10pct_7bulan.csv")
ap.add_argument("--limit", type=int, default=0, help="first N rows, as producer --limit (0 = all)")
ap.add_argument("--categories", default="ref/product_category.parquet")
args = ap.parse_args()

limit = f"LIMIT {args.limit}" if args.limit else ""
con = duckdb.connect()
con.execute(f"""
    CREATE TEMP TABLE ev AS
    SELECT CAST(product_id AS BIGINT) AS product_id, event_type,
           CAST(replace(event_time, ' UTC', '') AS TIMESTAMP) AS t
    FROM (SELECT * FROM read_csv('{args.sample}', header=true, all_varchar=true) {limit})
""")
ref = con.execute(f"""
    SELECT coalesce(c.category_top, 'unknown') AS category,
           count(*) FILTER (WHERE event_type = 'view') AS views,
           count(*) FILTER (WHERE event_type = 'cart') AS carts,
           count(*) FILTER (WHERE event_type = 'purchase') AS purchases
    FROM ev LEFT JOIN read_parquet('{args.categories}') c USING (product_id)
    GROUP BY 1
""").df()
n_ref, t_ref = con.execute("SELECT count(*), max(t) FROM ev").fetchone()

conn = psycopg2.connect(**DB)
with conn.cursor() as cur:
    cur.execute("SELECT category, views, carts, purchases FROM category_insights")
    got = pd.DataFrame(cur.fetchall(), columns=["category", "views", "carts", "purchases"])
    cur.execute("SELECT events_total, last_event_time AT TIME ZONE 'UTC' FROM pipeline_stats WHERE id = 1")
    n_got, t_got = cur.fetchone() or (None, None)
conn.close()

m = ref.merge(got, on="category", how="outer", suffixes=("_ref", "_db")).fillna(0)
bad = False
for c in ["views", "carts", "purchases"]:
    bad = bad | (m[f"{c}_ref"] != m[f"{c}_db"])
diff = m[bad]

pd.set_option("display.width", 200)
print(f"Events: reference {n_ref:,} | pipeline {n_got or 0:,}")
print(f"Last event time (UTC): reference {t_ref} | pipeline {t_got}")
print(f"Categories: reference {len(ref)} | pipeline {len(got)} | with different counts {len(diff)}")
if len(diff):
    print(diff.to_string(index=False))
ok = len(diff) == 0 and n_ref == n_got and pd.Timestamp(t_ref) == pd.Timestamp(t_got)
print("\nRESULT:", "MATCH" if ok else "DIFFERENCES")
sys.exit(0 if ok else 1)
