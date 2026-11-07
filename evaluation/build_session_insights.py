"""
Build the tables for the extra analyses (P2, P5 to P8) from the sample, in one pass with DuckDB.
Session rule and the first K events are the same as build_session_reference.py (sample-file order,
a gap of more than 30 minutes ends the session, later events are ignored).

    python -m evaluation.build_session_insights

Output:
  ref/session_insights.parquet   one row per session (all sessions, not only scored ones)
  ref/event_aggregates.parquet   event counts per month x category x brand x hour x weekday x event type
                                 (all events, like the stream's category_insights)
"""
import argparse
import os
import time

import duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="sample/sample_10pct_7bulan.csv")
ap.add_argument("--categories", default="ref/product_category.parquet")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--timeout-min", type=int, default=30)
ap.add_argument("--out-sessions", default="ref/session_insights.parquet")
ap.add_argument("--out-events", default="ref/event_aggregates.parquet")
ap.add_argument("--memory", default="6GB")
ap.add_argument("--threads", type=int, default=2)
args = ap.parse_args()
K, GAP = args.k, args.timeout_min * 60

t0 = time.time()
os.makedirs(os.path.dirname(args.out_sessions), exist_ok=True)
os.makedirs("duck_tmp", exist_ok=True)
# on-disk working database, so large intermediate tables can leave memory (an in-memory one ran out at 8 GB)
DBFILE = "duck_tmp/insights_work.duckdb"
for f in (DBFILE, DBFILE + ".wal"):
    if os.path.exists(f):
        os.remove(f)
con = duckdb.connect(DBFILE)
con.execute(f"PRAGMA memory_limit='{args.memory}'")
con.execute(f"PRAGMA threads={args.threads}")
con.execute("PRAGMA temp_directory='duck_tmp'")

con.execute(f"""
    CREATE TABLE ev AS
    SELECT row_number() OVER () AS rn, user_session AS sid, user_id AS uid, event_type AS et,
           CAST(replace(event_time, ' UTC', '') AS TIMESTAMP) AS t,
           TRY_CAST(price AS DOUBLE) AS price,
           TRY_CAST(product_id AS BIGINT) AS pid,
           nullif(category_id, '') AS cid,
           nullif(lower(brand), '') AS brand
    FROM read_csv('{args.sample}', header=true, all_varchar=true)
""")
con.execute("SET preserve_insertion_order=false")
con.execute(f"""
    CREATE TABLE ev2 AS
    SELECT e.*, coalesce(c.category_top, 'unknown') AS cat
    FROM ev e LEFT JOIN read_parquet('{args.categories}') c ON c.product_id = e.pid
""")
con.execute("DROP TABLE ev")
print(f"Loaded events ({time.time() - t0:.0f}s)")

# ---- event aggregates (all events)
con.execute(f"""
    COPY (
        SELECT strftime(t, '%Y-%m') AS month, cat AS category, coalesce(brand, 'unknown') AS brand,
               hour(t) AS hour_utc, isodow(t) AS dow, et AS event_type,
               count(*) AS events, sum(CASE WHEN price > 0 THEN price ELSE 0 END) AS value
        FROM ev2 GROUP BY ALL
    ) TO '{args.out_events}' (FORMAT PARQUET)
""")
print(f"Saved {args.out_events} ({time.time() - t0:.0f}s)")

# ---- sessions (gap rule, as in build_session_reference.py)
con.execute(f"""
    CREATE TABLE seq AS
    SELECT sid, uid, et, t, price, pid, cid, cat, brand,
           row_number() OVER w AS k,
           coalesce(epoch(t) - epoch(lag(t) OVER w), 0) > {GAP} AS is_gap
    FROM ev2 WHERE sid IS NOT NULL AND sid <> ''
    WINDOW w AS (PARTITION BY sid ORDER BY rn)
""")
con.execute("DROP TABLE ev2")
con.execute("""
    CREATE TABLE s AS
    WITH b AS (SELECT sid, min(k) FILTER (WHERE is_gap) AS brk FROM seq GROUP BY sid)
    SELECT q.* FROM seq q JOIN b USING (sid) WHERE b.brk IS NULL OR q.k < b.brk
""")
con.execute("DROP TABLE seq")
print(f"Built sessions ({time.time() - t0:.0f}s)")

# split into smaller aggregations; one big query with several DISTINCT counts ran out of memory
con.execute(f"""
    CREATE TABLE a_all AS
    SELECT sid, arg_min(uid, k) AS user_id, count(*) AS n, max(t) AS t_last,
           bool_or(et = 'purchase' AND k <= {K}) AS buy_k,
           bool_or(et = 'purchase' AND k > {K}) AS y,
           bool_or(et = 'cart') AS has_cart,
           bool_or(et = 'purchase') AS has_purchase,
           sum(CASE WHEN et = 'purchase' AND price > 0 THEN price ELSE 0 END) AS purchase_value,
           sum(CASE WHEN et = 'purchase' AND k > {K} AND price > 0 THEN price ELSE 0 END) AS purchase_value_after_k
    FROM s GROUP BY sid
""")
print(f"  session totals ({time.time() - t0:.0f}s)")
con.execute(f"""
    CREATE TABLE a_k AS
    SELECT sid,
           min(t) FILTER (WHERE k = 1) AS t1,
           min(t) FILTER (WHERE k = {K}) AS tk,
           count(*) FILTER (WHERE et = 'cart') AS c_k,
           count(DISTINCT cid) AS ncat_id_k,
           count(DISTINCT cat) FILTER (WHERE cat <> 'unknown') AS ncat_top_k,
           count(*) FILTER (WHERE cat = 'unknown') AS n_unknown_cat_k,
           count(DISTINCT pid) AS nprod_k,
           avg(price) FILTER (WHERE price > 0) AS avg_price_k,
           max(price) FILTER (WHERE price > 0) AS max_price_k
    FROM s WHERE k <= {K} GROUP BY sid
""")
print(f"  first {K} events ({time.time() - t0:.0f}s)")
con.execute("CREATE TABLE sc AS SELECT sid, pid, price, cat, brand, t, k FROM s WHERE et = 'cart'")
con.execute("DROP TABLE s")
con.execute("""
    CREATE TABLE a_cart AS
    WITH carted AS (       -- each carted product once per session, at its first cart price
        SELECT sid, pid, arg_min(price, k) AS price FROM sc GROUP BY sid, pid
    ),
    cv AS (SELECT sid, count(*) AS n_cart_products,
                  sum(CASE WHEN price > 0 THEN price ELSE 0 END) AS cart_value
           FROM carted GROUP BY sid),
    fc AS (SELECT sid, arg_min(t, k) AS t_first_cart, arg_min(cat, k) AS cat_first_cart,
                  arg_min(coalesce(brand, 'unknown'), k) AS brand_first_cart
           FROM sc GROUP BY sid)
    SELECT * FROM cv JOIN fc USING (sid)
""")
print(f"  carts ({time.time() - t0:.0f}s)")
con.execute(f"""
    COPY (
        SELECT a.sid, a.user_id, a.n, k.t1, k.tk, a.t_last, a.buy_k, a.y, a.has_cart, a.has_purchase,
               k.c_k, k.ncat_id_k, k.ncat_top_k, k.n_unknown_cat_k, k.nprod_k, k.avg_price_k, k.max_price_k,
               a.purchase_value, a.purchase_value_after_k,
               coalesce(c.n_cart_products, 0) AS n_cart_products, coalesce(c.cart_value, 0) AS cart_value,
               c.t_first_cart, c.cat_first_cart, c.brand_first_cart,
               (a.n >= {K} AND NOT a.buy_k) AS scored
        FROM a_all a JOIN a_k k USING (sid) LEFT JOIN a_cart c USING (sid)
    ) TO '{args.out_sessions}' (FORMAT PARQUET)
""")
n, sc, pos, buy = con.execute(f"""
    SELECT count(*), count(*) FILTER (WHERE scored), count(*) FILTER (WHERE scored AND y),
           count(*) FILTER (WHERE has_purchase)
    FROM '{args.out_sessions}'""").fetchone()
print(f"Saved {args.out_sessions}: {n:,} sessions | scored {sc:,} (purchase after event {K}: {pos:,}) | "
      f"sessions with a purchase {buy:,} ({time.time() - t0:.0f}s)")
con.close()
for f in (DBFILE, DBFILE + ".wal"):
    if os.path.exists(f):
        os.remove(f)
