"""
Reference ("answer key") for the stream processor, computed in batch with DuckDB.

Session rule: events of a user_session in sample-file order (the order the producer sends);
a gap of more than 30 minutes ends the session and later events are ignored. A session is
scored when it reaches the K-th event without a purchase among the first K events.
Features match make_features() in the stream processor:
  f_c = carts in first K, f_avg_price = log1p(mean valid price), f_max_price = log1p(max valid price),
  f_dur = log1p(seconds from event 1 to event K); valid price = price > 0.
Label y = purchase after the K-th event, before the session ends.
t_pred = time of the K-th event; t_close = last event of the session + timeout.

    python -m evaluation.build_session_reference                       # whole sample
    python -m evaluation.build_session_reference --limit 28890655      # first 5 months (regression run)
"""
import argparse
import os

import duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="sample/sample_10pct_7bulan.csv")
ap.add_argument("--limit", type=int, default=0, help="use only the first N rows (as producer --limit)")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--timeout-min", type=int, default=30)
ap.add_argument("--out", default="ref/session_reference.parquet")
args = ap.parse_args()
K, GAP = args.k, args.timeout_min * 60

os.makedirs(os.path.dirname(args.out), exist_ok=True)
os.makedirs("duck_tmp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA memory_limit='8GB'")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA temp_directory='duck_tmp'")

limit = f"AND rn <= {args.limit}" if args.limit else ""
con.execute(f"""
    CREATE TEMP TABLE seq AS
    WITH raw AS (
        SELECT row_number() OVER () AS rn, user_session AS sid, event_type AS et,
               CAST(replace(event_time, ' UTC', '') AS TIMESTAMP) AS t,
               TRY_CAST(price AS DOUBLE) AS price
        FROM read_csv('{args.sample}', header=true, all_varchar=true)
    ),
    ev AS (SELECT * FROM raw WHERE sid IS NOT NULL AND sid <> '' {limit})
    SELECT sid, et, t, price, row_number() OVER w AS k,
           coalesce(epoch(t) - epoch(lag(t) OVER w), 0) > {GAP} AS is_gap
    FROM ev WINDOW w AS (PARTITION BY sid ORDER BY rn)
""")
con.execute(f"""
    CREATE TEMP TABLE sess AS
    WITH b AS (SELECT sid, min(k) FILTER (WHERE is_gap) AS brk FROM seq GROUP BY sid),
    s AS (SELECT q.* FROM seq q JOIN b USING (sid) WHERE b.brk IS NULL OR q.k < b.brk)
    SELECT sid,
           count(*) AS n,
           min(t) FILTER (WHERE k = 1) AS t1,
           min(t) FILTER (WHERE k = {K}) AS tk,
           max(t) AS t_last,
           bool_or(et = 'purchase' AND k <= {K}) AS buy_k,
           count(*) FILTER (WHERE k <= {K} AND et = 'cart') AS c,
           sum(price) FILTER (WHERE k <= {K} AND price > 0) AS ps,
           count(*) FILTER (WHERE k <= {K} AND price > 0) AS pc,
           max(price) FILTER (WHERE k <= {K} AND price > 0) AS pmax,
           bool_or(et = 'purchase' AND k > {K}) AS y
    FROM s GROUP BY sid
""")
con.execute(f"""
    COPY (
        SELECT sid AS session_id,
               tk AS t_pred,
               t_last + INTERVAL {args.timeout_min} MINUTE AS t_close,
               c::DOUBLE AS f_c,
               CASE WHEN pc > 0 THEN ln(1 + ps / pc) ELSE 0 END AS f_avg_price,
               ln(1 + coalesce(pmax, 0)) AS f_max_price,
               ln(1 + greatest(epoch(tk) - epoch(t1), 0)) AS f_dur,
               y::INT AS y
        FROM sess
        WHERE n >= {K} AND NOT buy_k
        ORDER BY t_pred, session_id
    ) TO '{args.out}' (FORMAT PARQUET)
""")
n, pos, a, b = con.execute(
    f"SELECT count(*), sum(y), min(t_pred), max(t_pred) FROM '{args.out}'").fetchone()
print(f"Saved {args.out}: {n:,} sessions | positives {pos:,} ({pos / n:.2%}) | t_pred {a} .. {b}")