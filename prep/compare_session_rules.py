"""
Compare two rules for ending a session at a gap of more than 30 minutes (K = 3):
  A: any gap ends the session (later events of the same user_session are ignored)
  B: gaps before the K-th event are ignored; the first gap after it ends the session (v1-like)

    python -m prep.compare_session_rules
"""
import argparse

import duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="sample/sample_10pct_7bulan.csv")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--timeout-min", type=int, default=30)
args = ap.parse_args()
K, GAP = args.k, args.timeout_min * 60

con = duckdb.connect()
con.execute("PRAGMA memory_limit='8GB'")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA temp_directory='duck_tmp'")
con.execute(f"""
    CREATE TEMP TABLE seq AS
    WITH ev AS (
        SELECT row_number() OVER () AS rn, user_session AS sid, event_type AS et,
               CAST(replace(event_time, ' UTC', '') AS TIMESTAMP) AS t
        FROM read_csv('{args.sample}', header=true, all_varchar=true)
        WHERE user_session IS NOT NULL AND user_session <> ''
    )
    SELECT sid, et, row_number() OVER w AS k,
           coalesce(epoch(t) - epoch(lag(t) OVER w), 0) > {GAP} AS is_gap
    FROM ev WINDOW w AS (PARTITION BY sid ORDER BY rn)
""")
r = con.execute(f"""
    WITH b AS (
        SELECT sid,
               min(k) FILTER (WHERE is_gap) AS brk_a,
               min(k) FILTER (WHERE is_gap AND k > {K}) AS brk_b
        FROM seq GROUP BY sid
    ),
    s AS (
        SELECT q.sid,
               count(*) FILTER (WHERE b.brk_a IS NULL OR q.k < b.brk_a) AS n_a,
               bool_or(q.et = 'purchase' AND q.k <= {K}) AS buy_k,
               bool_or(q.et = 'purchase' AND q.k > {K} AND (b.brk_a IS NULL OR q.k < b.brk_a)) AS y_a,
               bool_or(q.et = 'purchase' AND q.k > {K} AND (b.brk_b IS NULL OR q.k < b.brk_b)) AS y_b,
               count(*) AS n_all
        FROM seq q JOIN b USING (sid) GROUP BY q.sid
    )
    SELECT count(*) FILTER (WHERE n_all >= {K} AND NOT buy_k) AS eligible_b,
           count(*) FILTER (WHERE n_a >= {K} AND NOT buy_k) AS eligible_a,
           count(*) FILTER (WHERE n_all >= {K} AND NOT buy_k AND y_b) AS positives_b,
           count(*) FILTER (WHERE n_a >= {K} AND NOT buy_k AND y_a) AS positives_a,
           count(*) FILTER (WHERE n_all >= {K} AND NOT buy_k AND n_a < {K}) AS dropped_by_a,
           count(*) FILTER (WHERE n_all >= {K} AND NOT buy_k AND n_a < {K} AND y_b) AS dropped_positives
    FROM s
""").fetchone()
eb, ea, pb, pa, d, dp = r
print(f"Rule B (v1-like): eligible {eb:,} | positives {pb:,} ({pb / eb:.2%})")
print(f"Rule A (any gap): eligible {ea:,} | positives {pa:,} ({pa / ea:.2%})")
print(f"Sessions dropped by A: {d:,} ({d / eb:.2%} of B) | of which positives under B: {dp:,} ({dp / max(pb, 1):.2%} of B positives)")