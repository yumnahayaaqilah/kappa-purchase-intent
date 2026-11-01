"""
Explain sessions the pipeline predicted that the reference does not have.

Written to diagnose the first full run: the stream processor then remembered a closed
user_session for only 7 days of event time, so an ID coming back later started a new session,
while the reference ignores every later event of a closed user_session, however late. The
stream processor now remembers closed IDs for good; after a re-compute this should report 0.

    python -m evaluation.explain_extra_sessions
"""
import argparse
import os

import duckdb
import pandas as pd
import psycopg2

from db_config import DB

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="sample/sample_10pct_7bulan.csv")
ap.add_argument("--reference", default="ref/session_reference_7m.parquet")
ap.add_argument("--retention-days", type=float, default=7)
ap.add_argument("--timeout-min", type=int, default=30)
args = ap.parse_args()

conn = psycopg2.connect(**DB)
with conn.cursor() as cur:
    cur.execute("SELECT session_id, t_pred AT TIME ZONE 'UTC' FROM eval_log "
                "UNION ALL SELECT session_id, t_pred AT TIME ZONE 'UTC' FROM active_predictions")
    pipe = pd.DataFrame(cur.fetchall(), columns=["sid", "t_pred_pipe"])
conn.close()

os.makedirs("duck_tmp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA memory_limit='6GB'")
con.execute("PRAGMA temp_directory='duck_tmp'")
con.register("pipe", pipe)
con.execute(f"CREATE TEMP TABLE ref AS SELECT session_id AS sid FROM read_parquet('{args.reference}')")
con.execute("CREATE TEMP TABLE extra AS SELECT * FROM pipe WHERE sid NOT IN (SELECT sid FROM ref)")
print(f"Pipeline sessions not in the reference: {con.execute('SELECT count(*) FROM extra').fetchone()[0]:,}")

# Every user_session: end of its first session (last event before the first gap > timeout),
# and the first event coming back more than retention-days after that end
con.execute(f"""
    CREATE TEMP TABLE seq AS
    WITH raw AS (
        SELECT row_number() OVER () AS rn, user_session AS sid,
               CAST(replace(event_time, ' UTC', '') AS TIMESTAMP) AS t
        FROM read_csv('{args.sample}', header=true, all_varchar=true)
    )
    SELECT sid, t, row_number() OVER w AS k,
           coalesce(epoch(t) - epoch(lag(t) OVER w), 0) > {args.timeout_min * 60} AS is_gap
    FROM raw WHERE sid IS NOT NULL AND sid <> ''
    WINDOW w AS (PARTITION BY sid ORDER BY rn)
""")
con.execute(f"""
    CREATE TEMP TABLE reuse AS
    WITH b AS (SELECT sid, min(k) FILTER (WHERE is_gap) AS brk FROM seq GROUP BY sid),
    e AS (SELECT s.sid, max(s.t) AS t_end FROM seq s JOIN b USING (sid)
          WHERE b.brk IS NOT NULL AND s.k < b.brk GROUP BY s.sid)
    SELECT e.sid, e.t_end, min(s.t) AS t_back, count(*) AS n_after
    FROM e JOIN seq s USING (sid)
    WHERE s.t > e.t_end + INTERVAL '{int(args.retention_days * 24)} hours'
    GROUP BY e.sid, e.t_end
""")
r = con.execute("""
    SELECT count(*) AS reused,
           count(*) FILTER (WHERE sid IN (SELECT sid FROM ref)) AS reused_in_ref,
           count(*) FILTER (WHERE n_after >= 3) AS reused_3plus
    FROM reuse
""").fetchone()
x = con.execute("""
    SELECT count(*) AS extra,
           count(r.sid) AS explained,
           count(*) FILTER (WHERE r.sid IS NOT NULL AND x.t_pred_pipe >= r.t_back) AS pred_after_return
    FROM extra x LEFT JOIN reuse r USING (sid)
""").fetchone()
gap = con.execute("""
    SELECT quantile_cont(epoch(t_back - t_end) / 86400, [0.0, 0.5, 1.0]) FROM reuse
""").fetchone()[0]

print(f"\nuser_session IDs that come back more than {args.retention_days:g} days after their session ended: {r[0]:,}")
print(f"  of which the first session is in the reference (predicted): {r[1]:,}")
print(f"  of which have 3+ events after coming back: {r[2]:,}")
if gap:
    print(f"  days between end and return: min {gap[0]:.1f}, median {gap[1]:.1f}, max {gap[2]:.1f}")
print(f"\nExtra pipeline sessions explained by such a return: {x[1]:,} of {x[0]:,} "
      f"(predicted after the return: {x[2]:,})")
