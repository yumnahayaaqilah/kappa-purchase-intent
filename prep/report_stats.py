"""
Statistics for the report, computed on the session sample:
  1. Exact duplicate rows
  2. Coverage of the K=3 prediction task
  3. Label definitions: full user_session vs 30-minute inactivity rule, and the
     labels the v1 pipeline actually logged (table eval_log_v1, if reachable)
  4. category_code consistency per category_id across months

    python -m prep.report_stats

Event order inside a session follows the sample file order, the same order the producer
sends to Kafka. Outputs go to hasil/report_*.csv.
"""
import argparse
import os
import time

import duckdb
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="sample/sample_10pct_7bulan.csv")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--timeout-min", type=int, default=30)
ap.add_argument("--eval-table", default="eval_log_v1")
ap.add_argument("--outdir", default="hasil")
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
os.makedirs("duck_tmp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA memory_limit='8GB'")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA temp_directory='duck_tmp'")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

SRC = f"""read_csv('{args.sample}', header=true, columns={{
    'event_time':'VARCHAR','event_type':'VARCHAR','product_id':'BIGINT',
    'category_id':'BIGINT','category_code':'VARCHAR','brand':'VARCHAR',
    'price':'DOUBLE','user_id':'BIGINT','user_session':'VARCHAR'}})"""
COLS = "event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session"
K, GAP = args.k, args.timeout_min * 60


def step(msg):
    print(f"\n=== {msg} ===", flush=True)
    return time.time()


# ------------------------------------------------------------------ 1. duplicates
t0 = step("1. Exact duplicate rows")
dup = con.execute(f"""
    SELECT (SELECT count(*) FROM {SRC}) AS rows,
           count(*) AS duplicate_groups,
           coalesce(sum(c - 1), 0) AS extra_rows,
           count(DISTINCT user_session) AS sessions_affected
    FROM (SELECT {COLS}, count(*) AS c FROM {SRC} GROUP BY {COLS} HAVING count(*) > 1)
""").df()
dup["extra_rows_pct"] = 100 * dup["extra_rows"] / dup["rows"]
dup.to_csv(os.path.join(args.outdir, "report_duplicates.csv"), index=False)
print(dup.to_string(index=False))
print(f"({time.time() - t0:.0f} s)")

# ------------------------------------------------------------------ 2-3. sessions
t0 = step(f"2. Coverage of the K={K} task")
con.execute(f"""
    CREATE TEMP TABLE ev AS
    SELECT row_number() OVER () AS rn, user_session AS sid, event_type AS et,
           CAST(replace(event_time, ' UTC', '') AS TIMESTAMP) AS t
    FROM {SRC}
    WHERE user_session IS NOT NULL AND user_session <> ''
""")
con.execute("""
    CREATE TEMP TABLE seq AS
    SELECT sid, et, t,
           row_number() OVER w AS k,
           epoch(t) - epoch(lag(t) OVER w) AS gap
    FROM ev
    WINDOW w AS (PARTITION BY sid ORDER BY rn)
""")
con.execute(f"""
    CREATE TEMP TABLE sess AS
    WITH b AS (SELECT sid, min(k) FILTER (WHERE k > {K} AND gap > {GAP}) AS brk FROM seq GROUP BY sid)
    SELECT q.sid,
           count(*) AS n,
           max(q.t) FILTER (WHERE q.k = {K}) AS t_k,
           bool_or(q.et = 'purchase') AS any_buy,
           bool_or(q.et = 'purchase' AND q.k <= {K}) AS buy_before_k,
           bool_or(q.et = 'purchase' AND q.k > {K}) AS y_full,
           bool_or(q.et = 'purchase' AND q.k > {K} AND (b.brk IS NULL OR q.k < b.brk)) AS y_timeout,
           bool_or(b.brk IS NOT NULL) AS has_gap_after_k
    FROM seq q JOIN b USING (sid)
    GROUP BY q.sid
""")
cov = con.execute(f"""
    SELECT count(*) AS sessions,
           count(*) FILTER (WHERE n >= {K}) AS sessions_ge_k,
           count(*) FILTER (WHERE n >= {K} AND buy_before_k) AS buy_within_first_k,
           count(*) FILTER (WHERE n >= {K} AND NOT buy_before_k) AS eligible,
           count(*) FILTER (WHERE any_buy) AS buy_sessions,
           count(*) FILTER (WHERE any_buy AND n < {K}) AS buy_sessions_short,
           count(*) FILTER (WHERE any_buy AND n >= {K} AND buy_before_k) AS buy_sessions_before_k,
           count(*) FILTER (WHERE n >= {K} AND NOT buy_before_k AND y_full) AS buy_sessions_covered
    FROM sess
""").df()
c = cov.iloc[0]
cov["eligible_pct_of_sessions"] = 100 * c.eligible / c.sessions
cov["covered_pct_of_buy_sessions"] = 100 * c.buy_sessions_covered / c.buy_sessions
cov.to_csv(os.path.join(args.outdir, "report_coverage.csv"), index=False)
print(cov.T.to_string(header=False))
print(f"({time.time() - t0:.0f} s)")

t0 = step(f"3. Label definitions (eligible sessions, timeout {args.timeout_min} min)")
lab = con.execute(f"""
    SELECT count(*) AS eligible,
           avg(y_full::INT) AS rate_full_session,
           avg(y_timeout::INT) AS rate_timeout_rule,
           count(*) FILTER (WHERE has_gap_after_k) AS with_gap_after_k,
           count(*) FILTER (WHERE y_full AND NOT y_timeout) AS buy_only_after_gap
    FROM sess WHERE n >= {K} AND NOT buy_before_k
""").df()
lab["buy_only_after_gap_pct_of_full_positives"] = (
    100 * lab["buy_only_after_gap"] / (lab["rate_full_session"] * lab["eligible"]))
print(lab.T.to_string(header=False))

try:
    import psycopg2
    from db_config import DB
    conn = psycopg2.connect(**DB)
    with conn.cursor() as cur:
        cur.execute(f"SELECT session_id, y FROM {args.eval_table}")
        logged = pd.DataFrame(cur.fetchall(), columns=["sid", "y_logged"])
    conn.close()
    con.register("logged", logged)
    cmp_ = con.execute(f"""
        SELECT count(*) AS logged_sessions,
               count(s.sid) AS matched_in_sample,
               count(*) FILTER (WHERE s.sid IS NOT NULL AND (l.y_logged = 1) = s.y_full) AS agree_full,
               count(*) FILTER (WHERE s.sid IS NOT NULL AND (l.y_logged = 1) = s.y_timeout) AS agree_timeout,
               count(*) FILTER (WHERE l.y_logged = 0 AND s.y_full) AS logged0_full1,
               count(*) FILTER (WHERE l.y_logged = 1 AND NOT s.y_full) AS logged1_full0,
               count(*) FILTER (WHERE l.y_logged = 0 AND s.y_timeout) AS logged0_timeout1,
               count(*) FILTER (WHERE l.y_logged = 1 AND NOT s.y_timeout) AS logged1_timeout0
        FROM logged l LEFT JOIN sess s ON s.sid = l.sid
    """).df()
    lab = pd.concat([lab, cmp_], axis=1)
    print(f"\nPipeline v1 labels ({args.eval_table}) vs recomputed labels:")
    print(cmp_.T.to_string(header=False))
except Exception as e:  # PostgreSQL is optional for this script
    print(f"\nSkipped comparison with {args.eval_table}: {e}")
lab.to_csv(os.path.join(args.outdir, "report_labels.csv"), index=False)
print(f"({time.time() - t0:.0f} s)")

# ------------------------------------------------------------------ 4. category_code
t0 = step("4. category_code consistency per category_id")
cc = con.execute(f"""
    SELECT category_id, left(event_time, 7) AS month, coalesce(category_code, '') AS code, count(*) AS n
    FROM {SRC} GROUP BY ALL
""").df()
named = cc[cc["code"] != ""]
per_id = named.groupby("category_id")["code"].nunique()
changed_ids = per_id[per_id > 1].index
rows = []
for cid in changed_ids:
    g = named[named["category_id"] == cid].sort_values("month")
    mapping = g.groupby("code")["month"].agg(lambda m: f"{m.min()}..{m.max()}")
    rows.append({"category_id": cid, "events": int(g["n"].sum()),
                 "codes": " | ".join(f"{k} [{v}]" for k, v in mapping.sort_values().items())})
chg = pd.DataFrame(rows, columns=["category_id", "events", "codes"]).sort_values("events", ascending=False)
chg.to_csv(os.path.join(args.outdir, "report_category_code_changes.csv"), index=False)
total_named = int(named["n"].sum())
print(f"category_id with more than one category_code: {len(chg):,} of {per_id.size:,} "
      f"({100 * chg['events'].sum() / max(total_named, 1):.1f}% of events that have a code)")
print(chg.head(10).to_string(index=False))

cc["top"] = cc["code"].str.split(".").str[0].replace("", "unknown")
share = cc.pivot_table(index="month", columns="top", values="n", aggfunc="sum", fill_value=0)
share = share.div(share.sum(axis=1), axis=0)
top_cols = share.mean().sort_values(ascending=False).index[:6]
share[top_cols].to_csv(os.path.join(args.outdir, "report_category_share_by_month.csv"))
print("\nTop-level category share by month:")
print((100 * share[top_cols]).round(1).to_string())
print(f"({time.time() - t0:.0f} s)")