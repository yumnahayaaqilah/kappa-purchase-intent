"""
Recompute metrics_daily from eval_log with UTC day boundaries (by time of prediction).
Needed once for runs made before the stream processor used UTC days; eval_log is not changed.

    python -m evaluation.rebuild_daily_metrics
"""
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from db_config import DB
from pipeline.learner import day_metrics

conn = psycopg2.connect(**DB)
with conn, conn.cursor() as cur:
    cur.execute("SELECT (t_pred AT TIME ZONE 'UTC')::date, prob, y FROM eval_log")
    df = pd.DataFrame(cur.fetchall(), columns=["day", "prob", "y"])
    rows = []
    for day, g in df.groupby("day"):
        m = day_metrics(g["prob"].to_numpy(), g["y"].to_numpy())
        rows.append((day, m["n"], m["pos_rate"], m["auc"], m["pr_auc"], m["p_at_5"]))
    cur.execute("TRUNCATE metrics_daily")
    execute_values(cur, """INSERT INTO metrics_daily (day, n, pos_rate, auc, pr_auc, p_at_5, updated_at)
                           VALUES %s""", rows, template="(%s, %s, %s, %s, %s, %s, now())")
conn.close()
print(f"metrics_daily rebuilt: {len(rows)} days from {len(df):,} sessions "
      f"({rows[0][0]} .. {rows[-1][0]})" if rows else "eval_log is empty")
