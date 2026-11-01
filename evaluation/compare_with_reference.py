"""
Regression check: compare what the stream processor wrote to PostgreSQL with the batch reference.

Every eligible session in the reference must appear exactly once, either finished (eval_log)
or still open (active_predictions). Finished sessions must match on t_pred, t_close, features and label.

    python -m evaluation.compare_with_reference --reference ref/session_reference_5m.parquet
"""
import argparse

import duckdb
import numpy as np
import pandas as pd
import psycopg2

from db_config import DB

ap = argparse.ArgumentParser()
ap.add_argument("--reference", default="ref/session_reference.parquet")
ap.add_argument("--tol", type=float, default=1e-9, help="tolerance for feature values")
args = ap.parse_args()

ref = duckdb.sql(f"SELECT * FROM '{args.reference}'").df().set_index("session_id")
COLS = ["session_id", "t_pred", "t_close", "f_c", "f_avg_price", "f_max_price", "f_dur", "y"]
conn = psycopg2.connect(**DB)
with conn.cursor() as cur:
    cur.execute(f"SELECT {', '.join(COLS)} FROM eval_log")
    ev = pd.DataFrame(cur.fetchall(), columns=COLS).set_index("session_id")
    cur.execute("SELECT session_id FROM active_predictions")
    act = pd.Series([r[0] for r in cur.fetchall()], dtype=object)
conn.close()

done, open_ = set(ev.index), set(act)
refset = set(ref.index)
print(f"Reference sessions: {len(refset):,} | finished in pipeline: {len(done):,} | still open: {len(open_):,}")
print(f"In both finished and open: {len(done & open_):,}")
print(f"In pipeline but not in reference: {len((done | open_) - refset):,}")
print(f"In reference but missing from pipeline: {len(refset - done - open_):,}")

common = ev.index.intersection(ref.index)
a, b = ev.loc[common], ref.loc[common]
for c in ("t_pred", "t_close"):
    a[c] = pd.to_datetime(a[c], utc=True)
    b[c] = pd.to_datetime(b[c]).dt.tz_localize("UTC") if b[c].dt.tz is None else b[c]
checks = {
    "t_pred": (a["t_pred"] != b["t_pred"]).sum(),
    "t_close": (a["t_close"] != b["t_close"]).sum(),
    "y": (a["y"].astype(int) != b["y"].astype(int)).sum(),
}
for f in ("f_c", "f_avg_price", "f_max_price", "f_dur"):
    checks[f] = int((np.abs(a[f].astype(float) - b[f].astype(float)) > args.tol).sum())
print(f"\nFinished sessions compared: {len(common):,}")
for k, v in checks.items():
    print(f"  mismatching {k}: {v:,}")
ok = (not (done & open_) and not ((done | open_) - refset) and not (refset - done - open_)
      and all(v == 0 for v in checks.values()))
print("\nRESULT:", "MATCH" if ok else "DIFFERENCES FOUND")
