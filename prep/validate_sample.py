"""
Compare the 10% session sample with the full population, per month and per category.

    python -m prep.validate_sample

Population and sample are filtered the same way (non-empty user_session).
Outputs: hasil/sample_validation_monthly.csv, hasil/sample_validation_category.csv
"""
import argparse
import os
import time

import duckdb
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--population", nargs="+", default=["data/20*.csv", "data/20*.csv.gz"])
ap.add_argument("--sample", default="sample/sample_10pct_7bulan.csv")
ap.add_argument("--outdir", default="hasil")
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
os.makedirs("duck_tmp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA memory_limit='8GB'")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA temp_directory='duck_tmp'")

COLS = """columns={'event_time':'VARCHAR','event_type':'VARCHAR','product_id':'BIGINT',
    'category_id':'BIGINT','category_code':'VARCHAR','brand':'VARCHAR',
    'price':'DOUBLE','user_id':'BIGINT','user_session':'VARCHAR'}"""


def source(files):
    lst = ", ".join(f"'{f}'" for f in files)
    return f"""(SELECT left(event_time, 7) AS month, event_type, user_session,
                       coalesce(nullif(split_part(category_code, '.', 1), ''), 'unknown') AS cat
                FROM read_csv([{lst}], header=true, {COLS})
                WHERE user_session IS NOT NULL AND user_session <> '')"""


def run(name, files):
    src = source(files)
    t0 = time.time()
    ev = con.execute(f"SELECT month, event_type, cat, count(*) AS n FROM {src} GROUP BY ALL").df()
    print(f"[{name}] event aggregate: {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    ss = con.execute(f"""
        SELECT month, count(*) AS sessions, sum(buy) AS buy_sessions, avg(n) AS events_per_session
        FROM (SELECT month, user_session, count(*) AS n,
                     max(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS buy
              FROM {src} GROUP BY month, user_session)
        GROUP BY month""").df()
    print(f"[{name}] session aggregate: {time.time() - t0:.0f} s", flush=True)
    return ev, ss


def monthly(ev, ss):
    m = ev.pivot_table(index="month", columns="event_type", values="n", aggfunc="sum", fill_value=0)
    m["events"] = m.sum(axis=1)
    for t in ("view", "cart", "purchase"):
        m[f"{t}_share"] = m.get(t, 0) / m["events"]
    m = m[["events", "view_share", "cart_share", "purchase_share"]].join(ss.set_index("month"))
    m["buy_session_rate"] = m["buy_sessions"] / m["sessions"]
    return m


ev_p, ss_p = run("population", args.population)
ev_s, ss_s = run("sample", [args.sample])
mp, ms = monthly(ev_p, ss_p), monthly(ev_s, ss_s)

mon = pd.DataFrame({
    "events_pop": mp["events"], "events_sample": ms["events"],
    "event_ratio": ms["events"] / mp["events"],
    "sessions_pop": mp["sessions"], "sessions_sample": ms["sessions"],
    "session_ratio": ms["sessions"] / mp["sessions"],
    "view_share_diff_pp": 100 * (ms["view_share"] - mp["view_share"]),
    "cart_share_diff_pp": 100 * (ms["cart_share"] - mp["cart_share"]),
    "purchase_share_diff_pp": 100 * (ms["purchase_share"] - mp["purchase_share"]),
    "buy_session_rate_pop": mp["buy_session_rate"], "buy_session_rate_sample": ms["buy_session_rate"],
    "events_per_session_pop": mp["events_per_session"], "events_per_session_sample": ms["events_per_session"],
})
tot = {"events_pop": mp["events"].sum(), "events_sample": ms["events"].sum()}
mon.to_csv(os.path.join(args.outdir, "sample_validation_monthly.csv"))

cp = ev_p.groupby("cat")["n"].sum()
cs = ev_s.groupby("cat")["n"].sum()
cat = pd.DataFrame({"share_pop": cp / cp.sum(), "share_sample": cs / cs.sum()}).fillna(0)
cat["diff_pp"] = 100 * (cat["share_sample"] - cat["share_pop"])
cat = cat.sort_values("share_pop", ascending=False)
cat.to_csv(os.path.join(args.outdir, "sample_validation_category.csv"))

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
print("\n=== Per month ===")
print(mon.round(4).to_string())
print(f"\nTotal events: population {tot['events_pop']:,} | sample {tot['events_sample']:,} "
      f"| ratio {tot['events_sample'] / tot['events_pop']:.4f}")
print("\n=== Category share (top 10 by population) ===")
print(cat.head(10).round(4).to_string())
print(f"Largest absolute category share difference: {cat['diff_pp'].abs().max():.3f} pp")