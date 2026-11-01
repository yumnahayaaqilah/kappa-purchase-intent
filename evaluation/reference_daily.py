"""
Daily view of the session reference (answer key): sessions, purchase rate and the cart signal
per day of prediction (UTC). Independent of the pipeline, so it shows whether a pattern in
metrics_daily comes from the data itself.

    python -m evaluation.reference_daily --start 2019-11-10 --end 2019-11-20
"""
import argparse

import duckdb
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--reference", default="ref/session_reference_7m.parquet")
ap.add_argument("--start", default="2019-10-01")
ap.add_argument("--end", default="2020-05-01", help="exclusive")
args = ap.parse_args()

df = duckdb.sql(f"""
    SELECT CAST(t_pred AS DATE) AS day,
           count(*) AS n,
           round(avg(y), 4) AS pos_rate,
           round(avg((f_c > 0)::INT), 4) AS cart_share,
           round(avg(y) FILTER (WHERE f_c > 0), 4) AS pos_rate_cart,
           round(avg(y) FILTER (WHERE f_c = 0), 4) AS pos_rate_no_cart
    FROM read_parquet('{args.reference}')
    WHERE t_pred >= TIMESTAMP '{args.start}' AND t_pred < TIMESTAMP '{args.end}'
    GROUP BY 1 ORDER BY 1
""").df()
pd.set_option("display.width", 200)
print(df.to_string(index=False))