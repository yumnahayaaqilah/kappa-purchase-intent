"""
Membuat SAMPEL SESSION dari seluruh CSV di data/ (session utuh, urut waktu).
Pemilihan session deterministik lewat hash(user_session), jadi session tidak terpotong
dan rentang waktu tetap seluruh periode.

    python make_sample.py --pct 10      # hasil: sample/sample_10pct.csv
"""
import argparse
import os
import time

import duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--pct", type=int, default=10, help="persen session yang diambil (1-100)")
ap.add_argument("--glob", default="data/20*.csv")
args = ap.parse_args()

os.makedirs("sample", exist_ok=True)
os.makedirs("duck_tmp", exist_ok=True)
out = f"sample/sample_{args.pct}pct.csv"

con = duckdb.connect()
con.execute("PRAGMA memory_limit='8GB'")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA temp_directory='duck_tmp'")

t0 = time.time()
print(f"Membuat sampel {args.pct}% session dari {args.glob} ... (beberapa menit)")
con.execute(f"""
    COPY (
        SELECT event_time, event_type, product_id, category_id, category_code,
               brand, price, user_id, user_session
        FROM read_csv('{args.glob}', header=true, columns={{
            'event_time':'VARCHAR','event_type':'VARCHAR','product_id':'BIGINT',
            'category_id':'BIGINT','category_code':'VARCHAR','brand':'VARCHAR',
            'price':'DOUBLE','user_id':'BIGINT','user_session':'VARCHAR'}})
        WHERE user_session IS NOT NULL AND user_session <> ''
          AND hash(user_session) % 100 < {args.pct}
        ORDER BY CAST(replace(event_time, ' UTC', '') AS TIMESTAMP), user_session, product_id
    ) TO '{out}' (HEADER, DELIMITER ',')
""")
n, a, b, s = con.execute(f"""
    SELECT count(*), min(event_time), max(event_time), count(DISTINCT user_session)
    FROM read_csv('{out}', header=true, all_varchar=true)
""").fetchone()
print(f"Selesai dalam {time.time() - t0:.0f} detik.")
print(f"File: {out} | {n:,} event | {s:,} session | {a} sampai {b}")
print(f"Ukuran file: {os.path.getsize(out) / 1e9:.2f} GB")
