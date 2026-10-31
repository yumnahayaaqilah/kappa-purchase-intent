"""Hitung nilai kosong per kolom pada semua CSV di data/. Hasil disimpan ke results/missing_values.csv"""
import duckdb

con = duckdb.connect()
con.execute("PRAGMA threads=4")
SRC = """read_csv('data/20*.csv', header=true, columns={
    'event_time':'VARCHAR','event_type':'VARCHAR','product_id':'BIGINT',
    'category_id':'BIGINT','category_code':'VARCHAR','brand':'VARCHAR',
    'price':'DOUBLE','user_id':'BIGINT','user_session':'VARCHAR'})"""

df = con.execute(f"""
    SELECT count(*) AS total_baris,
           count(*) FILTER (WHERE event_time IS NULL)    AS event_time_kosong,
           count(*) FILTER (WHERE event_type IS NULL)    AS event_type_kosong,
           count(*) FILTER (WHERE product_id IS NULL)    AS product_id_kosong,
           count(*) FILTER (WHERE category_code IS NULL OR category_code = '') AS category_code_kosong,
           count(*) FILTER (WHERE brand IS NULL OR brand = '')                 AS brand_kosong,
           count(*) FILTER (WHERE price IS NULL)         AS price_kosong,
           count(*) FILTER (WHERE price <= 0)            AS price_nol_atau_negatif,
           count(*) FILTER (WHERE user_id IS NULL)       AS user_id_kosong,
           count(*) FILTER (WHERE user_session IS NULL OR user_session = '') AS user_session_kosong
    FROM {SRC}
""").df()

t = df.iloc[0]
total = int(t["total_baris"])
out = []
for col in df.columns[1:]:
    n = int(t[col])
    out.append({"kolom": col, "jumlah": n, "persen": round(100.0 * n / total, 3)})
import pandas as pd
res = pd.DataFrame(out)
print(f"Total baris: {total:,}\n")
print(res.to_string(index=False))
res.to_csv("results/missing_values.csv", index=False)
