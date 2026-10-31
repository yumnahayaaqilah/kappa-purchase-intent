import duckdb

q = """
SELECT count(*) AS sesi,
       count(*) FILTER (WHERE hash(s) % 100 < 10) AS cocok
FROM (
  SELECT DISTINCT user_session AS s
  FROM (SELECT user_session
        FROM read_csv('sample/sample_10pct.csv', header=true, all_varchar=true)
        LIMIT 1000000)
)
"""
sesi, cocok = duckdb.sql(q).fetchone()
print(f"Sesi dicek: {sesi:,} | lolos aturan hash: {cocok:,} ({cocok / sesi:.1%})")