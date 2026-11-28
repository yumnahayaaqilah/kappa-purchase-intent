-- Session table for the Power BI report: one row per scored session.
DROP TABLE IF EXISTS pbi_sessions;
CREATE TABLE pbi_sessions AS
WITH b AS (
  SELECT *, percent_rank() OVER (ORDER BY t_pred) AS pr_time FROM eval_log
), c AS (
  SELECT *, CASE WHEN pr_time < 0.7 THEN 'Awal' ELSE 'Uji' END AS periode FROM b
)
SELECT session_id,
       (t_pred AT TIME ZONE 'UTC')::date AS tanggal,
       extract(hour FROM t_pred AT TIME ZONE 'UTC')::int AS jam_utc,
       f_c::int AS n_keranjang,
       round((exp(f_avg_price) - 1)::numeric, 2) AS harga_rata,
       round((exp(f_max_price) - 1)::numeric, 2) AS harga_max,
       round((exp(f_dur) - 1)::numeric, 0) AS durasi_detik,
       prob,
       y AS beli,
       periode,
       ntile(100) OVER (PARTITION BY periode ORDER BY prob DESC) AS persentil
FROM c;