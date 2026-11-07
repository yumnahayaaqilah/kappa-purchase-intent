"""
Extra analyses for the report (outside the pipeline, read-only). Needs build_session_insights.py first;
the 'seg' part also reads eval_log from PostgreSQL, 'anom' reads metrics_daily, 'early' reads the
session references for several K (build_session_reference.py --k N --out ref/session_reference_kN.parquet).

    python -m evaluation.analyze_insights                 # all parts
    python -m evaluation.analyze_insights --part p2 p5    # some parts

Parts:
  p2     purchase rate by number of distinct categories and by price band in the first K events (WS3 P2)
  p5     cart abandonment: share, value left in carts, by category, price band, hour and month
  p6     funnel per category and brand per month (views -> carts -> purchases)
  p7     time patterns: purchase rate of scored sessions and event volume by hour (UTC) and weekday
  anom   daily anomalies from metrics_daily (same rule as the dashboard)
  p8     returning users: purchase rate by visit number, visits before the first purchase
  seg    score segments and expected value ranking on the test period (eval_log)
  early  how early: model quality vs share of buyer sessions still reachable, for K = 3, 5, 7, 10
Output: results/insight_*.csv

Cart events are incomplete before 8 Nov 2019 (evaluation/check_cart_logging.py), so the report figures for
p2, p5, p6 and p7 come from:  python -m evaluation.analyze_insights --part p2 p5 p6 p7 --since 2019-11-08
"""
import argparse
import os

import duckdb
import numpy as np
import pandas as pd

PARTS = ["p2", "p5", "p6", "p7", "anom", "p8", "seg", "early"]
ap = argparse.ArgumentParser()
ap.add_argument("--part", nargs="+", default=PARTS, choices=PARTS)
ap.add_argument("--sessions", default="ref/session_insights.parquet")
ap.add_argument("--events", default="ref/event_aggregates.parquet")
ap.add_argument("--outdir", default="results")
ap.add_argument("--train-frac", type=float, default=0.7)
ap.add_argument("--early-k", nargs="+", type=int, default=[3, 5, 7, 10])
ap.add_argument("--since", help="YYYY-MM-DD: p2, p5, p6, p7 only use sessions starting on/after this date "
                                "(event aggregates: whole months from this date); files get a _since suffix")
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 60)
con = duckdb.connect()
con.execute(f"CREATE VIEW s_all AS SELECT * FROM read_parquet('{args.sessions}')")
SUFFIX, MONTH0 = "", None
if args.since:
    since = pd.Timestamp(args.since)
    MONTH0 = since.strftime("%Y-%m") if since.day == 1 else (since + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    SUFFIX = "_since" + since.strftime("%Y%m%d")
    con.execute(f"CREATE VIEW s AS SELECT * FROM s_all WHERE t1 >= TIMESTAMP '{since}'")
    print(f"--since {args.since}: sesi mulai {since.date()} (p2, p5, p7); agregat event mulai bulan {MONTH0} (p6, p7). "
          f"p8, anom, seg, early tidak difilter.")
else:
    con.execute("CREATE VIEW s AS SELECT * FROM s_all")
if any(p in args.part for p in ("p6", "p7")):
    month_filter = f"WHERE month >= '{MONTH0}'" if MONTH0 else ""
    con.execute(f"CREATE VIEW e AS SELECT * FROM read_parquet('{args.events}') {month_filter}")

PRICE_BANDS = """CASE WHEN {c} IS NULL THEN 'tanpa harga'
     WHEN {c} < 10 THEN 'a. < 10' WHEN {c} < 50 THEN 'b. 10-50' WHEN {c} < 100 THEN 'c. 50-100'
     WHEN {c} < 250 THEN 'd. 100-250' WHEN {c} < 500 THEN 'e. 250-500' WHEN {c} < 1000 THEN 'f. 500-1000'
     ELSE 'g. >= 1000' END"""
HARI = {1: "Senin", 2: "Selasa", 3: "Rabu", 4: "Kamis", 5: "Jumat", 6: "Sabtu", 7: "Minggu"}


FILTERED = {"p2", "p5", "p6", "p7"}


def save(df, name, title):
    if SUFFIX and name.split("_")[0] in FILTERED:
        name, title = name + SUFFIX, f"{title} [sejak {args.since}]"
    path = os.path.join(args.outdir, f"insight_{name}.csv")
    df.to_csv(path, index=False)
    print(f"\n== {title}  -> {path}")
    print(df.to_string(index=False))


def q(sql):
    return con.execute(sql).df()


# ------------------------------------------------------------------ P2
def p2():
    base = "FROM s WHERE scored"
    save(q(f"""
        SELECT ncat_id_k AS kategori_berbeda, CASE WHEN c_k > 0 THEN 'ada keranjang' ELSE 'tanpa keranjang' END AS keranjang,
               count(*) AS sesi, round(100 * count(*) / sum(count(*)) OVER (), 2) AS persen_sesi,
               round(100 * avg(y::INT), 2) AS persen_membeli
        {base} GROUP BY ALL ORDER BY keranjang, kategori_berbeda"""),
         "p2_categories", "P2: jumlah kategori berbeda (category_id) di event awal x keranjang")
    save(q(f"""
        SELECT ncat_id_k AS kategori_berbeda, count(*) AS sesi, round(100 * avg(y::INT), 2) AS persen_membeli
        {base} GROUP BY ALL ORDER BY 1"""),
         "p2_categories_total", "P2: jumlah kategori berbeda, semua sesi dinilai")
    save(q(f"""
        SELECT {PRICE_BANDS.format(c='avg_price_k')} AS harga_rata_rata,
               CASE WHEN c_k > 0 THEN 'ada keranjang' ELSE 'tanpa keranjang' END AS keranjang,
               count(*) AS sesi, round(100 * avg(y::INT), 2) AS persen_membeli
        {base} GROUP BY ALL ORDER BY keranjang, harga_rata_rata"""),
         "p2_price", "P2: rentang harga rata-rata di event awal x keranjang")

    # does the number of distinct categories add information? time split, as full_evaluation.py
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = q(f"""SELECT tk, c_k::DOUBLE AS f_c, ln(1 + coalesce(avg_price_k, 0)) AS f_avg_price,
                     ln(1 + coalesce(max_price_k, 0)) AS f_max_price,
                     ln(1 + greatest(epoch(tk) - epoch(t1), 0)) AS f_dur,
                     ncat_id_k::DOUBLE AS f_ncat, y::INT AS y {base} ORDER BY tk, sid""")
    cut = int(len(d) * args.train_frac)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    if len(tr) > 1_000_000:
        tr = tr.sample(1_000_000, random_state=42)
    rows = []
    for name, cols in [("4 fitur pipeline", ["f_c", "f_avg_price", "f_max_price", "f_dur"]),
                       ("4 fitur + kategori berbeda", ["f_c", "f_avg_price", "f_max_price", "f_dur", "f_ncat"]),
                       ("hanya kategori berbeda", ["f_ncat"])]:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[cols], tr.y)
        p = m.predict_proba(te[cols])[:, 1]
        rows.append({"fitur": name, "ROC-AUC": round(roc_auc_score(te.y, p), 4),
                     "PR-AUC": round(average_precision_score(te.y, p), 4)})
    save(pd.DataFrame(rows), "p2_feature_test",
         f"P2: apakah kategori berbeda menambah daya prediksi (latih {args.train_frac:.0%} awal, uji sisanya)")


# ------------------------------------------------------------------ P5
def p5():
    save(q("""
        SELECT count(*) FILTER (WHERE has_cart) AS sesi_dengan_keranjang,
               count(*) FILTER (WHERE has_cart AND NOT has_purchase) AS keranjang_ditinggal,
               round(100 * count(*) FILTER (WHERE has_cart AND NOT has_purchase) / count(*) FILTER (WHERE has_cart), 2) AS persen_ditinggal,
               round(sum(cart_value) FILTER (WHERE has_cart AND NOT has_purchase), 0) AS nilai_ditinggal,
               round(sum(purchase_value), 0) AS nilai_pembelian,
               round(median(cart_value) FILTER (WHERE has_cart AND NOT has_purchase), 2) AS median_nilai_per_sesi_ditinggal,
               round(100 * count(*) FILTER (WHERE has_purchase AND NOT has_cart) / count(*) FILTER (WHERE has_purchase), 2) AS persen_beli_tanpa_keranjang
        FROM s"""), "p5_summary", "P5: ringkasan keranjang ditinggal (semua sesi sampel; nilai dalam satuan harga dataset)")
    ab = "count(*) FILTER (WHERE NOT has_purchase)"
    save(q(f"""
        SELECT cat_first_cart AS kategori, count(*) AS sesi_keranjang, {ab} AS ditinggal,
               round(100 * {ab} / count(*), 2) AS persen_ditinggal,
               round(sum(cart_value) FILTER (WHERE NOT has_purchase), 0) AS nilai_ditinggal,
               round(100 * sum(cart_value) FILTER (WHERE NOT has_purchase) / sum(sum(cart_value) FILTER (WHERE NOT has_purchase)) OVER (), 2) AS persen_nilai
        FROM s WHERE has_cart GROUP BY 1 ORDER BY nilai_ditinggal DESC"""),
         "p5_category", "P5: per kategori produk yang pertama dimasukkan ke keranjang")
    save(q(f"""
        SELECT {PRICE_BANDS.format(c='nullif(cart_value, 0)')} AS nilai_keranjang, count(*) AS sesi_keranjang,
               round(100 * {ab} / count(*), 2) AS persen_ditinggal
        FROM s WHERE has_cart GROUP BY 1 ORDER BY 1"""),
         "p5_value_band", "P5: per rentang nilai keranjang")
    save(q(f"""
        SELECT hour(t_first_cart) AS jam_utc, count(*) AS sesi_keranjang, round(100 * {ab} / count(*), 2) AS persen_ditinggal
        FROM s WHERE has_cart GROUP BY 1 ORDER BY 1"""),
         "p5_hour", "P5: per jam (UTC) keranjang pertama")
    save(q(f"""
        SELECT strftime(t_first_cart, '%Y-%m') AS bulan, count(*) AS sesi_keranjang,
               round(100 * {ab} / count(*), 2) AS persen_ditinggal,
               round(sum(cart_value) FILTER (WHERE NOT has_purchase), 0) AS nilai_ditinggal
        FROM s WHERE has_cart GROUP BY 1 ORDER BY 1"""),
         "p5_month", "P5: per bulan")
    save(q("""
        SELECT count(*) FILTER (WHERE c_k > 0) AS sesi_dinilai_dengan_keranjang,
               count(*) FILTER (WHERE c_k > 0 AND NOT y) AS ditinggal,
               round(100 * count(*) FILTER (WHERE c_k > 0 AND NOT y) / count(*) FILTER (WHERE c_k > 0), 2) AS persen_ditinggal
        FROM s WHERE scored"""),
         "p5_scored", "P5: versi sesi yang dinilai model (keranjang di event awal, tanpa pembelian sesudahnya; sama dengan eval_log f_c > 0 dan y = 0)")


# ------------------------------------------------------------------ P6
def p6():
    piv = """sum(events) FILTER (WHERE event_type = 'view')::BIGINT AS lihat,
             sum(events) FILTER (WHERE event_type = 'cart')::BIGINT AS keranjang,
             sum(events) FILTER (WHERE event_type = 'purchase')::BIGINT AS beli"""
    rates = """round(100 * keranjang / nullif(lihat, 0), 2) AS persen_lihat_ke_keranjang,
               round(100 * beli / nullif(keranjang, 0), 2) AS persen_keranjang_ke_beli,
               round(100 * beli / nullif(lihat, 0), 3) AS persen_lihat_ke_beli"""
    cat = q(f"""WITH a AS (SELECT category AS kategori, {piv} FROM e GROUP BY 1)
                SELECT kategori, lihat, keranjang, beli, round(100 * lihat / sum(lihat) OVER (), 2) AS persen_lihat, {rates}
                FROM a ORDER BY lihat DESC""")
    known = cat[cat.kategori != "unknown"]
    med = known.persen_lihat_ke_beli.median()
    cat["catatan"] = np.where((cat.kategori != "unknown") & (cat.persen_lihat >= 5) & (cat.persen_lihat_ke_beli < med),
                              "banyak dilihat, konversi di bawah median", "")
    save(cat, "p6_category", f"P6: funnel per kategori (median lihat ke beli kategori dikenal = {med:.3f}%)")
    save(q(f"""WITH a AS (SELECT month AS bulan, category AS kategori, {piv} FROM e GROUP BY 1, 2)
               SELECT bulan, kategori, lihat, keranjang, beli, {rates} FROM a ORDER BY bulan, lihat DESC"""),
         "p6_category_month", "P6: funnel per kategori per bulan")
    br = q(f"""WITH a AS (SELECT brand, {piv} FROM e WHERE brand <> 'unknown' GROUP BY 1)
               SELECT brand, lihat, keranjang, beli, {rates} FROM a ORDER BY lihat DESC LIMIT 25""")
    save(br, "p6_brand", "P6: funnel 25 brand dengan event lihat terbanyak")


# ------------------------------------------------------------------ P7
def p7():
    save(q("""
        SELECT hour(tk) AS jam_utc, count(*) AS sesi_dinilai, round(100 * avg(y::INT), 2) AS persen_membeli,
               round(100 * avg((c_k > 0)::INT), 2) AS persen_ada_keranjang
        FROM s WHERE scored GROUP BY 1 ORDER BY 1"""),
         "p7_hour_sessions", "P7: tingkat pembelian sesi dinilai per jam (UTC, waktu event ke-3)")
    d = q("""SELECT isodow(tk) AS d, count(*) AS sesi_dinilai, round(100 * avg(y::INT), 2) AS persen_membeli
             FROM s WHERE scored GROUP BY 1 ORDER BY 1""")
    d.insert(0, "hari", d.pop("d").map(HARI))
    save(d, "p7_weekday_sessions", "P7: tingkat pembelian sesi dinilai per hari")
    save(q("""
        SELECT hour_utc AS jam_utc, sum(events)::BIGINT AS event,
               round(100 * sum(events) / sum(sum(events)) OVER (), 2) AS persen_event,
               sum(events) FILTER (WHERE event_type = 'purchase')::BIGINT AS pembelian
        FROM e GROUP BY 1 ORDER BY 1"""),
         "p7_hour_events", "P7: volume event per jam (UTC), semua event sampel")


# ------------------------------------------------------------------ anomalies
def anom():
    import psycopg2
    from db_config import DB
    conn = psycopg2.connect(**DB)
    with conn.cursor() as cur:
        cur.execute("SELECT day, n, pos_rate FROM metrics_daily ORDER BY day")
        met = pd.DataFrame(cur.fetchall(), columns=["day", "n", "pos_rate"])
    conn.close()
    met["pos_rate"] = met.pos_rate.astype(float)
    mn, mr = met.n.median(), met.pos_rate.median()
    rows = []
    for r in met.itertuples():
        why = []
        if r.n > 2 * mn: why.append("volume tinggi")
        if r.n < 0.3 * mn: why.append("volume rendah")
        if r.pos_rate == 0: why.append("tingkat pembelian 0%")
        elif r.pos_rate > 2 * mr: why.append("tingkat pembelian tinggi")
        elif r.pos_rate < 0.3 * mr: why.append("tingkat pembelian rendah")
        if why:
            rows.append({"tanggal": r.day, "sesi": r.n, "persen_membeli": round(100 * r.pos_rate, 2),
                         "jenis": ", ".join(why)})
    out = pd.DataFrame(rows)
    save(out, "anomalies", f"Anomali harian (aturan dashboard: 0%, < 0,3x atau > 2x median; median sesi {mn:,.0f}, "
                           f"median tingkat pembelian {100 * mr:.2f}%)")


# ------------------------------------------------------------------ P8
def p8():
    con.execute("""CREATE TEMP TABLE u AS
        SELECT user_id, has_purchase, row_number() OVER (PARTITION BY user_id ORDER BY t1, sid) AS visit
        FROM s_all WHERE user_id IS NOT NULL""")
    save(q("""
        SELECT CASE WHEN visit >= 10 THEN '10+' WHEN visit >= 5 THEN '5-9' ELSE visit::VARCHAR END AS kunjungan_ke,
               count(*) AS sesi, round(100 * avg(has_purchase::INT), 2) AS persen_membeli
        FROM u GROUP BY 1 ORDER BY min(visit)"""),
         "p8_visit", "P8: tingkat pembelian per urutan kunjungan pengguna")
    save(q("""
        WITH f AS (SELECT user_id, min(visit) FILTER (WHERE has_purchase) AS first_buy, count(*) AS visits FROM u GROUP BY 1)
        SELECT count(*) AS pengguna,
               count(*) FILTER (WHERE visits > 1) AS pengguna_berulang,
               round(100 * count(*) FILTER (WHERE visits > 1) / count(*), 2) AS persen_berulang,
               count(first_buy) AS pengguna_pembeli,
               round(100 * count(*) FILTER (WHERE first_buy = 1) / count(first_buy), 2) AS persen_beli_di_kunjungan_1,
               median(first_buy) AS median_kunjungan_pembelian_pertama,
               round(avg(first_buy), 2) AS rata_kunjungan_pembelian_pertama
        FROM f"""),
         "p8_summary", "P8: ringkasan pengguna berulang (dalam periode 7 bulan sampel)")


# ------------------------------------------------------------------ segments
def seg():
    import psycopg2
    from db_config import DB
    conn = psycopg2.connect(**DB)
    with conn.cursor() as cur:
        cur.execute("SELECT session_id, t_pred, prob, y, f_c FROM eval_log ORDER BY t_pred, session_id")
        lg = pd.DataFrame(cur.fetchall(), columns=["sid", "t_pred", "prob", "y", "f_c"])
    conn.close()
    lg["prob"], lg["f_c"], lg["y"] = lg.prob.astype(float), lg.f_c.astype(float), lg.y.astype(int)
    lg = lg.sort_values(["t_pred", "sid"]).reset_index(drop=True)
    con.register("lg", lg)
    d = q("""SELECT lg.*, s.max_price_k, s.cart_value, s.purchase_value_after_k
             FROM lg JOIN s_all s USING (sid) ORDER BY lg.t_pred, lg.sid""")
    if len(d) < len(lg):
        print(f"Peringatan: {len(lg) - len(d):,} sesi eval_log tidak ditemukan di session_insights")
    cut = int(len(d) * args.train_frac)
    early, test = d.iloc[:cut], d.iloc[cut:].copy()
    p80, p95 = np.percentile(early.prob, [80, 95])
    test["segmen"] = np.select([test.prob >= p95, test.prob >= p80], ["1. Tinggi (5% teratas)", "2. Menengah (5-20%)"],
                               "3. Rendah (80% sisanya)")
    tot_buy, tot_val = test.y.sum(), test.purchase_value_after_k.sum()
    g = test.groupby("segmen").agg(sesi=("y", "size"), pembeli=("y", "sum"), ada_keranjang=("f_c", lambda x: (x > 0).mean()),
                                   nilai_beli=("purchase_value_after_k", "sum"), skor_rata=("prob", "mean")).reset_index()
    g["persen_sesi"] = (100 * g.sesi / len(test)).round(2)
    g["persen_membeli"] = (100 * g.pembeli / g.sesi).round(2)
    g["persen_dari_semua_pembeli"] = (100 * g.pembeli / tot_buy).round(2)
    g["persen_dari_nilai_pembelian"] = (100 * g.nilai_beli / tot_val).round(2)
    g["persen_ada_keranjang"] = (100 * g.ada_keranjang).round(2)
    g["skor_rata"] = (100 * g.skor_rata).round(2)
    g["usulan_tindakan"] = g.segmen.map({"1. Tinggi (5% teratas)": "permudah checkout, tanpa diskon",
                                         "2. Menengah (5-20%)": "pengingat atau voucher terbatas",
                                         "3. Rendah (80% sisanya)": "rekomendasi produk, tanpa biaya promo"})
    save(g[["segmen", "sesi", "persen_sesi", "skor_rata", "persen_membeli", "persen_dari_semua_pembeli",
            "persen_dari_nilai_pembelian", "persen_ada_keranjang", "usulan_tindakan"]],
         "seg_segments", f"Segmen skor pada periode uji (ambang dari periode awal: p80 = {p80:.4f}, p95 = {p95:.4f})")

    # ranking by probability vs by expected value (probability x highest price seen so far; known at prediction time)
    test["ev"] = test.prob * test.max_price_k.fillna(0)
    rows = []
    for k in (1, 5, 10, 20):
        n = int(len(test) * k / 100)
        for name, col in (("peluang", "prob"), ("nilai harapan", "ev")):
            top = test.nlargest(n, col)
            rows.append({"persen_sesi_ditandai": k, "urutan_menurut": name,
                         "precision": round(100 * top.y.mean(), 2),
                         "persen_pembeli_terjangkau": round(100 * top.y.sum() / tot_buy, 2),
                         "persen_nilai_pembelian_terjangkau": round(100 * top.purchase_value_after_k.sum() / tot_val, 2)})
    save(pd.DataFrame(rows), "seg_expected_value",
         "Urutan menurut peluang vs nilai harapan (peluang x harga tertinggi di event awal), periode uji")


# ------------------------------------------------------------------ early
def early():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    buyers = con.execute("SELECT count(*) FROM s_all WHERE has_purchase").fetchone()[0]
    feats = ["f_c", "f_avg_price", "f_max_price", "f_dur"]
    rows = []
    for k in args.early_k:
        path = f"ref/session_reference_k{k}.parquet"
        if k == 3 and not os.path.exists(path) and os.path.exists("ref/session_reference_7m.parquet"):
            path = "ref/session_reference_7m.parquet"
        if not os.path.exists(path):
            print(f"Lewati K={k}: {path} belum ada (python -m evaluation.build_session_reference --k {k} --out {path})")
            continue
        d = pd.read_parquet(path).sort_values(["t_pred", "session_id"]).reset_index(drop=True)
        cut = int(len(d) * args.train_frac)
        tr, te = d.iloc[:cut], d.iloc[cut:]
        if len(tr) > 1_000_000:
            tr = tr.sample(1_000_000, random_state=42)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[feats], tr.y)
        p = m.predict_proba(te[feats])[:, 1]
        n5 = max(1, int(len(te) * 0.05))
        top5 = te.y.values[np.argsort(-p)[:n5]].mean()
        rows.append({"K": k, "sesi_dinilai": len(d), "pembeli_terjangkau": int(d.y.sum()),
                     "persen_dari_semua_sesi_pembeli": round(100 * d.y.sum() / buyers, 2),
                     "ROC-AUC": round(roc_auc_score(te.y, p), 4), "PR-AUC": round(average_precision_score(te.y, p), 4),
                     "Precision@5%": round(100 * top5, 2), "lift@5%": round(top5 / te.y.mean(), 2)})
    if rows:
        save(pd.DataFrame(rows), "early_k",
             f"Seberapa dini: kualitas model vs cakupan pembeli per K (logistic regression, latih {args.train_frac:.0%} awal; "
             f"semua sesi pembeli = {buyers:,})")


for part in args.part:
    globals()[part]()
