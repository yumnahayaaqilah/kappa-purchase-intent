"""Dashboard pemantauan prediksi pembelian sesi (satu layar, padat). Membaca hasil pipeline dari PostgreSQL.
Prinsip: setiap panel menjawab satu pertanyaan (pertanyaan analitik WS3 atau pemantauan pipeline), tanpa dekorasi.
Grafik dibuat dari HTML/CSS/SVG sendiri sehingga mengisi kartu dengan tinggi tetap di semua ukuran layar."""
import html
import math

import numpy as np
import pandas as pd
import psycopg2
import streamlit as st

from db_config import DB

st.set_page_config(page_title="Pemantauan Prediksi Pembelian", layout="wide", initial_sidebar_state="collapsed")

BG, CARD, LINE, TXT, MUTED, SUB = "#0b1220", "#111a2e", "#1f2b45", "#e8edf7", "#8a97b3", "#c3cbe0"
BLUE, ORANGE = "#3987e5", "#d95926"          # seri 1 dan 2 (sudah divalidasi pada permukaan gelap)
EDGES = (4.689940164029909, 5.635253781136272)  # batas kelompok harga dari 70% awal data (evaluation/tune_from_log.py)
H1, H2, H3 = 264, 264, 252                    # tinggi tiga baris kartu (px)

st.markdown(f"""<style>
.stApp{{background:{BG};color:{TXT};font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
header[data-testid="stHeader"]{{display:none}}
#MainMenu,footer{{visibility:hidden}}
.block-container{{padding:.45rem .9rem .3rem .9rem!important;max-width:100%!important}}
div[data-testid="stVerticalBlock"]{{gap:.4rem}}
div[data-testid="stHorizontalBlock"]{{gap:.45rem}}
div[data-testid="stMarkdownContainer"]{{margin-bottom:0!important}}
div[data-testid="stMarkdownContainer"] p{{margin:0}}
[role="tablist"]{{gap:2px;margin-bottom:-6px}}
[role="tab"]{{height:28px;padding:0 12px;color:{MUTED}}}
[role="tab"] p{{font-size:.82rem}}
[role="tab"][aria-selected="true"],[role="tab"][aria-selected="true"] p{{color:{TXT}!important}}
.react-aria-SelectionIndicator,[data-baseweb="tab-highlight"]{{background-color:{BLUE}!important}}
[data-baseweb="tab-border"]{{background-color:{LINE}!important}}
[role="tabpanel"]{{padding-top:.35rem}}
.hd{{display:flex;justify-content:space-between;align-items:center;height:30px}}
.hd b{{font-size:1.02rem;font-weight:700;color:{TXT}}}
.hd span.s{{color:{MUTED};font-size:.78rem;margin-left:10px}}
.live{{display:flex;align-items:center;gap:8px;font-size:.78rem;color:{SUB}}}
.dot{{width:8px;height:8px;border-radius:50%;background:{BLUE};animation:pl 1.8s infinite}}
@keyframes pl{{0%{{box-shadow:0 0 0 0 rgba(57,135,229,.6)}}100%{{box-shadow:0 0 0 8px rgba(57,135,229,0)}}}}
.card{{background:{CARD};border:1px solid {LINE};border-radius:8px;padding:8px 10px 6px 10px;box-sizing:border-box;display:flex;flex-direction:column;overflow:hidden}}
.ch b{{display:block;font-size:.8rem;font-weight:600;color:{TXT};line-height:1.2}}
.ch b .i{{display:inline !important;margin-top:0 !important;color:#6b7893;font-weight:400;font-size:.72rem;margin-left:3px;cursor:help}}
.ch span{{display:block;font-size:.66rem;color:{MUTED};line-height:1.25;margin-top:1px}}
.cb{{flex:1;min-height:0;position:relative;margin-top:4px}}
.kpi{{flex-direction:row;align-items:center;justify-content:space-between;gap:8px;height:70px}}
.kpi .l{{font-size:.62rem;letter-spacing:.05em;color:{MUTED};text-transform:uppercase}}
.kpi .v{{font-size:1.45rem;font-weight:700;color:{TXT};line-height:1.15}}
.kpi .d{{font-size:.64rem;color:{MUTED};line-height:1.2}}
.kpi svg{{width:78px;height:34px;flex:none}}
.kpi svg polyline{{vector-effect:non-scaling-stroke}}
.lc{{position:absolute;left:34px;right:6px;top:4px;bottom:16px}}
.lc .gl{{position:absolute;left:0;right:0;height:1px;background:{LINE}}}
.lc .yl{{position:absolute;right:100%;margin-right:5px;transform:translateY(-50%);font-size:.62rem;color:{MUTED};white-space:nowrap}}
.lc .xl{{position:absolute;top:100%;margin-top:3px;font-size:.62rem;color:{MUTED};white-space:nowrap}}
.lc svg{{position:absolute;inset:0;width:100%;height:100%;overflow:visible}}
.lc svg polyline{{vector-effect:non-scaling-stroke}}
.lc .rf{{position:absolute;left:0;right:0;height:1px;background:{ORANGE}}}
.lc .rt{{position:absolute;left:4px;transform:translateY(-120%);font-size:.62rem;color:{SUB};white-space:nowrap;background:{CARD};padding:0 3px;border-radius:3px;z-index:3}}
.lc .ev{{position:absolute;width:7px;height:7px;border-radius:50%;background:{BLUE};transform:translate(-50%,-50%);box-shadow:0 0 0 2px {CARD}}}
.lc .et{{position:absolute;transform:translate(-100%,-160%);font-size:.7rem;font-weight:600;color:{TXT};white-space:nowrap;background:{CARD};padding:0 3px;border-radius:3px;z-index:3}}
.lc .hc{{position:absolute;top:0;bottom:0}}
.lc .hc:hover{{background:rgba(57,135,229,.12)}}
.bv{{position:absolute;inset:0;display:flex;flex-direction:column}}
.bp{{flex:1;position:relative;display:flex;align-items:flex-end;gap:5px;border-bottom:1px solid {LINE}}}
.bp .rf{{position:absolute;left:0;right:0;height:1px;background:{ORANGE};z-index:2}}
.bp .rt{{position:absolute;transform:translateY(-120%);font-size:.62rem;color:{SUB};white-space:nowrap;z-index:3;background:{CARD};padding:0 3px;border-radius:3px}}
.bc{{flex:1;height:100%;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;min-width:0}}
.bc .bv2{{font-size:.68rem;color:{TXT};margin-bottom:2px;font-weight:600;position:relative;z-index:3;background:{CARD};padding:0 3px;border-radius:3px}}
.bc .bar{{width:72%;background:{BLUE};border-radius:3px 3px 0 0;min-height:1px}}
.bx{{display:flex;gap:5px;padding-top:3px}}
.bx div{{flex:1;text-align:center;font-size:.62rem;color:{MUTED};line-height:1.2;min-width:0}}
.bx div i{{display:block;font-style:normal;color:#6b7893;font-size:.58rem}}
.hm{{display:grid;gap:2px;align-items:stretch}}
.hm .c{{border-radius:3px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:.8rem;font-weight:600;line-height:1.1}}
.hm .c i{{font-style:normal;font-weight:400;font-size:.56rem;opacity:.85}}
.hm .l{{font-size:.62rem;color:{MUTED};display:flex;align-items:center;justify-content:flex-end;padding-right:5px;text-align:right;line-height:1.1}}
.hm .t{{font-size:.62rem;color:{MUTED};text-align:center;line-height:1.1;padding-bottom:2px}}
table.tb{{width:100%;border-collapse:collapse;font-size:.74rem}}
table.tb th{{text-align:right;color:{MUTED};font-weight:500;padding:2px 5px;border-bottom:1px solid {LINE};font-size:.64rem}}
table.tb th:first-child,table.tb td:first-child{{text-align:left}}
table.tb td{{text-align:right;padding:4px 5px;border-bottom:1px solid #16213a;color:{TXT}}}
table.tb tr.best td{{font-weight:700;background:rgba(57,135,229,.10)}}
.mono{{font-family:ui-monospace,Consolas,monospace;font-size:.68rem;color:{MUTED}}}
.pill{{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.64rem;font-weight:600}}
.p-hi{{background:{BLUE};color:#fff}}
.p-md{{border:1px solid {BLUE};color:{TXT}}}
.p-lo{{border:1px solid {LINE};color:{MUTED}}}
.hb{{margin:3px 0}}
.hb .t{{display:flex;justify-content:space-between;font-size:.72rem;color:{TXT};margin-bottom:2px}}
.hb .t span{{color:{MUTED};font-size:.66rem}}
.hb .k{{height:6px;background:#18233d;border-radius:3px;overflow:hidden}}
.hb .k div{{height:100%;background:{BLUE};border-radius:3px}}
.fi{{font-size:.74rem;line-height:1.35;color:{TXT};margin:5px 0}}
.fi b{{font-weight:700}}
.note{{font-size:.64rem;color:{MUTED};margin-top:3px;line-height:1.25}}
.big{{font-size:.8rem;color:{SUB};line-height:1.5}}
.big b{{color:{TXT}}}
.gl2{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;font-size:.78rem;color:{MUTED};line-height:1.45}}
.gl2 b{{color:{TXT}}}
</style>""", unsafe_allow_html=True)

BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


# ------------------------------------------------------------------ format
def idn(n):
    return f"{n:,.0f}".replace(",", ".")


def dec(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def short(n):
    n = float(n)
    if n >= 1e6:
        return f"{dec(n / 1e6, 1)} jt"
    if n >= 1e3:
        return f"{dec(n / 1e3, 0)} rb"
    return idn(n)


def pc(x, d=1):
    return f"{dec(100 * x, d)}%"


def tgl(d):
    return f"{d.day} {BULAN[d.month - 1]} {str(d.year)[2:]}"


# ------------------------------------------------------------------ data
def q(sql):
    conn = psycopg2.connect(**DB)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()
    for c in df.columns:  # numeric dari PostgreSQL datang sebagai Decimal
        if df[c].dtype == object and len(df) and df[c].map(lambda v: type(v).__name__ == "Decimal").any():
            df[c] = df[c].astype(float)
    return df


@st.cache_data(ttl=45, show_spinner=False)
def q_slow(sql):
    return q(sql)


def ramp(t):
    """Satu hue (biru), gelap ke terang. t 0..1."""
    stops = [(0, (26, 44, 76)), (.55, (42, 120, 214)), (1, (158, 197, 244))]
    t = min(max(t, 0), 1)
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if t <= b:
            f = (t - a) / (b - a)
            return tuple(int(ca[i] + (cb[i] - ca[i]) * f) for i in range(3))
    return stops[-1][1]


# ------------------------------------------------------------------ komponen
def card(title, sub, body, h, extra="", tip=""):
    t = (f'<b title="{html.escape(tip)}">{title} <span class="i">&#9432;</span></b>' if tip else f"<b>{title}</b>")
    return (f'<div class="card" style="height:{h}px"><div class="ch">{t}<span>{sub}</span></div>'
            f'<div class="cb">{body}</div>{extra}</div>')


def kpi(label, value, sub, vals=None, tip=""):
    sp = ""
    v = [x for x in (vals or []) if x == x]
    if len(v) > 2:
        lo, hi = min(v), max(v)
        rng = (hi - lo) or 1
        pts = " ".join(f"{i * 100 / (len(v) - 1):.1f},{92 - (x - lo) / rng * 84:.1f}" for i, x in enumerate(v))
        sp = (f'<svg viewBox="0 0 100 100" preserveAspectRatio="none"><polyline fill="none" stroke="{BLUE}" '
              f'stroke-width="1.6" stroke-linejoin="round" points="{pts}"/></svg>')
    return (f'<div class="card kpi" title="{html.escape(tip)}"><div><div class="l">{label}</div><div class="v">{value}</div>'
            f'<div class="d">{sub}</div></div>{sp}</div>')


def line_html(days, vals, lo, hi, fmt, ref=None, ref_label="", ticks=4, tip=None):
    n = len(vals)
    if n < 2:
        return '<div class="note">Belum cukup data.</div>'
    X = lambda i: i * 100 / (n - 1)
    Y = lambda v: 100 - (min(max(v, lo), hi) - lo) / (hi - lo) * 100
    out = ['<div class="lc">']
    for k in range(ticks + 1):
        v = lo + (hi - lo) * k / ticks
        out.append(f'<div class="gl" style="top:{Y(v):.1f}%"></div><div class="yl" style="top:{Y(v):.1f}%">{fmt(v)}</div>')
    for i in sorted({0, (n - 1) // 3, 2 * (n - 1) // 3, n - 1}):
        al = "left:0" if i == 0 else (f"right:0" if i == n - 1 else f"left:{X(i):.1f}%;transform:translateX(-50%)")
        out.append(f'<div class="xl" style="{al}">{tgl(days[i])}</div>')
    if ref is not None:
        out.append(f'<div class="rf" style="top:{Y(ref):.1f}%"></div><div class="rt" style="top:{Y(ref):.1f}%">{ref_label}</div>')
    pts = " ".join(f"{X(i):.2f},{Y(v):.2f}" for i, v in enumerate(vals) if v == v)
    out.append(f'<svg viewBox="0 0 100 100" preserveAspectRatio="none"><polyline fill="none" stroke="{BLUE}" '
               f'stroke-width="1.8" stroke-linejoin="round" points="{pts}"/></svg>')
    last = n - 1
    while last > 0 and vals[last] != vals[last]:
        last -= 1
    out.append(f'<div class="ev" style="left:{X(last):.1f}%;top:{Y(vals[last]):.1f}%"></div>'
               f'<div class="et" style="left:{X(last):.1f}%;top:{Y(vals[last]):.1f}%">{fmt(vals[last])}</div>')
    for i, v in enumerate(vals):
        if v == v:
            t = html.escape(tip(i) if tip else f"{tgl(days[i])}: {fmt(v)}")
            out.append(f'<i class="hc" style="left:{i * 100 / n:.2f}%;width:{100 / n:.2f}%" title="{t}"></i>')
    out.append("</div>")
    return "".join(out)


def bars_v(values, labels, fmt, base=None, base_label="", ymax=None, subs=None, label_ends=False, tips=None):
    ymax = ymax or max(list(values) + ([base] if base else []) + [1e-9]) * 1.18
    out = ['<div class="bv"><div class="bp">']
    if base is not None:
        side = "left:2px" if values[0] < base else "right:2px"
        out.append(f'<div class="rf" style="bottom:{base / ymax * 100:.1f}%"></div>'
                   f'<div class="rt" style="bottom:{base / ymax * 100:.1f}%;{side}">{base_label}</div>')
    for i, v in enumerate(values):
        show = (not label_ends) or i in (0, len(values) - 1)
        t = html.escape(tips[i] if tips else f"{labels[i]}: {fmt(v)}")
        out.append(f'<div class="bc" title="{t}"><div class="bv2">{fmt(v) if show else "&nbsp;"}</div>'
                   f'<div class="bar" style="height:calc({v / ymax * 100:.1f}% - 16px)"></div></div>')
    out.append('</div><div class="bx">')
    for i, lab in enumerate(labels):
        out.append(f'<div>{lab}' + (f'<i>{subs[i]}</i>' if subs else "") + "</div>")
    out.append("</div></div>")
    return "".join(out)


def heat(rows, cols, cell, ncol_first=62):
    """cell(r, c) -> (nilai 0..1 untuk warna, teks besar, teks kecil, tooltip) atau None."""
    out = [f'<div class="hm" style="grid-template-columns:{ncol_first}px repeat({len(cols)},1fr);height:100%;'
           f'grid-template-rows:auto repeat({len(rows)},1fr)"><div></div>']
    out += [f'<div class="t">{c}</div>' for c in cols]
    for r in rows:
        out.append(f'<div class="l">{r}</div>')
        for c in cols:
            x = cell(r, c)
            if x is None:
                out.append('<div class="c"></div>')
                continue
            t, big, small, tip = x
            rgb = ramp(t)
            fg = "#0b1220" if t > .72 else "#fff"
            out.append(f'<div class="c" title="{html.escape(tip)}" style="background:rgb{rgb};color:{fg}">{big}'
                       + (f"<i>{small}</i>" if small else "") + "</div>")
    out.append("</div>")
    return "".join(out)


def hbars(items, fmt, maxv=None):
    maxv = maxv or max([i[1] for i in items] + [1e-9])
    return "".join(
        f'<div class="hb"><div class="t"><span style="color:{TXT};font-size:.72rem">{html.escape(str(a))}</span>'
        f'<span>{fmt(b)} {m}</span></div><div class="k"><div style="width:{max(1.5, b / maxv * 100):.1f}%"></div></div></div>'
        for a, b, m in items)


# ------------------------------------------------------------------ perhitungan dari histogram skor
class Hist:
    """Histogram skor (2000 kelompok) -> precision/recall pada 'tandai X% sesi teratas', desil, dan ambang skor."""
    W = 2000

    def __init__(self, h):
        h = h.sort_values("b")
        self.n = h["n"].to_numpy(float)[::-1]
        self.buy = h["buy"].to_numpy(float)[::-1]
        self.bins = h["b"].to_numpy(float)[::-1]
        self.cn, self.cb = np.cumsum(self.n), np.cumsum(self.buy)
        self.N, self.B = float(self.cn[-1]), float(self.cb[-1])

    def top(self, share):
        """(jumlah sesi, jumlah pembeli) pada `share` teratas menurut skor (interpolasi dalam kelompok)."""
        share = min(max(share, 0.0), 1.0)
        tgt = share * self.N
        i = min(int(np.searchsorted(self.cn, tgt)), len(self.cn) - 1)
        pn = self.cn[i - 1] if i > 0 else 0.0
        pb = self.cb[i - 1] if i > 0 else 0.0
        fr = (tgt - pn) / self.n[i] if self.n[i] > 0 else 0.0
        return tgt, pb + min(max(fr, 0), 1) * self.buy[i]

    def score_at(self, share):
        i = min(int(np.searchsorted(self.cn, share * self.N)), len(self.cn) - 1)
        return self.bins[i] / self.W

    def ops(self, share):
        n, b = self.top(share)
        prec = b / n if n else 0.0
        rec = b / self.B if self.B else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return prec, rec, f1

    def deciles(self):
        """Tingkat beli per desil, dari skor terendah (1) ke tertinggi (10)."""
        out = []
        for d in range(1, 11):
            lo_share, hi_share = (10 - d) / 10, (11 - d) / 10
            n1, b1 = self.top(lo_share)
            n2, b2 = self.top(hi_share)
            out.append((b2 - b1) / (n2 - n1) if n2 > n1 else 0.0)
        return out


# ------------------------------------------------------------------ halaman
@st.fragment(run_every=3)
def live():
    s = q("SELECT * FROM pipeline_stats WHERE id = 1")
    if s.empty:
        st.info("Belum ada data. Jalankan python -m pipeline.stream_processor lalu python -m pipeline.producer.")
        return
    s = s.iloc[0]
    until = pd.Timestamp(s.last_event_time).tz_convert("UTC").strftime("%d %b %Y %H:%M") if s.last_event_time is not None else "-"

    met = q("SELECT day, n, pos_rate, auc, pr_auc, p_at_5 FROM metrics_daily ORDER BY day")
    met["day"] = pd.to_datetime(met["day"])
    if len(met) > 3 and met.n.iloc[-1] < 0.5 * met.n.median():
        met = met.iloc[:-1]  # hari terakhir biasanya belum penuh
    ok = met.dropna(subset=["auc"])
    wavg = lambda col: float((ok[col] * ok.n).sum() / ok.n.sum()) if len(ok) else float("nan")
    w_auc, w_p5 = wavg("auc"), wavg("p_at_5")
    hist_df = q_slow("SELECT floor(prob * 2000)::int AS b, count(*) AS n, sum(y)::float AS buy FROM eval_log GROUP BY 1 ORDER BY 1")
    act = int(q("SELECT count(*) AS n FROM session_state WHERE predicted AND NOT closed").iloc[0].n)

    st.markdown(
        f'<div class="hd"><div><b>Pemantauan Prediksi Pembelian Sesi</b>'
        f'<span class="s">Kafka &middot; Spark Structured Streaming &middot; River &middot; PostgreSQL</span></div>'
        f'<div class="live"><span class="dot"></span>Data hingga {until} UTC &middot; {idn(s.batches)} batch</div></div>',
        unsafe_allow_html=True)

    if hist_df.empty:
        st.info("Menunggu sesi pertama selesai (30 menit menurut waktu data).")
        return
    H = Hist(hist_df)
    base = H.B / H.N
    n_sess, n_buy = int(H.N), int(H.B)
    t5, t20 = H.score_at(0.05), H.score_at(0.20)
    dec_rates = H.deciles()

    cart = q_slow("SELECT least(f_c, 3)::int AS c, count(*) AS n, avg(y) AS rate FROM eval_log GROUP BY 1 ORDER BY 1")
    E0, E1 = EDGES
    heat_df = q_slow(f"""SELECT least(f_c, 2)::int AS c,
        CASE WHEN f_avg_price <= 0 THEN 3 WHEN f_avg_price < {E0} THEN 0 WHEN f_avg_price < {E1} THEN 1 ELSE 2 END AS pb,
        count(*) AS n, avg(y) AS rate FROM eval_log GROUP BY 1, 2""")
    dur_df = q_slow(f"""SELECT CASE WHEN f_dur < {math.log(11)} THEN 0 WHEN f_dur < {math.log(31)} THEN 1
                              WHEN f_dur < {math.log(121)} THEN 2 ELSE 3 END AS d, count(*) AS n, avg(y) AS rate
                        FROM eval_log GROUP BY 1 ORDER BY 1""")
    cat = q_slow("SELECT category, views, carts, purchases FROM category_insights")
    hm = q_slow("""SELECT extract(isodow FROM t_pred AT TIME ZONE 'UTC')::int AS d,
                          extract(hour FROM t_pred AT TIME ZONE 'UTC')::int AS h, count(*) AS n FROM eval_log GROUP BY 1, 2""")
    live_rows = q("""SELECT session_id, n_events, n_cart, prob FROM session_state
                     WHERE predicted AND NOT closed ORDER BY last_time DESC LIMIT 5""")

    # ---------------- baris ringkasan
    k = st.columns(6)
    k[0].markdown(kpi("Total Event", short(s.events_total), "view, cart, purchase", tip="Jumlah event view, cart, dan purchase yang telah diproses pipeline."), unsafe_allow_html=True)
    k[1].markdown(kpi("Sesi Dinilai", short(n_sess), "sesi selesai berlabel", met.n.tolist(),
                      tip="Sesi yang diprediksi pada event ke-3 dan telah selesai (30 menit tanpa event baru), sehingga labelnya (purchase atau tidak) diketahui. Garis kecil: jumlah per hari."), unsafe_allow_html=True)
    k[2].markdown(kpi("Tingkat Pembelian", pc(base), f"{short(n_buy)} sesi purchase", (met.pos_rate * 100).tolist(),
                      tip="Persentase sesi dinilai yang berakhir dengan purchase setelah event ke-3."), unsafe_allow_html=True)
    k[3].markdown(kpi("AUC Harian", dec(w_auc, 2) if w_auc == w_auc else "-", "rata-rata; acuan acak 0,5", ok.auc.tolist(),
                      tip="Rata-rata ROC-AUC harian, berbobot jumlah sesi. 0,5 = prediksi acak, 1 = sempurna."), unsafe_allow_html=True)
    k[4].markdown(kpi("Precision@5%", pc(w_p5, 0) if w_p5 == w_p5 else "-",
                      f"lift {dec(w_p5 / base, 1)}x terhadap acak" if w_p5 == w_p5 else "-", ok.p_at_5.tolist(),
                      tip="Rata-rata harian persentase purchase pada 5% sesi dengan skor prediksi tertinggi. Lift = nilai ini dibagi tingkat pembelian rata-rata."), unsafe_allow_html=True)
    k[5].markdown(kpi("Sesi Aktif", idn(act), "diprediksi, belum selesai", tip="Sesi yang sudah diprediksi dan masih menunggu sesi selesai (30 menit tanpa event baru)."), unsafe_allow_html=True)

    # ---------------- baris 1: kinerja model
    a = st.columns([1.2, 1.2, 1.1, 1.05])
    if len(ok) > 1:
        lo_a, hi_a = 0.5, min(1.0, math.ceil((ok.auc.max() + 0.02) * 20) / 20)
        a[0].markdown(card("AUC Harian", "ROC-AUC per hari; sumbu mulai dari 0,5 (acak)",
                           line_html(list(ok.day), ok.auc.astype(float).tolist(), lo_a, hi_a, lambda v: dec(v, 2)), H1,
                           tip="Area di bawah kurva ROC pada sesi hari tersebut. Mengukur kemampuan model menempatkan sesi purchase di atas sesi tanpa purchase."),
                      unsafe_allow_html=True)
        p5 = ok.p_at_5.astype(float).tolist()
        a[1].markdown(card("Precision@5% Harian", "Garis oranye: tingkat pembelian rata-rata semua sesi",
                           line_html(list(ok.day), p5, 0, min(1.0, max(max(p5), base) * 1.25),
                                     lambda v: pc(v, 0), ref=base, ref_label=f"rata-rata {pc(base)}"), H1,
                           tip="Persentase purchase pada 5% sesi berskor tertinggi di hari tersebut. Garis biru di atas garis oranye berarti model lebih baik daripada memilih sesi secara acak."),
                      unsafe_allow_html=True)
    else:
        a[0].markdown(card("AUC Harian", "", '<div class="note">Belum cukup hari data.</div>', H1), unsafe_allow_html=True)
        a[1].markdown(card("Precision@5% Harian", "", '<div class="note">Belum cukup hari data.</div>', H1), unsafe_allow_html=True)
    a[2].markdown(card("Tingkat Pembelian menurut Desil Skor", "Sesi dibagi 10 kelompok sama besar menurut skor prediksi",
                       bars_v(dec_rates, [str(i) for i in range(1, 11)], lambda v: pc(v, 0), base=base,
                              base_label=f"rata-rata {pc(base)}", label_ends=True,
                              subs=["rendah"] + [""] * 8 + ["tinggi"],
                              tips=[f"Desil {i + 1}: {pc(v)} purchase" for i, v in enumerate(dec_rates)]), H1,
                       tip="Sesi diurutkan dari skor terendah (desil 1) ke tertinggi (desil 10). Tinggi batang = persentase purchase. Batang yang naik ke kanan menunjukkan skor berurutan dengan benar."),
                  unsafe_allow_html=True)
    best = max(range(1, 51), key=lambda p: H.ops(p / 100)[2])
    shares = sorted({2, 5, 10, 20, best})
    trs = ""
    for p in shares:
        pr, rc, f1 = H.ops(p / 100)
        trs += (f'<tr class="{"best" if p == best else ""}"><td>Top {p}%{" *" if p == best else ""}</td><td>{pc(pr, 0)}</td>'
                f'<td>{pc(rc, 0)}</td><td>{dec(f1, 2)}</td><td>{dec(pr / base, 1)}x</td></tr>')
    a[3].markdown(card("Precision, Recall, F1 menurut Ambang", "Sesi dengan skor tertinggi ditandai sebagai calon pembeli",
                       '<table class="tb"><tr><th>Ditandai</th><th>Precision</th><th>Recall</th><th>F1</th><th>Lift</th></tr>'
                       + trs + '</table><div class="note">* F1 tertinggi. Lift = Precision / tingkat pembelian rata-rata.</div>', H1,
                       tip="Precision: proporsi sesi yang ditandai yang benar-benar purchase. Recall: proporsi seluruh sesi purchase yang berhasil ditandai. F1: rata-rata harmonik keduanya. Dihitung pada seluruh sesi dinilai."),
              unsafe_allow_html=True)

    # ---------------- baris 2: faktor yang berpengaruh
    b = st.columns([1, 1.15, 1, 1.2])
    cr = {int(r.c): (float(r.rate), int(r.n)) for r in cart.itertuples()}
    b[0].markdown(card("Tingkat Pembelian menurut Jumlah Cart", "Jumlah event cart pada 3 event pertama sesi",
                       bars_v([cr.get(i, (0, 0))[0] for i in range(4)], [f"{i} cart" for i in range(4)],
                              lambda v: pc(v), base=base, base_label=f"rata-rata {pc(base)}",
                              subs=[f"{short(cr.get(i, (0, 0))[1])} sesi" for i in range(4)]), H2,
                       tip="Prediksi dibuat pada 3 event pertama, sehingga jumlah view = 3 dikurangi jumlah cart."),
                  unsafe_allow_html=True)
    pr_names = [f"Rendah (&lt;{idn(math.exp(E0) - 1)})", f"Sedang ({idn(math.exp(E0) - 1)}-{idn(math.exp(E1) - 1)})", f"Tinggi (&ge;{idn(math.exp(E1) - 1)})"]
    grid = {(int(r.c), int(r.pb)): (float(r.rate), int(r.n)) for r in heat_df.itertuples()}
    valid_n = sum(v[1] for kx, v in grid.items() if kx[1] < 3)
    vals = [v[0] for kx, v in grid.items() if kx[1] < 3 and v[1] >= 0.005 * max(valid_n, 1)]
    vmax = max(vals + [1e-9])
    rows_lab = ["0 cart", "1 cart", "2+ cart"]

    def cell(r, c):
        key = (rows_lab.index(r), pr_names.index(c))
        if key not in grid:
            return None
        rate, n = grid[key]
        return rate / vmax, pc(rate, 0 if rate >= 0.1 else 1), f"{short(n)} sesi", f"{r}, harga {c}: {pc(rate)} purchase dari {idn(n)} sesi"

    no_price = sum(v[1] for kx, v in grid.items() if kx[1] == 3)
    b[1].markdown(card("Tingkat Pembelian menurut Cart dan Harga", "Warna = tingkat pembelian; angka kecil = jumlah sesi",
                       heat(rows_lab, pr_names, cell, 56), H2,
                       extra=f'<div class="note">Kelompok harga: tertil rata-rata harga produk pada 3 event pertama. Tanpa harga valid: {short(no_price)} sesi (tidak ditampilkan).</div>',
                       tip="Batas kelompok harga ditetapkan dari 70% awal data dan sama dengan yang dipakai model. Sel yang lebih gelap adalah segmen dengan tingkat pembelian lebih rendah."),
              unsafe_allow_html=True)
    dr = {int(r.d): (float(r.rate), int(r.n)) for r in dur_df.itertuples()}
    dlab = ["&le;10 dtk", "10-30 dtk", "30 dtk-2 mnt", "&gt;2 mnt"]
    b[2].markdown(card("Tingkat Pembelian menurut Durasi Awal", "Selang waktu dari event ke-1 hingga ke-3",
                       bars_v([dr.get(i, (0, 0))[0] for i in range(4)], dlab, lambda v: pc(v), base=base,
                              base_label=f"rata-rata {pc(base)}", subs=[f"{short(dr.get(i, (0, 0))[1])} sesi" for i in range(4)]), H2),
                  unsafe_allow_html=True)

    def lab(p):
        if p >= t5:
            return "Tinggi", "p-hi"
        if p >= t20:
            return "Sedang", "p-md"
        return "Rendah", "p-lo"

    if len(live_rows):
        body = "".join(
            f'<tr><td class="mono">{html.escape(str(r.session_id)[:9])}</td><td>{int(r.n_events)}</td><td>{int(r.n_cart)}</td>'
            f'<td>{r.prob * 100:.0f}%</td><td><span class="pill {lab(r.prob)[1]}">{lab(r.prob)[0]}</span></td></tr>'
            for r in live_rows.itertuples())
        tb = ('<table class="tb"><tr><th>Sesi</th><th>Event</th><th>Cart</th><th>Probabilitas</th><th>Status</th></tr>' + body + "</table>"
              f'<div class="note">Status menurut skor: Tinggi = 5% teratas (&ge; {pc(t5, 0)}), Sedang = 5-20% (&ge; {pc(t20, 0)}), Rendah = lainnya.</div>')
    else:
        tb = '<div class="note">Tidak ada sesi aktif.</div>'
    b[3].markdown(card("Sesi Aktif", "Sesi terbaru yang telah diprediksi dan belum selesai", tb, H2,
                       tip="Probabilitas purchase dihitung saat sesi mencapai event ke-3. Batas status ditentukan dari sebaran skor seluruh sesi dinilai."),
                  unsafe_allow_html=True)

    # ---------------- baris 3: konteks dan ringkasan
    c = st.columns([1, 1, 1.25, 1.1])
    v, ca, pu = int(cat.views.sum()), int(cat.carts.sum()), int(cat.purchases.sum())
    n_cart_s = sum(cr[i][1] for i in (1, 2, 3) if i in cr)
    fun = ('<div class="note" style="margin:0 0 1px 0">Event</div>'
           + hbars([("View", v, ""), ("Cart", ca, f"&middot; {pc(ca / v) if v else '-'} dari view"),
                    ("Purchase", pu, f"&middot; {pc(pu / ca) if ca else '-'} dari cart")], short)
           + '<div class="note" style="margin:6px 0 1px 0">Sesi</div>'
           + hbars([("Sesi dinilai", n_sess, ""), ("Sesi dengan cart", n_cart_s, f"&middot; {pc(n_cart_s / n_sess)} dari sesi dinilai"),
                    ("Sesi purchase", n_buy, f"&middot; {pc(base)} dari sesi dinilai")], short, maxv=n_sess))
    c[0].markdown(card("Funnel Konversi", "Jumlah per tahap pada level event dan level sesi", fun, H3,
                       tip="Level event: seluruh event yang diproses. Level sesi: sesi dinilai; cart dihitung pada 3 event pertama sesi."),
                  unsafe_allow_html=True)
    nm = lambda x: "(tanpa kategori)" if x == "unknown" else str(x).title()
    top = cat.assign(conv=lambda d: d.purchases / d.views.where(d.views > 0)).sort_values("purchases", ascending=False).head(6)
    c[1].markdown(card("Kategori Teratas menurut Purchase", "Jumlah purchase; persentase = purchase / view pada kategori",
                       hbars([(nm(r.category), int(r.purchases), f"&middot; {pc(r.conv)}") for r in top.itertuples()], short) if len(top)
                       else '<div class="note">Belum ada data.</div>', H3), unsafe_allow_html=True)
    g = {(int(r.d), int(r.h)): int(r.n) for r in hm.itertuples()}
    mx = max(g.values()) if g else 1
    dn = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
    cols_h = [str(h_) if h_ % 6 == 0 else "" for h_ in range(24)]
    out = ['<div class="hm" style="grid-template-columns:28px repeat(24,1fr);height:100%;grid-template-rows:auto repeat(7,1fr);gap:1px"><div></div>']
    out += [f'<div class="t">{x}</div>' for x in cols_h]
    for d in range(1, 8):
        out.append(f'<div class="l">{dn[d - 1]}</div>')
        for h_ in range(24):
            n = g.get((d, h_), 0)
            out.append(f'<div class="c" title="{dn[d - 1]} {h_:02d}:00: {idn(n)} sesi" style="background:rgb{ramp(n / mx)};border-radius:2px"></div>')
    out.append("</div>")
    c[2].markdown(card("Volume Sesi menurut Hari dan Jam", "Jumlah sesi dinilai per hari dan jam (UTC); warna lebih terang = lebih banyak",
                       "".join(out), H3), unsafe_allow_html=True)

    pr5, rc5, _ = H.ops(0.05)
    pbest, rbest, _ = H.ops(best / 100)
    fi = []
    if 0 in cr and cr[0][0] > 0:
        with_c = sum(cr[i][0] * cr[i][1] for i in (1, 2, 3) if i in cr) / max(sum(cr[i][1] for i in (1, 2, 3) if i in cr), 1)
        fi.append(f"Sesi dengan cart: purchase <b>{pc(with_c)}</b>; tanpa cart: <b>{pc(cr[0][0])}</b> (rasio {dec(with_c / cr[0][0], 1)}x).")
    fi.append(f"Desil skor tertinggi: purchase <b>{pc(dec_rates[-1])}</b>, {dec(dec_rates[-1] / base, 1)}x rata-rata semua sesi.")
    fi.append(f"Top 5%: Precision <b>{pc(pr5, 0)}</b>, Recall <b>{pc(rc5, 0)}</b>.")
    fi.append(f"F1 tertinggi pada Top {best}%: Precision {pc(pbest, 0)}, Recall {pc(rbest, 0)}.")
    if len(top):
        fi.append(f"Kategori dengan purchase terbanyak: <b>{html.escape(nm(top.iloc[0].category))}</b> ({short(top.iloc[0].purchases)}).")
    c[3].markdown(card("Ringkasan", "Dihitung otomatis dari data yang tampil", "".join(f'<div class="fi">{t}</div>' for t in fi), H3),
                  unsafe_allow_html=True)


live()
