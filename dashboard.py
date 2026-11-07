"""Dashboard pemantauan prediksi pembelian sesi. Membaca hasil pipeline dari PostgreSQL.

Tiga tampilan, satu pertanyaan per panel:
  Pipeline   apakah data mengalir dan diproses (status, volume harian, sesi aktif)
  Model      seberapa baik skor memisahkan pembeli (AUC, Precision@5%, desil, ambang, bobot model)
  Pelanggan  perilaku apa yang menyertai pembelian (keranjang dan harga, tahapan belanja, pembelian per kategori,
             keranjang tanpa pembelian, tingkat pembelian per jam)
Aturan warna: biru = data utama (seri 1); oranye, hijau toska, kuning = seri 2-4 hanya bila satu grafik memuat
beberapa seri (selalu dengan legenda); abu-abu = garis acuan; hijau status hanya untuk status pipeline.
Grafik dibuat dari HTML/CSS/SVG sendiri sehingga ukuran kartu tetap di semua layar.

    streamlit run dashboard.py
"""
import html
import math
import re
import time

import numpy as np
import pandas as pd
import psycopg2
import streamlit as st

from db_config import DB

st.set_page_config(page_title="Pemantauan Prediksi Pembelian", layout="wide", initial_sidebar_state="collapsed")

BG, CARD, LINE, TXT, MUTED, SUB, REF = "#0b1220", "#111a2e", "#1f2b45", "#e8edf7", "#8a97b3", "#c3cbe0", "#8a97b3"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"]   # seri 1-4, lolos uji buta warna pada permukaan kartu
BLUE, ORANGE = SERIES[0], SERIES[1]
GOOD = "#0ca30c"                                       # status: pipeline berjalan
WARN = "#fab219"                                       # status: anomali harian
EDGES = (4.689940164029909, 5.635253781136272)          # batas kelompok harga dari 70% awal data (evaluation/tune_from_log.py)
H = 350                                                # tinggi kartu (px)
LIVE_SEC = 90                                          # dianggap berjalan bila ada hasil batch dalam 90 detik terakhir
FEAT_NAMES = {"c": "Jumlah masuk keranjang", "avg_price": "Harga rata-rata", "max_price": "Harga tertinggi", "dur": "Durasi awal"}
CATEGORY_NAMES = {"electronics": "Elektronik", "appliances": "Peralatan rumah tangga", "computers": "Komputer", "apparel": "Pakaian",
                  "furniture": "Furnitur", "auto": "Otomotif", "construction": "Perkakas dan konstruksi", "kids": "Anak-anak",
                  "sport": "Olahraga", "accessories": "Aksesori", "country_yard": "Taman dan pekarangan", "medicine": "Kesehatan",
                  "stationery": "Alat tulis", "unknown": "Tanpa kategori"}

st.markdown(f"""<style>
.stApp{{background:{BG};color:{TXT};font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
header[data-testid="stHeader"]{{display:none}}
#MainMenu,footer{{visibility:hidden}}
.block-container{{padding:.5rem .9rem .4rem .9rem!important;max-width:100%!important}}
div[data-testid="stVerticalBlock"]{{gap:.45rem}}
div[data-testid="stHorizontalBlock"]{{gap:.45rem}}
div[data-testid="stMarkdownContainer"]{{margin-bottom:0!important}}
div[data-testid="stMarkdownContainer"] p{{margin:0}}
div[data-testid="stButtonGroup"] button{{min-height:30px;padding:0 14px;background:{CARD};border-color:{LINE};color:{MUTED}}}
div[data-testid="stButtonGroup"] button[aria-checked="true"]{{background:{LINE}!important;border-color:{BLUE}!important}}
div[data-testid="stButtonGroup"] button[aria-checked="true"] p{{color:{TXT}!important}}
div[data-testid="stButtonGroup"] button:hover p{{color:{TXT}!important}}
div[data-testid="stButtonGroup"] button:focus{{box-shadow:none!important}}
div[data-testid="stButtonGroup"] button p{{font-size:.8rem}}
.hd{{display:flex;justify-content:space-between;align-items:center;min-height:30px;gap:12px}}
.hd b{{font-size:1.02rem;font-weight:700;color:{TXT}}}
.hd span.s{{color:{MUTED};font-size:.76rem;margin-left:10px}}
.fresh{{display:flex;align-items:center;gap:8px;font-size:.76rem;color:{SUB};white-space:nowrap}}
.dot{{width:8px;height:8px;border-radius:50%;flex:none}}
.dot.on{{background:{GOOD};animation:pl 1.8s infinite}}
.dot.off{{background:none;border:1.5px solid {MUTED}}}
@keyframes pl{{0%{{box-shadow:0 0 0 0 rgba(12,163,12,.55)}}100%{{box-shadow:0 0 0 7px rgba(12,163,12,0)}}}}
.card{{position:relative;background:{CARD};border:1px solid {LINE};border-radius:8px;padding:9px 11px 7px 11px;box-sizing:border-box;display:flex;flex-direction:column;overflow:hidden}}
.i{{display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:50%;border:1px solid #3a4866;color:{SUB};font:700 .6rem/1 system-ui,sans-serif;margin-left:7px;cursor:pointer;vertical-align:1px;outline:none;user-select:none;text-transform:none;letter-spacing:0}}
.i:hover,.i:focus{{border-color:{BLUE};color:{TXT};background:rgba(57,135,229,.18)}}
.tip{{display:none;position:absolute;left:10px;right:10px;top:30px;z-index:30;background:#18233b;border:1px solid #2c3b5c;border-radius:6px;padding:9px 11px;font-size:.72rem;line-height:1.5;color:{TXT};box-shadow:0 10px 28px rgba(0,0,0,.45)}}
.tip p{{margin:0 0 5px 0!important}}
.tip p:last-child{{margin-bottom:0!important}}
.card:has(.i:hover) .tip,.card:has(.i:focus) .tip{{display:block}}
.kpi .tip{{top:calc(100% + 6px);left:0;right:auto;width:300px}}
.fl{{position:absolute;width:9px;height:9px;border-radius:50%;border:2px solid {WARN};background:{CARD};transform:translate(-50%,-50%);z-index:5}}
.ch b{{display:block;font-size:.82rem;font-weight:600;color:{TXT};line-height:1.25}}
.ch > span{{display:block;font-size:.68rem;color:{MUTED};line-height:1.3;margin-top:1px}}
.lg{{display:flex;flex-wrap:wrap;gap:4px 12px;margin-top:4px;font-size:.66rem;color:{SUB}}}
.lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
.lg i.c{{border-radius:50%;width:9px;height:9px}}
.lg i.r{{height:0;border-top:1px solid {REF};border-radius:0;vertical-align:3px;width:14px}}
.lg i.o{{width:7px;height:7px;border-radius:50%;border:2px solid {WARN};background:none}}
.tag{{display:inline-block;margin-left:8px;padding:0 7px;border-radius:999px;border:1px solid {BLUE};font-size:.6rem;font-weight:600;color:{TXT};vertical-align:1px}}
.cb{{flex:1;min-height:0;position:relative;margin-top:6px}}
.ft{{font-size:.64rem;color:{MUTED};margin-top:4px;line-height:1.3}}
.kpi{{flex-direction:row;align-items:center;justify-content:space-between;gap:8px;height:72px;overflow:visible}}
.kpi .l{{font-size:.62rem;letter-spacing:.05em;color:{MUTED};text-transform:uppercase}}
.kpi .v{{font-size:1.45rem;font-weight:700;color:{TXT};line-height:1.15}}
.kpi .d{{font-size:.64rem;color:{MUTED};line-height:1.2}}
.kpi svg{{width:80px;height:34px;flex:none}}
.kpi svg polyline,.lc svg polyline{{vector-effect:non-scaling-stroke}}
.lc{{position:absolute;left:38px;right:8px;top:6px;bottom:18px}}
.lc .gl{{position:absolute;left:0;right:0;height:1px;background:{LINE}}}
.lc .yl{{position:absolute;right:100%;margin-right:6px;transform:translateY(-50%);font-size:.62rem;color:{MUTED};white-space:nowrap}}
.lc .xl{{position:absolute;top:100%;margin-top:4px;font-size:.62rem;color:{MUTED};white-space:nowrap}}
.lc svg{{position:absolute;inset:0;width:100%;height:100%;overflow:visible}}
.rf{{position:absolute;left:0;right:0;height:0;border-top:1px solid {REF};z-index:1}}
.rt{{position:absolute;font-size:.62rem;color:{SUB};white-space:nowrap;background:{CARD};padding:0 4px;border-radius:3px;z-index:3;transform:translateY(-115%)}}
.ev{{position:absolute;width:8px;height:8px;border-radius:50%;transform:translate(-50%,-50%);box-shadow:0 0 0 2px {CARD};z-index:2}}
.et{{position:absolute;transform:translate(-100%,-165%);font-size:.7rem;font-weight:600;color:{TXT};white-space:nowrap;background:{CARD};padding:0 3px;border-radius:3px;z-index:3}}
.hc{{position:absolute;top:0;bottom:0;z-index:4}}
.dl{{position:absolute;left:calc(100% + 12px);transform:translateY(-50%);font-size:.68rem;color:{TXT};white-space:nowrap;line-height:1.2;z-index:3}}
.dl i{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:0}}
.dl b{{font-weight:600;margin-left:4px}}
.hc:hover{{background:rgba(138,151,179,.10)}}
.sm{{position:relative;height:calc(50% - 8px);margin-bottom:16px}}
.sm .t{{position:absolute;top:-2px;left:0;font-size:.66rem;color:{SUB}}}
.sm .lc{{top:24px}}
.bv{{position:absolute;inset:0;display:flex;flex-direction:column}}
.bp{{flex:1;position:relative;display:flex;gap:4px;border-bottom:1px solid {LINE}}}
.bc{{flex:1;position:relative;min-width:0}}
.bc .bar{{position:absolute;bottom:0;left:50%;transform:translateX(-50%);width:min(24px,72%);background:{BLUE};border-radius:4px 4px 0 0;min-height:1px}}
.bc .bt{{position:absolute;left:50%;transform:translateX(-50%);font-size:.66rem;font-weight:600;color:{TXT};white-space:nowrap;margin-bottom:3px;z-index:3}}
.bc .mk{{position:absolute;left:50%;width:9px;height:9px;border-radius:50%;background:{ORANGE};transform:translate(-50%,50%);box-shadow:0 0 0 2px {CARD};z-index:2}}
.bc .hit{{position:absolute;inset:0;z-index:4}}
.bc .hit:hover{{background:rgba(138,151,179,.10)}}
.bx{{display:flex;gap:4px;padding-top:4px}}
.bx div{{flex:1;text-align:center;font-size:.62rem;color:{MUTED};line-height:1.2;min-width:0}}
.ax{{display:flex;justify-content:space-between;font-size:.6rem;color:#6b7893;padding-top:2px}}
.hm{{display:grid;gap:2px;align-items:stretch}}
.hm .c{{border-radius:3px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:.82rem;font-weight:600;line-height:1.15}}
.hm .c i{{font-style:normal;font-weight:400;font-size:.6rem;opacity:.85}}
.hm .l{{font-size:.64rem;color:{MUTED};display:flex;align-items:center;justify-content:flex-end;padding-right:6px;text-align:right;line-height:1.15}}
.hm .t{{font-size:.62rem;color:{MUTED};text-align:center;line-height:1.15;padding-bottom:3px}}
table.tb{{width:100%;border-collapse:collapse;font-size:.74rem;font-variant-numeric:tabular-nums}}
table.tb th{{text-align:right;color:{MUTED};font-weight:500;padding:3px 6px;border-bottom:1px solid {LINE};font-size:.64rem}}
table.tb th:first-child,table.tb td:first-child{{text-align:left}}
table.tb td{{text-align:right;padding:5px 6px;border-bottom:1px solid #16213a;color:{TXT}}}
table.tb tr.best td{{font-weight:700;background:rgba(57,135,229,.10)}}
table.tb tr.rule td{{color:{SUB};border-top:1px solid {LINE}}}
.mono{{font-family:ui-monospace,Consolas,monospace;font-size:.7rem;color:{SUB}}}
.pill{{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.64rem;font-weight:600}}
.p-hi{{background:{BLUE};color:#fff}}
.p-md{{border:1px solid {BLUE};color:{TXT}}}
.p-lo{{border:1px solid {LINE};color:{MUTED}}}
.hb{{margin:4px 0 6px 0}}
.hb .t{{display:flex;justify-content:space-between;font-size:.74rem;color:{TXT};margin-bottom:3px;gap:8px}}
.hb .t span{{color:{MUTED};font-size:.68rem;white-space:nowrap}}
.hb .k{{height:8px;background:#18233d;border-radius:4px;overflow:hidden}}
.hb .k div{{height:100%;background:{BLUE};border-radius:4px}}
.grp{{font-size:.64rem;color:{MUTED};text-transform:uppercase;letter-spacing:.05em;margin:2px 0 2px 0}}
.st{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;height:100%}}
.st div{{border:1px solid {LINE};border-radius:6px;padding:7px 9px;display:flex;flex-direction:column;justify-content:center;min-width:0}}
.st .l{{font-size:.6rem;letter-spacing:.05em;color:{MUTED};text-transform:uppercase}}
.st .v{{font-size:1.05rem;font-weight:700;color:{TXT};line-height:1.25;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.st .d{{font-size:.62rem;color:{MUTED};line-height:1.2}}
.note{{font-size:.72rem;color:{MUTED};line-height:1.35}}
</style>""", unsafe_allow_html=True)

BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


# ------------------------------------------------------------------ format
def idn(n):
    return f"{n:,.0f}".replace(",", ".")


def dec(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def short(n):
    """Angka ringkas: 950 / 7,2 ribu / 105 ribu / 3,7 juta."""
    n = float(n)
    if n >= 1e6:
        return f"{dec(n / 1e6, 1)} juta"
    if n >= 1e4:
        return f"{dec(n / 1e3, 0)} ribu"
    if n >= 1e3:
        return f"{dec(n / 1e3, 1)} ribu"
    return idn(n)


def pc(x, d=1):
    return f"{dec(100 * x, d)}%"


def tgl(d):
    return f"{d.day} {BULAN[d.month - 1]} {d.year}"


def utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def waktu(ts):
    ts = utc(ts)
    return f"{ts.day} {BULAN[ts.month - 1]} {ts.year} {ts:%H:%M}"


def nz(x):
    return 0 if x is None or pd.isna(x) else x


def lalu(sec):
    if sec < 60:
        return f"{int(sec)} detik lalu"
    if sec < 3600:
        return f"{int(sec // 60)} menit lalu"
    if sec < 86400:
        return f"{int(sec // 3600)} jam lalu"
    return f"{int(sec // 86400)} hari lalu"


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


@st.cache_data(ttl=30, show_spinner=False)
def q_slow(sql):
    return q(sql)


def ramp(t):
    """Satu hue (biru), gelap = sedikit, terang = banyak. t 0..1."""
    stops = [(0, (26, 44, 76)), (.55, (42, 120, 214)), (1, (158, 197, 244))]
    t = min(max(t, 0), 1)
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if t <= b:
            f = (t - a) / (b - a)
            return tuple(int(ca[i] + (cb[i] - ca[i]) * f) for i in range(3))
    return stops[-1][1]


# ------------------------------------------------------------------ komponen
def info(tip):
    """Ikon informasi; penjelasan tampil saat ikon diarahkan atau diklik."""
    if not tip:
        return "", ""
    body = "".join(f"<p>{t}</p>" for t in (tip if isinstance(tip, (list, tuple)) else [tip]))
    return '<span class="i" tabindex="0" role="button" aria-label="Informasi">i</span>', f'<div class="tip">{body}</div>'


def card(title, sub, body, h=H, legend="", tip=""):
    icon, box = info(tip)
    return (f'<div class="card" style="height:{h}px"><div class="ch"><b>{title}{icon}</b><span>{sub}</span>{legend}</div>'
            f'<div class="cb">{body}</div>{box}</div>')


def legend(items):
    """items: (label, color, kind) dengan kind 'b' kotak, 'c' titik, 'r' garis acuan."""
    out = []
    for lab, col, kind in items:
        style = "" if kind in ("r", "o") else f' style="background:{col}"'
        out.append(f'<span><i class="{kind}"{style}></i>{lab}</span>')
    return f'<div class="lg">{"".join(out)}</div>'


def kpi(label, value, sub, vals=None, tip=""):
    sp = ""
    v = [x for x in (vals or []) if x == x]
    if len(v) > 2:
        lo, hi = min(v), max(v)
        rng = (hi - lo) or 1
        pts = " ".join(f"{i * 100 / (len(v) - 1):.1f},{92 - (x - lo) / rng * 84:.1f}" for i, x in enumerate(v))
        sp = (f'<svg viewBox="0 0 100 100" preserveAspectRatio="none"><polyline fill="none" stroke="{BLUE}" '
              f'stroke-width="1.6" stroke-linejoin="round" points="{pts}"/></svg>')
    icon, box = info(tip)
    return (f'<div class="card kpi"><div><div class="l">{label}{icon}</div><div class="v">{value}</div>'
            f'<div class="d">{sub}</div></div>{sp}{box}</div>')


def nice_range(lo, hi, ticks=4):
    """Batas sumbu dan jumlah tick dengan langkah bulat (1, 2, 2,5, 5 x 10^k), sedekat mungkin dengan data."""
    span = (hi - lo) or abs(hi) or 1
    best = None
    for n in range(max(ticks - 1, 2), ticks + 3):
        mag = 10 ** math.floor(math.log10(span / n))
        for m in (1, 2, 2.5, 5, 10):
            step = m * mag
            a, b = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
            k = round((b - a) / step)
            if max(ticks - 1, 2) <= k <= ticks + 2:
                waste = (b - a) - (hi - lo)
                if best is None or waste < best[0] - 1e-12:
                    best = (waste, a, b, k)
    return (best[1], best[2], best[3]) if best else (lo, hi, ticks)


def line_chart(xlab, series, lo, hi, ticks=4, fmt=str, ref=None, ref_label="", end_label=True, flags=None,
               labels_right=None, label_w=None):
    """xlab: label tiap titik x. series: [(nama, nilai, warna)]. Satu sumbu y untuk semua seri.
    labels_right: daftar teks label per seri, ditulis langsung di ujung kanan garis (pengganti legenda)."""
    n = len(xlab)
    if n < 2:
        return '<div class="note">Belum cukup data.</div>'
    X = lambda i: i * 100 / (n - 1)
    Y = lambda v: 100 - (min(max(v, lo), hi) - lo) / (hi - lo) * 100
    if labels_right and label_w is None:           # ruang kanan secukupnya untuk label terpanjang (rem: ikut ukuran huruf layar)
        chars = max(len(html.unescape(re.sub(r"<[^>]+>", "", t))) + 1 for t in labels_right)
        label_w = f"{1.6 + 0.3 * chars:.1f}rem"
    out = [f'<div class="lc" style="right:{label_w}">' if labels_right else '<div class="lc">']
    for k in range(ticks + 1):
        v = lo + (hi - lo) * k / ticks
        out.append(f'<div class="gl" style="top:{Y(v):.1f}%"></div><div class="yl" style="top:{Y(v):.1f}%">{fmt(v)}</div>')
    for i in sorted({0, (n - 1) // 3, 2 * (n - 1) // 3, n - 1}):
        al = "left:0" if i == 0 else ("right:0" if i == n - 1 else f"left:{X(i):.1f}%;transform:translateX(-50%)")
        out.append(f'<div class="xl" style="{al}">{xlab[i]}</div>')
    if ref is not None:
        out.append(f'<div class="rf" style="top:{Y(ref):.1f}%"></div>'
                   f'<div class="rt" style="top:{Y(ref):.1f}%;left:4px">{ref_label}</div>')
    svg = ['<svg viewBox="0 0 100 100" preserveAspectRatio="none">']
    for _, vals, col in series:
        run = []
        for i, v in enumerate(list(vals) + [float("nan")]):
            if v == v:
                run.append(f"{X(i):.2f},{Y(v):.2f}")
            elif run:
                svg.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" '
                           f'stroke-linecap="round" points="{" ".join(run)}"/>')
                run = []
    out.append("".join(svg) + "</svg>")
    for _, vals, col in series:
        last = max((i for i, v in enumerate(vals) if v == v), default=None)
        if last is None:
            continue
        out.append(f'<div class="ev" style="left:{X(last):.1f}%;top:{Y(vals[last]):.1f}%;background:{col}"></div>')
        if end_label and len(series) == 1 and not labels_right:
            out.append(f'<div class="et" style="left:{X(last):.1f}%;top:{Y(vals[last]):.1f}%">{fmt(vals[last])}</div>')
    if labels_right:                               # label langsung di ujung garis, digeser agar tidak bertumpuk
        pos = []
        for (nm, vals, col), text in zip(series, labels_right):
            last = max((i for i, v in enumerate(vals) if v == v), default=None)
            if last is not None:
                pos.append([Y(vals[last]), text, col, Y(vals[last])])
        pos.sort(key=lambda x: x[0])
        gap = 9.0                                  # jarak minimum antarlabel (% tinggi grafik)
        for i in range(1, len(pos)):
            pos[i][0] = max(pos[i][0], pos[i - 1][0] + gap)
        over = pos[-1][0] - 100 if pos else 0
        if over > 0:
            for x in pos:
                x[0] -= over
            for i in range(len(pos) - 2, -1, -1):
                pos[i][0] = min(pos[i][0], pos[i + 1][0] - gap)
        for y, text, col, y0 in pos:              # titik warna hanya bila label bergeser dari ujung garisnya
            dot = f'<i style="background:{col}"></i>' if abs(y - y0) > 2 else ""
            out.append(f'<div class="dl" style="top:{y:.1f}%">{dot}{text}</div>')
    for i, why in (flags or []):                   # penanda anomali pada seri pertama
        v = series[0][1][i]
        if v == v:
            out.append(f'<div class="fl" style="left:{X(i):.2f}%;top:{Y(v):.2f}%"></div>')
    for i in range(n):
        parts = [f"{nm}: {fmt(v[i])}" for nm, v, _ in series if v[i] == v[i]]
        parts += [why for j, why in (flags or []) if j == i]
        if parts:
            t = html.escape(f"{xlab[i]}  " + " | ".join(parts))
            out.append(f'<i class="hc" style="left:{i * 100 / n:.2f}%;width:{100 / n:.2f}%" title="{t}"></i>')
    out.append("</div>")
    return "".join(out)


def bars_v(values, labels, fmt, ref=None, ref_label="", markers=None, label_idx=None, tips=None, axis_note=None):
    """Kolom satu seri (biru); `markers` = titik seri kedua (oranye) pada skala yang sama; `ref` = garis acuan abu-abu."""
    top = max(list(values) + list(markers or []) + ([ref] if ref else []) + [1e-9]) * 1.2
    label_idx = range(len(values)) if label_idx is None else label_idx
    out = ['<div class="bv"><div class="bp">']
    if ref is not None:
        out.append(f'<div class="rf" style="top:{100 - ref / top * 100:.1f}%"></div>'
                   f'<div class="rt" style="top:{100 - ref / top * 100:.1f}%;left:2px">{ref_label}</div>')
    for i, v in enumerate(values):
        h_ = v / top * 100
        out.append('<div class="bc">'
                   f'<div class="bar" style="height:{h_:.1f}%"></div>'
                   + (f'<div class="bt" style="bottom:calc({max(h_, markers[i] / top * 100 if markers else 0):.1f}% + 6px)">{fmt(v)}</div>'
                      if i in label_idx else "")
                   + (f'<div class="mk" style="bottom:{markers[i] / top * 100:.1f}%"></div>' if markers else "")
                   + f'<div class="hit" title="{html.escape(tips[i] if tips else f"{labels[i]}: {fmt(v)}")}"></div></div>')
    out.append('</div><div class="bx">' + "".join(f"<div>{lab}</div>" for lab in labels) + "</div>")
    if axis_note:
        out.append(f'<div class="ax"><span>{axis_note[0]}</span><span>{axis_note[1]}</span></div>')
    out.append("</div>")
    return "".join(out)


def contrast(a, b):
    """Rasio kontras WCAG antara dua warna RGB."""
    def lum(c):
        c = [x / 255 for x in c]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def heat(rows, cols, cell, first_col=58, corner=""):
    """cell(r, c) -> (nilai 0..1 untuk warna, teks besar, teks kecil, tooltip) atau None."""
    out = [f'<div class="hm" style="grid-template-columns:{first_col}px repeat({len(cols)},1fr);height:100%;'
           f'grid-template-rows:auto repeat({len(rows)},1fr)"><div class="l" style="align-items:flex-end">{corner}</div>']
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
            fg = BG if contrast(rgb, (11, 18, 32)) > contrast(rgb, (255, 255, 255)) else "#fff"
            out.append(f'<div class="c" title="{html.escape(tip)}" style="background:rgb{rgb};color:{fg}">{big}'
                       + (f"<i>{small}</i>" if small else "") + "</div>")
    out.append("</div>")
    return "".join(out)


def hbars(items, fmt, maxv=None):
    """items: (label, nilai, keterangan) atau (label, nilai, keterangan, warna batang)."""
    maxv = maxv or max([i[1] for i in items] + [1e-9])
    out = []
    for it in items:
        a, b, m = it[:3]
        col = it[3] if len(it) > 3 else BLUE
        out.append(f'<div class="hb"><div class="t"><span style="color:{TXT};font-size:.74rem">{html.escape(str(a))}</span>'
                   f'<span>{fmt(b)}{" &middot; " + m if m else ""}</span></div>'
                   f'<div class="k"><div style="width:{max(1.5, b / maxv * 100):.1f}%;background:{col}"></div></div></div>')
    return "".join(out)


def tiles(items):
    """items: (label, nilai, keterangan)."""
    return '<div class="st">' + "".join(
        f'<div><span class="l">{a}</span><span class="v" title="{html.escape(str(b))}">{b}</span><span class="d">{c}</span></div>'
        for a, b, c in items) + "</div>"


# ------------------------------------------------------------------ skor: histogram -> precision/recall, desil, kalibrasi
class Hist:
    """Histogram skor (2000 kelompok) -> precision/recall pada 'tandai X% sesi teratas', desil, dan ambang skor."""
    W = 2000

    def __init__(self, h):
        h = h.sort_values("b", ascending=False)
        self.n, self.buy, self.ps = (h[c].to_numpy(float) for c in ("n", "buy", "ps"))
        self.bins = h["b"].to_numpy(float)
        self.cn, self.cb, self.cp = np.cumsum(self.n), np.cumsum(self.buy), np.cumsum(self.ps)
        self.N, self.B = float(self.cn[-1]), float(self.cb[-1])

    def top(self, share):
        """(sesi, pembeli, jumlah skor) pada `share` teratas menurut skor, interpolasi di dalam kelompok."""
        tgt = min(max(share, 0.0), 1.0) * self.N
        i = min(int(np.searchsorted(self.cn, tgt)), len(self.cn) - 1)
        pn, pb, pp = (a[i - 1] if i > 0 else 0.0 for a in (self.cn, self.cb, self.cp))
        fr = min(max((tgt - pn) / self.n[i], 0), 1) if self.n[i] > 0 else 0.0
        return tgt, pb + fr * self.buy[i], pp + fr * self.ps[i]

    def score_at(self, share):
        i = min(int(np.searchsorted(self.cn, share * self.N)), len(self.cn) - 1)
        return self.bins[i] / self.W

    def ops(self, share):
        n, b, _ = self.top(share)
        prec = b / n if n else 0.0
        rec = b / self.B if self.B else 0.0
        return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    def deciles(self):
        """(tingkat beli nyata, rata-rata skor) per desil, dari skor terendah (1) ke tertinggi (10)."""
        out = []
        for d in range(1, 11):
            n1, b1, p1 = self.top((10 - d) / 10)
            n2, b2, p2 = self.top((11 - d) / 10)
            out.append(((b2 - b1) / (n2 - n1), (p2 - p1) / (n2 - n1)) if n2 > n1 else (0.0, 0.0))
        return out


def f1_of(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def unusual_days(met):
    """Anomali harian: tingkat pembelian 0%, atau volume/tingkat pembelian < 0,3x atau > 2x median.
    Hasil: penanda per grafik {'n': [(posisi, alasan)], 'r': [...]} dan teks ringkas (hari berurutan digabung)."""
    mn, mr = met.n.median(), met.pos_rate.median()
    marks, found = {"n": [], "r": []}, []
    for pos, r in enumerate(met.itertuples()):
        for chart, why in (("n", "volume tinggi" if r.n > 2 * mn else "volume rendah" if r.n < 0.3 * mn else None),
                           ("r", "tingkat pembelian 0%" if r.pos_rate == 0 else "tingkat pembelian tinggi" if r.pos_rate > 2 * mr
                            else "tingkat pembelian rendah" if r.pos_rate < 0.3 * mr else None)):
            if why:
                marks[chart].append((pos, f"Anomali: {why}"))
                found.append((pd.Timestamp(r.day), why))
    groups = []
    for d, why in sorted(found, key=lambda x: (x[1], x[0])):
        if groups and groups[-1][2] == why and (d - groups[-1][1]).days == 1:
            groups[-1][1] = d
        else:
            groups.append([d, d, why])
    groups.sort(key=lambda g: g[0])
    text = [f"{tgl(a)} ({why})" if a == b else
            f"{a.day}{'' if a.month == b.month else ' ' + BULAN[a.month - 1]}-{tgl(b)} ({why})" for a, b, why in groups]
    return marks, text


# ------------------------------------------------------------------ tampilan
def view_pipeline(s, n_active, base, t5, t20, met, age):
    now, ev = time.time(), float(nz(s.events_total))
    hist_ev = [x for x in st.session_state.get("ev_hist", []) if now - x[0] <= 60 and x[1] <= ev] + [(now, ev)]
    st.session_state["ev_hist"] = hist_ev
    span = now - hist_ev[0][0]
    rate = (ev - hist_ev[0][1]) / span if span >= 5 else None
    running = age <= LIVE_SEC
    period = f"{tgl(met.day.min())} - {tgl(met.day.max())}" if len(met) else "-"

    a = st.columns([1, 1.35])
    a[0].markdown(card(
        "Status Pipeline", "Alur data: Kafka atau HDFS, Spark Structured Streaming, PostgreSQL",
        tiles([
            ("Status", "Berjalan" if running else "Berhenti", f"diperbarui {lalu(age)}"),
            ("Kecepatan", "-" if rate is None else f"{idn(rate if running else 0)} event/detik", "rata-rata 1 menit terakhir"),
            ("Event diproses", idn(nz(s.events_total)), f"offset Kafka terakhir {idn(s.last_offset) if pd.notna(s.last_offset) else '-'}"),
            ("Data hingga", waktu(s.last_event_time) if pd.notna(s.last_event_time) else "-", "waktu event terakhir (UTC)"),
            ("Micro-batch selesai", idn(nz(s.batches)), "query sesi"),
            ("Sesi dipelajari model", idn(nz(s.sessions_learned)), "pembelajaran online, satu kali per sesi selesai"),
            ("Sesi aktif", idn(n_active), "sudah diprediksi, hasil belum diketahui"),
            ("Periode data", period, "menurut waktu prediksi (UTC)"),
        ]),
        tip="Status Berjalan apabila PostgreSQL menerima hasil micro-batch dalam 90 detik terakhir. Offset Kafka dapat lebih besar "
            "dari jumlah event apabila sebagian event di awal topic dilewati saat pemrosesan ulang."), unsafe_allow_html=True)

    rows = q("SELECT session_id, n_cart, prob, t_pred FROM active_predictions ORDER BY t_pred DESC, session_id LIMIT 7")

    def lab(p):
        if p >= t5:
            return "Tinggi", "p-hi"
        if p >= t20:
            return "Sedang", "p-md"
        return "Rendah", "p-lo"

    if len(rows):
        body = "".join(
            f'<tr><td class="mono">{html.escape(str(r.session_id)[:13])}</td><td>{tgl(utc(r.t_pred))} '
            f'{utc(r.t_pred):%H:%M}</td><td>{int(r.n_cart)}</td><td>{pc(r.prob, 0)}</td>'
            f'<td><span class="pill {lab(r.prob)[1]}">{lab(r.prob)[0]}</span></td></tr>' for r in rows.itertuples())
        tb = ('<table class="tb"><tr><th>Sesi</th><th>Diprediksi (UTC)</th><th>Masuk keranjang</th><th>Probabilitas</th><th>Kategori skor</th></tr>'
              + body + "</table>")
    else:
        tb = '<div class="note">Tidak ada sesi aktif.</div>'
    a[1].markdown(card("Sesi Aktif Terbaru", "Diprediksi pada event ke-3; hasil diketahui setelah 30 menit tanpa aktivitas", tb,
                       tip=f"Probabilitas pembelian dihitung saat sesi mencapai event ke-3, sebelum hasilnya diketahui. Kategori skor: "
                           f"Tinggi = 5% skor teratas (minimal {pc(t5, 0)}), Sedang = 5-20% teratas (minimal {pc(t20, 0)}), Rendah = lainnya."),
                  unsafe_allow_html=True)

    if len(met) > 1:
        days = [tgl(d) for d in met.day]
        n_vals, r_vals = met.n.astype(float).tolist(), met.pos_rate.astype(float).tolist()
        marks, odd = unusual_days(met)
        body = (f'<div class="sm"><span class="t">Jumlah sesi dinilai per hari</span>'
                + line_chart(days, [("Sesi", n_vals, BLUE)], *nice_range(0, max(n_vals), 2), fmt=short, flags=marks["n"]) + "</div>"
                + f'<div class="sm"><span class="t">Tingkat pembelian per hari</span>'
                + line_chart(days, [("Tingkat pembelian", r_vals, BLUE)], *nice_range(0, max(r_vals), 2), fmt=lambda v: pc(v, 0),
                             ref=base, ref_label=f"rata-rata {pc(base)}", flags=marks["r"]) + "</div>")
    else:
        body, odd = '<div class="note">Data harian belum mencukupi.</div>', []
    st.markdown(card("Volume dan Tingkat Pembelian Harian", "Pemantauan data masuk dan deteksi anomali harian",
                     body, h=330,
                     legend=legend([(f"Anomali ({len(odd)})", WARN, "o")]) if odd else "",
                     tip=["Anomali: tingkat pembelian 0%, atau volume maupun tingkat pembelian di luar 0,3 sampai 2 kali median "
                          "(hari menurut waktu prediksi, UTC)."]
                         + (["Terdeteksi: " + "; ".join(odd) + "."] if odd else [])), unsafe_allow_html=True)


def view_model(Hs, base, met, ok, w_auc):
    a = st.columns([1, 1, 1.1])
    if len(ok) > 1:
        days = [tgl(d) for d in met.day]                 # semua hari: hari tanpa AUC tampil sebagai celah
        auc = met.auc.astype(float).tolist()
        gaps = [tgl(d) for d in met.day[met.auc.isna()]]
        gap_note = ("Hari tanpa nilai AUC: " + ", ".join(gaps) + ".") if gaps else ""
        a[0].markdown(card("AUC Harian", "ROC-AUC per hari; sumbu dimulai dari 0,5 (setara pemilihan acak)",
                           line_chart(days, [("AUC", auc, BLUE)], *nice_range(0.5, min(1.0, ok.auc.max() + 0.01)),
                                      fmt=lambda v: dec(v, 2), ref=w_auc, ref_label=f"rata-rata {dec(w_auc, 2)}"),
                           tip=["Peluang model memberi skor lebih tinggi kepada sesi yang berakhir dengan pembelian dibanding sesi lainnya, "
                                "dihitung per hari. Skor dibuat sebelum hasil sesi diketahui (prequential).",
                                "Hari tanpa pembelian atau dengan kurang dari 50 sesi tidak memiliki nilai AUC dan tampil sebagai celah, "
                                "termasuk pada grafik Precision@5%." + (" " + gap_note if gap_note else "")]),
                      unsafe_allow_html=True)
        p5 = met.p_at_5.astype(float).tolist()
        a[1].markdown(card("Precision@5% Harian", "Persentase pembeli di antara 5% sesi berskor tertinggi tiap hari",
                           line_chart(days, [("Precision@5%", p5, BLUE)], *nice_range(0, min(1.0, max(ok.p_at_5.max(), base) * 1.05)),
                                      fmt=lambda v: pc(v, 0), ref=base, ref_label=f"pemilihan acak ({pc(base)})"),
                           tip="Garis acuan adalah tingkat pembelian rata-rata, yaitu hasil yang diperoleh apabila sesi dipilih secara acak."),
                      unsafe_allow_html=True)
    else:
        for i, t in enumerate(["AUC Harian", "Precision@5% Harian"]):
            a[i].markdown(card(t, "", '<div class="note">Data harian belum mencukupi.</div>'), unsafe_allow_html=True)

    w = q_slow("SELECT batch, feature, weight FROM model_weights ORDER BY batch")
    if w.batch.nunique() > 1:
        pv = w.pivot(index="batch", columns="feature", values="weight").sort_index()
        feats = [f for f in FEAT_NAMES if f in pv.columns]
        series = [(FEAT_NAMES[f], pv[f].astype(float).tolist(), SERIES[i]) for i, f in enumerate(feats)]
        short_nm = {"c": "Keranjang", "avg_price": "Harga rata-rata", "max_price": "Harga tertinggi", "dur": "Durasi awal"}
        allv = [v for _, vals, _ in series for v in vals] + [0.0]
        lo_w, hi_w, nt = nice_range(min(allv), max(allv))
        last = {FEAT_NAMES[f]: float(pv[f].iloc[-1]) for f in feats}
        a[2].markdown(card("Bobot Model dari Waktu ke Waktu", "Bobot regresi logistik online pada skala terstandardisasi, dicatat setiap 10 batch",
                           line_chart([f"Batch {b}" for b in pv.index], series, lo_w, hi_w, nt, fmt=lambda v: dec(v, 1),
                                      labels_right=[f"{short_nm[f]}<b>{dec(last[FEAT_NAMES[f]], 2)}</b>" for f in feats]),
                           tip=["Bobot positif menaikkan peluang pembelian. Angka di ujung garis adalah bobot pada batch terakhir. "
                                "Keranjang = jumlah masuk keranjang pada 3 event pertama.",
                                "Besar bobot tidak mencerminkan tingkat kepentingan fitur apabila fitur saling berkorelasi. Harga rata-rata "
                                "dan harga tertinggi berkorelasi kuat, sehingga bobot keduanya perlu dibaca bersama."]),
                      unsafe_allow_html=True)
    else:
        a[2].markdown(card("Bobot Model dari Waktu ke Waktu", "", '<div class="note">Bobot dicatat setiap 10 batch; jumlah batch belum mencukupi.</div>'),
                      unsafe_allow_html=True)

    b = st.columns([1.25, 1])
    dd = Hs.deciles()
    rates, scores = [x[0] for x in dd], [x[1] for x in dd]
    b[0].markdown(card("Tingkat Pembelian menurut Desil Skor", "Sesi dibagi menjadi 10 kelompok sama besar menurut skor model",
                       bars_v(rates, [str(i) for i in range(1, 11)], lambda v: pc(v, 0), ref=base,
                              ref_label=f"rata-rata {pc(base)}", markers=scores, label_idx={0, 8, 9},
                              tips=[f"Desil {i + 1}: tingkat pembelian aktual {pc(r)}, rata-rata skor {pc(p)}" for i, (r, p) in enumerate(dd)],
                              axis_note=("skor terendah", "skor tertinggi")),
                       legend=legend([("Tingkat pembelian aktual", BLUE, "b"), ("Rata-rata skor model", ORANGE, "c"),
                                      ("Rata-rata semua sesi", REF, "r")]),
                       tip=["Desil 1 berisi 10% sesi dengan skor terendah, desil 10 berisi 10% sesi dengan skor tertinggi.",
                            "Titik menunjukkan rata-rata skor model pada desil tersebut. Kalibrasi tepat apabila titik berada di ujung batang."]),
                  unsafe_allow_html=True)

    best = max(range(1, 51), key=lambda p: Hs.ops(p / 100)[2])
    trs = ""
    for p in sorted({1, 2, 5, 10, 20, best}):
        pr, rc, f1 = Hs.ops(p / 100)
        trs += (f'<tr class="{"best" if p == best else ""}"><td>{p}% skor teratas'
                f'{"<span class=tag>F1 tertinggi model</span>" if p == best else ""}</td><td>{pc(pr, 0)}</td>'
                f'<td>{pc(rc, 0)}</td><td>{dec(f1, 2)}</td><td>{dec(pr / base, 1)}x</td></tr>')
    rule = q_slow("SELECT count(*) FILTER (WHERE f_c > 0) AS n, coalesce(sum(y) FILTER (WHERE f_c > 0), 0) AS b FROM eval_log").iloc[0]
    if rule.n:
        rp, rr = rule.b / rule.n, rule.b / Hs.B
        trs += (f'<tr class="rule"><td>Aturan keranjang ({pc(rule.n / Hs.N, 0)} sesi)</td><td>{pc(rp, 0)}</td><td>{pc(rr, 0)}</td>'
                f'<td>{dec(f1_of(rp, rr), 2)}</td><td>{dec(rp / base, 1)}x</td></tr>')
    b[1].markdown(card("Simulasi Penandaan Calon Pembeli", "Hasil apabila sebagian sesi dengan skor tertinggi ditandai sebagai calon pembeli",
                       '<table class="tb"><tr><th>Sesi ditandai</th><th>Precision (ketepatan)</th><th>Recall (cakupan)</th>'
                       '<th>F1</th><th>Lift</th></tr>'
                       + trs + "</table>",
                       tip=["Precision: dari sesi yang ditandai, berapa persen yang benar-benar membeli. Recall: dari seluruh pembeli, "
                            "berapa persen yang berhasil ditandai. F1: nilai tengah keduanya. Lift: precision dibagi tingkat pembelian rata-rata.",
                            "Semakin sedikit sesi yang ditandai, precision naik tetapi recall turun. Panel ini membantu memilih jumlah sesi "
                            "yang ditandai sesuai kapasitas tindakan, misalnya jumlah voucher atau agen layanan.",
                            "Aturan keranjang adalah pembanding tanpa model: seluruh sesi dengan aktivitas keranjang pada 3 event pertama ditandai."]),
                  unsafe_allow_html=True)


def view_customers(Hs, base):
    E0, E1 = EDGES
    a = st.columns([1.2, 1, 1])

    # 1. keranjang x harga
    heat_df = q_slow(f"""SELECT least(f_c, 2)::int AS c,
        CASE WHEN f_avg_price <= 0 THEN 3 WHEN f_avg_price < {E0} THEN 0 WHEN f_avg_price < {E1} THEN 1 ELSE 2 END AS pb,
        count(*) AS n, avg(y) AS rate FROM eval_log GROUP BY 1, 2""")
    pr_names = [f"Harga rendah<br>&lt; {idn(math.exp(E0) - 1)}", f"Harga sedang<br>{idn(math.exp(E0) - 1)}-{idn(math.exp(E1) - 1)}",
                f"Harga tinggi<br>&ge; {idn(math.exp(E1) - 1)}"]
    grid = {(int(r.c), int(r.pb)): (float(r.rate), int(r.n)) for r in heat_df.itertuples()}
    vmax = max([v[0] for k, v in grid.items() if k[1] < 3] + [1e-9])
    rows_lab = ["0 kali", "1 kali", "&ge; 2 kali"]

    def cell(r, c):
        key = (rows_lab.index(r), pr_names.index(c))
        if key not in grid:
            return None
        rate, n = grid[key]
        return (rate / vmax, pc(rate, 0 if rate >= 0.1 else 1), f"{short(n)} sesi",
                f"Masuk keranjang {html.unescape(r)}, {html.unescape(c.replace('<br>', ' '))}: tingkat pembelian {pc(rate)} dari {idn(n)} sesi")

    no_price = sum(v[1] for k, v in grid.items() if k[1] == 3)
    a[0].markdown(card("Tingkat Pembelian menurut Keranjang dan Harga",
                       "Persentase sesi yang membeli menurut jumlah masuk keranjang dan harga produk pada 3 aktivitas pertama",
                       heat(rows_lab, pr_names, cell, first_col=64, corner="Masuk<br>keranjang"), h=430,
                       tip=["Warna lebih terang berarti tingkat pembelian lebih tinggi. Angka kecil menunjukkan jumlah sesi pada kotak tersebut.",
                            f"Harga adalah rata-rata harga produk pada 3 aktivitas pertama, dibagi tiga kelompok menurut sebaran 70% awal data. "
                            f"Sesi tanpa harga yang valid ({short(no_price)} sesi) tidak ditampilkan."]),
                  unsafe_allow_html=True)

    # 2. tahapan belanja
    cat = q_slow("SELECT category, views, carts, purchases FROM category_insights")
    v, ca, pu = (int(cat[c].sum()) for c in ("views", "carts", "purchases"))
    sess = q_slow("SELECT count(*) AS n, count(*) FILTER (WHERE f_c > 0) AS nc, coalesce(sum(y), 0) AS nb FROM eval_log").iloc[0]
    fun = ('<div class="grp">Aktivitas</div>'
           + hbars([("Melihat produk", v, ""),
                    ("Masuk keranjang", ca, f"{pc(ca / v) if v else '-'} dari melihat produk"),
                    ("Membeli", pu, f"{pc(pu / ca) if ca else '-'} dari masuk keranjang")], short)
           + '<div class="grp" style="margin-top:10px">Sesi</div>'
           + hbars([("Sesi yang dinilai", int(sess.n), ""),
                    ("Masuk keranjang pada 3 aktivitas pertama", int(sess.nc), f"{pc(sess.nc / sess.n) if sess.n else '-'} dari sesi"),
                    ("Membeli", int(sess.nb), f"{pc(sess.nb / sess.n) if sess.n else '-'} dari sesi")], short, maxv=int(sess.n)))
    a[1].markdown(card("Tahapan Belanja", "Jumlah aktivitas dan sesi dari melihat produk hingga membeli", fun, h=430,
                       tip=["Aktivitas: seluruh aktivitas yang diproses pipeline. Sesi: sesi yang dinilai model, yaitu sesi dengan minimal "
                            "3 aktivitas tanpa pembelian pada 3 aktivitas pertama.",
                            "Persentase pada bagian aktivitas adalah perbandingan jumlah aktivitas antartahap, bukan persentase produk "
                            "di keranjang yang akhirnya dibeli."]),
                  unsafe_allow_html=True)

    # 3. pembelian per kategori
    nm = lambda x: CATEGORY_NAMES.get(x, str(x).replace("_", " ").capitalize())
    top = cat[cat.category != "unknown"].sort_values("purchases", ascending=False).head(8)
    unk = int(cat.loc[cat.category == "unknown", "purchases"].sum())
    items = [(nm(r.category), int(r.purchases), "") for r in top.itertuples()]
    if unk:
        items.append(("Tanpa kategori", unk, "", REF))
    a[2].markdown(card("Pembelian per Kategori", "Jumlah pembelian menurut kategori produk",
                       hbars(items, short) if items else '<div class="note">Belum ada data.</div>', h=430,
                       tip=["Delapan kategori dengan pembelian terbanyak, ditambah pembelian tanpa kategori pada baris terakhir (abu-abu).",
                            f"Tanpa kategori: produk yang kategorinya tidak dapat ditentukan ({pc(unk / pu) if pu else '-'} dari seluruh "
                            f"pembelian). Kode kategori pada data mulai Desember 2019 tidak konsisten, sehingga kategori diambil dari "
                            f"pemetaan produk Oktober-November 2019 dan produk di luar pemetaan tidak memiliki kategori."]),
                  unsafe_allow_html=True)

    b = st.columns([1, 1])

    # 4. keranjang tanpa pembelian
    ab = q_slow("""SELECT to_char(t_pred AT TIME ZONE 'UTC', 'YYYY-MM') AS m, count(*) FILTER (WHERE f_c > 0) AS n,
        count(*) FILTER (WHERE f_c > 0 AND y = 0) AS a FROM eval_log GROUP BY 1 ORDER BY 1""")
    n_c, n_a = int(ab.n.sum()), int(ab.a.sum())
    ab = ab[ab.n > 0]
    mlab = [f"{BULAN[int(m[5:7]) - 1]} {m[2:4]}" for m in ab.m]
    rates_ab = (ab.a / ab.n).astype(float).tolist()
    body = (f'<div style="height:62px">' + tiles([("Sesi dengan keranjang", short(n_c), "masuk keranjang pada 3 aktivitas pertama"),
                                                ("Tanpa pembelian", pc(n_a / n_c if n_c else 0), f"{short(n_a)} sesi")])
            + '</div><div style="position:relative;height:calc(100% - 72px);margin-top:10px">'
            + (bars_v(rates_ab, mlab, lambda v: pc(v, 0),
                      tips=[f"{lab}: {pc(r)} dari {idn(n)} sesi dengan keranjang tidak berlanjut ke pembelian"
                            for lab, r, n in zip(mlab, rates_ab, ab.n)])
               if len(ab) else '<div class="note">Belum ada data.</div>') + "</div>")
    b[0].markdown(card("Keranjang Tanpa Pembelian", "Sesi yang memasukkan produk ke keranjang tetapi tidak membeli, per bulan", body,
                       tip=["Dihitung dari sesi yang dinilai model: produk masuk keranjang pada 3 aktivitas pertama dan tidak ada pembelian "
                            "hingga sesi berakhir.",
                            "Pencatatan aktivitas keranjang pada data sumber belum lengkap sebelum 8 November 2019, sehingga nilai Oktober "
                            "dan awal November tidak sebanding dengan bulan lainnya."]),
                  unsafe_allow_html=True)

    # 5. tingkat pembelian per jam
    hr = q_slow("""SELECT extract(hour FROM t_pred AT TIME ZONE 'UTC')::int AS h, count(*) AS n, avg(y) AS r
        FROM eval_log GROUP BY 1""")
    hm = {int(r.h): (float(r.r), int(r.n)) for r in hr.itertuples()}
    rates_h = [hm.get(h, (0.0, 0))[0] for h in range(24)]
    if hm:
        hi_h = max(range(24), key=lambda h: rates_h[h])
        lo_h = min((h for h in range(24) if h in hm), key=lambda h: rates_h[h])
        body = bars_v(rates_h, [f"{h:02d}" if h % 3 == 0 else "" for h in range(24)], lambda v: pc(v, 0), ref=base,
                      ref_label=f"rata-rata {pc(base)}", label_idx={hi_h, lo_h},
                      tips=[f"Pukul {h:02d}.00-{h:02d}.59 UTC: tingkat pembelian {pc(rates_h[h])} dari {idn(hm.get(h, (0, 0))[1])} sesi"
                            for h in range(24)], axis_note=("pukul 00.00", "pukul 23.00"))
    else:
        body = '<div class="note">Belum ada data.</div>'
    b[1].markdown(card("Tingkat Pembelian per Jam", "Persentase sesi yang membeli menurut jam (waktu UTC)", body,
                       tip=["Jam dihitung saat sesi mencapai aktivitas ke-3, menurut waktu UTC pada data. Zona waktu toko tidak "
                            "disebutkan dalam dataset.",
                            "Garis abu-abu adalah rata-rata seluruh jam. Label angka ditampilkan pada jam dengan tingkat pembelian tertinggi "
                            "dan terendah."]),
                  unsafe_allow_html=True)


# ------------------------------------------------------------------ halaman
@st.fragment(run_every=3)
def live():
    try:
        render()
    except psycopg2.OperationalError:
        st.error("Tidak dapat terhubung ke PostgreSQL. Pastikan layanan PostgreSQL berjalan dan pengaturan di db_config.py "
                 "sudah benar. Halaman akan mencoba kembali secara otomatis.")


def render():
    s = q("SELECT * FROM pipeline_stats WHERE id = 1")
    head = ('<div class="hd"><div><b>Pemantauan Prediksi Pembelian Sesi</b><span class="s">Kafka &middot; HDFS &middot; '
            "Spark Structured Streaming &middot; River &middot; PostgreSQL &middot; sampel 10% sesi dataset REES46</span></div>{}</div>")
    if s.empty:
        st.markdown(head.format(""), unsafe_allow_html=True)
        st.info("Belum terdapat data di PostgreSQL. Jalankan python -m pipeline.stream_processor; "
                "untuk pemrosesan langsung, jalankan juga python -m pipeline.producer.")
        return
    s = s.iloc[0]
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(s.updated_at).tz_convert("UTC")).total_seconds()
    until = waktu(s.last_event_time) if pd.notna(s.last_event_time) else "-"
    st.markdown(head.format(
        f'<div class="fresh"><span class="dot {"on" if age <= LIVE_SEC else "off"}"></span>'
        f'{"Berjalan" if age <= LIVE_SEC else "Berhenti"} &middot; data hingga {until} UTC &middot; diperbarui {lalu(age)}</div>'),
        unsafe_allow_html=True)

    hist = q_slow("SELECT floor(prob * 2000)::int AS b, count(*) AS n, sum(y)::float AS buy, sum(prob) AS ps "
                  "FROM eval_log GROUP BY 1")
    if hist.empty:
        st.info("Menunggu sesi pertama selesai, yaitu 30 menit tanpa aktivitas menurut waktu data.")
        return
    Hs = Hist(hist)
    base = Hs.B / Hs.N
    met = q("SELECT day, n, pos_rate, auc, pr_auc, p_at_5 FROM metrics_daily ORDER BY day")
    met["day"] = pd.to_datetime(met["day"])
    if len(met) > 3 and met.n.iloc[-1] < 0.5 * met.n.median():
        met = met.iloc[:-1]                      # hari terakhir biasanya belum penuh
    ok = met.dropna(subset=["auc"])
    wavg = lambda col: float((ok[col] * ok.n).sum() / ok.n.sum()) if len(ok) else float("nan")
    w_auc, w_p5 = wavg("auc"), wavg("p_at_5")
    n_active = int(q("SELECT count(*) AS n FROM active_predictions").iloc[0].n)

    k = st.columns(5)
    k[0].markdown(kpi("Event Diproses", short(nz(s.events_total)), "lihat produk, keranjang, pembelian",
                      tip="Seluruh event yang telah dibaca dan diproses pipeline."), unsafe_allow_html=True)
    k[1].markdown(kpi("Sesi Dinilai", short(Hs.N), "diprediksi dan hasilnya telah diketahui", met.n.tolist(),
                      tip="Sesi dengan minimal 3 event tanpa pembelian pada 3 event pertama, yang telah selesai sehingga hasilnya diketahui. Grafik kecil menunjukkan jumlah per hari."),
                  unsafe_allow_html=True)
    k[2].markdown(kpi("Tingkat Pembelian", pc(base), f"{short(Hs.B)} sesi berakhir dengan pembelian", (met.pos_rate * 100).tolist(),
                      tip="Proporsi sesi dinilai yang melakukan pembelian setelah event ke-3. Grafik kecil menunjukkan nilai per hari."), unsafe_allow_html=True)
    k[3].markdown(kpi("AUC", dec(w_auc, 2) if w_auc == w_auc else "-", "rata-rata harian (acak = 0,5)", ok.auc.tolist(),
                      tip="Rata-rata ROC-AUC harian, berbobot jumlah sesi."), unsafe_allow_html=True)
    k[4].markdown(kpi("Precision@5%", pc(w_p5, 0) if w_p5 == w_p5 else "-",
                      f"lift {dec(w_p5 / base, 1)}x terhadap pemilihan acak" if w_p5 == w_p5 else "-", ok.p_at_5.tolist(),
                      tip="Rata-rata harian proporsi pembeli di antara 5% sesi dengan skor tertinggi."), unsafe_allow_html=True)

    view = st.segmented_control("Tampilan", ["Pipeline", "Model", "Pelanggan"], default="Pipeline", key="view",
                                label_visibility="collapsed") or "Pipeline"
    t5, t20 = Hs.score_at(0.05), Hs.score_at(0.20)
    if view == "Pipeline":
        view_pipeline(s, n_active, base, t5, t20, met, age)
    elif view == "Model":
        view_model(Hs, base, met, ok, w_auc)
    else:
        view_customers(Hs, base)


live()