"""
Menggabungkan beberapa file sampel yang sudah urut waktu menjadi satu file.
Header hanya diambil dari file pertama; header file berikutnya harus sama persis.

    python prep\\gabung_sampel.py --inputs sample/sample_10pct.csv sample/sample_10pct_marapr.csv --out sample/sample_10pct_7bulan.csv
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True, help="file sampel, urut dari periode paling awal")
ap.add_argument("--out", required=True, help="file hasil gabungan")
args = ap.parse_args()

if os.path.abspath(args.out) in [os.path.abspath(p) for p in args.inputs]:
    raise SystemExit("File keluaran tidak boleh sama dengan salah satu file masukan.")

CHUNK = 16 * 1024 * 1024
t0 = time.time()
header, total = None, 0
with open(args.out, "wb") as fo:
    for path in args.inputs:
        rows, last = 0, b"\n"
        with open(path, "rb") as fi:
            h = fi.readline()
            if header is None:
                header = h
                fo.write(h)
            elif h != header:
                raise SystemExit(f"Header {path} berbeda dengan file pertama, penggabungan dibatalkan.")
            while True:
                data = fi.read(CHUNK)
                if not data:
                    break
                rows += data.count(b"\n")
                last = data[-1:]
                fo.write(data)
        if last != b"\n":          # baris terakhir tanpa newline
            fo.write(b"\n")
            rows += 1
        total += rows
        print(f"{path}: {rows:,} baris data")

print(f"Total {args.out}: {total:,} baris data | {os.path.getsize(args.out) / 1e9:.2f} GB | {time.time() - t0:.0f} detik")