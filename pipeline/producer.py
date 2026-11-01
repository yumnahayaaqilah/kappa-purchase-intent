"""
Mengirim file sampel ke Kafka secepat mungkin (atau dibatasi --rate event/detik).

    python -m pipeline.producer --limit 2000000      # uji kecepatan 2 juta event
    python -m pipeline.producer                      # kirim semuanya
"""
import argparse
import time

import pandas as pd
from kafka import KafkaProducer

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="sample/sample_10pct_7bulan.csv")
ap.add_argument("--topic", default="clickstream")
ap.add_argument("--bootstrap", default="localhost:9092")
ap.add_argument("--limit", type=int, default=0, help="0 = kirim semua")
ap.add_argument("--rate", type=int, default=0, help="batas event per detik, 0 = tanpa batas")
args = ap.parse_args()

dtypes = {
    "event_time": str, "event_type": str, "product_id": "int64", "category_id": "int64",
    "category_code": str, "brand": str, "price": "float64", "user_id": "int64", "user_session": str,
}
producer = KafkaProducer(
    bootstrap_servers=args.bootstrap,
    linger_ms=20,
    batch_size=262144,
    acks=1,
)

sent, t0, last_print = 0, time.time(), 0
for chunk in pd.read_csv(args.file, chunksize=200_000, dtype=dtypes):
    if args.limit:
        chunk = chunk.iloc[: args.limit - sent]
    lines = chunk.to_json(orient="records", lines=True).splitlines()
    for ln in lines:
        producer.send(args.topic, ln.encode("utf-8"))
    sent += len(lines)

    if args.rate:
        ahead = sent / args.rate - (time.time() - t0)
        if ahead > 0:
            time.sleep(ahead)
    if sent - last_print >= 1_000_000:
        last_print = sent
        print(f"{sent:,} event terkirim | {sent / (time.time() - t0):,.0f} event/detik", flush=True)
    if args.limit and sent >= args.limit:
        break

producer.flush()
el = time.time() - t0
print(f"Selesai: {sent:,} event dalam {el:.0f} detik ({sent / max(el, 1e-9):,.0f} event/detik)")
