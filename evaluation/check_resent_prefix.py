"""
Check whether the start of the data lake holds an accidental re-send: the first N events
(offsets 0..N-1) appearing again at offsets N..2N-1, as when a producer is stopped and
started again from the top of the file.

    python -m evaluation.check_resent_prefix --n 120714
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--lake", default="hdfs://localhost:8020/datalake/raw/clickstream")
ap.add_argument("--n", type=int, required=True, help="suspected length of the re-sent prefix")
args = ap.parse_args()

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
if os.name == "nt":
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    os.environ["PATH"] += r";C:\hadoop\bin"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

COLS = ["event_time", "event_type", "product_id", "category_id", "category_code", "brand", "price",
        "user_id", "user_session"]
N = args.n

spark = (SparkSession.builder.appName("CheckResentPrefix").master("local[4]")
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(args.lake).where(F.col("kafka_offset") < 2 * N + 1).select("kafka_offset", *COLS)
first = df.where(F.col("kafka_offset") < N)
again = df.where((F.col("kafka_offset") >= N) & (F.col("kafka_offset") < 2 * N)) \
          .withColumn("kafka_offset", F.col("kafka_offset") - N)
pairs = first.alias("a").join(again.alias("b"), "kafka_offset")
same = F.lit(True)
for c in COLS:
    same = same & F.col(f"a.{c}").eqNullSafe(F.col(f"b.{c}"))
r = pairs.agg(F.count(F.lit(1)).alias("pairs"), F.sum(same.cast("int")).alias("identical")).first()

edge = {row["kafka_offset"]: row["event_time"]
        for row in df.where(F.col("kafka_offset").isin(0, N - 1, N, 2 * N)).collect()}
spark.stop()

print(f"Offsets 0..{N - 1} vs {N}..{2 * N - 1}: {r['pairs']:,} pairs, {r['identical'] or 0:,} identical")
print(f"event_time at offset 0: {edge.get(0)} | {N - 1}: {edge.get(N - 1)} | {N}: {edge.get(N)} | {2 * N}: {edge.get(2 * N)}")
ok = r["pairs"] == N and r["identical"] == N
print("\nRESULT:", f"offsets 0..{N - 1} were re-sent; use --start-offset {N}" if ok else "not a clean re-send")
