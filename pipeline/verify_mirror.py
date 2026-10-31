"""
Check the mirrored data in HDFS: row count, Kafka offset continuity and duplicates, files per month.

    python -m pipeline.verify_mirror
    python -m pipeline.verify_mirror --path /datalake/test/clickstream
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--hdfs", default="hdfs://localhost:8020")
ap.add_argument("--path", default="/datalake/raw/clickstream")
args = ap.parse_args()

os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ["PATH"] += r";C:\hadoop\bin"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.appName("VerifyMirror")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", args.hdfs)
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(args.hdfs + args.path)
r = df.agg(
    F.count("*").alias("rows"),
    F.countDistinct("kafka_partition", "kafka_offset").alias("distinct_offsets"),
    F.min("kafka_offset").alias("min_offset"),
    F.max("kafka_offset").alias("max_offset"),
    F.count(F.when(F.col("user_session").isNull(), 1)).alias("null_session"),
).first()
expected = r.max_offset - r.min_offset + 1
print(f"Rows: {r.rows:,} | distinct offsets: {r.distinct_offsets:,} | offsets {r.min_offset:,}..{r.max_offset:,}")
print(f"Duplicates: {r.rows - r.distinct_offsets:,} | gaps in offset range: {expected - r.distinct_offsets:,} "
      f"| rows without user_session: {r.null_session:,}")
print("\nRows and files per event_month:")
files = {}
for f in df.inputFiles():
    m = f.split("event_month=")[1].split("/")[0] if "event_month=" in f else "?"
    files[m] = files.get(m, 0) + 1
for row in df.groupBy("event_month").count().orderBy("event_month").collect():
    print(f"  {row.event_month}: {row['count']:,} rows in {files.get(row.event_month, 0)} file(s)")
spark.stop()