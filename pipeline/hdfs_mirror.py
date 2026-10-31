"""
Mirror raw events from the Kafka unified log to the HDFS data lake as Parquet.

Runs as its own Spark Structured Streaming query, independent of the processing job,
so the data lake keeps every event whatever happens downstream. Kafka partition and
offset are stored with each event so a replay from HDFS can restore the original order.

    python -m pipeline.hdfs_mirror                 # continuous
    python -m pipeline.hdfs_mirror --once          # mirror what Kafka holds now, then stop
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--bootstrap", default="localhost:9092")
ap.add_argument("--topic", default="ecommerce-v2")
ap.add_argument("--hdfs", default="hdfs://localhost:8020")
ap.add_argument("--path", default="/datalake/raw/clickstream")
ap.add_argument("--checkpoint", default="checkpoint/hdfs_mirror")
ap.add_argument("--min-offsets", type=int, default=2_000_000, help="events to collect before writing")
ap.add_argument("--max-offsets", type=int, default=3_000_000, help="upper bound of events per write")
ap.add_argument("--max-delay", default="2m", help="write anyway after waiting this long")
ap.add_argument("--once", action="store_true", help="process what is available, then stop")
args = ap.parse_args()

os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ["PATH"] += r";C:\hadoop\bin"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

SCHEMA = StructType([
    StructField("event_time", StringType()),
    StructField("event_type", StringType()),
    StructField("product_id", LongType()),
    StructField("category_id", LongType()),
    StructField("category_code", StringType()),
    StructField("brand", StringType()),
    StructField("price", DoubleType()),
    StructField("user_id", LongType()),
    StructField("user_session", StringType()),
])

spark = (
    SparkSession.builder.appName("HdfsMirror")
    .master("local[2]")
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0")
    .config("spark.driver.memory", "2g")
    .config("spark.hadoop.fs.defaultFS", args.hdfs)
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
    .config("spark.hadoop.dfs.replication", "1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

reader = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", args.bootstrap)
    .option("subscribe", args.topic)
    .option("startingOffsets", "earliest")
    .option("maxOffsetsPerTrigger", args.max_offsets)
    .option("failOnDataLoss", "true")      # the data lake must notice missing data
)
if not args.once:
    reader = reader.option("minOffsetsPerTrigger", args.min_offsets).option("maxTriggerDelay", args.max_delay)

events = (
    reader.load()
    .select(
        F.from_json(F.col("value").cast("string"), SCHEMA).alias("e"),
        F.col("partition").alias("kafka_partition"),
        F.col("offset").alias("kafka_offset"),
        F.col("timestamp").alias("kafka_timestamp"),
    )
    .select("e.*", "kafka_partition", "kafka_offset", "kafka_timestamp")
    .withColumn("event_month", F.substring("event_time", 1, 7))
)

writer = (
    events.writeStream.format("parquet")
    .option("path", args.hdfs + args.path)
    .option("checkpointLocation", args.checkpoint)
    .partitionBy("event_month")
    .outputMode("append")
)
writer = writer.trigger(availableNow=True) if args.once else writer.trigger(processingTime="10 seconds")
query = writer.start()
print(f"Mirroring topic '{args.topic}' to {args.hdfs}{args.path} (checkpoint: {args.checkpoint})", flush=True)

last = -1
try:
    while query.isActive:
        p = query.lastProgress
        if p and p["batchId"] != last and p["numInputRows"] > 0:
            last = p["batchId"]
            print(f"Batch {p['batchId']}: {p['numInputRows']:,} events written", flush=True)
        time.sleep(5)
    query.awaitTermination()
except KeyboardInterrupt:
    query.stop()
finally:
    spark.stop()