"""
Stream processor (Kappa processing layer): sessions and online purchase prediction.

Two streaming queries read the same source:
  sessions     Spark keeps session state per hash bucket of user_session (applyInPandasWithState,
               event-time timeout) and emits a row when a session reaches its 3rd event and when
               it ends; the driver applies them to an online logistic regression (River).
  descriptive  event counts per product category (broadcast join with the category table)
               and pipeline totals.
Both write to PostgreSQL. The same code reads either Kafka or the HDFS data lake (re-compute).

    python -m pipeline.stream_processor --reset                 # fresh run from Kafka
    python -m pipeline.stream_processor                         # continue after a stop
    python -m pipeline.stream_processor --source lake --reset   # re-compute from the HDFS mirror
"""
import argparse
import os
import shutil
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["kafka", "lake"], default="kafka")
ap.add_argument("--bootstrap", default="localhost:9092")
ap.add_argument("--topic", default="clickstream")
ap.add_argument("--lake", default="hdfs://localhost:8020/datalake/raw/clickstream")
ap.add_argument("--checkpoint", default="checkpoint/stream_processor", help="local folder")
ap.add_argument("--categories", default="ref/product_category.parquet", help="product_id -> category_top")
ap.add_argument("--max-offsets", type=int, default=500_000, help="max Kafka events per micro-batch")
ap.add_argument("--max-files", type=int, default=1, help="lake files per micro-batch")
ap.add_argument("--watermark", default="1 minute")
ap.add_argument("--partitions", type=int, default=4, help="state partitions (parallel tasks)")
ap.add_argument("--once", action="store_true", help="process what is available, then stop")
ap.add_argument("--reset", action="store_true", help="clear run tables and the checkpoint first")
args = ap.parse_args()

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ["PYTHONPATH"] = os.getcwd() + os.pathsep + os.environ.get("PYTHONPATH", "")   # workers import pipeline.*
if os.name == "nt":
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    os.environ["PATH"] += r";C:\hadoop\bin"

import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming.state import GroupStateTimeout
from pyspark.sql.types import (DoubleType, IntegerType, LongType, StringType, StructField, StructType,
                               TimestampType)

from db_config import DB
from pipeline.descriptive import Descriptive
from pipeline.learner import Learner
from pipeline.sessions import N_BUCKETS, OUTPUT_SCHEMA, STATE_SCHEMA, update_bucket

EVENT_FIELDS = [
    StructField("event_time", StringType()),
    StructField("event_type", StringType()),
    StructField("product_id", LongType()),
    StructField("category_id", LongType()),
    StructField("category_code", StringType()),
    StructField("brand", StringType()),
    StructField("price", DoubleType()),
    StructField("user_id", LongType()),
    StructField("user_session", StringType()),
]
LAKE_SCHEMA = StructType(EVENT_FIELDS + [
    StructField("kafka_partition", IntegerType()),
    StructField("kafka_offset", LongType()),
    StructField("kafka_timestamp", TimestampType()),
    StructField("event_month", StringType()),
])
RUN_TABLES = ["eval_log", "active_predictions", "model_state", "stream_progress", "metrics_daily",
              "category_insights", "pipeline_stats", "model_weights"]


def reset():
    conn = psycopg2.connect(**DB)
    with conn, conn.cursor() as cur:
        cur.execute("TRUNCATE " + ", ".join(RUN_TABLES))
    conn.close()
    shutil.rmtree(args.checkpoint, ignore_errors=True)
    print(f"Reset: truncated {', '.join(RUN_TABLES)} and removed {args.checkpoint}", flush=True)


def read_events(spark):
    if args.source == "kafka":
        raw = (spark.readStream.format("kafka")
               .option("kafka.bootstrap.servers", args.bootstrap)
               .option("subscribe", args.topic)
               .option("startingOffsets", "earliest")
               .option("maxOffsetsPerTrigger", args.max_offsets)
               .option("failOnDataLoss", "true")
               .load())
        return (raw.select(F.from_json(F.col("value").cast("string"), StructType(EVENT_FIELDS)).alias("e"),
                           F.col("offset").alias("kafka_offset"))
                .select("e.*", "kafka_offset"))
    return (spark.readStream.schema(LAKE_SCHEMA)
            .option("maxFilesPerTrigger", args.max_files)
            .parquet(args.lake))


def main():
    if args.reset:
        reset()
    builder = SparkSession.builder.appName("StreamProcessor").master(f"local[{args.partitions}]")
    if args.source == "kafka":
        builder = builder.config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0")
    spark = (
        builder
        .config("spark.driver.memory", "4g")
        # on Windows, Arrow needs pyarrow.util.download_tzdata_on_windows() once for this to work
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(args.partitions))
        .config("spark.sql.streaming.minBatchesToRetain", "20")   # bucket state is rewritten every batch
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    sessions = session_query(spark)
    descriptive = descriptive_query(spark)
    print(f"Queries started (source: {args.source}): sessions, descriptive", flush=True)
    try:
        if args.once:
            sessions.awaitTermination()
            descriptive.awaitTermination()
        else:
            spark.streams.awaitAnyTermination()     # stop everything if one query fails
    except KeyboardInterrupt:
        pass
    finally:
        for q in spark.streams.active:
            q.stop()
        spark.stop()


def parse_time(df):
    return df.withColumn("event_ts", F.to_timestamp(F.regexp_replace("event_time", " UTC$", ""),
                                                    "yyyy-MM-dd HH:mm:ss"))


def start(df, handler, name):
    writer = (df.writeStream.queryName(name)
              .foreachBatch(handler)
              .option("checkpointLocation", os.path.join(args.checkpoint, name)))
    writer = writer.trigger(availableNow=True) if args.once else writer.trigger(processingTime="5 seconds")
    return writer.start()


def session_query(spark):
    events = (
        parse_time(read_events(spark).where(F.col("user_session").isNotNull() & (F.col("user_session") != "")))
        .withColumn("ts_ms", F.unix_millis("event_ts"))
        .withWatermark("event_ts", args.watermark)
        .withColumn("bucket", F.pmod(F.hash("user_session"), F.lit(N_BUCKETS)))
        .select("bucket", "user_session", "event_type", "price", "ts_ms", "kafka_offset", "event_ts")
    )
    sessions = events.groupBy("bucket").applyInPandasWithState(
        update_bucket, OUTPUT_SCHEMA, STATE_SCHEMA, "append", GroupStateTimeout.EventTimeTimeout)
    return start(sessions, Learner(DB), "sessions")


def descriptive_query(spark):
    categories = (spark.read.parquet(args.categories)
                  .select(F.col("product_id").cast("long"), "category_top")
                  .cache())
    print(f"Category table: {categories.count():,} products", flush=True)
    events = parse_time(read_events(spark)).select("product_id", "event_type", "event_ts", "kafka_offset")
    return start(events, Descriptive(DB, categories), "descriptive")


if __name__ == "__main__":
    main()