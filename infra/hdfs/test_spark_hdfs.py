"""Smoke test: write and read a small Parquet file on HDFS from Spark on Windows.

    python infra/hdfs/test_spark_hdfs.py
"""
import os

os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ["PATH"] += r";C:\hadoop\bin"

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("hdfs-smoke-test")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:8020")
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
    .config("spark.hadoop.dfs.replication", "1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

path = "hdfs://localhost:8020/datalake/_smoke_test"
spark.range(1000).withColumnRenamed("id", "n").write.mode("overwrite").parquet(path)
print(f"Rows written: 1000 | rows read back: {spark.read.parquet(path).count()}")
spark.stop()