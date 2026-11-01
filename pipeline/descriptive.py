"""
Driver-side handler for the descriptive stream (foreachBatch): event counts per category.

Each micro-batch is aggregated in Spark (category x event type, plus the batch's latest event
time and offset); only the small result reaches the driver. Counts, pipeline totals and the
batch id are committed in one PostgreSQL transaction, so a repeated batch is skipped safely.
"""
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import execute_values
from pyspark.sql import functions as F

QUERY = "descriptive"
TYPES = {"view": 0, "cart": 1, "purchase": 2}


class Descriptive:
    def __init__(self, db, categories):
        """categories: small DataFrame (product_id, category_top), broadcast into every batch."""
        self.db = db
        self.categories = F.broadcast(categories)
        conn = psycopg2.connect(**db)
        with conn, conn.cursor() as cur:
            cur.execute("SELECT last_batch_id FROM stream_progress WHERE query = %s", (QUERY,))
            r = cur.fetchone()
            self.last_batch = r[0] if r else -1
        conn.close()

    def __call__(self, batch_df, batch_id):
        if batch_id <= self.last_batch:
            print(f"Descriptive batch {batch_id}: already committed, skipped", flush=True)
            return
        rows = (batch_df.join(self.categories, "product_id", "left")
                .groupBy(F.coalesce("category_top", F.lit("unknown")).alias("category"), "event_type")
                .agg(F.count(F.lit(1)).alias("n"), F.max(F.unix_millis("event_ts")).alias("t_max"),
                     F.max("kafka_offset").alias("offset_max"))
                .collect())
        if not rows:
            return
        counts = {}
        for r in rows:
            i = TYPES.get(r["event_type"])
            if i is not None:
                counts.setdefault(r["category"], [0, 0, 0])[i] += r["n"]
        n_events = sum(r["n"] for r in rows)
        t_max = max((r["t_max"] for r in rows if r["t_max"] is not None), default=None)
        t_max = datetime.fromtimestamp(t_max / 1000, tz=timezone.utc) if t_max is not None else None
        offset_max = max(r["offset_max"] for r in rows)

        conn = psycopg2.connect(**self.db)
        try:
            with conn, conn.cursor() as cur:
                if counts:
                    execute_values(cur, """
                        INSERT INTO category_insights (category, views, carts, purchases) VALUES %s
                        ON CONFLICT (category) DO UPDATE SET
                          views = category_insights.views + EXCLUDED.views,
                          carts = category_insights.carts + EXCLUDED.carts,
                          purchases = category_insights.purchases + EXCLUDED.purchases""",
                                   [(c, *v) for c, v in counts.items()])
                cur.execute("""
                    INSERT INTO pipeline_stats (id, events_total, last_event_time, last_offset, updated_at)
                    VALUES (1, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET
                      events_total = coalesce(pipeline_stats.events_total, 0) + EXCLUDED.events_total,
                      last_event_time = greatest(pipeline_stats.last_event_time, EXCLUDED.last_event_time),
                      last_offset = greatest(pipeline_stats.last_offset, EXCLUDED.last_offset),
                      updated_at = now()""", (n_events, t_max, offset_max))
                cur.execute("""
                    INSERT INTO stream_progress (query, last_batch_id, updated_at) VALUES (%s, %s, now())
                    ON CONFLICT (query) DO UPDATE SET last_batch_id = EXCLUDED.last_batch_id,
                                                      updated_at = EXCLUDED.updated_at""", (QUERY, batch_id))
        finally:
            conn.close()
        self.last_batch = batch_id
        print(f"Descriptive batch {batch_id}: {n_events:,} events | {len(counts)} categories", flush=True)
