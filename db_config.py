"""Satu-satunya tempat mengisi koneksi Postgres. Edit nilai default di bawah."""
import os

DB = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "ecommerce_db"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", ""),
)
