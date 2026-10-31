-- Jalankan sekali di pgAdmin (Query Tool) pada database proyekmu.

-- State per session (satu baris per session)
CREATE TABLE IF NOT EXISTS session_state (
  session_id        TEXT PRIMARY KEY,
  n_events          INT NOT NULL DEFAULT 0,
  first_time        TIMESTAMPTZ,
  last_time         TIMESTAMPTZ,
  k_time            TIMESTAMPTZ,
  n_cart            INT NOT NULL DEFAULT 0,
  price_sum         DOUBLE PRECISION NOT NULL DEFAULT 0,
  price_cnt         INT NOT NULL DEFAULT 0,
  price_max         DOUBLE PRECISION NOT NULL DEFAULT 0,
  purchase_before_k BOOLEAN NOT NULL DEFAULT FALSE,
  purchase_after    BOOLEAN NOT NULL DEFAULT FALSE,
  predicted         BOOLEAN NOT NULL DEFAULT FALSE,
  prob              DOUBLE PRECISION,
  closed            BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_session_open
  ON session_state (last_time) WHERE predicted AND NOT closed;

-- Catatan evaluasi: satu baris per session yang sudah selesai (prediksi dibuat SEBELUM label diketahui)
CREATE TABLE IF NOT EXISTS eval_log (
  session_id   TEXT PRIMARY KEY,
  t_pred       TIMESTAMPTZ NOT NULL,
  t_close      TIMESTAMPTZ NOT NULL,
  f_c          DOUBLE PRECISION,
  f_avg_price  DOUBLE PRECISION,
  f_max_price  DOUBLE PRECISION,
  f_dur        DOUBLE PRECISION,
  prob         DOUBLE PRECISION NOT NULL,
  y            INT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_tpred ON eval_log (t_pred);

-- Metrik per hari (dihitung oleh pipeline)
CREATE TABLE IF NOT EXISTS metrics_daily (
  day        DATE PRIMARY KEY,
  n          INT,
  pos_rate   DOUBLE PRECISION,
  auc        DOUBLE PRECISION,
  pr_auc     DOUBLE PRECISION,
  p_at_5     DOUBLE PRECISION,
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Wawasan kategori (kumulatif)
CREATE TABLE IF NOT EXISTS category_insights (
  category  TEXT PRIMARY KEY,
  views     BIGINT NOT NULL DEFAULT 0,
  carts     BIGINT NOT NULL DEFAULT 0,
  purchases BIGINT NOT NULL DEFAULT 0
);

-- Statistik pipeline (satu baris)
CREATE TABLE IF NOT EXISTS pipeline_stats (
  id               INT PRIMARY KEY,
  events_total     BIGINT,
  last_event_time  TIMESTAMPTZ,
  batches          BIGINT,
  sessions_learned BIGINT,
  last_offset      BIGINT,
  updated_at       TIMESTAMPTZ
);

-- Bobot fitur model dari waktu ke waktu
CREATE TABLE IF NOT EXISTS model_weights (
  batch   BIGINT,
  feature TEXT,
  weight  DOUBLE PRECISION,
  PRIMARY KEY (batch, feature)
);
