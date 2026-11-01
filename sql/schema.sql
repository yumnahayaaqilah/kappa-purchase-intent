-- PostgreSQL serving layer for the stream processor. Safe to run more than once.

-- One row per finished, predicted session (prediction made before the label was known)
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

-- Predicted sessions that have not ended yet
CREATE TABLE IF NOT EXISTS active_predictions (
  session_id  TEXT PRIMARY KEY,
  t_pred      TIMESTAMPTZ NOT NULL,
  n_events    INT,
  n_cart      INT,
  prob        DOUBLE PRECISION NOT NULL,
  last_time   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_active_last ON active_predictions (last_time);

-- Online model, saved in the same transaction as the batch it learned from
CREATE TABLE IF NOT EXISTS model_state (
  id          INT PRIMARY KEY,
  batch_id    BIGINT,
  model       BYTEA NOT NULL,
  n_learned   BIGINT,
  updated_at  TIMESTAMPTZ
);

-- Last committed micro-batch per streaming query (skip batches replayed after a restart)
CREATE TABLE IF NOT EXISTS stream_progress (
  query          TEXT PRIMARY KEY,
  last_batch_id  BIGINT NOT NULL,
  updated_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS metrics_daily (
  day        DATE PRIMARY KEY,
  n          INT,
  pos_rate   DOUBLE PRECISION,
  auc        DOUBLE PRECISION,
  pr_auc     DOUBLE PRECISION,
  p_at_5     DOUBLE PRECISION,
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS category_insights (
  category  TEXT PRIMARY KEY,
  views     BIGINT NOT NULL DEFAULT 0,
  carts     BIGINT NOT NULL DEFAULT 0,
  purchases BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pipeline_stats (
  id               INT PRIMARY KEY,
  events_total     BIGINT,
  last_event_time  TIMESTAMPTZ,
  batches          BIGINT,
  sessions_learned BIGINT,
  last_offset      BIGINT,
  updated_at       TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS model_weights (
  batch   BIGINT,
  feature TEXT,
  weight  DOUBLE PRECISION,
  PRIMARY KEY (batch, feature)
);

-- Dashboard compatibility: the v1 table session_state becomes a view over active_predictions
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = 'session_state') THEN
    EXECUTE 'DROP TABLE session_state';
  END IF;
END $$;
CREATE OR REPLACE VIEW session_state AS
  SELECT session_id, n_events, n_cart, prob, last_time, TRUE AS predicted, FALSE AS closed
  FROM active_predictions;
