-- CashYar Database Schema for Supabase + Lovable
-- Paste this in Supabase SQL Editor and click Run

-- ============================================================
-- USERS
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
  user_id              INTEGER PRIMARY KEY,
  uuid                 UUID NOT NULL UNIQUE,
  age                  INTEGER NOT NULL,
  gender               TEXT NOT NULL,
  city                 TEXT NOT NULL,
  university           TEXT NOT NULL,
  major                TEXT NOT NULL,
  study_year           INTEGER NOT NULL,
  living_status        TEXT NOT NULL,
  scholarship          INTEGER NOT NULL DEFAULT 0,
  part_time_income     INTEGER NOT NULL DEFAULT 0,
  other_income         INTEGER NOT NULL DEFAULT 0,
  total_income         INTEGER NOT NULL,
  monthly_budget       NUMERIC(10,2) NOT NULL,
  financial_goal       TEXT NOT NULL,
  goal_amount          INTEGER NOT NULL,
  target_months        INTEGER NOT NULL,
  financial_personality TEXT NOT NULL,
  created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- TRANSACTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS transactions (
  transaction_id       INTEGER PRIMARY KEY,
  user_id              INTEGER NOT NULL REFERENCES users(user_id),
  datetime             TIMESTAMPTZ NOT NULL,
  category             TEXT NOT NULL,
  merchant             TEXT NOT NULL,
  amount               NUMERIC(10,2) NOT NULL,
  payment_method       TEXT NOT NULL,
  planned_purchase     BOOLEAN NOT NULL DEFAULT FALSE,
  online_purchase      BOOLEAN NOT NULL DEFAULT FALSE,
  purchase_location    TEXT NOT NULL,
  season               TEXT NOT NULL,
  event                TEXT NOT NULL DEFAULT 'None',
  day_of_week          TEXT NOT NULL,
  hour                 INTEGER NOT NULL,
  remaining_budget     NUMERIC(12,2),
  monthly_spending     NUMERIC(12,2),
  monthly_saving       NUMERIC(12,2),
  goal_progress        NUMERIC(5,3),
  financial_score      INTEGER,
  financial_risk       TEXT,
  recommended_action   TEXT
);

CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_datetime ON transactions(datetime);
CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);

-- ============================================================
-- BEHAVIORAL SUMMARY
-- ============================================================
CREATE TABLE IF NOT EXISTS behavioral_summary (
  id                       BIGSERIAL PRIMARY KEY,
  user_id                  INTEGER NOT NULL REFERENCES users(user_id),
  month                    INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
  total_income             INTEGER NOT NULL,
  total_spending           NUMERIC(12,2) NOT NULL,
  total_saving             NUMERIC(12,2) NOT NULL,
  saving_rate              NUMERIC(10,3),
  restaurant_ratio         NUMERIC(6,3),
  coffee_ratio             NUMERIC(6,3),
  shopping_ratio           NUMERIC(6,3),
  transport_ratio          NUMERIC(6,3),
  electronics_ratio        NUMERIC(6,3),
  entertainment_ratio      NUMERIC(6,3),
  weekend_spending_ratio   NUMERIC(6,3),
  planned_purchase_ratio   NUMERIC(6,3),
  online_purchase_ratio    NUMERIC(6,3),
  budget_adherence         NUMERIC(6,3),
  financial_score          NUMERIC(6,2),
  goal_progress            NUMERIC(6,3),
  consecutive_budget_success INTEGER DEFAULT 0,
  spending_trend           TEXT,
  financial_personality    TEXT,
  target_months            INTEGER,
  financial_risk           TEXT,
  will_reach_goal          TEXT,
  behavior_label           TEXT,
  recommended_action       TEXT,
  UNIQUE (user_id, month)
);

CREATE INDEX IF NOT EXISTS idx_summary_user_id ON behavioral_summary(user_id);
CREATE INDEX IF NOT EXISTS idx_summary_month ON behavioral_summary(month);

-- ============================================================
-- PUBLIC READ (for Lovable prototype)
-- ============================================================
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE behavioral_summary ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read users" ON users FOR SELECT USING (true);
CREATE POLICY "Public read transactions" ON transactions FOR SELECT USING (true);
CREATE POLICY "Public read summary" ON behavioral_summary FOR SELECT USING (true);
