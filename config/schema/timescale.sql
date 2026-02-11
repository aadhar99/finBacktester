-- TimescaleDB Schema for Warm Data (Audit Trails & Time-Series)
-- High-volume time-series data with automatic compression and retention

-- ============================================================================
-- SETUP TIMESCALEDB EXTENSION
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- TRADE DECISIONS (Complete audit trail for all AI decisions)
-- ============================================================================

CREATE TABLE IF NOT EXISTS trade_decisions (
    id BIGSERIAL,
    decision_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Trade details
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL', 'HOLD', 'EXIT')),
    quantity INTEGER,
    entry_price DECIMAL(10,2),
    exit_price DECIMAL(10,2),
    exit_time TIMESTAMPTZ,

    -- AI decision data
    agent VARCHAR(50) NOT NULL,
    ai_reasoning TEXT NOT NULL,
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    risk_score INTEGER NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),

    -- Market context at decision time
    market_regime VARCHAR(20),
    vix DECIMAL(5,2),
    sector_sentiment VARCHAR(20),
    nifty_change_pct DECIMAL(5,2),

    -- Expected outcome (AI prediction)
    expected_return_pct DECIMAL(5,2),
    expected_hold_hours DECIMAL(5,2),

    -- Actual outcome (filled after trade closes)
    realized_pnl DECIMAL(10,2),
    realized_pnl_pct DECIMAL(5,2),
    actual_hold_hours DECIMAL(5,2),
    outcome VARCHAR(20) CHECK (outcome IN ('WIN', 'LOSS', 'BREAKEVEN', 'PENDING')),

    -- Human interaction
    required_human_approval BOOLEAN NOT NULL,
    human_decision VARCHAR(20) CHECK (human_decision IN ('APPROVED', 'REJECTED', 'MODIFIED', NULL)),
    human_decision_time TIMESTAMPTZ,
    human_notes TEXT,

    -- SEBI compliance
    algo_id VARCHAR(50),
    client_id VARCHAR(50),
    exchange_order_id VARCHAR(100),

    -- Metadata
    llm_provider VARCHAR(20),
    llm_cost DECIMAL(8,4),
    decision_latency_ms INTEGER,

    PRIMARY KEY (timestamp, decision_id)
);

-- Convert to hypertable (automatic time-based partitioning)
SELECT create_hypertable('trade_decisions', 'timestamp', if_not_exists => TRUE);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_trade_decisions_symbol ON trade_decisions(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trade_decisions_agent ON trade_decisions(agent, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trade_decisions_outcome ON trade_decisions(outcome, timestamp DESC) WHERE outcome IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trade_decisions_decision_id ON trade_decisions(decision_id);

-- ============================================================================
-- DATA RETENTION POLICY (Archive to S3/GCS after 90 days)
-- ============================================================================

SELECT add_retention_policy('trade_decisions', INTERVAL '90 days', if_not_exists => TRUE);

-- ============================================================================
-- COMPRESSION POLICY (Compress data older than 7 days for 10-100x savings)
-- ============================================================================

ALTER TABLE trade_decisions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'agent, symbol',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('trade_decisions', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================================================
-- CONTINUOUS AGGREGATES (Pre-computed analytics for fast queries)
-- ============================================================================

-- Hourly aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS trade_decisions_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS hour,
    agent,
    symbol,
    COUNT(*) as num_decisions,
    AVG(confidence) as avg_confidence,
    AVG(risk_score) as avg_risk_score,
    COUNT(*) FILTER (WHERE outcome = 'WIN') as num_wins,
    COUNT(*) FILTER (WHERE outcome = 'LOSS') as num_losses,
    COUNT(*) FILTER (WHERE required_human_approval = TRUE) as num_requiring_approval,
    AVG(realized_pnl_pct) FILTER (WHERE outcome IN ('WIN', 'LOSS')) as avg_return_pct,
    SUM(llm_cost) as total_llm_cost
FROM trade_decisions
GROUP BY hour, agent, symbol
WITH NO DATA;

-- Refresh policy (update every hour)
SELECT add_continuous_aggregate_policy('trade_decisions_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Daily aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS trade_decisions_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS day,
    agent,
    COUNT(*) as num_decisions,
    AVG(confidence) as avg_confidence,
    AVG(risk_score) as avg_risk_score,
    COUNT(*) FILTER (WHERE outcome = 'WIN') as num_wins,
    COUNT(*) FILTER (WHERE outcome = 'LOSS') as num_losses,
    CASE
        WHEN COUNT(*) FILTER (WHERE outcome IN ('WIN', 'LOSS')) > 0
        THEN COUNT(*) FILTER (WHERE outcome = 'WIN')::DECIMAL / COUNT(*) FILTER (WHERE outcome IN ('WIN', 'LOSS'))
        ELSE NULL
    END as win_rate,
    AVG(realized_pnl_pct) FILTER (WHERE outcome IN ('WIN', 'LOSS')) as avg_return_pct,
    SUM(realized_pnl) FILTER (WHERE outcome IN ('WIN', 'LOSS')) as total_pnl,
    SUM(llm_cost) as total_llm_cost
FROM trade_decisions
GROUP BY day, agent
WITH NO DATA;

SELECT add_continuous_aggregate_policy('trade_decisions_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ============================================================================
-- AGENT ACTIONS LOG (High-volume agent activity log)
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_actions (
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_name VARCHAR(100) NOT NULL,
    action_type VARCHAR(50) NOT NULL,  -- ANALYSIS, DECISION, LEARNING, PATTERN_RECOGNITION, etc.

    -- Context and result (JSONB for flexibility)
    context JSONB,
    result JSONB,

    -- Performance metrics
    latency_ms INTEGER,
    llm_cost DECIMAL(8,4),
    llm_provider VARCHAR(20),

    -- Error tracking
    error_occurred BOOLEAN DEFAULT FALSE,
    error_message TEXT,

    PRIMARY KEY (timestamp, agent_name, action_type)
);

-- Convert to hypertable
SELECT create_hypertable('agent_actions', 'timestamp', if_not_exists => TRUE);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_agent_actions_agent ON agent_actions(agent_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_actions_type ON agent_actions(action_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_actions_error ON agent_actions(error_occurred, timestamp DESC) WHERE error_occurred = TRUE;

-- Retention (90 days)
SELECT add_retention_policy('agent_actions', INTERVAL '90 days', if_not_exists => TRUE);

-- Compression
ALTER TABLE agent_actions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'agent_name, action_type',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('agent_actions', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================================================
-- MARKET DATA SNAPSHOTS (Historical market data for backtesting)
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,

    -- OHLCV data
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,

    -- Technical indicators
    vwap DECIMAL(10,2),
    rsi DECIMAL(5,2),
    macd DECIMAL(10,4),
    macd_signal DECIMAL(10,4),

    -- Market context
    market_regime VARCHAR(20),
    sector VARCHAR(50),

    PRIMARY KEY (timestamp, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('market_snapshots', 'timestamp', if_not_exists => TRUE);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_market_snapshots_symbol ON market_snapshots(symbol, timestamp DESC);

-- Retention (180 days - longer for backtesting)
SELECT add_retention_policy('market_snapshots', INTERVAL '180 days', if_not_exists => TRUE);

-- Compression
ALTER TABLE market_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('market_snapshots', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================================================
-- TRADE LEARNINGS (Post-trade analysis results)
-- ============================================================================

CREATE TABLE IF NOT EXISTS trade_learnings (
    id BIGSERIAL,
    trade_id VARCHAR(50) NOT NULL,  -- References trade_decisions.decision_id
    analysis_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Analysis results
    prediction_accuracy VARCHAR(20) CHECK (prediction_accuracy IN ('ACCURATE', 'PARTIALLY_ACCURATE', 'INACCURATE')),

    -- What went right/wrong (JSONB arrays)
    what_went_right JSONB,
    what_went_wrong JSONB,
    missed_signals JSONB,
    lessons_learned JSONB,

    -- Pattern detection
    pattern_name VARCHAR(100),
    pattern_description TEXT,
    pattern_confidence DECIMAL(3,2),
    similar_trades JSONB,  -- Array of trade IDs with similar patterns

    -- Recommendation
    recommendation TEXT,
    analysis_confidence DECIMAL(3,2),

    -- LLM metadata
    llm_provider VARCHAR(20),
    llm_cost DECIMAL(8,4),

    PRIMARY KEY (analysis_timestamp, trade_id)
);

-- Convert to hypertable
SELECT create_hypertable('trade_learnings', 'analysis_timestamp', if_not_exists => TRUE);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_trade_learnings_trade_id ON trade_learnings(trade_id);
CREATE INDEX IF NOT EXISTS idx_trade_learnings_pattern ON trade_learnings(pattern_name, analysis_timestamp DESC) WHERE pattern_name IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trade_learnings_accuracy ON trade_learnings(prediction_accuracy, analysis_timestamp DESC);

-- Retention
SELECT add_retention_policy('trade_learnings', INTERVAL '90 days', if_not_exists => TRUE);

-- Compression
ALTER TABLE trade_learnings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'pattern_name',
    timescaledb.compress_orderby = 'analysis_timestamp DESC'
);

SELECT add_compression_policy('trade_learnings', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================================================
-- LLM COSTS TRACKING (Track LLM API costs over time)
-- ============================================================================

CREATE TABLE IF NOT EXISTS llm_costs (
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provider VARCHAR(20) NOT NULL,
    model VARCHAR(50) NOT NULL,

    -- Usage
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,

    -- Cost
    cost DECIMAL(8,4) NOT NULL,

    -- Context
    agent_name VARCHAR(100),
    task_type VARCHAR(50),

    PRIMARY KEY (timestamp, provider)
);

-- Convert to hypertable
SELECT create_hypertable('llm_costs', 'timestamp', if_not_exists => TRUE);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_llm_costs_provider ON llm_costs(provider, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_llm_costs_agent ON llm_costs(agent_name, timestamp DESC);

-- Continuous aggregate for daily costs
CREATE MATERIALIZED VIEW IF NOT EXISTS llm_costs_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS day,
    provider,
    agent_name,
    SUM(cost) as total_cost,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    COUNT(*) as num_calls
FROM llm_costs
GROUP BY day, provider, agent_name
WITH NO DATA;

SELECT add_continuous_aggregate_policy('llm_costs_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Retention (365 days for cost tracking)
SELECT add_retention_policy('llm_costs', INTERVAL '365 days', if_not_exists => TRUE);

-- Compression
ALTER TABLE llm_costs SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'provider, agent_name',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('llm_costs', INTERVAL '30 days', if_not_exists => TRUE);

-- ============================================================================
-- USEFUL VIEWS
-- ============================================================================

-- Recent trade decisions with outcomes
CREATE OR REPLACE VIEW recent_trade_decisions AS
SELECT
    timestamp,
    decision_id,
    symbol,
    action,
    agent,
    confidence,
    risk_score,
    market_regime,
    outcome,
    realized_pnl,
    realized_pnl_pct,
    required_human_approval,
    human_decision
FROM trade_decisions
WHERE timestamp >= NOW() - INTERVAL '7 days'
ORDER BY timestamp DESC;

-- Performance by agent (last 30 days)
CREATE OR REPLACE VIEW agent_performance_30d AS
SELECT
    agent,
    COUNT(*) as total_decisions,
    COUNT(*) FILTER (WHERE outcome = 'WIN') as wins,
    COUNT(*) FILTER (WHERE outcome = 'LOSS') as losses,
    CASE
        WHEN COUNT(*) FILTER (WHERE outcome IN ('WIN', 'LOSS')) > 0
        THEN COUNT(*) FILTER (WHERE outcome = 'WIN')::DECIMAL / COUNT(*) FILTER (WHERE outcome IN ('WIN', 'LOSS'))
        ELSE 0
    END as win_rate,
    AVG(confidence) as avg_confidence,
    AVG(risk_score) as avg_risk_score,
    AVG(realized_pnl_pct) FILTER (WHERE outcome IN ('WIN', 'LOSS')) as avg_return_pct,
    SUM(realized_pnl) FILTER (WHERE outcome IN ('WIN', 'LOSS')) as total_pnl,
    SUM(llm_cost) as total_llm_cost
FROM trade_decisions
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY agent
ORDER BY total_pnl DESC NULLS LAST;

-- LLM cost summary (last 30 days)
CREATE OR REPLACE VIEW llm_cost_summary_30d AS
SELECT
    provider,
    SUM(cost) as total_cost,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    COUNT(*) as num_calls,
    AVG(cost) as avg_cost_per_call
FROM llm_costs
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY provider
ORDER BY total_cost DESC;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to insert trade decision with automatic archival metadata
CREATE OR REPLACE FUNCTION log_trade_decision(
    p_decision_id VARCHAR,
    p_symbol VARCHAR,
    p_action VARCHAR,
    p_agent VARCHAR,
    p_ai_reasoning TEXT,
    p_confidence DECIMAL,
    p_risk_score INTEGER,
    p_market_regime VARCHAR DEFAULT NULL,
    p_vix DECIMAL DEFAULT NULL,
    p_required_approval BOOLEAN DEFAULT TRUE
) RETURNS BIGINT AS $$
DECLARE
    inserted_id BIGINT;
BEGIN
    INSERT INTO trade_decisions (
        decision_id, symbol, action, agent, ai_reasoning,
        confidence, risk_score, market_regime, vix,
        required_human_approval, outcome
    ) VALUES (
        p_decision_id, p_symbol, p_action, p_agent, p_ai_reasoning,
        p_confidence, p_risk_score, p_market_regime, p_vix,
        p_required_approval, 'PENDING'
    ) RETURNING id INTO inserted_id;

    RETURN inserted_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE trade_decisions IS 'Complete audit trail of all AI trading decisions (hypertable)';
COMMENT ON TABLE agent_actions IS 'High-volume log of all agent activities (hypertable)';
COMMENT ON TABLE market_snapshots IS 'Historical market data for backtesting (hypertable)';
COMMENT ON TABLE trade_learnings IS 'Post-trade analysis and learnings (hypertable)';
COMMENT ON TABLE llm_costs IS 'LLM API cost tracking (hypertable)';
