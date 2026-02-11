-- PostgreSQL Schema for Hot Data (Active Trading)
-- Fast queries on small, frequently accessed data

-- ============================================================================
-- ACTIVE TRADES (Small table, real-time updates)
-- ============================================================================

CREATE TABLE IF NOT EXISTS active_trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL')),

    -- Entry details
    entry_time TIMESTAMP NOT NULL DEFAULT NOW(),
    entry_price DECIMAL(10,2) NOT NULL,
    quantity INTEGER NOT NULL,

    -- Exit targets
    stop_loss DECIMAL(10,2),
    take_profit DECIMAL(10,2),

    -- Current state
    current_price DECIMAL(10,2),
    current_pnl DECIMAL(10,2),
    current_pnl_pct DECIMAL(5,2),

    -- Metadata
    agent VARCHAR(50) NOT NULL,
    confidence DECIMAL(3,2),
    risk_score INTEGER,

    -- Timestamps
    last_updated TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Indexes for fast lookups
    CONSTRAINT valid_quantity CHECK (quantity > 0),
    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

CREATE INDEX idx_active_trades_symbol ON active_trades(symbol);
CREATE INDEX idx_active_trades_entry_time ON active_trades(entry_time DESC);
CREATE INDEX idx_active_trades_agent ON active_trades(agent);

-- ============================================================================
-- PORTFOLIO STATE (Single row, frequently updated)
-- ============================================================================

CREATE TABLE IF NOT EXISTS portfolio_state (
    id INTEGER PRIMARY KEY DEFAULT 1,

    -- Capital
    cash DECIMAL(12,2) NOT NULL,
    total_value DECIMAL(12,2) NOT NULL,
    invested_value DECIMAL(12,2) NOT NULL DEFAULT 0,

    -- Daily metrics
    daily_pnl DECIMAL(10,2) NOT NULL DEFAULT 0,
    daily_pnl_pct DECIMAL(5,2) NOT NULL DEFAULT 0,

    -- Position metrics
    total_exposure_pct DECIMAL(5,2) NOT NULL DEFAULT 0,
    num_positions INTEGER NOT NULL DEFAULT 0,

    -- Performance metrics
    total_pnl DECIMAL(10,2) NOT NULL DEFAULT 0,
    total_pnl_pct DECIMAL(5,2) NOT NULL DEFAULT 0,
    max_drawdown_pct DECIMAL(5,2) NOT NULL DEFAULT 0,

    -- Timestamps
    last_updated TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Ensure only one row
    CONSTRAINT single_row CHECK (id = 1)
);

-- Initialize with default values
INSERT INTO portfolio_state (id, cash, total_value)
VALUES (1, 100000.00, 100000.00)
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- DAILY SUMMARY (One row per day)
-- ============================================================================

CREATE TABLE IF NOT EXISTS daily_summary (
    date DATE PRIMARY KEY,

    -- Capital snapshots
    starting_capital DECIMAL(12,2) NOT NULL,
    ending_capital DECIMAL(12,2) NOT NULL,

    -- P&L
    pnl DECIMAL(10,2) NOT NULL,
    pnl_pct DECIMAL(5,2) NOT NULL,

    -- Trade statistics
    num_trades INTEGER NOT NULL DEFAULT 0,
    num_wins INTEGER NOT NULL DEFAULT 0,
    num_losses INTEGER NOT NULL DEFAULT 0,
    win_rate DECIMAL(4,3),

    -- Best/worst trades
    best_trade_pnl DECIMAL(10,2),
    worst_trade_pnl DECIMAL(10,2),

    -- Risk metrics
    sharpe_ratio DECIMAL(5,2),
    max_drawdown_pct DECIMAL(5,2),
    max_exposure_pct DECIMAL(5,2),

    -- Agent performance
    best_agent VARCHAR(50),
    worst_agent VARCHAR(50),

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_daily_summary_date ON daily_summary(date DESC);

-- ============================================================================
-- STRATEGY PERFORMANCE (Aggregate metrics per strategy)
-- ============================================================================

CREATE TABLE IF NOT EXISTS strategy_performance (
    strategy_name VARCHAR(100) PRIMARY KEY,

    -- Trade counts
    total_trades INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,

    -- P&L metrics
    total_pnl DECIMAL(10,2) NOT NULL DEFAULT 0,
    avg_win_pct DECIMAL(5,2),
    avg_loss_pct DECIMAL(5,2),

    -- Performance metrics
    sharpe_ratio DECIMAL(5,2),
    win_rate DECIMAL(4,3),
    profit_factor DECIMAL(5,2),
    max_drawdown_pct DECIMAL(5,2),

    -- Timing
    avg_hold_time_hours DECIMAL(5,2),
    last_trade_time TIMESTAMP,

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_strategy_performance_sharpe ON strategy_performance(sharpe_ratio DESC NULLS LAST);
CREATE INDEX idx_strategy_performance_active ON strategy_performance(is_active, sharpe_ratio DESC);

-- ============================================================================
-- USER CONFIGURATION (Key-value store)
-- ============================================================================

CREATE TABLE IF NOT EXISTS user_config (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    last_updated TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Insert default configuration
INSERT INTO user_config (key, value, description) VALUES
    ('initial_capital', '100000', 'Initial trading capital in INR'),
    ('max_position_size_pct', '5.0', 'Max position size as % of portfolio'),
    ('max_total_exposure_pct', '30.0', 'Max total exposure as % of portfolio'),
    ('max_daily_loss_pct', '5.0', 'Daily loss limit (circuit breaker)'),
    ('max_drawdown_pct', '15.0', 'Max drawdown before halt'),
    ('max_concurrent_positions', '6', 'Maximum number of open positions'),
    ('autonomy_level', '0', 'Current AI autonomy level (0-4)')
ON CONFLICT (key) DO NOTHING;

-- ============================================================================
-- AUTONOMY SETTINGS (AI autonomy configuration)
-- ============================================================================

CREATE TABLE IF NOT EXISTS autonomy_settings (
    level INTEGER PRIMARY KEY CHECK (level >= 0 AND level <= 4),
    level_name VARCHAR(50) NOT NULL,
    auto_execute_threshold INTEGER NOT NULL,  -- Max risk score for auto-execution
    description TEXT,

    -- Performance gates to reach this level
    min_win_rate DECIMAL(4,3),
    min_sharpe_ratio DECIMAL(5,2),
    min_trades_required INTEGER,

    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Insert default autonomy levels
INSERT INTO autonomy_settings (level, level_name, auto_execute_threshold, description, min_win_rate, min_sharpe_ratio, min_trades_required) VALUES
    (0, 'Manual', 0, '100% human approval required', NULL, NULL, 0),
    (1, 'Assisted', 20, 'Very low risk only', 0.55, 1.2, 20),
    (2, 'Semi-Autonomous', 40, 'Low to medium risk', 0.58, 1.4, 50),
    (3, 'Autonomous', 70, 'Most trades automated', 0.60, 1.5, 100),
    (4, 'Fully Autonomous', 90, 'Almost everything automated', 0.62, 1.6, 200)
ON CONFLICT (level) DO NOTHING;

-- ============================================================================
-- SECTOR EXPOSURE (Real-time sector allocation tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS sector_exposure (
    sector VARCHAR(50) PRIMARY KEY,
    exposure_amount DECIMAL(12,2) NOT NULL DEFAULT 0,
    exposure_pct DECIMAL(5,2) NOT NULL DEFAULT 0,
    num_positions INTEGER NOT NULL DEFAULT 0,

    -- Limits
    max_allowed_pct DECIMAL(5,2) NOT NULL DEFAULT 20.0,

    last_updated TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Insert default sectors
INSERT INTO sector_exposure (sector, exposure_amount, exposure_pct, num_positions) VALUES
    ('IT', 0, 0, 0),
    ('Banking', 0, 0, 0),
    ('Energy', 0, 0, 0),
    ('Auto', 0, 0, 0),
    ('Pharma', 0, 0, 0),
    ('FMCG', 0, 0, 0),
    ('Metals', 0, 0, 0),
    ('Telecom', 0, 0, 0),
    ('Infrastructure', 0, 0, 0),
    ('Other', 0, 0, 0)
ON CONFLICT (sector) DO NOTHING;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to update portfolio state
CREATE OR REPLACE FUNCTION update_portfolio_state()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE portfolio_state
    SET
        num_positions = (SELECT COUNT(*) FROM active_trades),
        invested_value = (SELECT COALESCE(SUM(entry_price * quantity), 0) FROM active_trades),
        total_value = cash + (SELECT COALESCE(SUM(current_price * quantity), 0) FROM active_trades),
        total_exposure_pct = ((SELECT COALESCE(SUM(current_price * quantity), 0) FROM active_trades) / NULLIF(total_value, 0)) * 100,
        last_updated = NOW()
    WHERE id = 1;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update portfolio state when active_trades changes
CREATE TRIGGER trigger_update_portfolio
AFTER INSERT OR UPDATE OR DELETE ON active_trades
FOR EACH STATEMENT
EXECUTE FUNCTION update_portfolio_state();

-- Function to update sector exposure
CREATE OR REPLACE FUNCTION update_sector_exposure()
RETURNS TRIGGER AS $$
DECLARE
    sector_name VARCHAR(50);
BEGIN
    -- Update all sectors
    UPDATE sector_exposure se
    SET
        exposure_amount = COALESCE(sector_totals.total, 0),
        exposure_pct = (COALESCE(sector_totals.total, 0) / NULLIF((SELECT total_value FROM portfolio_state WHERE id = 1), 0)) * 100,
        num_positions = COALESCE(sector_totals.count, 0),
        last_updated = NOW()
    FROM (
        SELECT
            -- You'll need to join with a symbols table that has sector info
            -- For now, using a placeholder
            'Other' as sector,
            SUM(current_price * quantity) as total,
            COUNT(*) as count
        FROM active_trades
        GROUP BY sector
    ) sector_totals
    WHERE se.sector = sector_totals.sector;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS (Convenient queries)
-- ============================================================================

-- Current portfolio overview
CREATE OR REPLACE VIEW portfolio_overview AS
SELECT
    ps.cash,
    ps.total_value,
    ps.invested_value,
    ps.daily_pnl,
    ps.daily_pnl_pct,
    ps.total_exposure_pct,
    ps.num_positions,
    ps.total_pnl,
    ps.total_pnl_pct,
    ps.max_drawdown_pct,
    (SELECT COUNT(*) FROM active_trades WHERE current_pnl > 0) as winning_positions,
    (SELECT COUNT(*) FROM active_trades WHERE current_pnl < 0) as losing_positions
FROM portfolio_state ps
WHERE ps.id = 1;

-- Active trades with P&L
CREATE OR REPLACE VIEW active_trades_summary AS
SELECT
    trade_id,
    symbol,
    action,
    entry_time,
    entry_price,
    current_price,
    quantity,
    current_pnl,
    current_pnl_pct,
    agent,
    confidence,
    risk_score,
    EXTRACT(EPOCH FROM (NOW() - entry_time)) / 3600 as hours_held
FROM active_trades
ORDER BY entry_time DESC;

-- Performance summary (last 30 days)
CREATE OR REPLACE VIEW performance_last_30_days AS
SELECT
    COUNT(*) as total_days,
    SUM(num_trades) as total_trades,
    SUM(num_wins) as total_wins,
    SUM(num_losses) as total_losses,
    AVG(win_rate) as avg_win_rate,
    SUM(pnl) as total_pnl,
    AVG(pnl_pct) as avg_daily_return,
    AVG(sharpe_ratio) as avg_sharpe_ratio,
    MIN(max_drawdown_pct) as worst_drawdown
FROM daily_summary
WHERE date >= CURRENT_DATE - INTERVAL '30 days';

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Additional composite indexes for common queries
CREATE INDEX idx_active_trades_symbol_action ON active_trades(symbol, action);
CREATE INDEX idx_strategy_performance_updated ON strategy_performance(updated_at DESC);

-- ============================================================================
-- COMMENTS (Documentation)
-- ============================================================================

COMMENT ON TABLE active_trades IS 'Real-time active trading positions';
COMMENT ON TABLE portfolio_state IS 'Current portfolio state (single row)';
COMMENT ON TABLE daily_summary IS 'End-of-day performance summary';
COMMENT ON TABLE strategy_performance IS 'Aggregate performance metrics per strategy';
COMMENT ON TABLE user_config IS 'User configuration key-value store';
COMMENT ON TABLE autonomy_settings IS 'AI autonomy level configuration';
COMMENT ON TABLE sector_exposure IS 'Real-time sector allocation tracking';
