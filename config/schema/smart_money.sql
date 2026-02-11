-- Smart Money Schema — Bulk Deals, FII/DII Flows, Corporate Actions, Promoter Holdings
-- Tracks institutional activity signals for the Smart Money trading strategy

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- BULK DEALS (High-volume institutional trades from NSE)
-- ============================================================================

CREATE TABLE IF NOT EXISTS bulk_deals (
    date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    client_name VARCHAR(200) NOT NULL,
    deal_type VARCHAR(4) NOT NULL CHECK (deal_type IN ('BUY', 'SELL')),
    quantity BIGINT NOT NULL CHECK (quantity > 0),
    price DECIMAL(10,2) NOT NULL CHECK (price > 0),
    value DECIMAL(16,2) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (date, symbol, client_name, deal_type, quantity)
);

SELECT create_hypertable('bulk_deals', 'date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_bulk_deals_symbol_date ON bulk_deals(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_bulk_deals_client ON bulk_deals(client_name, date DESC);
CREATE INDEX IF NOT EXISTS idx_bulk_deals_deal_type ON bulk_deals(deal_type, date DESC);

-- Retention: 365 days
SELECT add_retention_policy('bulk_deals', INTERVAL '365 days', if_not_exists => TRUE);

-- Compression after 30 days
ALTER TABLE bulk_deals SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'date DESC'
);

SELECT add_compression_policy('bulk_deals', INTERVAL '30 days', if_not_exists => TRUE);

-- ============================================================================
-- FII/DII FLOWS (Foreign & Domestic Institutional Investor daily activity)
-- ============================================================================

CREATE TABLE IF NOT EXISTS fii_dii_flows (
    date DATE NOT NULL,
    category VARCHAR(3) NOT NULL CHECK (category IN ('FII', 'DII')),
    buy_value DECIMAL(16,2) NOT NULL,
    sell_value DECIMAL(16,2) NOT NULL,
    net_value DECIMAL(16,2) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (date, category)
);

SELECT create_hypertable('fii_dii_flows', 'date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_fii_dii_flows_category_date ON fii_dii_flows(category, date DESC);

-- Retention: 365 days
SELECT add_retention_policy('fii_dii_flows', INTERVAL '365 days', if_not_exists => TRUE);

-- Compression after 30 days
ALTER TABLE fii_dii_flows SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'category',
    timescaledb.compress_orderby = 'date DESC'
);

SELECT add_compression_policy('fii_dii_flows', INTERVAL '30 days', if_not_exists => TRUE);

-- ============================================================================
-- CORPORATE ACTIONS (Dividends, buybacks, board meetings, etc.)
-- ============================================================================

CREATE TABLE IF NOT EXISTS corporate_actions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    announcement_date DATE NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    description TEXT,
    subject TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (symbol, announcement_date, action_type, subject)
);

CREATE INDEX IF NOT EXISTS idx_corporate_actions_symbol_date ON corporate_actions(symbol, announcement_date DESC);
CREATE INDEX IF NOT EXISTS idx_corporate_actions_type ON corporate_actions(action_type, announcement_date DESC);

-- ============================================================================
-- PROMOTER HOLDINGS (Quarterly promoter stake changes from SEBI filings)
-- ============================================================================

CREATE TABLE IF NOT EXISTS promoter_holdings (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    quarter VARCHAR(10) NOT NULL,  -- e.g. 'Q3FY25'
    promoter_pct DECIMAL(5,2) NOT NULL,
    change_pct DECIMAL(5,2),  -- Change from previous quarter
    pledge_pct DECIMAL(5,2),  -- Percentage of promoter shares pledged

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (symbol, quarter)
);

CREATE INDEX IF NOT EXISTS idx_promoter_holdings_symbol ON promoter_holdings(symbol, quarter DESC);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE bulk_deals IS 'NSE bulk/block deal records (hypertable)';
COMMENT ON TABLE fii_dii_flows IS 'FII/DII daily buy/sell flows (hypertable)';
COMMENT ON TABLE corporate_actions IS 'Corporate actions — dividends, buybacks, board meetings';
COMMENT ON TABLE promoter_holdings IS 'Quarterly promoter stake changes from SEBI filings';
