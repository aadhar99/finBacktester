#!/usr/bin/env python3
"""
Database Verification Script

Connects to the trading system database and verifies:
- TimescaleDB extension is enabled
- All expected tables exist
- All expected hypertables exist
- Default data is present
"""

import sys
import psycopg2

DATABASE_URL = "postgresql://trading_user:trading_password_dev@localhost:5432/trading_system"

EXPECTED_TABLES = [
    "active_trades",
    "agent_actions",
    "autonomy_settings",
    "daily_summary",
    "llm_costs",
    "market_snapshots",
    "portfolio_state",
    "sector_exposure",
    "strategy_performance",
    "trade_decisions",
    "trade_learnings",
    "user_config",
]

EXPECTED_HYPERTABLES = [
    "agent_actions",
    "llm_costs",
    "market_snapshots",
    "trade_decisions",
    "trade_learnings",
]

SMART_MONEY_TABLES = [
    "bulk_deals",
    "corporate_actions",
    "fii_dii_flows",
    "promoter_holdings",
]

SMART_MONEY_HYPERTABLES = [
    "bulk_deals",
    "fii_dii_flows",
]


def verify():
    ok = True

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

    # Check TimescaleDB extension
    cur.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb';")
    if cur.fetchone():
        print("✅ TimescaleDB extension enabled")
    else:
        print("❌ TimescaleDB extension NOT enabled")
        ok = False

    # Check tables
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name;"
    )
    tables = {row[0] for row in cur.fetchall()}

    print(f"\n📋 Tables ({len(tables)} found):")
    for t in sorted(EXPECTED_TABLES + SMART_MONEY_TABLES):
        if t in tables:
            print(f"  ✅ {t}")
        else:
            print(f"  ❌ {t} MISSING")
            ok = False

    # Check hypertables
    cur.execute(
        "SELECT hypertable_name FROM timescaledb_information.hypertables ORDER BY hypertable_name;"
    )
    hypertables = {row[0] for row in cur.fetchall()}

    print(f"\n⏱️  Hypertables ({len(hypertables)} found):")
    for h in sorted(EXPECTED_HYPERTABLES + SMART_MONEY_HYPERTABLES):
        if h in hypertables:
            print(f"  ✅ {h}")
        else:
            print(f"  ❌ {h} MISSING")
            ok = False

    # Check default data
    cur.execute("SELECT COUNT(*) FROM portfolio_state;")
    ps_count = cur.fetchone()[0]
    print(f"\n📊 Default data:")
    print(f"  {'✅' if ps_count == 1 else '❌'} portfolio_state: {ps_count} row(s)")

    cur.execute("SELECT COUNT(*) FROM autonomy_settings;")
    as_count = cur.fetchone()[0]
    print(f"  {'✅' if as_count == 5 else '❌'} autonomy_settings: {as_count} row(s)")

    cur.execute("SELECT COUNT(*) FROM user_config;")
    uc_count = cur.fetchone()[0]
    print(f"  {'✅' if uc_count >= 7 else '❌'} user_config: {uc_count} row(s)")

    cur.execute("SELECT COUNT(*) FROM sector_exposure;")
    se_count = cur.fetchone()[0]
    print(f"  {'✅' if se_count >= 10 else '❌'} sector_exposure: {se_count} row(s)")

    cur.close()
    conn.close()

    print(f"\n{'✅ All checks passed!' if ok else '❌ Some checks failed.'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    verify()
