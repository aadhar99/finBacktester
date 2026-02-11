#!/usr/bin/env python3
"""
Comprehensive System Validation

Single script that tests every module end-to-end and confirms readiness
for a test run.  12 test sections, synchronous, print-based (matches
existing test_safeguards.py pattern).  No API keys required. No live
NSE data required.  Fast (< 60 s).

Usage:
    python3 scripts/test_system.py
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_test(name: str, fn) -> tuple:
    """Run a single test, return (passed, detail)."""
    try:
        passed, detail = fn()
        return passed, detail
    except Exception as e:
        return False, f"Exception: {e}"


def run_async_test(name: str, coro_fn) -> tuple:
    """Run an async test wrapped in asyncio.run()."""
    try:
        passed, detail = asyncio.run(coro_fn())
        return passed, detail
    except Exception as e:
        return False, f"Exception: {e}"


# ====================================================================
# Test 1: Configuration
# ====================================================================

def test_config() -> tuple:
    from config.settings import SystemConfig, DatabaseConfig, get_config

    cfg = get_config()
    assert isinstance(cfg, SystemConfig), "get_config() must return SystemConfig"

    db = cfg.database
    assert isinstance(db, DatabaseConfig), "database config missing"
    assert db.url and "postgresql" in db.url, f"Bad DB URL: {db.url}"

    # validation runs in __post_init__; if we got here it passed
    cfg._validate_config()
    return True, f"SystemConfig loaded, DB URL present, validation passed"


# ====================================================================
# Test 2: Data Fetcher
# ====================================================================

def test_data_fetcher() -> tuple:
    from data.fetcher import DataFetcher, sanitize_symbol

    # sanitize_symbol
    assert sanitize_symbol("RELIANCE") == "RELIANCE"
    assert sanitize_symbol("../../etc/passwd") == "etcpasswd"

    # synthetic data creation
    fetcher = DataFetcher()
    df = fetcher._generate_synthetic_data("TEST", "2023-01-01", "2023-06-01")
    assert len(df) > 50, f"Expected >50 rows, got {len(df)}"

    # OHLCV validate
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column: {col}"

    df = fetcher._validate_ohlcv_data(df, "TEST")
    assert len(df) > 0, "Validation removed all rows"
    assert (df["high"] >= df["low"]).all(), "OHLC integrity fail"

    return True, f"Synthetic data ({len(df)} rows), sanitize_symbol, OHLCV validation OK"


# ====================================================================
# Test 3: Data Preprocessor
# ====================================================================

def test_preprocessor() -> tuple:
    from data.fetcher import DataFetcher
    from data.preprocessor import DataPreprocessor

    fetcher = DataFetcher()
    df = fetcher._generate_synthetic_data("TEST", "2023-01-01", "2024-01-01")

    pp = DataPreprocessor()
    df = pp.add_sma(df, 20)
    df = pp.add_rsi(df)
    df = pp.add_atr(df)
    df = pp.add_bollinger_bands(df)

    checks = {
        "sma_20": "sma_20" in df.columns,
        "rsi": "rsi" in df.columns,
        "atr": "atr" in df.columns,
        "bb_upper": "bb_upper" in df.columns,
        "bb_lower": "bb_lower" in df.columns,
    }

    failed = [k for k, v in checks.items() if not v]
    if failed:
        return False, f"Missing indicators: {failed}"

    # Spot-check: SMA should be close to recent closes
    last_sma = df["sma_20"].dropna().iloc[-1]
    last_close = df["close"].iloc[-1]
    assert abs(last_sma - last_close) / last_close < 0.30, "SMA is wildly off"

    return True, f"SMA, RSI, ATR, Bollinger Bands added ({df.shape[1]} columns)"


# ====================================================================
# Test 4: Trading Agents
# ====================================================================

def test_trading_agents() -> tuple:
    from data.fetcher import DataFetcher
    from data.preprocessor import DataPreprocessor
    from agents.momentum_agent import MomentumAgent
    from agents.reversion_agent import ReversionAgent

    fetcher = DataFetcher()
    pp = DataPreprocessor()
    df = fetcher._generate_synthetic_data("TEST", "2023-01-01", "2024-01-01")
    df = pp.prepare_for_backtest(df)

    mom = MomentumAgent()
    rev = ReversionAgent()

    mom_signals = mom.generate_signals(df, {}, 100000.0, "trending_up")
    rev_signals = rev.generate_signals(df, {}, 100000.0, "ranging")

    # We don't require signals (depends on random data), just that they
    # return lists without crashing.
    assert isinstance(mom_signals, list), "MomentumAgent must return list"
    assert isinstance(rev_signals, list), "ReversionAgent must return list"

    return True, f"MomentumAgent: {len(mom_signals)} signals, ReversionAgent: {len(rev_signals)} signals"


# ====================================================================
# Test 5: Portfolio
# ====================================================================

def test_portfolio() -> tuple:
    import pandas as pd
    from execution.portfolio import Portfolio

    pf = Portfolio(initial_capital=100000.0)
    assert pf.cash == 100000.0

    ts = pd.Timestamp("2024-01-15")

    # Buy
    ok = pf.execute_buy("RELIANCE", 2500.0, 10, ts, strategy="test")
    assert ok, "Buy should succeed"
    assert pf.cash < 100000.0, "Cash should decrease after buy"

    # Update value
    pf.update_portfolio_value({"RELIANCE": 2600.0}, ts)
    assert pf.total_value > 100000.0, "Portfolio should be up"

    # Sell
    ok = pf.execute_sell("RELIANCE", 2600.0, ts, reason="test exit")
    assert ok, "Sell should succeed"

    # Position should be closed
    has = pf.position_tracker.has_position("RELIANCE")
    assert not has, "Position should be closed after sell"

    return True, f"Buy/sell executed, cash tracks, positions open/close correctly"


# ====================================================================
# Test 6: Risk Manager
# ====================================================================

def test_risk_manager() -> tuple:
    import pandas as pd
    from execution.portfolio import Portfolio
    from risk.manager import RiskManager
    from agents.base_agent import Signal, SignalType

    pf = Portfolio(initial_capital=100000.0)
    rm = RiskManager(pf)
    ts = pd.Timestamp("2024-01-15")

    # A normal sized signal should pass
    # Constraints: >= 5000 min position AND <= 5% of 100k (5000) max position
    sig = Signal(
        signal_type=SignalType.BUY,
        symbol="ITC",
        timestamp=ts,
        price=500.0,
        size=10,  # 500 * 10 = 5000 — exactly at both limits
        confidence=0.8,
        reason="test",
    )
    approved, reason = rm.check_signal(sig, {"ITC": 500.0}, ts)
    assert approved, f"Normal signal should pass: {reason}"

    # An oversized signal should fail (> 5% of capital)
    big_sig = Signal(
        signal_type=SignalType.BUY,
        symbol="INFY",
        timestamp=ts,
        price=1500.0,
        size=1000,  # 1500*1000 = 1.5M >> 5% of 100k
        confidence=0.8,
        reason="test",
    )
    approved, reason = rm.check_signal(big_sig, {"INFY": 1500.0}, ts)
    assert not approved, "Oversized signal should be rejected"

    # Halt state
    rm._halt_trading("test halt")
    assert rm.trading_halted
    approved, reason = rm.check_signal(sig, {"TCS": 3500.0}, ts)
    assert not approved, "Should reject when halted"

    rm._resume_trading()
    assert not rm.trading_halted

    return True, "Oversized rejected, halt state works, normal signal approved"


# ====================================================================
# Test 7: Transaction Costs
# ====================================================================

def test_transaction_costs() -> tuple:
    from risk.transaction_costs import TransactionCostModel

    model = TransactionCostModel(is_intraday=True)
    rt = model.calculate_round_trip_cost(20, 2500.0, 2525.0)

    assert rt["total_cost"] > 0, "Round-trip cost must be > 0"
    assert rt["gross_pnl"] == 500.0, f"Gross PnL should be 500, got {rt['gross_pnl']}"

    validation = model.validate_strategy_profitability(1.5, 50000)
    assert validation["is_profitable"], "1.5% avg profit should be profitable"

    return True, f"Round-trip cost: {rt['total_cost']}, strategy validation OK"


# ====================================================================
# Test 8: Kill Switch (async)
# ====================================================================

async def _test_kill_switch() -> tuple:
    from execution.kill_switch import KillSwitch

    class MockOrderManager:
        async def cancel_all_orders(self):
            await asyncio.sleep(0.01)
            return ["o1"]

    class MockPositionManager:
        async def close_all_positions(self):
            await asyncio.sleep(0.01)
            return ["p1"]

    class MockAgentManager:
        async def disable_all_agents(self):
            await asyncio.sleep(0.01)
            return ["a1"]

    ks = KillSwitch()
    ks.inject_dependencies(
        order_manager=MockOrderManager(),
        position_manager=MockPositionManager(),
        agent_manager=MockAgentManager(),
    )

    result = await ks.test_weekly()
    passed = result["test_passed"]
    response_time = result["total_response_time"]
    fast_enough = response_time < 30

    if not passed:
        return False, f"Weekly test failed"
    if not fast_enough:
        return False, f"Response time {response_time}s > 30s target"

    return True, f"Weekly test passed, response time {response_time}s < 30s"


def test_kill_switch() -> tuple:
    return asyncio.run(_test_kill_switch())


# ====================================================================
# Test 9: Circuit Breakers (async)
# ====================================================================

async def _test_circuit_breakers() -> tuple:
    from execution.kill_switch import KillSwitch
    from risk.circuit_breakers import CircuitBreakers

    ks = KillSwitch()
    breakers = CircuitBreakers(ks)

    # Normal: no breakers should trip
    breakers.update_portfolio_metrics(current_value=100500, daily_pnl=500, starting_capital=100000)
    triggered = await breakers.check_all_breakers()
    if len(triggered) > 0:
        return False, f"Breakers tripped on normal conditions: {triggered}"

    # Daily loss: should trip
    breakers.update_portfolio_metrics(current_value=94500, daily_pnl=-5500, starting_capital=100000)
    triggered = await breakers.check_all_breakers()
    if len(triggered) == 0:
        return False, "Daily loss breaker did NOT trip at -5.5%"

    status = breakers.get_status()
    dl_tripped = status["breakers"]["daily_loss"]["tripped"]
    if not dl_tripped:
        return False, "daily_loss breaker not marked as tripped"

    return True, f"Normal OK, daily loss -5.5% triggers breaker"


def test_circuit_breakers() -> tuple:
    return asyncio.run(_test_circuit_breakers())


# ====================================================================
# Test 10: Complexity Enforcer
# ====================================================================

def test_complexity_enforcer() -> tuple:
    from validation.complexity_enforcer import ComplexityEnforcer

    enforcer = ComplexityEnforcer()

    simple_code = """
class SimpleStrategy:
    def __init__(self):
        self.rsi_period = 14
    def should_buy(self, data):
        rsi = calculate_rsi(data, self.rsi_period)
        return rsi < 30
"""
    passes, analysis = enforcer.validate_strategy(simple_code, "Buy when RSI < 30", "SimpleRSI")
    if not passes:
        return False, f"Simple strategy should pass, got score {analysis.score}"

    complex_code = """
class KitchenSink:
    def __init__(self):
        self.sma1=10; self.sma2=20; self.sma3=50; self.sma4=100; self.sma5=200
        self.rsi=14; self.macd_f=12; self.macd_s=26; self.bb=20; self.atr=14
        self.stoch=14; self.cci=20; self.williams=14; self.adx=14
    def should_buy(self, data):
        sma = calculate_sma(data, self.sma1)
        rsi = calculate_rsi(data, self.rsi)
        macd = calculate_macd(data, self.macd_f, self.macd_s)
        bb = calculate_bollinger(data, self.bb)
        atr = calculate_atr(data, self.atr)
        stoch = calculate_stochastic(data, self.stoch)
        cci = calculate_cci(data, self.cci)
        williams = calculate_williams(data, self.williams)
        adx = calculate_adx(data, self.adx)
        return sma and rsi < 30 and macd > 0 and bb and atr and stoch and cci and williams and adx
"""
    _, analysis2 = enforcer.validate_strategy(
        complex_code, "Kitchen sink multi-indicator", "KitchenSink"
    )
    # Complex strategy should have warnings or fail
    has_warnings = len(analysis2.warnings) > 0 or not analysis2.passes
    if not has_warnings:
        return False, "Complex strategy should have warnings"

    return True, f"Simple passes (score {analysis.score:.0f}), complex warns ({len(analysis2.warnings)} warnings)"


# ====================================================================
# Test 11: Database
# ====================================================================

def test_database() -> tuple:
    import psycopg2
    from config.settings import DatabaseConfig

    db_url = DatabaseConfig().url

    try:
        conn = psycopg2.connect(db_url)
    except Exception as e:
        return False, f"Cannot connect to DB: {e}"

    cur = conn.cursor()

    # Check all 16 expected tables
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name;"
    )
    tables = {row[0] for row in cur.fetchall()}

    expected_tables = {
        "active_trades", "agent_actions", "autonomy_settings", "daily_summary",
        "llm_costs", "market_snapshots", "portfolio_state", "sector_exposure",
        "strategy_performance", "trade_decisions", "trade_learnings", "user_config",
        "bulk_deals", "corporate_actions", "fii_dii_flows", "promoter_holdings",
    }
    missing_tables = expected_tables - tables
    if missing_tables:
        cur.close(); conn.close()
        return False, f"Missing tables: {sorted(missing_tables)}"

    # Check hypertables (7 expected)
    cur.execute(
        "SELECT hypertable_name FROM timescaledb_information.hypertables ORDER BY hypertable_name;"
    )
    hypertables = {row[0] for row in cur.fetchall()}

    expected_hypertables = {
        "agent_actions", "llm_costs", "market_snapshots",
        "trade_decisions", "trade_learnings",
        "bulk_deals", "fii_dii_flows",
    }
    missing_ht = expected_hypertables - hypertables
    if missing_ht:
        cur.close(); conn.close()
        return False, f"Missing hypertables: {sorted(missing_ht)}"

    # Smart money tables present
    sm_tables = {"bulk_deals", "corporate_actions", "fii_dii_flows", "promoter_holdings"}
    missing_sm = sm_tables - tables
    if missing_sm:
        cur.close(); conn.close()
        return False, f"Missing smart money tables: {sorted(missing_sm)}"

    cur.close()
    conn.close()

    return True, f"{len(tables)} tables, {len(hypertables)} hypertables, smart money tables present"


# ====================================================================
# Test 12: Smart Money Imports
# ====================================================================

def test_smart_money_imports() -> tuple:
    from ai.agents.scrapers.nse_bulk_deals import BulkDealsScraper, BulkDeal
    from ai.agents.scrapers.nse_fii_dii import FIIDIIScraper, FIIDIIFlow
    from ai.agents.scrapers.nse_corporate_actions import CorporateActionsScraper, CorporateAction
    from ai.agents.parsers.csv_parser import parse_bulk_deals_csv, parse_fii_dii_csv
    from ai.agents.parsers.html_parser import parse_nse_table
    from ai.agents.parsers.excel_parser import parse_fii_dii_excel, parse_bulk_deals_excel

    # Scrapers instantiate
    bd = BulkDealsScraper()
    fii = FIIDIIScraper()
    ca = CorporateActionsScraper()

    # Dataclasses create
    deal = BulkDeal(date=date.today(), symbol="TEST", client_name="X",
                    deal_type="BUY", quantity=100, price=500.0, value=50000.0)
    flow = FIIDIIFlow(date=date.today(), category="FII",
                      buy_value=1000.0, sell_value=800.0, net_value=200.0)
    action = CorporateAction(symbol="TEST", announcement_date=date.today(),
                             action_type="DIVIDEND", description="test", subject="test")

    assert deal.symbol == "TEST"
    assert flow.category == "FII"
    assert action.action_type == "DIVIDEND"

    return True, "All scraper/parser classes import, instantiate, dataclasses create"


# ====================================================================
# Main runner
# ====================================================================

TESTS = [
    ("Config", test_config),
    ("Data Fetcher", test_data_fetcher),
    ("Data Preprocessor", test_preprocessor),
    ("Trading Agents", test_trading_agents),
    ("Portfolio", test_portfolio),
    ("Risk Manager", test_risk_manager),
    ("Transaction Costs", test_transaction_costs),
    ("Kill Switch", test_kill_switch),
    ("Circuit Breakers", test_circuit_breakers),
    ("Complexity Enforcer", test_complexity_enforcer),
    ("Database", test_database),
    ("Smart Money Imports", test_smart_money_imports),
]


def main():
    print("=" * 70)
    print("COMPREHENSIVE SYSTEM VALIDATION")
    print("=" * 70)
    print()

    results = []

    for idx, (name, fn) in enumerate(TESTS, 1):
        label = f"[{idx:>2}/{len(TESTS)}] {name}"
        print(f"{label}...")
        print("-" * 70)

        passed, detail = run_test(name, fn)
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {detail}")
        print()
        results.append((name, passed, detail))

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    pass_count = 0
    fail_count = 0
    for name, passed, detail in results:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name}")
        if passed:
            pass_count += 1
        else:
            fail_count += 1

    print()
    print(f"  Passed: {pass_count}/{len(results)}")
    print(f"  Failed: {fail_count}/{len(results)}")
    print("=" * 70)

    if fail_count == 0:
        print("ALL TESTS PASSED — system ready for test run")
    else:
        print("SOME TESTS FAILED — fix before proceeding")

    print("=" * 70)
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    main()
