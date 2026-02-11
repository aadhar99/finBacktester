"""
Main entry point for the trading system.

Run backtests, optimize parameters, and analyze results.
"""

import logging
from datetime import datetime

from config import get_config, update_config
from agents import MomentumAgent, ReversionAgent, EnsembleAgent, NiftyShortAgent
from execution import BacktestEngine
from metrics import MetricsCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def run_simple_backtest():
    """Run a simple backtest with default parameters."""
    logger.info("Starting simple backtest...")

    # Configuration
    config = get_config()
    initial_capital = config.capital.initial_capital
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    symbols = config.universe.initial_universe[:5]  # Use first 5 symbols for quick test

    # Create agents
    momentum_agent = MomentumAgent(
        lookback_period=config.agents.momentum_lookback_period,
        exit_period=config.agents.momentum_exit_period,
        atr_period=config.agents.momentum_atr_period
    )

    reversion_agent = ReversionAgent(
        bb_period=config.agents.bb_period,
        bb_std_dev=config.agents.bb_std_dev,
        rsi_period=config.agents.rsi_period,
        rsi_oversold=config.agents.rsi_oversold,
        rsi_overbought=config.agents.rsi_overbought
    )

    # Create ensemble
    ensemble = EnsembleAgent(
        agents=[momentum_agent, reversion_agent],
        weights={
            momentum_agent.name: config.agents.momentum_weight,
            reversion_agent.name: config.agents.reversion_weight
        }
    )

    # Run backtest
    engine = BacktestEngine(
        initial_capital=initial_capital,
        agents=[ensemble],
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        enable_regime_filter=True
    )

    metrics = engine.run()

    # Print detailed metrics
    metrics_calc = MetricsCalculator()
    metrics_calc.print_metrics(metrics)

    return metrics, engine


def run_mvp_backtest():
    """Run MVP backtest as specified in requirements."""
    logger.info("=" * 70)
    logger.info("RUNNING MVP BACKTEST")
    logger.info("2 Agents (Momentum + Reversion) + Regime Filter")
    logger.info("=" * 70)

    config = get_config()

    # MVP Configuration
    initial_capital = 100_000  # ₹1 lakh
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    symbols = [
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "HINDUNILVR",
        "ITC",
        "SBIN",
        "BHARTIARTL",
        "KOTAKBANK"
    ]

    # Create individual agents (not ensemble for MVP)
    momentum_agent = MomentumAgent()
    reversion_agent = ReversionAgent()

    # Run backtest
    engine = BacktestEngine(
        initial_capital=initial_capital,
        agents=[momentum_agent, reversion_agent],  # Both agents run independently
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        enable_regime_filter=True  # Basic regime filter enabled
    )

    metrics = engine.run()

    # Print results
    metrics_calc = MetricsCalculator()
    metrics_calc.print_metrics(metrics)

    # Check if meets targets
    logger.info("\n" + "=" * 70)
    logger.info("TARGET VALIDATION")
    logger.info("=" * 70)

    target_return = config.target_monthly_return_pct * 3  # 3 months
    target_sharpe = config.target_sharpe_ratio
    target_max_dd = config.target_max_drawdown_pct

    logger.info(f"Target Return (3 months):  {target_return:.2f}% | Actual: {metrics.total_return_pct:.2f}% | {'✓ PASS' if metrics.total_return_pct >= target_return else '✗ FAIL'}")
    logger.info(f"Target Sharpe Ratio:       {target_sharpe:.2f} | Actual: {metrics.sharpe_ratio:.2f} | {'✓ PASS' if metrics.sharpe_ratio >= target_sharpe else '✗ FAIL'}")
    logger.info(f"Target Max Drawdown:       {target_max_dd:.2f}% | Actual: {abs(metrics.max_drawdown_pct):.2f}% | {'✓ PASS' if abs(metrics.max_drawdown_pct) <= target_max_dd else '✗ FAIL'}")

    logger.info("=" * 70)

    return metrics, engine


def run_parameter_optimization():
    """Run parameter optimization for agents."""
    logger.info("Parameter optimization not yet implemented")
    logger.info("This would test different combinations of:")
    logger.info("  - Momentum lookback periods (20, 55, 100)")
    logger.info("  - RSI thresholds (25/75, 30/70, 35/65)")
    logger.info("  - BB standard deviations (1.5, 2.0, 2.5)")
    logger.info("  - Risk management parameters")


def run_intraday_backtest():
    """Run intraday Nifty short strategy backtest."""
    logger.info("=" * 70)
    logger.info("RUNNING INTRADAY NIFTY SHORT BACKTEST")
    logger.info("=" * 70)

    from data.fetcher import DataFetcher
    from execution.intraday_engine import IntradayBacktestEngine
    from utils.sqlite_store import SQLiteStore

    config = get_config()
    fetcher = DataFetcher()

    print("\nFetching 15-min Nifty data (last 60 days)...")
    intraday_data = fetcher.fetch_intraday_data("NIFTY50", days_back=60, interval="15m")
    print(f"Loaded {len(intraday_data)} candles")

    agent = NiftyShortAgent(
        min_first_candle_range=75.0,
        entry_candle_index=3,
        swing_high_lookback=5,
        lot_size=config.intraday.nifty_lot_size
    )

    store = SQLiteStore()
    engine = IntradayBacktestEngine(agent=agent, store=store)
    result = engine.run(intraday_data, symbol="NIFTY50")

    print(f"\nResults: {result.total_trades} trades, "
          f"P&L={result.total_pnl_points:+.0f} pts (Rs.{result.total_pnl_rupees:+,.0f}), "
          f"Win rate={result.win_rate:.1f}%, Sharpe={result.sharpe_ratio:.2f}")
    print(f"Run ID: {result.run_id} | View in dashboard: streamlit run dashboard/app.py")

    return result


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("QUANTITATIVE TRADING SYSTEM FOR NSE")
    print("=" * 70)
    print("\nOptions:")
    print("1. Run Simple Backtest (quick test)")
    print("2. Run MVP Backtest (full 10 stocks, 1 year)")
    print("3. Run Parameter Optimization")
    print("4. Exit")
    print("5. Run Intraday Backtest (Nifty short strategy)")

    choice = input("\nSelect option (1-5): ").strip()

    if choice == "1":
        metrics, engine = run_simple_backtest()
    elif choice == "2":
        metrics, engine = run_mvp_backtest()
    elif choice == "3":
        run_parameter_optimization()
    elif choice == "4":
        print("Exiting...")
        return
    elif choice == "5":
        run_intraday_backtest()
        return
    else:
        print("Invalid choice")
        return

    # Optional: Save results
    save = input("\nSave results to CSV? (y/n): ").strip().lower()
    if save == 'y':
        if hasattr(engine, 'portfolio'):
            trades_df = engine.portfolio.get_trades_dataframe()
            if len(trades_df) > 0:
                filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                trades_df.to_csv(filename)
                print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
