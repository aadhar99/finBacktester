#!/usr/bin/env python3
"""
CLI entry point for intraday Nifty short strategy backtest.

Usage:
    python scripts/run_intraday.py                        # Defaults: 60 days yfinance
    python scripts/run_intraday.py --csv data/nifty.csv   # Historical CSV
    python scripts/run_intraday.py --min-range 0          # Disable 75-pt filter
    python scripts/run_intraday.py --dashboard             # Launch Streamlit after
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.nifty_short_agent import NiftyShortAgent
from execution.intraday_engine import IntradayBacktestEngine
from data.fetcher import DataFetcher
from utils.sqlite_store import SQLiteStore
from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('intraday_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_backtest(args):
    """Run the intraday backtest with given arguments."""
    config = get_config()
    fetcher = DataFetcher()

    # Load data
    if args.csv:
        logger.info(f"Loading intraday data from CSV: {args.csv}")
        intraday_data = fetcher.load_intraday_csv(args.csv)
    else:
        logger.info(f"Fetching {args.days} days of {args.interval} data for {args.symbol}")
        intraday_data = fetcher.fetch_intraday_data(
            symbol=args.symbol,
            days_back=args.days,
            interval=args.interval
        )

    logger.info(f"Loaded {len(intraday_data)} candles")

    # Optionally load daily data
    daily_data = None
    if not args.csv:
        try:
            end_date = intraday_data.index.max().strftime("%Y-%m-%d")
            start_date = intraday_data.index.min().strftime("%Y-%m-%d")
            daily_data = fetcher.fetch_historical_data(args.symbol, start_date, end_date)
        except Exception as e:
            logger.warning(f"Could not fetch daily data, will derive from intraday: {e}")

    # Create agent
    agent = NiftyShortAgent(
        min_first_candle_range=args.min_range,
        entry_candle_index=args.entry_candle,
        swing_high_lookback=args.swing_lookback,
        lot_size=args.lot_size,
        entry_cutoff_time=args.entry_cutoff,
        stop_loss_points=args.stop_loss
    )

    # Create engine
    store = SQLiteStore(args.db)
    engine = IntradayBacktestEngine(
        agent=agent,
        store=store,
        initial_capital=args.capital
    )

    # Run
    result = engine.run(intraday_data, daily_data=daily_data, symbol=args.symbol)

    # Print summary
    print("\n" + "=" * 70)
    print("INTRADAY BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Strategy:          {agent.name}")
    print(f"  Symbol:            {args.symbol}")
    print(f"  Min Range Filter:  {args.min_range} pts")
    print(f"  Entry Candle:      #{args.entry_candle}")
    print(f"  Swing Lookback:    {args.swing_lookback} candles")
    print(f"  Lot Size:          {args.lot_size}")
    print(f"  Entry Cutoff:      {args.entry_cutoff}")
    print(f"  Stop Loss:         {args.stop_loss} pts")
    print()
    print(f"  Total Trades:      {result.total_trades}")
    print(f"  Winning Trades:    {result.winning_trades}")
    print(f"  Losing Trades:     {result.losing_trades}")
    print(f"  Win Rate:          {result.win_rate:.1f}%")
    print()
    print(f"  Total P&L:         {result.total_pnl_points:+.0f} pts (Rs.{result.total_pnl_rupees:+,.0f})")
    print(f"  Avg Win:           {result.avg_win_points:+.1f} pts")
    print(f"  Avg Loss:          {result.avg_loss_points:+.1f} pts")
    print(f"  Max Win:           {result.max_win_points:+.1f} pts")
    print(f"  Max Loss:          {result.max_loss_points:+.1f} pts")
    print(f"  Profit Factor:     {result.profit_factor:.2f}")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:      {result.max_drawdown_points:.0f} pts")
    print()
    print(f"  Run ID:            {result.run_id}")
    print(f"  Audit DB:          {args.db}")
    print("=" * 70)

    # Export trades CSV
    if result.trades:
        csv_path = f"trades_run_{result.run_id}.csv"
        store.export_trades_csv(result.run_id, csv_path)
        print(f"  Trades exported:   {csv_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Intraday Nifty Short Strategy Backtester"
    )
    parser.add_argument('--symbol', default='NIFTY50', help='Symbol to backtest (default: NIFTY50)')
    parser.add_argument('--csv', default=None, help='Path to historical CSV file')
    parser.add_argument('--days', type=int, default=60, help='Days of data to fetch (default: 60)')
    parser.add_argument('--interval', default='15m', help='Candle interval (default: 15m)')
    parser.add_argument('--min-range', type=float, default=75.0,
                        help='Min 1st candle range in points, 0 to disable (default: 75)')
    parser.add_argument('--entry-candle', type=int, default=3,
                        help='Which candle triggers entry, 1-indexed (default: 3)')
    parser.add_argument('--swing-lookback', type=int, default=5,
                        help='Candles to look back for swing high exit (default: 5)')
    parser.add_argument('--lot-size', type=int, default=25, help='Lot size (default: 25)')
    parser.add_argument('--entry-cutoff', default='14:00',
                        help='Entry cutoff time HH:MM, "00:00" to disable (default: 14:00)')
    parser.add_argument('--stop-loss', type=float, default=0,
                        help='Stop loss in points above entry, 0 to disable (default: 0)')
    parser.add_argument('--capital', type=float, default=100_000, help='Initial capital (default: 100000)')
    parser.add_argument('--db', default='data/backtest_audit.db', help='SQLite DB path')
    parser.add_argument('--dashboard', action='store_true', help='Launch Streamlit dashboard after backtest')

    args = parser.parse_args()

    result = run_backtest(args)

    if args.dashboard:
        print("\nLaunching dashboard...")
        dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


if __name__ == "__main__":
    main()
