#!/usr/bin/env python3
"""
Pre-download intraday data for all supported symbols and intervals.

Usage:
    python3 scripts/prefetch_data.py
    python3 scripts/prefetch_data.py --symbols NIFTY50,BANKNIFTY --intervals 5m,15m,30m,1h
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["NIFTY50", "BANKNIFTY"]
DEFAULT_INTERVALS = ["5m", "15m", "30m", "1h"]
DAYS_BACK = 60


def prefetch(symbols, intervals):
    fetcher = DataFetcher()
    results = []

    for symbol in symbols:
        # Intraday intervals
        for interval in intervals:
            logger.info(f"Fetching {symbol} {interval}...")
            try:
                df = fetcher.fetch_intraday_data(symbol, days_back=DAYS_BACK, interval=interval)
                first = str(df.index.min().date())
                last = str(df.index.max().date())
                results.append({
                    "symbol": symbol,
                    "interval": interval,
                    "candles": len(df),
                    "from": first,
                    "to": last,
                    "status": "OK",
                })
                logger.info(f"  {symbol} {interval}: {len(df)} candles, {first} to {last}")
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "interval": interval,
                    "candles": 0,
                    "from": "-",
                    "to": "-",
                    "status": f"FAIL: {e}",
                })
                logger.error(f"  {symbol} {interval}: FAILED - {e}")

        # Daily data
        logger.info(f"Fetching {symbol} daily...")
        try:
            from datetime import datetime, timedelta
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            df = fetcher.fetch_historical_data(symbol, start, end)
            first = str(df.index.min().date())
            last = str(df.index.max().date())
            results.append({
                "symbol": symbol,
                "interval": "1d",
                "candles": len(df),
                "from": first,
                "to": last,
                "status": "OK",
            })
            logger.info(f"  {symbol} daily: {len(df)} bars, {first} to {last}")
        except Exception as e:
            results.append({
                "symbol": symbol,
                "interval": "1d",
                "candles": 0,
                "from": "-",
                "to": "-",
                "status": f"FAIL: {e}",
            })
            logger.error(f"  {symbol} daily: FAILED - {e}")

    # Print summary
    print("\n" + "=" * 75)
    print("PREFETCH SUMMARY")
    print("=" * 75)
    print(f"{'Symbol':<12} {'Interval':<10} {'Candles':>8} {'From':<12} {'To':<12} {'Status'}")
    print("-" * 75)
    for r in results:
        print(f"{r['symbol']:<12} {r['interval']:<10} {r['candles']:>8} {r['from']:<12} {r['to']:<12} {r['status']}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Prefetch market data for all symbols and intervals")
    parser.add_argument('--symbols', default=','.join(DEFAULT_SYMBOLS),
                        help=f'Comma-separated symbols (default: {",".join(DEFAULT_SYMBOLS)})')
    parser.add_argument('--intervals', default=','.join(DEFAULT_INTERVALS),
                        help=f'Comma-separated intervals (default: {",".join(DEFAULT_INTERVALS)})')
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]
    intervals = [i.strip() for i in args.intervals.split(',')]

    print(f"Prefetching data for {symbols} with intervals {intervals}...")
    prefetch(symbols, intervals)


if __name__ == "__main__":
    main()
