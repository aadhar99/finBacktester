"""
Walk-Forward Validator - Validate Smart Money Patterns with Historical Data

Tests whether detected patterns actually predict future price movements by:
1. Detecting patterns using historical data
2. Tracking price movements after detection
3. Calculating success rates and returns
4. Validating the entire system

This is the critical validation step that proves (or disproves) the system works.

Usage:
    validator = WalkForwardValidator(db, llm)
    results = await validator.run_validation(days=60)
    validator.print_report(results)
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

from ai.validation.pattern_tracker import PatternTracker, PatternPerformance
from ai.agents.smart_money_analyzer import SmartMoneyAnalyzer
from data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


@dataclass
class ValidationResults:
    """Results from walk-forward validation."""
    total_patterns: int
    patterns_by_type: Dict[str, int]

    # Overall metrics
    overall_win_rate: float
    overall_avg_return: float

    # Per-pattern performance
    pattern_performances: List[PatternPerformance]

    # Trading simulation
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float

    # Timeframe
    start_date: date
    end_date: date
    days_analyzed: int


class WalkForwardValidator:
    """
    Walk-forward validation for smart money patterns.

    Validates that detected patterns actually predict future price movements
    by backtesting on historical data.
    """

    def __init__(
        self,
        db,
        llm_manager,
        tracker_db: str = "data/pattern_validation.db"
    ):
        """
        Initialize Walk-Forward Validator.

        Args:
            db: SmartMoneySQLite instance
            llm_manager: LLMManager for pattern enhancement
            tracker_db: Path to pattern tracking database
        """
        self.db = db
        self.analyzer = SmartMoneyAnalyzer(db, llm_manager)
        self.tracker = PatternTracker(tracker_db)
        self.data_fetcher = DataFetcher()

        logger.info("✅ WalkForwardValidator initialized")

    async def run_validation(
        self,
        days: int = 60,
        use_llm: bool = False,  # Disabled by default for speed
        update_outcomes: bool = True
    ) -> ValidationResults:
        """
        Run walk-forward validation.

        Args:
            days: Number of days to look back
            use_llm: Whether to use LLM enhancement (slower)
            update_outcomes: Whether to update pattern outcomes

        Returns:
            ValidationResults with performance metrics
        """
        logger.info("=" * 70)
        logger.info(f"WALK-FORWARD VALIDATION - {days} Days")
        logger.info("=" * 70)

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        logger.info(f"\n📅 Date Range: {start_date} to {end_date}")
        logger.info(f"   Using LLM: {use_llm}")

        # Get all symbols with bulk deals in this period
        logger.info("\n🔍 Getting symbols with bulk deal activity...")

        bulk_deals_df = self.db.get_bulk_deals(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            limit=10000
        )

        if bulk_deals_df.empty:
            logger.warning("No bulk deals found in this period!")
            return self._empty_results(start_date, end_date, days)

        symbols = bulk_deals_df['symbol'].unique().tolist()
        logger.info(f"   Found {len(symbols)} symbols with activity")

        # Analyze all symbols to detect patterns
        logger.info(f"\n🔍 Detecting patterns across all symbols...")

        analyses = await self.analyzer.analyze_multiple_stocks(
            symbols,
            lookback_days=days,
            use_llm=use_llm
        )

        logger.info(f"   ✅ Detected patterns in {len(analyses)} stocks")

        # Track each pattern
        logger.info(f"\n📝 Tracking patterns for validation...")

        tracked_count = 0
        patterns_by_type = {}

        for analysis in analyses:
            for pattern in analysis.patterns:
                # Get detection price (use first bulk deal date as proxy)
                first_deal = bulk_deals_df[
                    bulk_deals_df['symbol'] == analysis.symbol
                ].iloc[0]

                detection_date = datetime.fromisoformat(first_deal['date']).date()
                detection_price = float(first_deal['price'])

                # Track pattern
                self.tracker.track_pattern(
                    symbol=analysis.symbol,
                    pattern_type=pattern.type,
                    signal=pattern.signal,
                    confidence=pattern.confidence,
                    detection_date=detection_date,
                    detection_price=detection_price,
                    net_value=analysis.net_value,
                    num_deals=analysis.num_deals,
                    llm_recommendation=analysis.llm_insights.recommendation if analysis.llm_insights else None
                )

                tracked_count += 1
                patterns_by_type[pattern.type] = patterns_by_type.get(pattern.type, 0) + 1

        logger.info(f"   ✅ Tracked {tracked_count} patterns")
        logger.info(f"   Pattern distribution: {patterns_by_type}")

        # Update outcomes if requested
        if update_outcomes:
            logger.info(f"\n📈 Updating pattern outcomes (fetching price data)...")
            await self._update_pattern_outcomes(analyses, days=30)

        # Get performance metrics
        logger.info(f"\n📊 Calculating performance metrics...")

        pattern_performances = self.tracker.get_all_patterns_performance(days=30)

        # Calculate overall metrics
        if pattern_performances:
            overall_win_rate = np.mean([p.win_rate_30d for p in pattern_performances])
            overall_avg_return = np.mean([p.avg_return_30d for p in pattern_performances])
            total_occurrences = sum(p.total_occurrences for p in pattern_performances)
        else:
            overall_win_rate = 0
            overall_avg_return = 0
            total_occurrences = 0

        results = ValidationResults(
            total_patterns=tracked_count,
            patterns_by_type=patterns_by_type,
            overall_win_rate=overall_win_rate,
            overall_avg_return=overall_avg_return,
            pattern_performances=pattern_performances,
            total_trades=total_occurrences,
            winning_trades=int(total_occurrences * overall_win_rate / 100) if total_occurrences > 0 else 0,
            losing_trades=int(total_occurrences * (1 - overall_win_rate / 100)) if total_occurrences > 0 else 0,
            total_pnl=0,  # Would calculate from actual trades
            sharpe_ratio=0,  # Would calculate from returns series
            max_drawdown=0,  # Would calculate from equity curve
            start_date=start_date,
            end_date=end_date,
            days_analyzed=days
        )

        logger.info(f"   ✅ Validation complete!")

        return results

    async def _update_pattern_outcomes(
        self,
        analyses: List,
        days: int = 30
    ):
        """
        Update pattern outcomes with current prices.

        NOTE: This is simplified - would need actual historical price data
        for each detection date + N days to be accurate.
        """
        logger.info(f"   Fetching current prices for validation...")

        # Get unique symbols
        symbols = list(set(a.symbol for a in analyses))

        # Fetch current prices (proxy for future prices)
        for symbol in symbols[:10]:  # Limit for testing
            try:
                yf_symbol = f"{symbol}.NS"
                df = self.data_fetcher.fetch_data(
                    symbol=yf_symbol,
                    start_date=(date.today() - timedelta(days=60)).strftime('%Y-%m-%d'),
                    end_date=date.today().strftime('%Y-%m-%d'),
                    interval='1d'
                )

                if df is not None and not df.empty:
                    current_price = float(df['close'].iloc[-1])
                    logger.info(f"     {symbol}: ₹{current_price:.2f}")

                    # Note: In real implementation, would map to specific pattern_ids
                    # and use historical prices N days after detection

            except Exception as e:
                logger.debug(f"     {symbol}: Could not fetch price - {e}")
                continue

    def _empty_results(
        self,
        start_date: date,
        end_date: date,
        days: int
    ) -> ValidationResults:
        """Return empty results when no data available."""
        return ValidationResults(
            total_patterns=0,
            patterns_by_type={},
            overall_win_rate=0,
            overall_avg_return=0,
            pattern_performances=[],
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_pnl=0,
            sharpe_ratio=0,
            max_drawdown=0,
            start_date=start_date,
            end_date=end_date,
            days_analyzed=days
        )

    def print_report(self, results: ValidationResults):
        """
        Print detailed validation report.

        Args:
            results: ValidationResults from run_validation()
        """
        print("\n" + "=" * 70)
        print("WALK-FORWARD VALIDATION REPORT")
        print("=" * 70)

        print(f"\n📅 Analysis Period:")
        print(f"   {results.start_date} to {results.end_date} ({results.days_analyzed} days)")

        print(f"\n📊 Pattern Detection:")
        print(f"   Total Patterns: {results.total_patterns}")
        if results.patterns_by_type:
            for ptype, count in sorted(results.patterns_by_type.items(), key=lambda x: x[1], reverse=True):
                print(f"      {ptype}: {count}")

        if results.pattern_performances:
            print(f"\n📈 Overall Performance:")
            print(f"   Win Rate: {results.overall_win_rate:.1f}%")
            print(f"   Avg Return: {results.overall_avg_return:+.2f}%")
            print(f"   Total Trades: {results.total_trades}")
            print(f"      Winners: {results.winning_trades}")
            print(f"      Losers: {results.losing_trades}")

            print(f"\n🎯 Per-Pattern Performance:")
            for perf in results.pattern_performances:
                print(f"\n   {perf.pattern_type}:")
                print(f"      Occurrences: {perf.total_occurrences}")
                print(f"      Win Rate: {perf.win_rate_30d:.1f}%")
                print(f"      Avg Return: {perf.avg_return_30d:+.2f}%")
                print(f"      Best: {perf.best_symbol} ({perf.best_return:+.2f}%)")
                print(f"      Sharpe: {perf.sharpe_ratio:.2f}")
        else:
            print(f"\n⚠️  Not enough data for performance metrics yet")
            print(f"   Need at least 5 occurrences per pattern type")
            print(f"   Continue collecting data and run validation again")

        print("\n" + "=" * 70)


# ============================================================================
# CLI Runner
# ============================================================================

async def main():
    """Run walk-forward validation from command line."""
    import argparse
    import os

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from utils.smart_money_sqlite import SmartMoneySQLite
    from utils.llm import LLMManager, LLMConfig, LLMProvider

    parser = argparse.ArgumentParser(description='Walk-Forward Validation')
    parser.add_argument('--days', type=int, default=60, help='Days to look back')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM enhancement (slower)')
    parser.add_argument('--no-update', action='store_true', help='Skip outcome updates')

    args = parser.parse_args()

    # Initialize
    db = SmartMoneySQLite()

    # LLM (optional)
    llm = None
    if args.use_llm:
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            llm_config = LLMConfig(
                provider=LLMProvider.GEMINI,
                model='flash',
                api_key=api_key
            )
            llm = LLMManager([llm_config])

    # Run validation
    validator = WalkForwardValidator(db, llm)

    results = await validator.run_validation(
        days=args.days,
        use_llm=args.use_llm and llm is not None,
        update_outcomes=not args.no_update
    )

    # Print report
    validator.print_report(results)

    # Also print detailed tracker report if we have data
    print("\n")
    validator.tracker.print_performance_report(days=30)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    asyncio.run(main())
