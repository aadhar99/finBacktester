"""
Smart Money Tracker - Daily Scanner for Top Opportunities

Scans all NSE stocks with recent bulk deal activity and identifies
top trading opportunities based on smart money patterns.

Features:
- Daily scanning of all active stocks
- Pattern detection + AI enhancement
- Top 10 opportunities ranked by confidence
- Optional alerts (Telegram/Email)
- Caching for performance

Usage:
    tracker = SmartMoneyTracker(db, llm)
    opportunities = await tracker.get_daily_signals()
    # Returns top opportunities sorted by confidence
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
import json

from ai.agents.smart_money_analyzer import SmartMoneyAnalyzer, StockAnalysis

logger = logging.getLogger(__name__)


@dataclass
class DailyReport:
    """Daily smart money opportunities report."""
    date: date
    opportunities: List[StockAnalysis]
    total_stocks_scanned: int
    patterns_detected: int
    generated_at: datetime = field(default_factory=datetime.now)

    def get_top_opportunities(self, top_n: int = 10) -> List[StockAnalysis]:
        """Get top N opportunities by confidence."""
        return self.opportunities[:top_n]

    def get_bullish_signals(self) -> List[StockAnalysis]:
        """Get only bullish signals."""
        return [opp for opp in self.opportunities if opp.signal == 'BULLISH']

    def get_bearish_signals(self) -> List[StockAnalysis]:
        """Get only bearish signals."""
        return [opp for opp in self.opportunities if opp.signal == 'BEARISH']


class SmartMoneyTracker:
    """
    Daily scanner for smart money opportunities.

    Scans all stocks with recent bulk deal activity and identifies
    the best trading opportunities based on institutional patterns.
    """

    def __init__(self, db, llm_manager, cache_hours: int = 24):
        """
        Initialize Smart Money Tracker.

        Args:
            db: SmartMoneySQLite instance
            llm_manager: LLMManager instance
            cache_hours: Hours to cache daily reports (default 24)
        """
        self.db = db
        self.analyzer = SmartMoneyAnalyzer(db, llm_manager)
        self.cache_hours = cache_hours
        self._cache: Dict[str, DailyReport] = {}

        logger.info("✅ SmartMoneyTracker initialized")

    async def get_daily_signals(
        self,
        lookback_days: int = 7,
        min_confidence: float = 70.0,
        use_llm: bool = True,
        use_cache: bool = True
    ) -> DailyReport:
        """
        Get today's smart money signals across all active stocks.

        Args:
            lookback_days: Days to look back for bulk deals (default 7)
            min_confidence: Minimum confidence threshold (default 70%)
            use_llm: Whether to use LLM enhancement (default True)
            use_cache: Whether to use cached results (default True)

        Returns:
            DailyReport with top opportunities sorted by confidence
        """
        cache_key = f"{date.today()}_{lookback_days}_{use_llm}"

        # Check cache
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            age_hours = (datetime.now() - cached.generated_at).total_seconds() / 3600

            if age_hours < self.cache_hours:
                logger.info(f"  📦 Using cached report ({age_hours:.1f}h old)")
                return cached

        logger.info("🔍 Scanning for smart money opportunities...")

        # 1. Get all symbols with recent bulk deals
        start_date = (date.today() - timedelta(days=lookback_days)).isoformat()
        end_date = date.today().isoformat()

        bulk_deals_df = self.db.get_bulk_deals(
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )

        if bulk_deals_df.empty:
            logger.warning("  ⚠️  No bulk deals found in database")
            return DailyReport(
                date=date.today(),
                opportunities=[],
                total_stocks_scanned=0,
                patterns_detected=0
            )

        symbols = bulk_deals_df['symbol'].unique().tolist()
        logger.info(f"  📊 Found {len(symbols)} stocks with activity")

        # 2. Analyze all symbols
        analyses = await self.analyzer.analyze_multiple_stocks(
            symbols,
            lookback_days=lookback_days,
            use_llm=use_llm
        )

        # 3. Filter by minimum confidence
        opportunities = [a for a in analyses if a.confidence >= min_confidence]

        logger.info(f"  ✅ {len(opportunities)} opportunities found (>{min_confidence}% confidence)")

        # 4. Create report
        report = DailyReport(
            date=date.today(),
            opportunities=opportunities,
            total_stocks_scanned=len(symbols),
            patterns_detected=len(analyses)
        )

        # Cache it
        self._cache[cache_key] = report

        return report

    async def get_top_opportunities(
        self,
        top_n: int = 10,
        lookback_days: int = 7,
        use_llm: bool = True
    ) -> List[StockAnalysis]:
        """
        Get top N opportunities.

        Args:
            top_n: Number of top opportunities to return (default 10)
            lookback_days: Days to look back (default 7)
            use_llm: Whether to use LLM enhancement (default True)

        Returns:
            List of top StockAnalysis sorted by confidence
        """
        report = await self.get_daily_signals(
            lookback_days=lookback_days,
            use_llm=use_llm
        )

        return report.get_top_opportunities(top_n)

    def print_daily_report(self, report: DailyReport, top_n: int = 10):
        """
        Pretty print daily opportunities report.

        Args:
            report: DailyReport to print
            top_n: Number of top opportunities to show (default 10)
        """
        print("\n" + "=" * 70)
        print(f"📊 SMART MONEY DAILY REPORT - {report.date}")
        print("=" * 70)

        print(f"\n📈 Scan Summary:")
        print(f"   Stocks scanned: {report.total_stocks_scanned}")
        print(f"   Patterns detected: {report.patterns_detected}")
        print(f"   Opportunities: {len(report.opportunities)}")

        bullish = report.get_bullish_signals()
        bearish = report.get_bearish_signals()

        print(f"\n   Bullish signals: {len(bullish)}")
        print(f"   Bearish signals: {len(bearish)}")

        # Top opportunities
        top = report.get_top_opportunities(top_n)

        if not top:
            print("\n⚠️  No opportunities found above threshold")
            print("=" * 70)
            return

        print(f"\n🎯 Top {min(top_n, len(top))} Opportunities:\n")

        for i, opp in enumerate(top, 1):
            print(f"{i}. {opp.symbol:12s} [{opp.signal:7s}] {opp.confidence:.0f}% confidence")

            # Show patterns
            if opp.patterns:
                pattern = opp.patterns[0]  # Show first pattern
                print(f"   Pattern: {pattern.type}")

                if 'net_value' in pattern.evidence:
                    print(f"   Net value: ₹{pattern.evidence['net_value']:,.0f}")

                if 'buy_sell_ratio' in pattern.evidence:
                    ratio = pattern.evidence['buy_sell_ratio']
                    print(f"   Buy/Sell: {ratio:.2f}:1")

            # Show LLM recommendation if available
            if opp.llm_insights:
                print(f"   AI: {opp.llm_insights.recommendation} - {opp.llm_insights.reasoning[:60]}...")

            print()

        print("=" * 70)

        # Show detailed analysis for top opportunity
        if top:
            print(f"\n💡 Detailed Analysis of Top Opportunity:\n")
            self.analyzer.print_analysis(top[0])

    async def send_alerts(
        self,
        report: DailyReport,
        top_n: int = 5,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None
    ):
        """
        Send alerts for top opportunities.

        Args:
            report: DailyReport to send
            top_n: Number of top opportunities to include (default 5)
            telegram_token: Telegram bot token (optional)
            telegram_chat_id: Telegram chat ID (optional)
        """
        top = report.get_top_opportunities(top_n)

        if not top:
            logger.info("  ℹ️  No opportunities to alert")
            return

        # Format message
        message = self._format_alert_message(report, top)

        # Send via Telegram if configured
        if telegram_token and telegram_chat_id:
            try:
                await self._send_telegram(message, telegram_token, telegram_chat_id)
                logger.info(f"  ✅ Telegram alert sent ({len(top)} opportunities)")
            except Exception as e:
                logger.error(f"  ❌ Telegram alert failed: {e}")

        # For now, just print to console
        print("\n" + "=" * 70)
        print("📢 ALERT: Smart Money Opportunities Detected")
        print("=" * 70)
        print(message)
        print("=" * 70)

    def _format_alert_message(self, report: DailyReport, opportunities: List[StockAnalysis]) -> str:
        """Format alert message for Telegram/Email."""
        lines = [
            f"🔔 Smart Money Alert - {report.date}",
            f"",
            f"📊 {len(opportunities)} Top Opportunities:",
            f""
        ]

        for i, opp in enumerate(opportunities, 1):
            lines.append(f"{i}. {opp.symbol} [{opp.signal}] {opp.confidence:.0f}%")

            if opp.patterns:
                pattern = opp.patterns[0]
                lines.append(f"   {pattern.type}")

                if 'net_value' in pattern.evidence:
                    value_cr = pattern.evidence['net_value'] / 10000000
                    lines.append(f"   Net: ₹{value_cr:.1f} Cr")

            if opp.llm_insights:
                rec = opp.llm_insights.recommendation
                lines.append(f"   AI: {rec}")

            lines.append("")

        lines.append(f"Scan: {report.total_stocks_scanned} stocks, {report.patterns_detected} patterns")
        lines.append(f"")
        lines.append(f"🤖 Generated by Smart Money Tracker")

        return "\n".join(lines)

    async def _send_telegram(self, message: str, token: str, chat_id: str):
        """Send message via Telegram bot."""
        import aiohttp

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    raise Exception(f"Telegram API error: {response.status}")

    def save_report(self, report: DailyReport, filename: Optional[str] = None):
        """
        Save daily report to JSON file.

        Args:
            report: DailyReport to save
            filename: Output filename (default: smart_money_report_YYYY-MM-DD.json)
        """
        if filename is None:
            filename = f"data/smart_money_report_{report.date}.json"

        # Convert to dict
        report_dict = {
            'date': str(report.date),
            'generated_at': report.generated_at.isoformat(),
            'total_stocks_scanned': report.total_stocks_scanned,
            'patterns_detected': report.patterns_detected,
            'opportunities': [
                {
                    'symbol': opp.symbol,
                    'signal': opp.signal,
                    'confidence': opp.confidence,
                    'num_deals': opp.num_deals,
                    'net_value': opp.net_value,
                    'patterns': [
                        {
                            'type': p.type,
                            'signal': p.signal,
                            'confidence': p.confidence
                        }
                        for p in opp.patterns
                    ],
                    'llm_recommendation': opp.llm_insights.recommendation if opp.llm_insights else None
                }
                for opp in report.opportunities
            ]
        }

        # Save to file
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"  💾 Report saved to {filename}")


# ============================================================================
# Daily Runner Script
# ============================================================================

async def run_daily_scan():
    """Run daily smart money scan and generate report."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from utils.smart_money_sqlite import SmartMoneySQLite
    from utils.llm import LLMManager, LLMConfig, LLMProvider
    import os

    print("=" * 70)
    print("🔍 SMART MONEY TRACKER - Daily Scan")
    print("=" * 70)

    # Initialize database
    db = SmartMoneySQLite()

    # Initialize LLM
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n⚠️  GEMINI_API_KEY not found - running without LLM enhancement")
        llm = None
        use_llm = False
    else:
        llm_config = LLMConfig(
            provider=LLMProvider.GEMINI,
            model='flash',
            api_key=api_key
        )
        llm = LLMManager([llm_config])
        use_llm = True

    # Initialize tracker
    tracker = SmartMoneyTracker(db, llm)

    # Run daily scan
    print("\n🔍 Running daily scan...\n")

    report = await tracker.get_daily_signals(
        lookback_days=7,
        min_confidence=70.0,
        use_llm=use_llm
    )

    # Print report
    tracker.print_daily_report(report, top_n=10)

    # Save report
    tracker.save_report(report)

    # Send alerts (console only for now)
    await tracker.send_alerts(report, top_n=5)

    print("\n✅ Daily scan complete!")
    print("=" * 70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    asyncio.run(run_daily_scan())
