"""
Smart Money Analyzer - Complete Stock Analysis Orchestrator

Combines pattern detection + AI enhancement for comprehensive analysis:
1. Fetch bulk deals from database
2. Detect patterns (rule-based)
3. Enhance with LLM (AI-powered insights)
4. Return complete analysis with signals and recommendations

Single API: analyze_stock('BIOPOL') → StockAnalysis
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any

from ai.agents.pattern_detector import PatternDetector, Pattern, BulkDealRecord
from ai.agents.llm_pattern_enhancer import LLMPatternEnhancer, LLMInsight

logger = logging.getLogger(__name__)


@dataclass
class StockAnalysis:
    """Complete analysis result for a stock."""
    symbol: str
    signal: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float  # 0-100 (aggregated from all patterns)
    patterns: List[Pattern]
    llm_insights: Optional[LLMInsight]
    analyzed_at: datetime = field(default_factory=datetime.now)

    # Summary stats
    num_deals: int = 0
    net_value: float = 0
    buy_sell_ratio: float = 0

    def __str__(self):
        return f"{self.symbol} [{self.signal}] {self.confidence:.0f}%"


class SmartMoneyAnalyzer:
    """
    Complete stock analysis orchestrator.

    Combines:
    - Database queries (bulk deals)
    - Pattern detection (rules)
    - LLM enhancement (AI insights)

    Into single comprehensive analysis.
    """

    def __init__(self, db, llm_manager):
        """
        Initialize Smart Money Analyzer.

        Args:
            db: SmartMoneySQLite instance
            llm_manager: LLMManager instance (for AI enhancement)
        """
        self.db = db
        self.pattern_detector = PatternDetector()
        self.llm_enhancer = LLMPatternEnhancer(llm_manager)

        logger.info("✅ SmartMoneyAnalyzer initialized")

    async def analyze_stock(
        self,
        symbol: str,
        lookback_days: int = 30,
        use_llm: bool = True
    ) -> Optional[StockAnalysis]:
        """
        Analyze a stock for smart money patterns.

        Args:
            symbol: Stock symbol (e.g., 'BIOPOL')
            lookback_days: Days to look back for bulk deals (default 30)
            use_llm: Whether to enhance with LLM insights (default True)

        Returns:
            StockAnalysis with detected patterns and recommendations,
            or None if no bulk deals found
        """
        logger.info(f"🔍 Analyzing {symbol}...")

        # 1. Fetch bulk deals from database
        start_date = (date.today() - timedelta(days=lookback_days)).isoformat()
        end_date = date.today().isoformat()

        bulk_deals_df = self.db.get_bulk_deals(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )

        if bulk_deals_df.empty:
            logger.info(f"  ⏭️  No bulk deals found for {symbol}")
            return None

        # Convert to BulkDealRecord objects
        deals = []
        for _, row in bulk_deals_df.iterrows():
            deals.append(BulkDealRecord(
                date=date.fromisoformat(row['date']),
                symbol=row['symbol'],
                client_name=row['client_name'],
                deal_type=row['deal_type'],
                quantity=int(row['quantity']),
                price=float(row['price']),
                value=float(row['value'])
            ))

        logger.info(f"  📊 Found {len(deals)} bulk deals")

        # 2. Detect patterns (rule-based)
        patterns = self.pattern_detector.detect_all_patterns(deals)

        if not patterns:
            logger.info(f"  ⏭️  No patterns detected for {symbol}")
            return None

        logger.info(f"  ✅ Detected {len(patterns)} pattern(s)")

        # 3. Enhance with LLM (if enabled)
        llm_insight = None
        if use_llm and patterns:
            # Use highest confidence pattern for LLM enhancement
            primary_pattern = max(patterns, key=lambda p: p.confidence)

            try:
                llm_insight = await self.llm_enhancer.enhance_pattern(
                    symbol,
                    primary_pattern
                )
                logger.info(f"  🤖 LLM recommendation: {llm_insight.recommendation}")
            except Exception as e:
                logger.warning(f"  ⚠️  LLM enhancement failed: {e}")

        # 4. Aggregate signal and confidence
        signal, confidence = self._aggregate_signals(patterns, llm_insight)

        # 5. Calculate summary stats
        buy_deals = [d for d in deals if d.deal_type == 'BUY']
        sell_deals = [d for d in deals if d.deal_type == 'SELL']
        buy_value = sum(d.value for d in buy_deals)
        sell_value = sum(d.value for d in sell_deals)

        analysis = StockAnalysis(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            patterns=patterns,
            llm_insights=llm_insight,
            num_deals=len(deals),
            net_value=buy_value - sell_value,
            buy_sell_ratio=buy_value / max(sell_value, 1)
        )

        logger.info(f"  ✅ Analysis complete: {analysis}")

        return analysis

    def _aggregate_signals(
        self,
        patterns: List[Pattern],
        llm_insight: Optional[LLMInsight]
    ) -> tuple[str, float]:
        """
        Aggregate multiple patterns into single signal and confidence.

        Rules:
        1. If LLM provided, use its recommendation
        2. Otherwise, use majority vote of patterns
        3. Confidence is weighted average
        """
        if not patterns:
            return 'NEUTRAL', 0

        # Count signals
        bullish_count = sum(1 for p in patterns if p.signal == 'BULLISH')
        bearish_count = sum(1 for p in patterns if p.signal == 'BEARISH')

        # LLM recommendation overrides if available
        if llm_insight and llm_insight.recommendation in ['BUY', 'AVOID']:
            final_signal = 'BULLISH' if llm_insight.recommendation == 'BUY' else 'BEARISH'

            # Blend pattern confidence with LLM confidence
            pattern_conf = sum(p.confidence for p in patterns) / len(patterns)
            final_confidence = (pattern_conf * 0.6 + llm_insight.target_confidence * 0.4)

        else:
            # Majority vote
            if bullish_count > bearish_count:
                final_signal = 'BULLISH'
            elif bearish_count > bullish_count:
                final_signal = 'BEARISH'
            else:
                final_signal = 'NEUTRAL'

            # Weighted average confidence
            final_confidence = sum(p.confidence for p in patterns) / len(patterns)

        return final_signal, final_confidence

    async def analyze_multiple_stocks(
        self,
        symbols: List[str],
        lookback_days: int = 30,
        use_llm: bool = True
    ) -> List[StockAnalysis]:
        """
        Analyze multiple stocks in parallel.

        Args:
            symbols: List of stock symbols
            lookback_days: Days to look back
            use_llm: Whether to use LLM enhancement

        Returns:
            List of StockAnalysis (sorted by confidence descending)
        """
        logger.info(f"🔍 Analyzing {len(symbols)} stocks...")

        # Analyze in parallel
        tasks = [
            self.analyze_stock(symbol, lookback_days, use_llm)
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks)

        # Filter out None results and sort by confidence
        analyses = [r for r in results if r is not None]
        analyses.sort(key=lambda a: a.confidence, reverse=True)

        logger.info(f"  ✅ {len(analyses)} stocks with patterns detected")

        return analyses

    def print_analysis(self, analysis: StockAnalysis):
        """Pretty print a stock analysis."""
        print("\n" + "=" * 70)
        print(f"SMART MONEY ANALYSIS: {analysis.symbol}")
        print("=" * 70)

        print(f"\n📊 Signal: {analysis.signal} ({analysis.confidence:.0f}% confidence)")

        print(f"\n💰 Bulk Deal Summary:")
        print(f"   Total deals: {analysis.num_deals}")
        print(f"   Net value: ₹{analysis.net_value:,.0f}")
        print(f"   Buy/Sell ratio: {analysis.buy_sell_ratio:.2f}:1")

        print(f"\n🔍 Patterns Detected ({len(analysis.patterns)}):")
        for i, pattern in enumerate(analysis.patterns, 1):
            print(f"\n   {i}. {pattern.type} [{pattern.signal}] {pattern.confidence:.0f}%")
            if 'net_value' in pattern.evidence:
                print(f"      Net value: ₹{pattern.evidence['net_value']:,.0f}")
            if 'buy_sell_ratio' in pattern.evidence:
                print(f"      Buy/Sell: {pattern.evidence['buy_sell_ratio']:.2f}:1")
            if 'unique_buyers' in pattern.evidence:
                print(f"      Buyers: {pattern.evidence['unique_buyers']}")

        if analysis.llm_insights:
            insight = analysis.llm_insights
            print(f"\n🤖 AI-Powered Insights:")
            print(f"\n   💡 Reasoning:")
            print(f"      {insight.reasoning}")

            if insight.significance:
                print(f"\n   ⭐ Significance:")
                print(f"      {insight.significance}")

            if insight.risk_factors:
                print(f"\n   ⚠️  Risk Factors:")
                for i, risk in enumerate(insight.risk_factors, 1):
                    print(f"      {i}. {risk}")

            print(f"\n   📈 Recommendation: {insight.recommendation}")
            print(f"      LLM Confidence: {insight.target_confidence:.0f}%")

            if insight.action_items:
                print(f"\n   ✅ Action Items:")
                for i, action in enumerate(insight.action_items, 1):
                    print(f"      {i}. {action}")

        print("\n" + "=" * 70)


# ============================================================================
# Testing with Real Data
# ============================================================================

async def test_analyzer():
    """Test SmartMoneyAnalyzer with real BIOPOL data."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from utils.smart_money_sqlite import SmartMoneySQLite
    from utils.llm import LLMManager, LLMConfig, LLMProvider
    import os

    print("=" * 70)
    print("SMART MONEY ANALYZER - Testing with Real Data")
    print("=" * 70)

    # Initialize database
    db = SmartMoneySQLite()

    # Check for Gemini API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n❌ GEMINI_API_KEY not found in .env")
        print("   Continuing without LLM enhancement...")
        llm = None
        use_llm = False
    else:
        # Initialize LLM
        llm_config = LLMConfig(
            provider=LLMProvider.GEMINI,
            model='flash',
            api_key=api_key
        )
        llm = LLMManager([llm_config])
        use_llm = True

    # Initialize analyzer
    analyzer = SmartMoneyAnalyzer(db, llm)

    # Get stocks with activity
    bulk_deals_df = db.get_bulk_deals(limit=1000)
    symbols = bulk_deals_df['symbol'].unique().tolist()

    print(f"\n📊 Found {len(symbols)} symbols with bulk deals")
    print(f"   Symbols: {', '.join(symbols[:10])}")

    # Analyze top 3 stocks
    print(f"\n🔍 Analyzing top 3 stocks...\n")

    for symbol in symbols[:3]:
        analysis = await analyzer.analyze_stock(symbol, lookback_days=30, use_llm=use_llm)

        if analysis:
            analyzer.print_analysis(analysis)

    print("\n✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    asyncio.run(test_analyzer())
