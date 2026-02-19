"""
Pattern Detector - Rule-based Smart Money Pattern Detection

Detects institutional trading patterns from bulk deals and FII/DII flows:
- Sustained Accumulation (repeated buying)
- Distribution (smart money exiting)
- Unusual Activity (volume spikes)
- Clustered Buying (multiple buyers same day)
- FII Reversal (trend change)

Fast, deterministic detection before LLM enhancement.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Dict, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """A detected smart money pattern."""
    type: str  # SUSTAINED_ACCUMULATION, DISTRIBUTION, etc.
    signal: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float  # 0-100
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)

    def __str__(self):
        return f"{self.type} [{self.signal}] {self.confidence:.0f}%"


@dataclass
class BulkDealRecord:
    """Simplified bulk deal for pattern detection."""
    date: date
    symbol: str
    client_name: str
    deal_type: str  # BUY or SELL
    quantity: int
    price: float
    value: float


class PatternDetector:
    """
    Rule-based pattern detector for smart money signals.

    Uses deterministic logic to identify patterns, then LLM adds context.
    """

    # Pattern thresholds (can be tuned)
    MIN_ACCUMULATION_DEALS = 3
    MIN_ACCUMULATION_VALUE = 5_00_00_000  # ₹5 Cr
    MIN_ACCUMULATION_RATIO = 3.0  # Buy/sell ratio

    MIN_DISTRIBUTION_DEALS = 3
    MIN_DISTRIBUTION_VALUE = 5_00_00_000
    MIN_DISTRIBUTION_RATIO = 3.0

    MIN_CLUSTERED_BUYERS = 3
    MIN_CLUSTERED_VALUE = 10_00_00_000  # ₹10 Cr

    UNUSUAL_ACTIVITY_MULTIPLIER = 3.0  # 3x average

    def __init__(self):
        logger.info("✅ PatternDetector initialized")

    # ========================================================================
    # Pattern 1: Sustained Accumulation
    # ========================================================================

    def detect_accumulation(self, deals: List[BulkDealRecord]) -> Optional[Pattern]:
        """
        Detect sustained accumulation pattern.

        Criteria:
        - Multiple buy deals (3+)
        - High buy/sell ratio (3:1 or better)
        - Significant net buying (> ₹5 Cr)

        Returns:
            Pattern if detected, None otherwise
        """
        if not deals:
            return None

        buy_deals = [d for d in deals if d.deal_type == 'BUY']
        sell_deals = [d for d in deals if d.deal_type == 'SELL']

        # Must have minimum buy deals
        if len(buy_deals) < self.MIN_ACCUMULATION_DEALS:
            return None

        # Calculate values
        buy_value = sum(d.value for d in buy_deals)
        sell_value = sum(d.value for d in sell_deals)
        net_value = buy_value - sell_value

        # Must have significant net buying
        if net_value < self.MIN_ACCUMULATION_VALUE:
            return None

        # Calculate buy/sell ratio
        buy_sell_ratio = buy_value / max(sell_value, 1)

        # Must have strong buy bias
        if buy_sell_ratio < self.MIN_ACCUMULATION_RATIO:
            return None

        # Get unique buyers
        unique_buyers = set(d.client_name for d in buy_deals)

        # Calculate confidence (70-95%)
        # Higher ratio = higher confidence
        confidence = 70 + min(buy_sell_ratio * 5, 25)

        return Pattern(
            type="SUSTAINED_ACCUMULATION",
            signal="BULLISH",
            confidence=confidence,
            evidence={
                'buy_deals': len(buy_deals),
                'sell_deals': len(sell_deals),
                'buy_value': buy_value,
                'sell_value': sell_value,
                'net_value': net_value,
                'buy_sell_ratio': round(buy_sell_ratio, 2),
                'unique_buyers': len(unique_buyers),
                'top_buyers': self._get_top_buyers(buy_deals, top_n=3),
                'date_range': f"{min(d.date for d in deals)} to {max(d.date for d in deals)}"
            }
        )

    # ========================================================================
    # Pattern 2: Distribution (Smart Money Exiting)
    # ========================================================================

    def detect_distribution(self, deals: List[BulkDealRecord]) -> Optional[Pattern]:
        """
        Detect distribution pattern (smart money selling).

        Criteria:
        - Multiple sell deals (3+)
        - High sell/buy ratio (3:1 or better)
        - Significant net selling (> ₹5 Cr)

        Returns:
            Pattern if detected, None otherwise
        """
        if not deals:
            return None

        buy_deals = [d for d in deals if d.deal_type == 'BUY']
        sell_deals = [d for d in deals if d.deal_type == 'SELL']

        # Must have minimum sell deals
        if len(sell_deals) < self.MIN_DISTRIBUTION_DEALS:
            return None

        # Calculate values
        buy_value = sum(d.value for d in buy_deals)
        sell_value = sum(d.value for d in sell_deals)
        net_value = sell_value - buy_value

        # Must have significant net selling
        if net_value < self.MIN_DISTRIBUTION_VALUE:
            return None

        # Calculate sell/buy ratio
        sell_buy_ratio = sell_value / max(buy_value, 1)

        # Must have strong sell bias
        if sell_buy_ratio < self.MIN_DISTRIBUTION_RATIO:
            return None

        # Get unique sellers
        unique_sellers = set(d.client_name for d in sell_deals)

        # Calculate confidence
        confidence = 70 + min(sell_buy_ratio * 5, 25)

        return Pattern(
            type="DISTRIBUTION",
            signal="BEARISH",
            confidence=confidence,
            evidence={
                'buy_deals': len(buy_deals),
                'sell_deals': len(sell_deals),
                'buy_value': buy_value,
                'sell_value': sell_value,
                'net_value': -net_value,  # Negative for selling
                'sell_buy_ratio': round(sell_buy_ratio, 2),
                'unique_sellers': len(unique_sellers),
                'top_sellers': self._get_top_sellers(sell_deals, top_n=3),
                'date_range': f"{min(d.date for d in deals)} to {max(d.date for d in deals)}"
            }
        )

    # ========================================================================
    # Pattern 3: Clustered Buying (Multiple Buyers Same Day)
    # ========================================================================

    def detect_clustered_buying(self, deals: List[BulkDealRecord]) -> Optional[Pattern]:
        """
        Detect clustered buying pattern.

        Criteria:
        - Multiple unique buyers (3+) on same day
        - Significant total value (> ₹10 Cr)

        Returns:
            Pattern if detected, None otherwise
        """
        if not deals:
            return None

        # Group by date
        deals_by_date = defaultdict(list)
        for deal in deals:
            if deal.deal_type == 'BUY':
                deals_by_date[deal.date].append(deal)

        # Check each date for clustering
        for deal_date, day_deals in deals_by_date.items():
            unique_buyers = set(d.client_name for d in day_deals)
            total_value = sum(d.value for d in day_deals)

            if len(unique_buyers) >= self.MIN_CLUSTERED_BUYERS and \
               total_value >= self.MIN_CLUSTERED_VALUE:

                # Calculate confidence based on number of buyers and value
                confidence = 70 + min(len(unique_buyers) * 3, 15) + \
                            min(total_value / 10_00_00_000, 10)

                return Pattern(
                    type="CLUSTERED_BUYING",
                    signal="BULLISH",
                    confidence=min(confidence, 95),
                    evidence={
                        'date': str(deal_date),
                        'unique_buyers': len(unique_buyers),
                        'total_deals': len(day_deals),
                        'total_value': total_value,
                        'buyers': list(unique_buyers)[:5],  # Top 5
                        'avg_deal_size': total_value / len(day_deals)
                    }
                )

        return None

    # ========================================================================
    # Pattern 4: Unusual Activity (Volume Spike)
    # ========================================================================

    def detect_unusual_activity(
        self,
        recent_deals: List[BulkDealRecord],
        historical_avg: float
    ) -> Optional[Pattern]:
        """
        Detect unusual activity based on volume spike.

        Criteria:
        - Recent volume > 3x historical average

        Args:
            recent_deals: Deals from recent period (e.g., last day)
            historical_avg: Average daily value from historical period

        Returns:
            Pattern if detected, None otherwise
        """
        if not recent_deals or historical_avg <= 0:
            return None

        recent_value = sum(d.value for d in recent_deals)

        if recent_value < historical_avg * self.UNUSUAL_ACTIVITY_MULTIPLIER:
            return None

        multiplier = recent_value / historical_avg

        # Determine signal based on buy/sell ratio
        buy_value = sum(d.value for d in recent_deals if d.deal_type == 'BUY')
        sell_value = sum(d.value for d in recent_deals if d.deal_type == 'SELL')

        if buy_value > sell_value * 2:
            signal = "BULLISH"
        elif sell_value > buy_value * 2:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        confidence = 60 + min(multiplier * 5, 30)

        return Pattern(
            type="UNUSUAL_ACTIVITY",
            signal=signal,
            confidence=min(confidence, 90),
            evidence={
                'recent_value': recent_value,
                'historical_avg': historical_avg,
                'multiplier': round(multiplier, 2),
                'buy_value': buy_value,
                'sell_value': sell_value,
                'num_deals': len(recent_deals)
            }
        )

    # ========================================================================
    # Pattern 5: Pure Accumulation (No Selling)
    # ========================================================================

    def detect_pure_accumulation(self, deals: List[BulkDealRecord]) -> Optional[Pattern]:
        """
        Detect pure accumulation (all buying, no selling).

        Criteria:
        - Multiple buy deals (2+)
        - Zero sell deals
        - Significant value (> ₹5 Cr)

        Returns:
            Pattern if detected, None otherwise
        """
        if not deals:
            return None

        buy_deals = [d for d in deals if d.deal_type == 'BUY']
        sell_deals = [d for d in deals if d.deal_type == 'SELL']

        # Must have selling
        if len(sell_deals) > 0:
            return None

        # Must have multiple buys
        if len(buy_deals) < 2:
            return None

        total_value = sum(d.value for d in buy_deals)

        if total_value < self.MIN_ACCUMULATION_VALUE:
            return None

        unique_buyers = set(d.client_name for d in buy_deals)

        # Higher confidence for pure accumulation
        confidence = 85 + min(len(unique_buyers) * 3, 10)

        return Pattern(
            type="PURE_ACCUMULATION",
            signal="BULLISH",
            confidence=min(confidence, 95),
            evidence={
                'buy_deals': len(buy_deals),
                'sell_deals': 0,
                'total_value': total_value,
                'unique_buyers': len(unique_buyers),
                'buyers': list(unique_buyers)[:5]
            }
        )

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _get_top_buyers(self, buy_deals: List[BulkDealRecord], top_n: int = 3) -> List[Dict]:
        """Get top buyers by value."""
        buyer_values = defaultdict(float)
        for deal in buy_deals:
            buyer_values[deal.client_name] += deal.value

        top_buyers = sorted(buyer_values.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return [
            {'name': name, 'value': value}
            for name, value in top_buyers
        ]

    def _get_top_sellers(self, sell_deals: List[BulkDealRecord], top_n: int = 3) -> List[Dict]:
        """Get top sellers by value."""
        seller_values = defaultdict(float)
        for deal in sell_deals:
            seller_values[deal.client_name] += deal.value

        top_sellers = sorted(seller_values.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return [
            {'name': name, 'value': value}
            for name, value in top_sellers
        ]

    # ========================================================================
    # Main Detection Method
    # ========================================================================

    def detect_all_patterns(
        self,
        deals: List[BulkDealRecord],
        historical_avg: Optional[float] = None
    ) -> List[Pattern]:
        """
        Detect all patterns in the given deals.

        Args:
            deals: List of bulk deals to analyze
            historical_avg: Optional historical average for unusual activity detection

        Returns:
            List of detected patterns (may be empty)
        """
        patterns = []

        # Pattern 1: Pure Accumulation (check first - most bullish)
        pattern = self.detect_pure_accumulation(deals)
        if pattern:
            patterns.append(pattern)
            logger.info(f"  ✅ Detected: {pattern}")

        # Pattern 2: Sustained Accumulation
        if not patterns:  # Don't double-count
            pattern = self.detect_accumulation(deals)
            if pattern:
                patterns.append(pattern)
                logger.info(f"  ✅ Detected: {pattern}")

        # Pattern 3: Distribution
        pattern = self.detect_distribution(deals)
        if pattern:
            patterns.append(pattern)
            logger.info(f"  ✅ Detected: {pattern}")

        # Pattern 4: Clustered Buying
        pattern = self.detect_clustered_buying(deals)
        if pattern:
            patterns.append(pattern)
            logger.info(f"  ✅ Detected: {pattern}")

        # Pattern 5: Unusual Activity (if historical data provided)
        if historical_avg:
            pattern = self.detect_unusual_activity(deals, historical_avg)
            if pattern:
                patterns.append(pattern)
                logger.info(f"  ✅ Detected: {pattern}")

        if not patterns:
            logger.debug(f"  No patterns detected ({len(deals)} deals analyzed)")

        return patterns


# ============================================================================
# Testing with Real Data
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from utils.smart_money_sqlite import SmartMoneySQLite
    from datetime import date

    print("=" * 70)
    print("PATTERN DETECTOR - Testing with Real Data")
    print("=" * 70)

    # Load database
    db = SmartMoneySQLite()

    # Initialize detector
    detector = PatternDetector()

    # Get stocks with activity
    print("\n📊 Analyzing stocks with bulk deals...")
    bulk_deals_df = db.get_bulk_deals(limit=1000)

    if bulk_deals_df.empty:
        print("  ❌ No bulk deals in database")
        sys.exit(1)

    # Group by symbol
    symbols = bulk_deals_df['symbol'].unique()
    print(f"  Found {len(symbols)} symbols with activity\n")

    results = []

    for symbol in symbols:
        symbol_deals_df = bulk_deals_df[bulk_deals_df['symbol'] == symbol]

        # Convert to BulkDealRecord objects
        deals = []
        for _, row in symbol_deals_df.iterrows():
            deals.append(BulkDealRecord(
                date=date.fromisoformat(row['date']),
                symbol=row['symbol'],
                client_name=row['client_name'],
                deal_type=row['deal_type'],
                quantity=row['quantity'],
                price=row['price'],
                value=row['value']
            ))

        # Detect patterns
        print(f"🔍 Analyzing {symbol} ({len(deals)} deals)...")
        patterns = detector.detect_all_patterns(deals)

        if patterns:
            results.append((symbol, patterns))

    # Summary
    print("\n" + "=" * 70)
    print("PATTERN DETECTION SUMMARY")
    print("=" * 70)

    if results:
        print(f"\n✅ Found patterns in {len(results)} stocks:\n")

        for symbol, patterns in results:
            for pattern in patterns:
                print(f"📈 {symbol:12s} {pattern}")
                print(f"   Evidence: {pattern.evidence}")
                print()
    else:
        print("\n❌ No patterns detected")
        print("   Try running backfill to get more data")

    print("=" * 70)
