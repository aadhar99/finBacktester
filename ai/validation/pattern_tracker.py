"""
Pattern Tracker - Track Smart Money Pattern Outcomes

Tracks the performance of detected patterns by monitoring:
1. Pattern detection date/price
2. Price movements in following days (7, 14, 30 days)
3. Whether pattern predicted correct direction
4. Actual returns achieved

This builds the evidence base to validate which patterns work best.

Usage:
    tracker = PatternTracker(db)
    tracker.track_pattern(symbol, pattern, detection_date, detection_price)
    results = tracker.get_pattern_performance(pattern_type, days=30)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class PatternOutcome:
    """Outcome tracking for a detected pattern."""
    pattern_id: int
    symbol: str
    pattern_type: str
    signal: str  # BULLISH or BEARISH
    confidence: float
    detection_date: date
    detection_price: float

    # Future prices
    price_7d: Optional[float] = None
    price_14d: Optional[float] = None
    price_30d: Optional[float] = None

    # Returns
    return_7d: Optional[float] = None
    return_14d: Optional[float] = None
    return_30d: Optional[float] = None

    # Success flags
    success_7d: Optional[bool] = None
    success_14d: Optional[bool] = None
    success_30d: Optional[bool] = None

    # Metadata
    net_value: float = 0
    num_deals: int = 0
    llm_recommendation: Optional[str] = None


@dataclass
class PatternPerformance:
    """Aggregated performance metrics for a pattern type."""
    pattern_type: str
    total_occurrences: int

    # Win rates
    win_rate_7d: float
    win_rate_14d: float
    win_rate_30d: float

    # Average returns
    avg_return_7d: float
    avg_return_14d: float
    avg_return_30d: float

    # Winners vs losers
    avg_winning_return: float
    avg_losing_return: float

    # Best/worst
    best_return: float
    worst_return: float
    best_symbol: str
    worst_symbol: str

    # Risk metrics
    sharpe_ratio: float
    max_drawdown: float


class PatternTracker:
    """
    Track and analyze smart money pattern outcomes.

    Maintains a database of detected patterns and their subsequent
    price movements to validate which patterns actually work.
    """

    def __init__(self, db_path: str = "data/pattern_validation.db"):
        """
        Initialize Pattern Tracker.

        Args:
            db_path: Path to SQLite database for pattern tracking
        """
        self.db_path = db_path
        self._init_database()

        logger.info("✅ PatternTracker initialized")
        logger.info(f"   Database: {db_path}")

    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create pattern_outcomes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_outcomes (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                detection_date TEXT NOT NULL,
                detection_price REAL NOT NULL,

                -- Future prices
                price_7d REAL,
                price_14d REAL,
                price_30d REAL,

                -- Returns
                return_7d REAL,
                return_14d REAL,
                return_30d REAL,

                -- Success flags
                success_7d INTEGER,
                success_14d INTEGER,
                success_30d INTEGER,

                -- Metadata
                net_value REAL DEFAULT 0,
                num_deals INTEGER DEFAULT 0,
                llm_recommendation TEXT,

                -- Tracking
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(symbol, pattern_type, detection_date)
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_type
            ON pattern_outcomes(pattern_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detection_date
            ON pattern_outcomes(detection_date)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol
            ON pattern_outcomes(symbol)
        """)

        conn.commit()
        conn.close()

        logger.info("  📊 Database tables created/verified")

    def track_pattern(
        self,
        symbol: str,
        pattern_type: str,
        signal: str,
        confidence: float,
        detection_date: date,
        detection_price: float,
        net_value: float = 0,
        num_deals: int = 0,
        llm_recommendation: Optional[str] = None
    ) -> int:
        """
        Record a detected pattern for future tracking.

        Args:
            symbol: Stock symbol
            pattern_type: Type of pattern (e.g., CLUSTERED_BUYING)
            signal: BULLISH or BEARISH
            confidence: Pattern confidence (0-100)
            detection_date: Date pattern was detected
            detection_price: Stock price at detection
            net_value: Net institutional value
            num_deals: Number of bulk deals
            llm_recommendation: AI recommendation (BUY/HOLD/AVOID)

        Returns:
            pattern_id: ID of inserted pattern
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO pattern_outcomes (
                    symbol, pattern_type, signal, confidence,
                    detection_date, detection_price,
                    net_value, num_deals, llm_recommendation,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                symbol, pattern_type, signal, confidence,
                detection_date.isoformat(), detection_price,
                net_value, num_deals, llm_recommendation
            ))

            pattern_id = cursor.lastrowid
            conn.commit()

            logger.info(f"  📝 Tracked: {symbol} {pattern_type} @ ₹{detection_price:.2f}")

            return pattern_id

        finally:
            conn.close()

    def update_outcome(
        self,
        pattern_id: int,
        days: int,
        current_price: float
    ):
        """
        Update pattern outcome with current price after N days.

        Args:
            pattern_id: Pattern ID to update
            days: Number of days since detection (7, 14, or 30)
            current_price: Current stock price
        """
        if days not in [7, 14, 30]:
            raise ValueError("days must be 7, 14, or 30")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get pattern details
            cursor.execute("""
                SELECT detection_price, signal
                FROM pattern_outcomes
                WHERE pattern_id = ?
            """, (pattern_id,))

            row = cursor.fetchone()
            if not row:
                logger.warning(f"Pattern {pattern_id} not found")
                return

            detection_price, signal = row

            # Calculate return
            return_pct = ((current_price - detection_price) / detection_price) * 100

            # Determine success (did it move in predicted direction?)
            if signal == 'BULLISH':
                success = return_pct > 0
            else:  # BEARISH
                success = return_pct < 0

            # Update database
            price_col = f"price_{days}d"
            return_col = f"return_{days}d"
            success_col = f"success_{days}d"

            cursor.execute(f"""
                UPDATE pattern_outcomes
                SET {price_col} = ?,
                    {return_col} = ?,
                    {success_col} = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE pattern_id = ?
            """, (current_price, return_pct, 1 if success else 0, pattern_id))

            conn.commit()

            logger.info(f"  ✅ Updated pattern {pattern_id}: {days}d return = {return_pct:+.2f}%")

        finally:
            conn.close()

    def get_pattern_performance(
        self,
        pattern_type: Optional[str] = None,
        days: int = 30,
        min_occurrences: int = 5
    ) -> Optional[PatternPerformance]:
        """
        Get aggregated performance for a pattern type.

        Args:
            pattern_type: Specific pattern type (None for all patterns)
            days: Timeframe to analyze (7, 14, or 30)
            min_occurrences: Minimum occurrences needed for valid stats

        Returns:
            PatternPerformance with aggregated metrics
        """
        if days not in [7, 14, 30]:
            raise ValueError("days must be 7, 14, or 30")

        conn = sqlite3.connect(self.db_path)

        # Build query
        where_clause = ""
        params = []

        if pattern_type:
            where_clause = "WHERE pattern_type = ?"
            params.append(pattern_type)

        # Get patterns with outcomes
        query = f"""
            SELECT
                symbol,
                return_7d, return_14d, return_30d,
                success_7d, success_14d, success_30d
            FROM pattern_outcomes
            {where_clause}
            AND return_{days}d IS NOT NULL
        """

        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()

        if len(df) < min_occurrences:
            logger.warning(f"Not enough data: {len(df)} < {min_occurrences}")
            return None

        # Calculate metrics
        returns = df[f'return_{days}d']
        successes = df[f'success_{days}d']

        win_rate = successes.mean() * 100
        avg_return = returns.mean()

        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]

        avg_winning = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_losing = losing_returns.mean() if len(losing_returns) > 0 else 0

        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()

        # Sharpe ratio (annualized, assuming daily returns)
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Max drawdown
        cumulative = (1 + returns / 100).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100

        return PatternPerformance(
            pattern_type=pattern_type or "ALL",
            total_occurrences=len(df),
            win_rate_7d=df['success_7d'].mean() * 100 if 'success_7d' in df else 0,
            win_rate_14d=df['success_14d'].mean() * 100 if 'success_14d' in df else 0,
            win_rate_30d=df['success_30d'].mean() * 100 if 'success_30d' in df else 0,
            avg_return_7d=df['return_7d'].mean() if 'return_7d' in df else 0,
            avg_return_14d=df['return_14d'].mean() if 'return_14d' in df else 0,
            avg_return_30d=df['return_30d'].mean() if 'return_30d' in df else 0,
            avg_winning_return=avg_winning,
            avg_losing_return=avg_losing,
            best_return=returns.max(),
            worst_return=returns.min(),
            best_symbol=df.loc[best_idx, 'symbol'],
            worst_symbol=df.loc[worst_idx, 'symbol'],
            sharpe_ratio=sharpe,
            max_drawdown=max_dd
        )

    def get_all_patterns_performance(
        self,
        days: int = 30
    ) -> List[PatternPerformance]:
        """
        Get performance for all pattern types.

        Args:
            days: Timeframe to analyze

        Returns:
            List of PatternPerformance for each pattern type
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get unique pattern types
        cursor.execute("SELECT DISTINCT pattern_type FROM pattern_outcomes")
        pattern_types = [row[0] for row in cursor.fetchall()]

        conn.close()

        performances = []
        for pattern_type in pattern_types:
            perf = self.get_pattern_performance(pattern_type, days)
            if perf:
                performances.append(perf)

        # Sort by win rate
        performances.sort(key=lambda p: p.win_rate_30d, reverse=True)

        return performances

    def print_performance_report(self, days: int = 30):
        """
        Print detailed performance report for all patterns.

        Args:
            days: Timeframe to report on
        """
        performances = self.get_all_patterns_performance(days)

        if not performances:
            print("No pattern performance data available yet.")
            return

        print("=" * 80)
        print(f"SMART MONEY PATTERN PERFORMANCE REPORT - {days} Days")
        print("=" * 80)

        for perf in performances:
            print(f"\n📊 {perf.pattern_type}")
            print(f"   Total Occurrences: {perf.total_occurrences}")
            print(f"   Win Rate: {perf.win_rate_30d:.1f}%")
            print(f"   Avg Return: {perf.avg_return_30d:+.2f}%")
            print(f"   Avg Win: {perf.avg_winning_return:+.2f}%")
            print(f"   Avg Loss: {perf.avg_losing_return:+.2f}%")
            print(f"   Best: {perf.best_symbol} ({perf.best_return:+.2f}%)")
            print(f"   Worst: {perf.worst_symbol} ({perf.worst_return:+.2f}%)")
            print(f"   Sharpe: {perf.sharpe_ratio:.2f}")
            print(f"   Max DD: {perf.max_drawdown:.2f}%")

        # Overall stats
        total_occurrences = sum(p.total_occurrences for p in performances)
        avg_win_rate = np.mean([p.win_rate_30d for p in performances])
        avg_return = np.mean([p.avg_return_30d for p in performances])

        print("\n" + "=" * 80)
        print("OVERALL PERFORMANCE")
        print("=" * 80)
        print(f"   Total Patterns Tracked: {total_occurrences}")
        print(f"   Average Win Rate: {avg_win_rate:.1f}%")
        print(f"   Average Return: {avg_return:+.2f}%")
        print("=" * 80)


# ============================================================================
# Testing
# ============================================================================

def test_pattern_tracker():
    """Test PatternTracker with sample data."""
    print("=" * 70)
    print("PATTERN TRACKER - Test")
    print("=" * 70)

    tracker = PatternTracker(db_path="data/pattern_validation_test.db")

    # Track some sample patterns
    print("\n📝 Tracking sample patterns...")

    patterns = [
        ("APEX", "CLUSTERED_BUYING", "BULLISH", 95, "2026-02-01", 150.0),
        ("BIOPOL", "SUSTAINED_ACCUMULATION", "BULLISH", 92, "2026-02-01", 155.0),
        ("XYZ", "DISTRIBUTION", "BEARISH", 88, "2026-02-01", 200.0),
    ]

    pattern_ids = []
    for symbol, ptype, signal, conf, det_date, price in patterns:
        pid = tracker.track_pattern(
            symbol=symbol,
            pattern_type=ptype,
            signal=signal,
            confidence=conf,
            detection_date=datetime.fromisoformat(det_date).date(),
            detection_price=price
        )
        pattern_ids.append(pid)

    # Simulate outcomes
    print("\n📈 Simulating price movements...")

    # APEX went up 10%
    tracker.update_outcome(pattern_ids[0], 7, 165.0)
    tracker.update_outcome(pattern_ids[0], 30, 172.0)

    # BIOPOL went up 5%
    tracker.update_outcome(pattern_ids[1], 7, 160.0)
    tracker.update_outcome(pattern_ids[1], 30, 163.0)

    # XYZ went down 8% (bearish pattern was correct)
    tracker.update_outcome(pattern_ids[2], 7, 184.0)
    tracker.update_outcome(pattern_ids[2], 30, 180.0)

    # Print report
    print("\n")
    tracker.print_performance_report(days=30)

    print("\n✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_pattern_tracker()
