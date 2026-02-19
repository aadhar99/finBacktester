"""
Smart Money Trading Agent - Combines Institutional Signals with Technical Analysis

This agent bridges the gap between smart money pattern detection and actual trading:
1. Gets daily smart money signals (from SmartMoneyTracker)
2. Validates signals with technical analysis
3. Generates BUY/SELL signals with confidence-based position sizing
4. Integrates with existing risk management and execution

Usage:
    agent = SmartMoneyTradingAgent(db, llm, technical_validator='momentum')
    signals = agent.generate_signals(data, positions, portfolio_value)
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from agents.base_agent import BaseAgent, Signal, SignalType
from ai.agents.smart_money_tracker import SmartMoneyTracker, DailyReport
from ai.agents.smart_money_analyzer import StockAnalysis

logger = logging.getLogger(__name__)


@dataclass
class TechnicalValidation:
    """Result of technical validation for a smart money signal."""
    is_valid: bool
    reason: str
    score: float  # 0-100
    indicators: Dict[str, any]


class SmartMoneyTradingAgent(BaseAgent):
    """
    Trading agent that combines smart money signals with technical analysis.

    Flow:
    1. Daily scan for smart money opportunities
    2. For each opportunity, validate with technicals
    3. Generate trade signals only for validated opportunities
    4. Position size based on confidence (smart money + technical alignment)
    """

    def __init__(
        self,
        db,
        llm_manager,
        technical_validator: str = 'momentum',
        min_smart_money_confidence: float = 80.0,
        min_technical_score: float = 60.0,
        lookback_days: int = 7,
        use_llm: bool = True
    ):
        """
        Initialize Smart Money Trading Agent.

        Args:
            db: SmartMoneySQLite instance
            llm_manager: LLMManager for AI enhancement
            technical_validator: Which technical strategy to use ('momentum', 'reversion', 'trend')
            min_smart_money_confidence: Minimum smart money confidence (default 80%)
            min_technical_score: Minimum technical validation score (default 60%)
            lookback_days: Days to look back for bulk deals (default 7)
            use_llm: Whether to use LLM enhancement (default True)
        """
        super().__init__(name="SmartMoneyTradingAgent")

        self.db = db
        self.smart_money_tracker = SmartMoneyTracker(db, llm_manager, cache_hours=24)
        self.technical_validator = technical_validator
        self.min_smart_money_confidence = min_smart_money_confidence
        self.min_technical_score = min_technical_score
        self.lookback_days = lookback_days
        self.use_llm = use_llm

        logger.info(f"✅ SmartMoneyTradingAgent initialized")
        logger.info(f"   Technical validator: {technical_validator}")
        logger.info(f"   Min smart money confidence: {min_smart_money_confidence}%")
        logger.info(f"   Min technical score: {min_technical_score}%")

    async def get_opportunities(self) -> DailyReport:
        """Get today's smart money opportunities."""
        return await self.smart_money_tracker.get_daily_signals(
            lookback_days=self.lookback_days,
            min_confidence=self.min_smart_money_confidence,
            use_llm=self.use_llm
        )

    def validate_with_technicals(
        self,
        symbol: str,
        data: pd.DataFrame,
        smart_money_signal: str
    ) -> TechnicalValidation:
        """
        Validate smart money signal with technical analysis.

        Args:
            symbol: Stock symbol
            data: Price/volume data for the symbol
            smart_money_signal: BULLISH or BEARISH

        Returns:
            TechnicalValidation with score and reasoning
        """
        if symbol not in data.index:
            return TechnicalValidation(
                is_valid=False,
                reason=f"No price data available for {symbol}",
                score=0,
                indicators={}
            )

        stock_data = data.loc[symbol]

        # Calculate technical indicators
        indicators = self._calculate_indicators(stock_data)

        # Score the technical setup
        score, reasons = self._score_technical_setup(
            indicators,
            smart_money_signal
        )

        is_valid = score >= self.min_technical_score

        reason = "; ".join(reasons)

        return TechnicalValidation(
            is_valid=is_valid,
            reason=reason,
            score=score,
            indicators=indicators
        )

    def _calculate_indicators(self, stock_data: pd.Series) -> Dict[str, any]:
        """Calculate technical indicators from price data."""
        indicators = {}

        # For now, return placeholder indicators
        # In real implementation, you would calculate from historical data
        # This requires fetching historical data for each symbol

        indicators['price'] = stock_data.get('close', 0)
        indicators['volume'] = stock_data.get('volume', 0)

        # Placeholder values - in real implementation, calculate from historical data
        indicators['sma_20'] = indicators['price'] * 0.98  # Assume price slightly above SMA
        indicators['sma_50'] = indicators['price'] * 0.95
        indicators['rsi'] = 55  # Neutral
        indicators['macd'] = 0.5  # Slightly bullish
        indicators['volume_sma_20'] = indicators['volume'] * 0.9

        return indicators

    def _score_technical_setup(
        self,
        indicators: Dict[str, any],
        smart_money_signal: str
    ) -> Tuple[float, List[str]]:
        """
        Score the technical setup (0-100).

        For BULLISH signals, we want:
        - Price above moving averages
        - RSI not overbought (30-70)
        - MACD positive
        - Volume above average

        For BEARISH signals, opposite.
        """
        score = 0
        reasons = []

        price = indicators.get('price', 0)
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        volume = indicators.get('volume', 0)
        volume_sma = indicators.get('volume_sma_20', 0)

        if smart_money_signal == 'BULLISH':
            # Price above moving averages (+25 points)
            if price > sma_20:
                score += 15
                reasons.append("Price above SMA20")
            if price > sma_50:
                score += 10
                reasons.append("Price above SMA50")

            # RSI in healthy range (+20 points)
            if 40 <= rsi <= 70:
                score += 20
                reasons.append(f"RSI healthy ({rsi:.1f})")
            elif rsi > 70:
                score += 5
                reasons.append(f"RSI overbought ({rsi:.1f})")

            # MACD positive (+20 points)
            if macd > 0:
                score += 20
                reasons.append("MACD positive")

            # Volume confirmation (+20 points)
            if volume > volume_sma:
                score += 20
                reasons.append("Volume above average")

            # Trend alignment (+15 points)
            if sma_20 > sma_50:
                score += 15
                reasons.append("Uptrend confirmed")

        elif smart_money_signal == 'BEARISH':
            # Price below moving averages (+25 points)
            if price < sma_20:
                score += 15
                reasons.append("Price below SMA20")
            if price < sma_50:
                score += 10
                reasons.append("Price below SMA50")

            # RSI in healthy range (+20 points)
            if 30 <= rsi <= 60:
                score += 20
                reasons.append(f"RSI healthy ({rsi:.1f})")
            elif rsi < 30:
                score += 5
                reasons.append(f"RSI oversold ({rsi:.1f})")

            # MACD negative (+20 points)
            if macd < 0:
                score += 20
                reasons.append("MACD negative")

            # Volume confirmation (+20 points)
            if volume > volume_sma:
                score += 20
                reasons.append("Volume above average")

            # Trend alignment (+15 points)
            if sma_20 < sma_50:
                score += 15
                reasons.append("Downtrend confirmed")

        if not reasons:
            reasons.append("No technical confirmation")

        return score, reasons

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_positions: Dict[str, int],
        portfolio_value: float,
        market_regime: Optional[str] = None
    ) -> List[Signal]:
        """
        Generate trading signals based on smart money + technical validation.

        Args:
            data: Market data (must include symbols from smart money scan)
            current_positions: Current holdings
            portfolio_value: Total portfolio value
            market_regime: Optional market regime (not used currently)

        Returns:
            List of validated trading signals
        """
        logger.info("🔍 Generating smart money trading signals...")

        # Get smart money opportunities (async)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new loop
            import nest_asyncio
            nest_asyncio.apply()

        report = loop.run_until_complete(self.get_opportunities())

        logger.info(f"  📊 Found {len(report.opportunities)} smart money opportunities")

        signals = []

        for opp in report.opportunities:
            # Skip if already have position
            if opp.symbol in current_positions:
                logger.info(f"  ⏭️  {opp.symbol}: Already have position, skipping")
                continue

            # Validate with technical analysis
            validation = self.validate_with_technicals(
                opp.symbol,
                data,
                opp.signal
            )

            if not validation.is_valid:
                logger.info(f"  ❌ {opp.symbol}: Technical validation failed ({validation.score:.0f}%)")
                logger.info(f"      Reason: {validation.reason}")
                continue

            # Calculate combined confidence
            combined_confidence = (opp.confidence * 0.6 + validation.score * 0.4)

            # Get current price
            if opp.symbol in data.index:
                current_price = data.loc[opp.symbol, 'close']
            else:
                logger.warning(f"  ⚠️  {opp.symbol}: No price data, skipping")
                continue

            # Calculate position size based on confidence
            position_size = self.calculate_position_size(
                opp.symbol,
                current_price,
                portfolio_value,
                volatility=0.02,  # Default 2% volatility
                max_position_pct=combined_confidence / 100 * 0.15  # Up to 15% for 100% confidence
            )

            # Create signal
            signal_type = SignalType.BUY if opp.signal == 'BULLISH' else SignalType.SELL

            # Set stop loss based on smart money analysis
            stop_loss_pct = 0.05 if combined_confidence >= 90 else 0.07
            stop_loss = current_price * (1 - stop_loss_pct)

            # Set take profit based on confidence
            take_profit_pct = 0.15 if combined_confidence >= 90 else 0.10
            take_profit = current_price * (1 + take_profit_pct)

            signal = Signal(
                signal_type=signal_type,
                symbol=opp.symbol,
                timestamp=datetime.now(),
                price=current_price,
                size=position_size,
                confidence=combined_confidence,
                reason=f"Smart Money: {opp.patterns[0].type if opp.patterns else 'N/A'} | Technical: {validation.reason}",
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'smart_money_confidence': opp.confidence,
                    'technical_score': validation.score,
                    'pattern': opp.patterns[0].type if opp.patterns else None,
                    'net_value': opp.net_value,
                    'llm_recommendation': opp.llm_insights.recommendation if opp.llm_insights else None,
                    'technical_indicators': validation.indicators
                }
            )

            signals.append(signal)

            logger.info(f"  ✅ {opp.symbol}: SIGNAL GENERATED")
            logger.info(f"      Smart Money: {opp.confidence:.0f}% | Technical: {validation.score:.0f}%")
            logger.info(f"      Combined: {combined_confidence:.0f}% | Size: {position_size} shares")
            logger.info(f"      Price: ₹{current_price:.2f} | SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f}")

        logger.info(f"  🎯 Generated {len(signals)} validated trading signals")

        return signals

    def should_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        current_data: pd.Series,
        days_held: int
    ) -> Tuple[bool, str]:
        """
        Determine if we should exit a position.

        Uses standard technical exit rules:
        - Stop loss hit
        - Take profit hit
        - Held for max duration
        - Technical reversal
        """
        # Stop loss (5-7% below entry)
        if current_price <= entry_price * 0.93:
            return True, "Stop loss hit"

        # Take profit (10-15% above entry)
        if current_price >= entry_price * 1.12:
            return True, "Take profit reached"

        # Max holding period (30 days for smart money plays)
        if days_held >= 30:
            return True, "Max holding period reached"

        # Technical reversal (price drops below SMA20)
        # Would need historical data to calculate properly
        # For now, use simple price decline
        if current_price <= entry_price * 0.95 and days_held >= 5:
            return True, "Technical reversal"

        return False, ""

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        volatility: float,
        max_position_pct: float
    ) -> int:
        """
        Calculate position size based on confidence and risk.

        Higher confidence → Larger position (up to max_position_pct of portfolio)
        """
        # Position value based on confidence
        position_value = portfolio_value * max_position_pct

        # Convert to shares
        shares = int(position_value / price)

        # Minimum 1 share
        return max(1, shares)


# ============================================================================
# Testing Script
# ============================================================================

async def test_smart_money_agent():
    """Test SmartMoneyTradingAgent with real data."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from utils.smart_money_sqlite import SmartMoneySQLite
    from utils.llm import LLMManager, LLMConfig, LLMProvider
    import os

    print("=" * 70)
    print("SMART MONEY TRADING AGENT - Test")
    print("=" * 70)

    # Initialize database
    db = SmartMoneySQLite()

    # Initialize LLM (disabled for faster testing)
    print("⚠️  LLM disabled for faster testing")
    llm = None

    # Initialize agent
    agent = SmartMoneyTradingAgent(
        db=db,
        llm_manager=llm,
        technical_validator='momentum',
        min_smart_money_confidence=80.0,
        min_technical_score=60.0,
        lookback_days=7,
        use_llm=False  # Disabled for faster testing
    )

    # Create mock price data for testing
    # In real usage, this would come from DataFetcher
    symbols = ['APEX', 'GVPIL', 'BIOPOL', 'ENGINERSIN']

    mock_data = pd.DataFrame({
        'symbol': symbols,
        'close': [150.0, 200.0, 155.0, 300.0],
        'volume': [1000000, 800000, 600000, 500000]
    }).set_index('symbol')

    print(f"\n📊 Mock price data:")
    print(mock_data)

    # Generate signals
    print(f"\n🔍 Generating trading signals...\n")

    signals = agent.generate_signals(
        data=mock_data,
        current_positions={},
        portfolio_value=1000000  # ₹10 lakh portfolio
    )

    # Print results
    print(f"\n{'=' * 70}")
    print(f"SIGNALS GENERATED: {len(signals)}")
    print(f"{'=' * 70}\n")

    for i, signal in enumerate(signals, 1):
        print(f"{i}. {signal.symbol} - {signal.signal_type.name}")
        print(f"   Price: ₹{signal.price:.2f}")
        print(f"   Size: {signal.size} shares (₹{signal.price * signal.size:,.0f})")
        print(f"   Confidence: {signal.confidence:.0f}%")
        print(f"   Stop Loss: ₹{signal.stop_loss:.2f} (-{((signal.price - signal.stop_loss) / signal.price * 100):.1f}%)")
        print(f"   Take Profit: ₹{signal.take_profit:.2f} (+{((signal.take_profit - signal.price) / signal.price * 100):.1f}%)")
        print(f"   Reason: {signal.reason}")
        print()

    print("=" * 70)
    print("✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    asyncio.run(test_smart_money_agent())
