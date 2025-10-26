"""
Momentum Agent implementing Turtle Trader strategy.

Entry: Price breaks above 55-day high (or 20-day high for secondary entries)
Exit: Price breaks below 20-day low
Stop Loss: 2 x ATR from entry
Position Sizing: Based on volatility (ATR)

Best for: Trending markets
Avoid: Ranging/choppy markets
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent, Signal, SignalType
from config import get_config


class MomentumAgent(BaseAgent):
    """Turtle Trader momentum strategy."""

    def __init__(
        self,
        name: str = "MomentumAgent",
        lookback_period: int = 55,
        exit_period: int = 20,
        atr_period: int = 20,
        atr_multiplier: float = 2.0
    ):
        """
        Initialize the momentum agent.

        Args:
            name: Agent name
            lookback_period: Breakout period (default 55 days)
            exit_period: Exit period (default 20 days)
            atr_period: ATR calculation period
            atr_multiplier: Stop loss multiplier for ATR
        """
        super().__init__(name)
        self.lookback_period = lookback_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.config_obj = get_config()

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_positions: Dict[str, int],
        portfolio_value: float,
        market_regime: Optional[str] = None
    ) -> list[Signal]:
        """
        Generate momentum signals based on Turtle Trader rules.

        Entry Signal:
        - Price breaks above 55-day high (NEW high)
        - Must be a fresh breakout (not seen in last 2 days)

        Args:
            data: DataFrame with OHLCV and indicators (single symbol)
            current_positions: Current positions {symbol: quantity}
            portfolio_value: Portfolio value
            market_regime: Market regime

        Returns:
            List of signals
        """
        signals = []

        if len(data) < self.lookback_period:
            return signals

        # Get current (latest) data
        current = data.iloc[-1]
        symbol = current.name if hasattr(current, 'name') else 'UNKNOWN'

        # Check if symbol is in data columns
        if 'symbol' in data.columns:
            symbol = current['symbol']

        # Get required indicators
        current_price = current['close']
        channel_high = current.get('channel_high', None)
        channel_low = current.get('channel_low', None)
        exit_low = current.get('exit_low', None)
        atr = current.get('atr', None)

        # If indicators not present, calculate them
        if channel_high is None or pd.isna(channel_high):
            channel_high = data['high'].rolling(window=self.lookback_period).max().iloc[-1]
            channel_low = data['low'].rolling(window=self.lookback_period).min().iloc[-1]
            exit_low = data['low'].rolling(window=self.exit_period).min().iloc[-1]

        if atr is None or pd.isna(atr):
            atr = self._calculate_atr(data)

        # Check current position
        position_qty = current_positions.get(symbol, 0)
        has_position = position_qty != 0

        # ENTRY LOGIC: Breakout above 55-day high
        if not has_position:
            prev_high = data['high'].iloc[-2] if len(data) > 1 else 0

            # Breakout: current price > channel high AND previous wasn't
            if current_price > channel_high and prev_high <= channel_high:
                # Calculate position size
                size = self.calculate_position_size(
                    symbol=symbol,
                    price=current_price,
                    portfolio_value=portfolio_value,
                    volatility=atr
                )

                if size > 0:
                    # Calculate stop loss and take profit
                    stop_loss = self.calculate_stop_loss(
                        entry_price=current_price,
                        atr=atr,
                        multiplier=self.atr_multiplier,
                        is_long=True
                    )

                    risk = current_price - stop_loss
                    take_profit = self.calculate_take_profit(
                        entry_price=current_price,
                        risk=risk,
                        reward_ratio=2.5,  # 2.5:1 reward-to-risk
                        is_long=True
                    )

                    signal = Signal(
                        signal_type=SignalType.BUY,
                        symbol=symbol,
                        timestamp=current.name if isinstance(current.name, pd.Timestamp) else pd.Timestamp.now(),
                        price=current_price,
                        size=size,
                        confidence=0.8,
                        reason=f"Turtle breakout: price {current_price:.2f} > 55d high {channel_high:.2f}",
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'strategy': 'turtle_trader',
                            'channel_high': channel_high,
                            'atr': atr
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        self.logger.info(f"Generated BUY signal for {symbol} at {current_price:.2f}")

        # EXIT LOGIC: Break below 20-day low OR stop loss hit
        else:
            # Get entry price (would come from position tracker in real implementation)
            # For now, approximate from recent data
            entry_price = data['close'].iloc[-10:].mean()  # Placeholder

            should_exit, reason = self.should_exit(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                current_data=current,
                days_held=0  # Would come from position tracker
            )

            if should_exit:
                signal = Signal(
                    signal_type=SignalType.EXIT_LONG,
                    symbol=symbol,
                    timestamp=current.name if isinstance(current.name, pd.Timestamp) else pd.Timestamp.now(),
                    price=current_price,
                    size=abs(position_qty),
                    confidence=1.0,
                    reason=reason,
                    metadata={'strategy': 'turtle_trader'}
                )

                if self.validate_signal(signal):
                    signals.append(signal)
                    self.logger.info(f"Generated EXIT signal for {symbol}: {reason}")

        return signals

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        volatility: float,
        max_position_pct: float = 5.0
    ) -> int:
        """
        Calculate position size using ATR-based volatility sizing.

        Position size = (Portfolio Value * Risk % per Trade) / (ATR * Multiplier)

        Args:
            symbol: Stock symbol
            price: Current price
            portfolio_value: Portfolio value
            volatility: ATR value
            max_position_pct: Max position size as % of portfolio

        Returns:
            Number of shares
        """
        # Risk per trade (from config)
        risk_per_trade_pct = self.config_obj.risk.max_loss_per_trade_pct / 100  # 2% -> 0.02
        risk_amount = portfolio_value * risk_per_trade_pct

        # Position size based on ATR
        # Risk per share = ATR * multiplier
        if volatility > 0:
            risk_per_share = volatility * self.atr_multiplier
            shares_by_risk = int(risk_amount / risk_per_share)
        else:
            shares_by_risk = 0

        # Also respect max position size
        max_position_value = portfolio_value * (max_position_pct / 100)
        shares_by_max = int(max_position_value / price) if price > 0 else 0

        # Take the smaller of the two
        shares = min(shares_by_risk, shares_by_max)

        # Respect minimum position size
        min_position_value = self.config_obj.risk.min_position_value
        min_shares = int(min_position_value / price) if price > 0 else 0

        if shares < min_shares:
            return 0  # Position too small

        return shares

    def should_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        current_data: pd.Series,
        days_held: int
    ) -> Tuple[bool, str]:
        """
        Determine if we should exit using Turtle exit rules.

        Exit conditions:
        1. Price breaks below 20-day low
        2. Stop loss hit (entry - 2*ATR)
        3. Take profit hit (optional)

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            current_price: Current price
            current_data: Current market data
            days_held: Days held

        Returns:
            (should_exit, reason)
        """
        exit_low = current_data.get('exit_low', None)
        atr = current_data.get('atr', None)

        # Exit 1: Break below 20-day low
        if exit_low is not None and not pd.isna(exit_low):
            if current_price < exit_low:
                return True, f"Price {current_price:.2f} < 20d low {exit_low:.2f}"

        # Exit 2: Stop loss (entry - 2*ATR)
        if atr is not None and not pd.isna(atr):
            stop_loss = entry_price - (self.atr_multiplier * atr)
            if current_price < stop_loss:
                loss_pct = ((current_price - entry_price) / entry_price) * 100
                return True, f"Stop loss hit: {current_price:.2f} < {stop_loss:.2f} ({loss_pct:.1f}%)"

        # Exit 3: Take profit (optional, for momentum we usually let winners run)
        # Could add trailing stop here

        return False, ""

    def is_enabled_for_regime(self, market_regime: Optional[str]) -> bool:
        """
        Momentum strategies work best in trending markets.

        Args:
            market_regime: Current market regime

        Returns:
            True if should trade in this regime
        """
        if market_regime is None:
            return True

        # Enable for trending regimes
        trending_regimes = ['trending_up', 'trending_down']
        return market_regime.lower() in trending_regimes

    def _calculate_atr(self, data: pd.DataFrame, period: Optional[int] = None) -> float:
        """
        Calculate ATR from OHLC data.

        Args:
            data: OHLCV DataFrame
            period: ATR period (uses self.atr_period if not specified)

        Returns:
            ATR value
        """
        if period is None:
            period = self.atr_period

        if len(data) < period:
            return 0.0

        high = data['high']
        low = data['low']
        close = data['close']

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr.iloc[-1] if len(atr) > 0 else 0.0


def test_momentum_agent():
    """Test the momentum agent with synthetic data."""
    from data.fetcher import DataFetcher
    from data.preprocessor import DataPreprocessor

    # Generate test data
    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()

    df = fetcher.fetch_historical_data("TEST", "2023-01-01", "2024-01-01")
    df = preprocessor.prepare_for_backtest(df)

    print(f"Test data: {len(df)} rows")
    print(df[['close', 'channel_high', 'exit_low', 'atr']].tail())

    # Create agent
    agent = MomentumAgent()

    # Generate signals
    signals = agent.generate_signals(
        data=df,
        current_positions={},
        portfolio_value=100000.0,
        market_regime='trending_up'
    )

    print(f"\nGenerated {len(signals)} signals")
    for signal in signals:
        print(f"  {signal.signal_type.value}: {signal.symbol} @ {signal.price:.2f} x {signal.size} shares")
        print(f"    Reason: {signal.reason}")
        print(f"    Stop Loss: {signal.stop_loss:.2f}, Take Profit: {signal.take_profit:.2f}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_momentum_agent()
