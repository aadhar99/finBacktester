"""
Mean Reversion Agent using Bollinger Bands and RSI.

Entry (Long):
- Price touches or breaks below lower Bollinger Band (oversold)
- RSI < 30 (oversold confirmation)

Exit (Long):
- Price reaches middle Bollinger Band (mean reversion complete)
- RSI > 70 (overbought)
- Stop loss: Below recent low

Best for: Ranging/consolidating markets
Avoid: Strong trending markets
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent, Signal, SignalType
from config import get_config


class ReversionAgent(BaseAgent):
    """Mean reversion strategy using Bollinger Bands and RSI."""

    def __init__(
        self,
        name: str = "ReversionAgent",
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0
    ):
        """
        Initialize the reversion agent.

        Args:
            name: Agent name
            bb_period: Bollinger Bands period
            bb_std_dev: Number of standard deviations for BB
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
        """
        super().__init__(name)
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.config_obj = get_config()

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_positions: Dict[str, int],
        portfolio_value: float,
        market_regime: Optional[str] = None
    ) -> list[Signal]:
        """
        Generate mean reversion signals.

        Entry Signal (Long):
        - Price <= Lower BB AND RSI < 30 (oversold)

        Entry Signal (Short - not implemented yet):
        - Price >= Upper BB AND RSI > 70 (overbought)

        Args:
            data: DataFrame with OHLCV and indicators
            current_positions: Current positions
            portfolio_value: Portfolio value
            market_regime: Market regime

        Returns:
            List of signals
        """
        signals = []

        if len(data) < max(self.bb_period, self.rsi_period):
            return signals

        # Get current data
        current = data.iloc[-1]
        symbol = current.name if hasattr(current, 'name') else 'UNKNOWN'

        if 'symbol' in data.columns:
            symbol = current['symbol']

        # Get indicators
        current_price = current['close']
        bb_upper = current.get('bb_upper', None)
        bb_middle = current.get('bb_middle', None)
        bb_lower = current.get('bb_lower', None)
        rsi = current.get('rsi', None)
        atr = current.get('atr', None)

        # Calculate if missing
        if bb_lower is None or pd.isna(bb_lower):
            bb_middle = data['close'].rolling(window=self.bb_period).mean().iloc[-1]
            rolling_std = data['close'].rolling(window=self.bb_period).std().iloc[-1]
            bb_upper = bb_middle + (self.bb_std_dev * rolling_std)
            bb_lower = bb_middle - (self.bb_std_dev * rolling_std)

        if rsi is None or pd.isna(rsi):
            rsi = self._calculate_rsi(data)

        if atr is None or pd.isna(atr):
            atr = self._calculate_atr(data)

        # Check current position
        position_qty = current_positions.get(symbol, 0)
        has_position = position_qty != 0

        # ENTRY LOGIC: Oversold conditions
        if not has_position:
            # Long entry: Price at/below lower BB AND RSI oversold
            if current_price <= bb_lower and rsi < self.rsi_oversold:
                # Calculate position size
                size = self.calculate_position_size(
                    symbol=symbol,
                    price=current_price,
                    portfolio_value=portfolio_value,
                    volatility=atr
                )

                if size > 0:
                    # Stop loss: Below recent swing low
                    recent_low = data['low'].tail(10).min()
                    stop_loss = recent_low * 0.98  # 2% below recent low

                    # Take profit: BB middle (mean reversion target)
                    take_profit = bb_middle

                    signal = Signal(
                        signal_type=SignalType.BUY,
                        symbol=symbol,
                        timestamp=current.name if isinstance(current.name, pd.Timestamp) else pd.Timestamp.now(),
                        price=current_price,
                        size=size,
                        confidence=0.75,
                        reason=f"Mean reversion: price {current_price:.2f} <= BB lower {bb_lower:.2f}, RSI {rsi:.1f} < {self.rsi_oversold}",
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'strategy': 'mean_reversion',
                            'bb_lower': bb_lower,
                            'bb_middle': bb_middle,
                            'rsi': rsi
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        self.logger.info(f"Generated BUY signal for {symbol} at {current_price:.2f} (oversold)")

        # EXIT LOGIC: Mean reversion complete or overbought
        else:
            entry_price = data['close'].iloc[-10:].mean()  # Placeholder

            should_exit, reason = self.should_exit(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                current_data=current,
                days_held=0
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
                    metadata={'strategy': 'mean_reversion'}
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
        Calculate position size for mean reversion trade.

        For mean reversion, we use slightly more conservative sizing since
        we're betting against the trend.

        Args:
            symbol: Stock symbol
            price: Current price
            portfolio_value: Portfolio value
            volatility: ATR
            max_position_pct: Max position size %

        Returns:
            Number of shares
        """
        # Use 1.5% risk instead of 2% (more conservative)
        risk_per_trade_pct = 0.015  # 1.5%
        risk_amount = portfolio_value * risk_per_trade_pct

        # Position size based on ATR
        if volatility > 0:
            risk_per_share = volatility * 1.5  # 1.5 ATR stop
            shares_by_risk = int(risk_amount / risk_per_share)
        else:
            shares_by_risk = 0

        # Respect max position size
        max_position_value = portfolio_value * (max_position_pct / 100)
        shares_by_max = int(max_position_value / price) if price > 0 else 0

        shares = min(shares_by_risk, shares_by_max)

        # Minimum position size
        min_position_value = self.config_obj.risk.min_position_value
        min_shares = int(min_position_value / price) if price > 0 else 0

        if shares < min_shares:
            return 0

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
        Determine exit using mean reversion rules.

        Exit conditions:
        1. Price reaches BB middle (mean reversion complete)
        2. RSI becomes overbought (> 70)
        3. Stop loss hit
        4. Max hold period exceeded (optional)

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            current_price: Current price
            current_data: Current market data
            days_held: Days held

        Returns:
            (should_exit, reason)
        """
        bb_middle = current_data.get('bb_middle', None)
        rsi = current_data.get('rsi', None)

        # Exit 1: Price reached BB middle (target achieved)
        if bb_middle is not None and not pd.isna(bb_middle):
            if current_price >= bb_middle:
                profit_pct = ((current_price - entry_price) / entry_price) * 100
                return True, f"Mean reversion target: price {current_price:.2f} >= BB middle {bb_middle:.2f} (+{profit_pct:.1f}%)"

        # Exit 2: RSI overbought (momentum shift)
        if rsi is not None and not pd.isna(rsi):
            if rsi > self.rsi_overbought:
                return True, f"RSI overbought: {rsi:.1f} > {self.rsi_overbought}"

        # Exit 3: Stop loss (would use actual stop loss from position)
        loss_pct = ((current_price - entry_price) / entry_price) * 100
        if loss_pct < -3.0:  # -3% stop loss
            return True, f"Stop loss hit: {loss_pct:.1f}%"

        # Exit 4: Max hold period (mean reversion should happen quickly)
        if days_held > 10:  # Exit if not reverted in 10 days
            return True, f"Max hold period exceeded: {days_held} days"

        return False, ""

    def is_enabled_for_regime(self, market_regime: Optional[str]) -> bool:
        """
        Mean reversion works best in ranging markets.

        Args:
            market_regime: Current market regime

        Returns:
            True if should trade in this regime
        """
        if market_regime is None:
            return True

        # Enable for ranging/low volatility
        favorable_regimes = ['ranging', 'low_volatility']
        return market_regime.lower() in favorable_regimes

    def _calculate_rsi(self, data: pd.DataFrame) -> float:
        """Calculate RSI from price data."""
        if len(data) < self.rsi_period + 1:
            return 50.0  # Neutral

        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if len(rsi) > 0 else 50.0

    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate ATR."""
        if len(data) < 20:
            return 0.0

        high = data['high']
        low = data['low']
        close = data['close']

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=20, adjust=False).mean()

        return atr.iloc[-1] if len(atr) > 0 else 0.0


def test_reversion_agent():
    """Test the reversion agent."""
    from data.fetcher import DataFetcher
    from data.preprocessor import DataPreprocessor

    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()

    df = fetcher.fetch_historical_data("TEST", "2023-01-01", "2024-01-01")
    df = preprocessor.prepare_for_backtest(df)

    print(f"Test data: {len(df)} rows")
    print(df[['close', 'bb_upper', 'bb_middle', 'bb_lower', 'rsi']].tail())

    agent = ReversionAgent()

    signals = agent.generate_signals(
        data=df,
        current_positions={},
        portfolio_value=100000.0,
        market_regime='ranging'
    )

    print(f"\nGenerated {len(signals)} signals")
    for signal in signals:
        print(f"  {signal.signal_type.value}: {signal.symbol} @ {signal.price:.2f} x {signal.size} shares")
        print(f"    Reason: {signal.reason}")
        if signal.stop_loss:
            print(f"    Stop Loss: {signal.stop_loss:.2f}, Take Profit: {signal.take_profit:.2f}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_reversion_agent()
