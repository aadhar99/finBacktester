"""
Market Regime Filter.

Classifies market conditions to optimize agent selection:
- Trend: Trending up/down (favor momentum strategies)
- Range: Sideways/consolidating (favor mean reversion)
- Volatility: High/low volatility (affects position sizing)

Uses Nifty 50 index and India VIX for regime classification.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from enum import Enum
import logging

from config import MarketRegime, get_config

logger = logging.getLogger(__name__)


class RegimeFilter:
    """Detect and classify market regimes."""

    def __init__(
        self,
        trend_fast_ma: int = 20,
        trend_slow_ma: int = 50,
        trend_threshold: float = 0.02,
        vix_high_threshold: float = 20.0,
        vix_low_threshold: float = 12.0,
        lookback_days: int = 30
    ):
        """
        Initialize the regime filter.

        Args:
            trend_fast_ma: Fast moving average period
            trend_slow_ma: Slow moving average period
            trend_threshold: Threshold for trend classification (2%)
            vix_high_threshold: VIX threshold for high volatility
            vix_low_threshold: VIX threshold for low volatility
            lookback_days: Days to look back for regime classification
        """
        self.trend_fast_ma = trend_fast_ma
        self.trend_slow_ma = trend_slow_ma
        self.trend_threshold = trend_threshold
        self.vix_high_threshold = vix_high_threshold
        self.vix_low_threshold = vix_low_threshold
        self.lookback_days = lookback_days

        self.config = get_config()

    def detect_regime(
        self,
        nifty_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None
    ) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Detect current market regime.

        Args:
            nifty_data: Nifty 50 OHLCV data
            vix_data: India VIX data (optional)
            current_date: Date to detect regime for (uses latest if None)

        Returns:
            Tuple of (regime, metrics_dict)
        """
        if current_date is None:
            current_date = nifty_data.index[-1]

        # Get data up to current date
        data_slice = nifty_data[nifty_data.index <= current_date].tail(self.lookback_days + self.trend_slow_ma)

        if len(data_slice) < self.trend_slow_ma:
            logger.warning("Insufficient data for regime detection")
            return MarketRegime.RANGING, {}

        # 1. Detect trend direction
        trend_regime, trend_metrics = self._detect_trend(data_slice)

        # 2. Detect volatility regime
        volatility_regime, vol_metrics = self._detect_volatility(data_slice, vix_data, current_date)

        # 3. Combine into primary regime
        primary_regime = self._combine_regimes(trend_regime, volatility_regime)

        # Combine metrics
        metrics = {**trend_metrics, **vol_metrics, 'regime': primary_regime.value}

        logger.info(f"Detected regime: {primary_regime.value} (trend={trend_regime.value}, vol={volatility_regime.value})")
        return primary_regime, metrics

    def _detect_trend(self, data: pd.DataFrame) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Detect trend regime using moving averages.

        Trend Classification:
        - Trending Up: Fast MA > Slow MA by > threshold AND price > both MAs
        - Trending Down: Fast MA < Slow MA by > threshold AND price < both MAs
        - Ranging: Otherwise

        Args:
            data: OHLCV data

        Returns:
            (trend_regime, metrics)
        """
        close = data['close']

        # Calculate moving averages
        fast_ma = close.rolling(window=self.trend_fast_ma).mean()
        slow_ma = close.rolling(window=self.trend_slow_ma).mean()

        # Get current values
        current_price = close.iloc[-1]
        current_fast_ma = fast_ma.iloc[-1]
        current_slow_ma = slow_ma.iloc[-1]

        # Calculate MA difference (percentage)
        ma_diff_pct = ((current_fast_ma - current_slow_ma) / current_slow_ma) * 100

        # Calculate price position relative to MAs
        price_vs_fast = ((current_price - current_fast_ma) / current_fast_ma) * 100
        price_vs_slow = ((current_price - current_slow_ma) / current_slow_ma) * 100

        # Classify trend
        if ma_diff_pct > (self.trend_threshold * 100) and current_price > current_fast_ma:
            regime = MarketRegime.TRENDING_UP
        elif ma_diff_pct < -(self.trend_threshold * 100) and current_price < current_fast_ma:
            regime = MarketRegime.TRENDING_DOWN
        else:
            regime = MarketRegime.RANGING

        metrics = {
            'fast_ma': current_fast_ma,
            'slow_ma': current_slow_ma,
            'ma_diff_pct': ma_diff_pct,
            'price_vs_fast_pct': price_vs_fast,
            'price_vs_slow_pct': price_vs_slow,
            'trend': regime.value
        }

        return regime, metrics

    def _detect_volatility(
        self,
        data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame],
        current_date: pd.Timestamp
    ) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Detect volatility regime.

        Uses:
        1. India VIX if available
        2. Historical volatility (std of returns) as fallback

        Args:
            data: OHLCV data
            vix_data: VIX data (optional)
            current_date: Current date

        Returns:
            (volatility_regime, metrics)
        """
        # Try to use VIX first
        if vix_data is not None and len(vix_data) > 0:
            vix_slice = vix_data[vix_data.index <= current_date]
            if len(vix_slice) > 0:
                current_vix = vix_slice['vix'].iloc[-1]

                if current_vix > self.vix_high_threshold:
                    regime = MarketRegime.HIGH_VOLATILITY
                elif current_vix < self.vix_low_threshold:
                    regime = MarketRegime.LOW_VOLATILITY
                else:
                    regime = MarketRegime.RANGING

                metrics = {
                    'vix': current_vix,
                    'vix_source': 'india_vix',
                    'volatility': regime.value
                }
                return regime, metrics

        # Fallback: Calculate historical volatility
        returns = data['close'].pct_change()
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100  # Annualized %

        current_vol = rolling_vol.iloc[-1]
        avg_vol = rolling_vol.mean()

        # Classify based on relative to average
        if current_vol > avg_vol * 1.5:
            regime = MarketRegime.HIGH_VOLATILITY
        elif current_vol < avg_vol * 0.7:
            regime = MarketRegime.LOW_VOLATILITY
        else:
            regime = MarketRegime.RANGING

        metrics = {
            'historical_volatility': current_vol,
            'avg_volatility': avg_vol,
            'vix_source': 'historical',
            'volatility': regime.value
        }

        return regime, metrics

    def _combine_regimes(
        self,
        trend_regime: MarketRegime,
        volatility_regime: MarketRegime
    ) -> MarketRegime:
        """
        Combine trend and volatility regimes into primary regime.

        Priority:
        1. Trend regimes (trending up/down) take precedence
        2. High volatility overrides ranging with caution
        3. Low volatility with ranging = favorable for mean reversion

        Args:
            trend_regime: Detected trend
            volatility_regime: Detected volatility

        Returns:
            Primary market regime
        """
        # Trending regimes take precedence
        if trend_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            return trend_regime

        # High volatility in ranging market = be cautious
        if volatility_regime == MarketRegime.HIGH_VOLATILITY:
            return MarketRegime.HIGH_VOLATILITY

        # Low volatility in ranging market = good for mean reversion
        if volatility_regime == MarketRegime.LOW_VOLATILITY:
            return MarketRegime.LOW_VOLATILITY

        # Default to ranging
        return MarketRegime.RANGING

    def get_recommended_strategies(self, regime: MarketRegime) -> list[str]:
        """
        Get recommended strategies for a given regime.

        Args:
            regime: Market regime

        Returns:
            List of recommended strategy names
        """
        recommendations = {
            MarketRegime.TRENDING_UP: ['momentum', 'trend_following'],
            MarketRegime.TRENDING_DOWN: ['momentum', 'short_selling'],
            MarketRegime.RANGING: ['mean_reversion', 'pairs_trading'],
            MarketRegime.HIGH_VOLATILITY: ['conservative', 'reduced_size'],
            MarketRegime.LOW_VOLATILITY: ['mean_reversion', 'carry_trades']
        }

        return recommendations.get(regime, ['balanced'])

    def should_reduce_exposure(self, regime: MarketRegime) -> bool:
        """
        Check if we should reduce exposure in this regime.

        Args:
            regime: Market regime

        Returns:
            True if should reduce exposure
        """
        return regime == MarketRegime.HIGH_VOLATILITY

    def get_position_size_multiplier(self, regime: MarketRegime) -> float:
        """
        Get position size multiplier based on regime.

        Args:
            regime: Market regime

        Returns:
            Multiplier for position sizing (0.5 to 1.5)
        """
        multipliers = {
            MarketRegime.TRENDING_UP: 1.2,  # Slightly larger in strong trends
            MarketRegime.TRENDING_DOWN: 0.8,  # Smaller in downtrends
            MarketRegime.RANGING: 1.0,  # Normal
            MarketRegime.HIGH_VOLATILITY: 0.5,  # Much smaller in high vol
            MarketRegime.LOW_VOLATILITY: 1.3  # Slightly larger in low vol
        }

        return multipliers.get(regime, 1.0)


def test_regime_filter():
    """Test the regime filter."""
    from data.fetcher import DataFetcher

    fetcher = DataFetcher()

    # Generate test data
    nifty_data = fetcher.fetch_nifty_data("2023-01-01", "2024-01-01")
    vix_data = fetcher.fetch_vix_data("2023-01-01", "2024-01-01")

    print(f"Nifty data: {len(nifty_data)} rows")
    print(f"VIX data: {len(vix_data)} rows")

    # Create filter
    regime_filter = RegimeFilter()

    # Test regime detection for various dates
    test_dates = nifty_data.index[-60::20]  # Last 60 days, every 20 days

    print("\nRegime Detection Tests:")
    print("-" * 80)

    for date in test_dates:
        regime, metrics = regime_filter.detect_regime(nifty_data, vix_data, date)

        print(f"\nDate: {date.date()}")
        print(f"Regime: {regime.value}")
        print(f"Trend: {metrics.get('trend', 'N/A')}")
        print(f"Volatility: {metrics.get('volatility', 'N/A')}")
        print(f"MA Diff: {metrics.get('ma_diff_pct', 0):.2f}%")
        print(f"VIX: {metrics.get('vix', metrics.get('historical_volatility', 0)):.2f}")

        strategies = regime_filter.get_recommended_strategies(regime)
        print(f"Recommended strategies: {', '.join(strategies)}")

        size_mult = regime_filter.get_position_size_multiplier(regime)
        print(f"Position size multiplier: {size_mult:.2f}x")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_regime_filter()
