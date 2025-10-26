"""
Data preprocessor for adding technical indicators and preparing data for trading.

This module:
- Adds technical indicators (ATR, Bollinger Bands, RSI, SMA, etc.)
- Cleans and validates data
- Handles missing values
- Prepares data for strategy consumption
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess market data and add technical indicators."""

    def __init__(self):
        """Initialize the preprocessor."""
        pass

    def add_all_indicators(
        self,
        df: pd.DataFrame,
        atr_period: int = 20,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        sma_periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe.

        Args:
            df: DataFrame with OHLCV data
            atr_period: ATR calculation period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation
            rsi_period: RSI calculation period
            sma_periods: List of SMA periods (default: [20, 50, 200])

        Returns:
            DataFrame with added indicators
        """
        df = df.copy()

        if sma_periods is None:
            sma_periods = [20, 50, 200]

        # Add indicators
        df = self.add_atr(df, period=atr_period)
        df = self.add_bollinger_bands(df, period=bb_period, std_dev=bb_std)
        df = self.add_rsi(df, period=rsi_period)

        for period in sma_periods:
            df = self.add_sma(df, period=period)

        # Add basic features
        df = self.add_returns(df)
        df = self.add_volatility(df, window=20)

        logger.info(f"Added all indicators. DataFrame shape: {df.shape}")
        return df

    def add_sma(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Add Simple Moving Average.

        Args:
            df: DataFrame with OHLCV data
            period: Moving average period

        Returns:
            DataFrame with SMA column
        """
        df = df.copy()
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df

    def add_ema(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Add Exponential Moving Average.

        Args:
            df: DataFrame with OHLCV data
            period: EMA period

        Returns:
            DataFrame with EMA column
        """
        df = df.copy()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR) indicator.

        ATR = Moving average of True Range
        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))

        Args:
            df: DataFrame with OHLCV data
            period: ATR period

        Returns:
            DataFrame with ATR column
        """
        df = df.copy()

        # Calculate True Range
        prev_close = df['close'].shift(1)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - prev_close)
        low_close = np.abs(df['low'] - prev_close)

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR is exponential moving average of True Range
        df['atr'] = true_range.ewm(span=period, adjust=False).mean()

        return df

    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands.

        Middle Band = SMA(period)
        Upper Band = SMA(period) + std_dev * StdDev(period)
        Lower Band = SMA(period) - std_dev * StdDev(period)

        Args:
            df: DataFrame with OHLCV data
            period: Moving average period
            std_dev: Number of standard deviations

        Returns:
            DataFrame with BB columns (bb_upper, bb_middle, bb_lower)
        """
        df = df.copy()

        # Middle band (SMA)
        df['bb_middle'] = df['close'].rolling(window=period).mean()

        # Standard deviation
        rolling_std = df['close'].rolling(window=period).std()

        # Upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (std_dev * rolling_std)
        df['bb_lower'] = df['bb_middle'] - (std_dev * rolling_std)

        # Add bandwidth (useful for volatility analysis)
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Add %B (position within bands)
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss

        Args:
            df: DataFrame with OHLCV data
            period: RSI period

        Returns:
            DataFrame with RSI column
        """
        df = df.copy()

        # Calculate price changes
        delta = df['close'].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate exponential moving averages
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def add_macd(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence).

        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD, signal_period)
        Histogram = MACD Line - Signal Line

        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            DataFrame with MACD columns
        """
        df = df.copy()

        # Calculate EMAs
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()

        # MACD line
        df['macd'] = ema_fast - ema_slow

        # Signal line
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()

        # Histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add return calculations.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with returns columns
        """
        df = df.copy()

        # Simple returns
        df['returns'] = df['close'].pct_change()

        # Log returns (better for analysis)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        return df

    def add_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Add rolling volatility (standard deviation of returns).

        Args:
            df: DataFrame with OHLCV data
            window: Rolling window for volatility calculation

        Returns:
            DataFrame with volatility column
        """
        df = df.copy()

        if 'returns' not in df.columns:
            df = self.add_returns(df)

        df['volatility'] = df['returns'].rolling(window=window).std() * np.sqrt(252)  # Annualized

        return df

    def add_high_low_channels(
        self,
        df: pd.DataFrame,
        lookback_period: int = 55,
        exit_period: int = 20
    ) -> pd.DataFrame:
        """
        Add high/low channels for Turtle Trader strategy.

        Args:
            df: DataFrame with OHLCV data
            lookback_period: Period for breakout (typically 55)
            exit_period: Period for exit (typically 20)

        Returns:
            DataFrame with channel columns
        """
        df = df.copy()

        # Entry channels (55-day high/low)
        df['channel_high'] = df['high'].rolling(window=lookback_period).max()
        df['channel_low'] = df['low'].rolling(window=lookback_period).min()

        # Exit channels (20-day high/low)
        df['exit_high'] = df['high'].rolling(window=exit_period).max()
        df['exit_low'] = df['low'].rolling(window=exit_period).min()

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.

        - Remove duplicates
        - Handle missing values
        - Remove invalid prices
        - Ensure proper data types

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        original_len = len(df)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Remove rows with invalid prices
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

        # Ensure OHLC relationships
        df = df[df['high'] >= df['low']]
        df = df[df['high'] >= df['open']]
        df = df[df['high'] >= df['close']]
        df = df[df['low'] <= df['open']]
        df = df[df['low'] <= df['close']]

        # Forward fill missing values (max 5 days)
        df = df.ffill(limit=5)

        # Drop remaining NaN rows
        df = df.dropna()

        cleaned_len = len(df)
        if cleaned_len < original_len:
            logger.warning(f"Cleaned data: {original_len} -> {cleaned_len} rows ({original_len - cleaned_len} removed)")

        return df

    def prepare_for_backtest(
        self,
        df: pd.DataFrame,
        warmup_days: int = 200
    ) -> pd.DataFrame:
        """
        Prepare data for backtesting by adding all indicators and cleaning.

        Args:
            df: Raw OHLCV DataFrame
            warmup_days: Days needed for indicator warmup

        Returns:
            Prepared DataFrame ready for backtesting
        """
        # Clean data
        df = self.clean_data(df)

        # Add all indicators
        df = self.add_all_indicators(df)
        df = self.add_high_low_channels(df)
        df = self.add_macd(df)

        # Drop warmup period (rows with NaN indicators)
        df = df.dropna()

        logger.info(f"Data prepared for backtest: {len(df)} rows after warmup")
        return df


def test_preprocessor():
    """Test the preprocessor with sample data."""
    from data.fetcher import DataFetcher

    # Generate sample data
    fetcher = DataFetcher()
    df = fetcher.fetch_historical_data("TEST", "2023-01-01", "2024-01-01")

    print("\nOriginal data:")
    print(df.head())

    # Preprocess
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.prepare_for_backtest(df)

    print("\nProcessed data with indicators:")
    print(df_processed.head())
    print(f"\nColumns: {df_processed.columns.tolist()}")
    print(f"\nShape: {df_processed.shape}")

    # Check indicators
    print("\nSample indicator values:")
    print(df_processed[['close', 'sma_20', 'rsi', 'atr', 'bb_upper', 'bb_lower']].tail())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_preprocessor()
