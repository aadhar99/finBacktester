"""
Data fetcher module for retrieving market data.

This module handles data fetching from various sources (in priority order):
1. yfinance API for real NSE historical data (PRIMARY)
2. Zerodha Kite API for live/historical data
3. CSV files for backtesting
4. Synthetic data generation (FALLBACK ONLY)

Uses real market data by default for realistic backtesting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    logger.info("yfinance is available - will use real market data")
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available - install with: pip install yfinance")


class DataFetcher:
    """Fetch market data from various sources."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        data_dir: str = "data/historical"
    ):
        """
        Initialize the data fetcher.

        Args:
            api_key: Zerodha API key (optional for backtest mode)
            access_token: Zerodha access token (optional for backtest mode)
            data_dir: Directory to store/load historical data
        """
        self.api_key = api_key
        self.access_token = access_token
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Kite connection if credentials provided
        self.kite = None
        if api_key and access_token:
            try:
                from kiteconnect import KiteConnect
                self.kite = KiteConnect(api_key=api_key)
                self.kite.set_access_token(access_token)
                logger.info("Successfully connected to Zerodha Kite API")
            except ImportError:
                logger.warning("kiteconnect not installed. Install with: pip install kiteconnect")
            except Exception as e:
                logger.error(f"Failed to initialize Kite API: {e}")

    def fetch_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "day"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.

        Data source priority:
        1. Cache (if available)
        2. yfinance API (real NSE data) - PRIMARY
        3. Zerodha Kite API (if credentials provided)
        4. Synthetic data (fallback only)

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('day', 'minute', 'hour')

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        # Try to load from cache first
        cache_file = self.data_dir / f"{symbol}_{start_date}_{end_date}_{interval}.csv"

        if cache_file.exists():
            logger.info(f"✓ Loading cached data for {symbol}")
            return pd.read_csv(cache_file, parse_dates=['date'], index_col='date')

        # Try yfinance first (REAL MARKET DATA)
        if YFINANCE_AVAILABLE:
            try:
                data = self._fetch_from_yfinance(symbol, start_date, end_date, interval)
                if data is not None and len(data) > 0:
                    # Cache the data
                    data.to_csv(cache_file)
                    logger.info(f"✓ Fetched REAL data for {symbol} from yfinance ({len(data)} days)")
                    return data
            except Exception as e:
                logger.warning(f"yfinance fetch failed for {symbol}: {e}")

        # Try fetching from Kite API
        if self.kite:
            try:
                data = self._fetch_from_kite(symbol, start_date, end_date, interval)
                # Cache the data
                data.to_csv(cache_file)
                logger.info(f"✓ Fetched data for {symbol} from Kite API")
                return data
            except Exception as e:
                logger.error(f"Kite API fetch failed for {symbol}: {e}")

        # Fallback: Generate synthetic data (ONLY IF NOTHING ELSE WORKS)
        logger.warning(f"⚠️  Using SYNTHETIC data for {symbol} - install yfinance for real data!")
        data = self._generate_synthetic_data(symbol, start_date, end_date)

        # Cache synthetic data
        data.to_csv(cache_file)
        return data

    def _fetch_from_yfinance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "day"
    ) -> pd.DataFrame:
        """
        Fetch real market data from yfinance (Yahoo Finance).

        For NSE stocks, appends .NS to symbol (e.g., RELIANCE.NS).
        For Nifty index, uses ^NSEI.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d' for day, '1h' for hour, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        # Convert symbol to yfinance format
        if symbol == "NIFTY50" or symbol == "NIFTY":
            yf_symbol = "^NSEI"  # Nifty 50 index
        elif symbol == "BANKNIFTY":
            yf_symbol = "^NSEBANK"  # Bank Nifty index
        elif symbol.startswith("^"):
            yf_symbol = symbol  # Already in index format
        else:
            yf_symbol = f"{symbol}.NS"  # NSE stocks

        # Convert interval to yfinance format
        interval_map = {
            "day": "1d",
            "hour": "1h",
            "minute": "1m"
        }
        yf_interval = interval_map.get(interval, "1d")

        logger.info(f"Fetching {yf_symbol} from yfinance ({start_date} to {end_date})...")

        # Fetch data using yfinance
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=yf_interval,
            auto_adjust=False  # Keep raw prices
        )

        if df.empty:
            logger.warning(f"No data returned from yfinance for {yf_symbol}")
            return None

        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()

        # Rename columns to match our format
        column_map = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }

        # Select and rename only the columns we need
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()

        # Ensure index is named 'date'
        df.index.name = 'date'

        # Remove timezone info if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Data quality checks
        df = self._validate_ohlcv_data(df, symbol)

        return df

    def _validate_ohlcv_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol (for logging)

        Returns:
            Cleaned DataFrame
        """
        original_len = len(df)

        # Remove rows with NaN values
        df = df.dropna()

        # Remove rows with zero or negative prices
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

        # Validate OHLC relationships
        df = df[df['high'] >= df['low']]
        df = df[df['high'] >= df['open']]
        df = df[df['high'] >= df['close']]
        df = df[df['low'] <= df['open']]
        df = df[df['low'] <= df['close']]

        # Remove extreme outliers (>50% daily change)
        returns = df['close'].pct_change()
        df = df[abs(returns) < 0.5]

        cleaned_len = len(df)
        if cleaned_len < original_len:
            removed = original_len - cleaned_len
            logger.info(f"Cleaned {symbol}: removed {removed} invalid rows ({original_len} -> {cleaned_len})")

        return df

    def _fetch_from_kite(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Fetch data from Zerodha Kite API.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            DataFrame with OHLCV data
        """
        from datetime import datetime

        # Convert symbol to Kite instrument token format
        # For NSE stocks, format is typically NSE:SYMBOL
        instrument_token = f"NSE:{symbol}"

        # Convert date strings to datetime
        from_date = datetime.strptime(start_date, "%Y-%m-%d")
        to_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Fetch historical data
        records = self.kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )

        # Convert to DataFrame
        df = pd.DataFrame(records)
        df.rename(columns={
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }, inplace=True)

        df.set_index('date', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]

    def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        base_price: float = 1000.0
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic OHLCV data for testing.

        Uses geometric Brownian motion with realistic parameters:
        - Drift: ~15% annual return
        - Volatility: ~25% annual
        - Volume: Random walk around base volume

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            base_price: Starting price

        Returns:
            DataFrame with synthetic OHLCV data
        """
        # Generate date range (trading days only)
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.bdate_range(start=start, end=end)

        n_days = len(dates)

        # Parameters for geometric Brownian motion
        annual_return = 0.15  # 15% annual return
        annual_volatility = 0.25  # 25% annual volatility
        dt = 1 / 252  # Daily time step (252 trading days/year)

        # Generate returns
        drift = (annual_return - 0.5 * annual_volatility**2) * dt
        shock = annual_volatility * np.sqrt(dt) * np.random.randn(n_days)
        returns = drift + shock

        # Generate close prices
        close = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC from close
        daily_volatility = annual_volatility * np.sqrt(dt)
        high = close * (1 + np.abs(np.random.randn(n_days)) * daily_volatility * 0.5)
        low = close * (1 - np.abs(np.random.randn(n_days)) * daily_volatility * 0.5)

        # Open is previous close with small gap
        open_prices = np.roll(close, 1)
        open_prices[0] = base_price
        open_prices = open_prices * (1 + np.random.randn(n_days) * 0.002)  # Small gaps

        # Ensure OHLC consistency
        for i in range(n_days):
            high[i] = max(high[i], open_prices[i], close[i])
            low[i] = min(low[i], open_prices[i], close[i])

        # Generate realistic volume (around 1M shares/day with variation)
        base_volume = 1_000_000
        volume = base_volume * (1 + 0.5 * np.random.randn(n_days))
        volume = np.maximum(volume, 100_000)  # Minimum 100k volume

        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume.astype(int)
        })

        df.set_index('date', inplace=True)
        logger.info(f"Generated {n_days} days of synthetic data for {symbol}")
        return df

    def fetch_nifty_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch Nifty 50 index data for regime detection.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with Nifty OHLCV data
        """
        return self.fetch_historical_data("NIFTY50", start_date, end_date)

    def fetch_vix_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch India VIX data for volatility regime detection.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with VIX data
        """
        # VIX data - if not available from Kite, generate synthetic
        cache_file = self.data_dir / f"INDIAVIX_{start_date}_{end_date}.csv"

        if cache_file.exists():
            return pd.read_csv(cache_file, parse_dates=['date'], index_col='date')

        # Generate synthetic VIX (mean-reverting process)
        dates = pd.bdate_range(start=start_date, end=end_date)
        n_days = len(dates)

        # VIX typically ranges 10-40, mean around 15-18
        vix_mean = 16.0
        vix_volatility = 5.0
        mean_reversion_speed = 0.3

        vix = np.zeros(n_days)
        vix[0] = vix_mean

        for i in range(1, n_days):
            # Mean-reverting Ornstein-Uhlenbeck process
            drift = mean_reversion_speed * (vix_mean - vix[i-1])
            shock = vix_volatility * np.random.randn()
            vix[i] = max(8, min(50, vix[i-1] + drift + shock))  # Bound between 8-50

        df = pd.DataFrame({
            'date': dates,
            'vix': vix
        })
        df.set_index('date', inplace=True)

        df.to_csv(cache_file)
        return df

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.fetch_historical_data(symbol, start_date, end_date)
                logger.info(f"Successfully fetched data for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")

        return data


def test_data_fetcher():
    """Test the data fetcher with synthetic data."""
    fetcher = DataFetcher()

    # Test single symbol
    symbol = "RELIANCE"
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    df = fetcher.fetch_historical_data(symbol, start_date, end_date)
    print(f"\nFetched {len(df)} days of data for {symbol}")
    print(df.head())
    print(f"\nData range: {df.index.min()} to {df.index.max()}")
    print(f"Price range: ₹{df['close'].min():.2f} to ₹{df['close'].max():.2f}")

    # Test multiple symbols
    symbols = ["RELIANCE", "TCS", "INFY"]
    data = fetcher.fetch_multiple_symbols(symbols, start_date, end_date)
    print(f"\nFetched data for {len(data)} symbols")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_data_fetcher()
