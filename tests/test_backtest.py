"""
Unit tests for the trading system.

Run with: pytest tests/test_backtest.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import get_config
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from agents import MomentumAgent, ReversionAgent
from execution import Portfolio, BacktestEngine
from risk import RiskManager
from metrics import MetricsCalculator


class TestDataFetcher:
    """Test data fetching and generation."""

    def test_synthetic_data_generation(self):
        """Test that synthetic data is generated correctly."""
        fetcher = DataFetcher()
        df = fetcher.fetch_historical_data("TEST", "2023-01-01", "2023-12-31")

        assert len(df) > 0
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert (df['high'] >= df['low']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['low'] <= df['close']).all()

    def test_vix_data_generation(self):
        """Test VIX data generation."""
        fetcher = DataFetcher()
        vix_df = fetcher.fetch_vix_data("2023-01-01", "2023-12-31")

        assert len(vix_df) > 0
        assert 'vix' in vix_df.columns
        assert (vix_df['vix'] > 0).all()


class TestDataPreprocessor:
    """Test data preprocessing and indicators."""

    def test_add_indicators(self):
        """Test that indicators are added correctly."""
        fetcher = DataFetcher()
        preprocessor = DataPreprocessor()

        df = fetcher.fetch_historical_data("TEST", "2023-01-01", "2023-12-31")
        df = preprocessor.add_all_indicators(df)

        required_indicators = ['sma_20', 'rsi', 'atr', 'bb_upper', 'bb_lower']
        for indicator in required_indicators:
            assert indicator in df.columns
            assert not df[indicator].isna().all()

    def test_clean_data(self):
        """Test data cleaning."""
        preprocessor = DataPreprocessor()

        # Create dirty data
        df = pd.DataFrame({
            'open': [100, 101, np.nan, 103],
            'high': [105, 106, 107, 108],
            'low': [95, 96, 97, 98],
            'close': [102, 103, 104, 105],
            'volume': [1000, 1000, 1000, 1000]
        })

        df_clean = preprocessor.clean_data(df)
        assert df_clean.isna().sum().sum() == 0


class TestAgents:
    """Test trading agents."""

    def test_momentum_agent_signal_generation(self):
        """Test momentum agent generates signals."""
        fetcher = DataFetcher()
        preprocessor = DataPreprocessor()
        agent = MomentumAgent()

        df = fetcher.fetch_historical_data("TEST", "2023-01-01", "2023-12-31")
        df = preprocessor.prepare_for_backtest(df)

        signals = agent.generate_signals(
            data=df,
            current_positions={},
            portfolio_value=100000,
            market_regime='trending_up'
        )

        assert isinstance(signals, list)

    def test_reversion_agent_signal_generation(self):
        """Test reversion agent generates signals."""
        fetcher = DataFetcher()
        preprocessor = DataPreprocessor()
        agent = ReversionAgent()

        df = fetcher.fetch_historical_data("TEST", "2023-01-01", "2023-12-31")
        df = preprocessor.prepare_for_backtest(df)

        signals = agent.generate_signals(
            data=df,
            current_positions={},
            portfolio_value=100000,
            market_regime='ranging'
        )

        assert isinstance(signals, list)


class TestPortfolio:
    """Test portfolio management."""

    def test_portfolio_initialization(self):
        """Test portfolio is initialized correctly."""
        portfolio = Portfolio(100000)

        assert portfolio.initial_capital == 100000
        assert portfolio.cash == 100000
        assert portfolio.total_value == 100000

    def test_buy_execution(self):
        """Test buying shares."""
        portfolio = Portfolio(100000)

        success = portfolio.execute_buy(
            symbol="TEST",
            price=100,
            quantity=10,
            timestamp=pd.Timestamp("2023-01-01"),
            commission_pct=0.1
        )

        assert success
        assert portfolio.cash < 100000
        assert portfolio.position_tracker.has_position("TEST")

    def test_sell_execution(self):
        """Test selling shares."""
        portfolio = Portfolio(100000)

        # First buy
        portfolio.execute_buy(
            symbol="TEST",
            price=100,
            quantity=10,
            timestamp=pd.Timestamp("2023-01-01")
        )

        # Then sell
        success = portfolio.execute_sell(
            symbol="TEST",
            price=110,
            timestamp=pd.Timestamp("2023-01-02"),
            reason="Test exit"
        )

        assert success
        assert not portfolio.position_tracker.has_position("TEST")
        assert portfolio.cash > 100000  # Made profit


class TestRiskManager:
    """Test risk management."""

    def test_risk_manager_initialization(self):
        """Test risk manager initializes correctly."""
        portfolio = Portfolio(100000)
        risk_mgr = RiskManager(portfolio)

        assert risk_mgr.portfolio == portfolio
        assert not risk_mgr.trading_halted

    def test_position_size_check(self):
        """Test position size limits."""
        portfolio = Portfolio(100000)
        risk_mgr = RiskManager(portfolio)

        from agents.base_agent import Signal, SignalType

        # Test oversized position
        big_signal = Signal(
            signal_type=SignalType.BUY,
            symbol="TEST",
            timestamp=pd.Timestamp.now(),
            price=100,
            size=1000,  # ₹100,000 position (100%)
            confidence=1.0
        )

        approved, reason = risk_mgr.check_signal(big_signal, {}, pd.Timestamp.now())
        assert not approved  # Should be rejected


class TestMetricsCalculator:
    """Test metrics calculation."""

    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        calc = MetricsCalculator()

        # Create sample data
        portfolio_values = pd.Series(
            [100000, 102000, 101000, 105000, 110000],
            index=pd.date_range('2023-01-01', periods=5)
        )

        trades = pd.DataFrame()
        closed_positions = []

        metrics = calc.calculate_all_metrics(
            portfolio_values=portfolio_values,
            trades=trades,
            closed_positions=closed_positions,
            initial_capital=100000
        )

        assert metrics.total_return_pct == 10.0  # 10% return
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown_pct, float)


class TestBacktestEngine:
    """Test backtest engine."""

    @pytest.mark.slow
    def test_simple_backtest(self):
        """Test running a simple backtest."""
        agent = MomentumAgent()

        engine = BacktestEngine(
            initial_capital=100000,
            agents=[agent],
            start_date="2023-01-01",
            end_date="2023-03-31",  # Short period for testing
            symbols=["RELIANCE", "TCS"],
            enable_regime_filter=False
        )

        metrics = engine.run()

        assert metrics is not None
        assert isinstance(metrics.total_return_pct, float)
        assert metrics.total_trades >= 0


def test_config_validation():
    """Test configuration validation."""
    config = get_config()

    assert config.capital.initial_capital > 0
    assert config.risk.max_position_size_pct > 0
    assert config.risk.max_position_size_pct <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
