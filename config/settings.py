"""
Configuration settings for the quantitative trading system.

This module contains all configuration parameters including:
- Broker-specific costs and constraints
- Risk management parameters
- Capital allocation settings
- Backtest parameters
"""

from dataclasses import dataclass, field
from typing import List
from enum import Enum


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class ZerodhaConfig:
    """Zerodha broker-specific configuration."""

    # Transaction costs (percentage)
    brokerage_per_trade: float = 0.03  # 0.03% or ₹20 per executed order, whichever is lower
    stt: float = 0.025  # Securities Transaction Tax (0.025% on sell side for equity delivery)
    exchange_txn_charge: float = 0.00325  # NSE: 0.00325%
    gst: float = 18.0  # 18% GST on brokerage + txn charges
    sebi_charges: float = 0.0001  # 0.0001% (₹10 per crore)
    stamp_duty: float = 0.003  # 0.003% on buy side (₹1500 per crore capped at ₹30,000)

    # Combined effective cost per round trip (buy + sell)
    effective_cost_per_round_trip: float = 0.37  # ~0.37-0.40% realistic

    # Slippage model
    slippage_bps: float = 5.0  # 5 basis points (0.05%) average slippage

    # API rate limits
    max_orders_per_second: int = 10
    max_positions: int = 100


@dataclass
class RiskConfig:
    """Risk management parameters."""

    # Position sizing
    max_position_size_pct: float = 5.0  # Max 5% of capital per position
    max_total_exposure_pct: float = 30.0  # Max 30% total capital deployed

    # Stop losses
    max_loss_per_trade_pct: float = 2.0  # Max 2% loss per trade
    max_daily_loss_pct: float = 5.0  # Max 5% daily loss limit
    max_drawdown_pct: float = 15.0  # Max 15% drawdown before stopping

    # Position limits
    max_concurrent_positions: int = 6  # Max 6 positions at once
    min_position_value: float = 5000.0  # Min ₹5000 per position

    # Correlation limits (future enhancement)
    max_correlation: float = 0.7  # Don't take highly correlated positions


@dataclass
class CapitalConfig:
    """Capital allocation and scaling configuration."""

    initial_capital: float = 100000.0  # ₹1 lakh
    target_capital: float = 1000000.0  # ₹10 lakh (scaling target)
    min_cash_reserve_pct: float = 20.0  # Keep 20% in cash

    # Profit reinvestment
    reinvest_profits: bool = True
    reinvest_pct: float = 50.0  # Reinvest 50% of profits


@dataclass
class BacktestConfig:
    """Backtest-specific parameters."""

    # Time periods
    warmup_period_days: int = 200  # Days needed for indicator calculation
    walk_forward_train_months: int = 6  # Train on 6 months
    walk_forward_test_months: int = 1  # Test on 1 month

    # Data
    data_start_date: str = "2020-01-01"
    data_end_date: str = "2024-10-27"

    # Execution simulation
    use_realistic_fills: bool = True  # Model slippage and partial fills
    market_impact_enabled: bool = True  # Model price impact

    # Performance optimization
    enable_multiprocessing: bool = True
    cache_indicators: bool = True


@dataclass
class AgentConfig:
    """Configuration for trading agents."""

    # Momentum Agent (Turtle Trader)
    momentum_lookback_period: int = 55  # Turtle 55-day breakout
    momentum_exit_period: int = 20  # 20-day exit
    momentum_atr_period: int = 20  # ATR period for position sizing
    momentum_atr_multiplier: float = 2.0  # Stop loss = 2 x ATR

    # Mean Reversion Agent (Bollinger + RSI)
    bb_period: int = 20  # 20-day Bollinger Bands
    bb_std_dev: float = 2.0  # 2 standard deviations
    rsi_period: int = 14  # 14-day RSI
    rsi_oversold: float = 30.0  # RSI < 30 = oversold
    rsi_overbought: float = 70.0  # RSI > 70 = overbought

    # Agent weights (for ensemble)
    momentum_weight: float = 0.5
    reversion_weight: float = 0.5


@dataclass
class RegimeConfig:
    """Market regime detection configuration."""

    # Trend detection
    trend_fast_ma: int = 20  # Fast moving average
    trend_slow_ma: int = 50  # Slow moving average
    trend_threshold: float = 0.02  # 2% difference = trending

    # Volatility detection (using Nifty VIX India)
    vix_high_threshold: float = 20.0  # VIX > 20 = high volatility
    vix_low_threshold: float = 12.0  # VIX < 12 = low volatility

    # Lookback for regime classification
    regime_lookback_days: int = 30


@dataclass
class UniverseConfig:
    """Stock universe configuration."""

    # MVP: Start with 10 liquid Nifty stocks
    initial_universe: List[str] = field(default_factory=lambda: [
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "HINDUNILVR",
        "ITC",
        "SBIN",
        "BHARTIARTL",
        "KOTAKBANK"
    ])

    # Liquidity filters
    min_daily_volume: float = 1000000.0  # Min 10 lakh shares daily volume
    min_market_cap_cr: float = 10000.0  # Min ₹10,000 crore market cap

    # Expand to full Nifty 50 after MVP validation
    use_full_nifty50: bool = False


@dataclass
class IntradayConfig:
    """Intraday backtesting configuration."""
    candle_interval: str = "15m"
    session_start: str = "09:15"    # NSE market open
    session_end: str = "15:30"      # NSE market close
    nifty_lot_size: int = 25        # Nifty futures lot size
    force_close_time: str = "15:15" # Force close 15 min before market close
    default_lookback_days: int = 60 # yfinance 15m limit
    brokerage_per_order: float = 40.0  # Zerodha futures flat fee per order


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    url: str = "postgresql://trading_user:trading_password_dev@localhost:5432/trading_system"
    min_pool_size: int = 5
    max_pool_size: int = 20


@dataclass
class SystemConfig:
    """Master configuration combining all sub-configs."""

    zerodha: ZerodhaConfig = field(default_factory=ZerodhaConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    intraday: IntradayConfig = field(default_factory=IntradayConfig)

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/trading_system.log"

    # Performance targets
    target_monthly_return_pct: float = 3.33  # ~10% in 3 months
    target_sharpe_ratio: float = 1.5
    target_max_drawdown_pct: float = 10.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Check capital allocation makes sense
        assert self.capital.initial_capital > 0, "Initial capital must be positive"
        assert self.capital.min_cash_reserve_pct < 100, "Cash reserve cannot be 100%"

        # Check risk parameters
        assert self.risk.max_position_size_pct > 0, "Position size must be positive"
        assert self.risk.max_total_exposure_pct <= 100, "Total exposure cannot exceed 100%"
        assert self.risk.max_loss_per_trade_pct < 100, "Max loss per trade must be < 100%"

        # Check agent weights sum to 1
        total_weight = self.agents.momentum_weight + self.agents.reversion_weight
        assert abs(total_weight - 1.0) < 0.01, f"Agent weights must sum to 1, got {total_weight}"


# Global configuration instance
config = SystemConfig()


def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> SystemConfig:
    """Update configuration parameters dynamically."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config._validate_config()
    return config
