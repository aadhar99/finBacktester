"""
Base agent class for trading strategies.

All trading agents inherit from this abstract base class and implement:
- Signal generation (buy/sell/hold)
- Position sizing
- Exit logic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"


@dataclass
class Signal:
    """Trading signal with metadata."""
    signal_type: SignalType
    symbol: str
    timestamp: pd.Timestamp
    price: float
    size: int  # Number of shares
    confidence: float = 1.0  # Signal confidence (0-1)
    reason: str = ""  # Why this signal was generated
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """
    Abstract base class for trading agents.

    Each agent implements a specific trading strategy (momentum, mean reversion, etc.)
    and generates trading signals based on market data.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the agent.

        Args:
            name: Agent name (e.g., "MomentumAgent")
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.logger.info(f"Initialized {name}")

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        current_positions: Dict[str, int],
        portfolio_value: float,
        market_regime: Optional[str] = None
    ) -> list[Signal]:
        """
        Generate trading signals based on current market data.

        This is the core method that each agent must implement.

        Args:
            data: DataFrame with OHLCV and indicator data (latest row is current)
            current_positions: Dictionary of current positions {symbol: quantity}
            portfolio_value: Current total portfolio value
            market_regime: Current market regime (optional, for regime filtering)

        Returns:
            List of Signal objects
        """
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        volatility: float,
        max_position_pct: float = 5.0
    ) -> int:
        """
        Calculate position size for a trade.

        Args:
            symbol: Stock symbol
            price: Current price
            portfolio_value: Total portfolio value
            volatility: Asset volatility (ATR or similar)
            max_position_pct: Maximum position size as % of portfolio

        Returns:
            Number of shares to trade
        """
        pass

    @abstractmethod
    def should_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        current_data: pd.Series,
        days_held: int
    ) -> Tuple[bool, str]:
        """
        Determine if we should exit an existing position.

        Args:
            symbol: Stock symbol
            entry_price: Entry price of the position
            current_price: Current market price
            current_data: Current row of market data
            days_held: Number of days position has been held

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        pass

    def is_enabled_for_regime(self, market_regime: Optional[str]) -> bool:
        """
        Check if this agent should trade in the current market regime.

        Override this method to implement regime filtering.

        Args:
            market_regime: Current market regime

        Returns:
            True if agent should trade, False otherwise
        """
        # By default, trade in all regimes
        return True

    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate a signal before sending it to execution.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid, False otherwise
        """
        # Basic validation
        if signal.price <= 0:
            self.logger.warning(f"Invalid signal: price <= 0")
            return False

        if signal.size <= 0:
            self.logger.warning(f"Invalid signal: size <= 0")
            return False

        if not 0 <= signal.confidence <= 1:
            self.logger.warning(f"Invalid signal: confidence not in [0, 1]")
            return False

        return True

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        multiplier: float = 2.0,
        is_long: bool = True
    ) -> float:
        """
        Calculate stop loss price based on ATR.

        Args:
            entry_price: Entry price
            atr: Average True Range
            multiplier: ATR multiplier (typically 2-3)
            is_long: True for long positions, False for short

        Returns:
            Stop loss price
        """
        if is_long:
            return entry_price - (multiplier * atr)
        else:
            return entry_price + (multiplier * atr)

    def calculate_take_profit(
        self,
        entry_price: float,
        risk: float,
        reward_ratio: float = 2.0,
        is_long: bool = True
    ) -> float:
        """
        Calculate take profit price based on risk-reward ratio.

        Args:
            entry_price: Entry price
            risk: Risk amount (entry_price - stop_loss for long)
            reward_ratio: Reward-to-risk ratio (typically 2-3)
            is_long: True for long positions, False for short

        Returns:
            Take profit price
        """
        if is_long:
            return entry_price + (risk * reward_ratio)
        else:
            return entry_price - (risk * reward_ratio)

    def get_strategy_description(self) -> str:
        """
        Get a human-readable description of this strategy.

        Returns:
            Strategy description
        """
        return f"{self.name}: {self.__class__.__doc__}"

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"


class EnsembleAgent(BaseAgent):
    """
    Ensemble agent that combines signals from multiple sub-agents.

    This agent runs multiple strategies in parallel and aggregates their signals
    based on configured weights.
    """

    def __init__(
        self,
        name: str = "EnsembleAgent",
        agents: Optional[list[BaseAgent]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the ensemble agent.

        Args:
            name: Agent name
            agents: List of sub-agents to combine
            weights: Dictionary mapping agent names to weights
        """
        super().__init__(name)
        self.agents = agents or []
        self.weights = weights or {}

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def add_agent(self, agent: BaseAgent, weight: float = 1.0):
        """Add a sub-agent to the ensemble."""
        self.agents.append(agent)
        self.weights[agent.name] = weight
        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_positions: Dict[str, int],
        portfolio_value: float,
        market_regime: Optional[str] = None
    ) -> list[Signal]:
        """
        Generate aggregated signals from all sub-agents.

        Args:
            data: Market data
            current_positions: Current positions
            portfolio_value: Portfolio value
            market_regime: Market regime

        Returns:
            List of aggregated signals
        """
        all_signals = []

        # Collect signals from each agent
        for agent in self.agents:
            if agent.is_enabled_for_regime(market_regime):
                try:
                    signals = agent.generate_signals(
                        data, current_positions, portfolio_value, market_regime
                    )
                    # Weight the signals
                    weight = self.weights.get(agent.name, 0)
                    for signal in signals:
                        signal.confidence *= weight
                        signal.metadata['source_agent'] = agent.name
                        all_signals.append(signal)
                except Exception as e:
                    self.logger.error(f"Error generating signals from {agent.name}: {e}")

        # Aggregate signals by symbol
        aggregated = self._aggregate_signals(all_signals)
        return aggregated

    def _aggregate_signals(self, signals: list[Signal]) -> list[Signal]:
        """
        Aggregate multiple signals for the same symbol.

        Args:
            signals: List of signals

        Returns:
            List of aggregated signals
        """
        # Group by symbol
        by_symbol = {}
        for signal in signals:
            if signal.symbol not in by_symbol:
                by_symbol[signal.symbol] = []
            by_symbol[signal.symbol].append(signal)

        aggregated = []
        for symbol, symbol_signals in by_symbol.items():
            # Aggregate confidence scores
            buy_confidence = sum(s.confidence for s in symbol_signals if s.signal_type == SignalType.BUY)
            sell_confidence = sum(s.confidence for s in symbol_signals if s.signal_type == SignalType.SELL)

            # Determine final signal
            if buy_confidence > sell_confidence and buy_confidence > 0.3:  # Threshold
                # Take the strongest buy signal
                strongest = max(
                    [s for s in symbol_signals if s.signal_type == SignalType.BUY],
                    key=lambda s: s.confidence
                )
                strongest.confidence = buy_confidence
                aggregated.append(strongest)
            elif sell_confidence > buy_confidence and sell_confidence > 0.3:
                strongest = max(
                    [s for s in symbol_signals if s.signal_type == SignalType.SELL],
                    key=lambda s: s.confidence
                )
                strongest.confidence = sell_confidence
                aggregated.append(strongest)

        return aggregated

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        volatility: float,
        max_position_pct: float = 5.0
    ) -> int:
        """Calculate position size (delegates to first agent)."""
        if self.agents:
            return self.agents[0].calculate_position_size(
                symbol, price, portfolio_value, volatility, max_position_pct
            )
        return 0

    def should_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        current_data: pd.Series,
        days_held: int
    ) -> Tuple[bool, str]:
        """Check if should exit (aggregates from all agents)."""
        exit_votes = 0
        reasons = []

        for agent in self.agents:
            should_exit, reason = agent.should_exit(
                symbol, entry_price, current_price, current_data, days_held
            )
            if should_exit:
                exit_votes += self.weights.get(agent.name, 0)
                reasons.append(f"{agent.name}: {reason}")

        if exit_votes > 0.5:  # Majority vote
            return True, "; ".join(reasons)

        return False, ""
