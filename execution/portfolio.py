"""
Portfolio management module.

Manages cash, positions, and overall portfolio state including:
- Cash management and allocation
- Portfolio valuation
- Performance tracking
- Transaction history
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import logging

from execution.position import Position, PositionTracker
from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Snapshot of portfolio state at a point in time."""
    timestamp: pd.Timestamp
    cash: float
    positions_value: float
    total_value: float
    daily_pnl: float = 0.0
    daily_return_pct: float = 0.0
    positions_count: int = 0


class Portfolio:
    """
    Portfolio manager handling cash, positions, and P&L.

    Tracks:
    - Cash balance
    - Open positions (via PositionTracker)
    - Historical portfolio values
    - Performance metrics
    """

    def __init__(self, initial_capital: float):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_tracker = PositionTracker()
        self.config = get_config()

        # Historical tracking
        self.value_history: List[PortfolioState] = []
        self.trade_history: List[dict] = []

        # Performance metrics
        self.peak_value = initial_capital
        self.peak_date: Optional[pd.Timestamp] = None

        logger.info(f"Initialized portfolio with ₹{initial_capital:,.0f}")

    def update_portfolio_value(self, prices: Dict[str, float], timestamp: pd.Timestamp):
        """
        Update portfolio valuation with current prices.

        Args:
            prices: Dictionary of current prices {symbol: price}
            timestamp: Current timestamp
        """
        # Update position prices
        self.position_tracker.update_prices(prices, timestamp)

        # Calculate total value
        positions_value = self.position_tracker.get_total_exposure(prices)
        total_value = self.cash + positions_value

        # Calculate daily P&L and return
        daily_pnl = 0.0
        daily_return_pct = 0.0
        if len(self.value_history) > 0:
            prev_value = self.value_history[-1].total_value
            daily_pnl = total_value - prev_value
            daily_return_pct = (daily_pnl / prev_value) * 100 if prev_value > 0 else 0.0

        # Track peak
        if total_value > self.peak_value:
            self.peak_value = total_value
            self.peak_date = timestamp

        # Record state
        state = PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=positions_value,
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            positions_count=self.position_tracker.get_position_count()
        )
        self.value_history.append(state)

    def execute_buy(
        self,
        symbol: str,
        price: float,
        quantity: int,
        timestamp: pd.Timestamp,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: str = "unknown",
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0
    ) -> bool:
        """
        Execute a buy order.

        Args:
            symbol: Stock symbol
            price: Buy price
            quantity: Number of shares
            timestamp: Execution timestamp
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy: Strategy name
            commission_pct: Commission percentage
            slippage_pct: Slippage percentage

        Returns:
            True if executed successfully
        """
        # Calculate costs
        base_cost = price * quantity
        commission = base_cost * (commission_pct / 100)
        slippage_cost = (price * (slippage_pct / 100)) * quantity
        total_cost = base_cost + commission + slippage_cost

        # Check if we have enough cash
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash: need ₹{total_cost:.2f}, have ₹{self.cash:.2f}")
            return False

        # Check if we already have a position
        if self.position_tracker.has_position(symbol):
            logger.warning(f"Already have position in {symbol}, skipping buy")
            return False

        # Deduct cash
        self.cash -= total_cost

        # Open position
        self.position_tracker.open_position(
            symbol=symbol,
            entry_price=price,
            quantity=quantity,
            timestamp=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy,
            commission=commission,
            slippage=slippage_cost / quantity if quantity > 0 else 0
        )

        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'BUY',
            'price': price,
            'quantity': quantity,
            'value': base_cost,
            'commission': commission,
            'slippage': slippage_cost,
            'total_cost': total_cost,
            'strategy': strategy
        }
        self.trade_history.append(trade_record)

        logger.info(f"BUY: {symbol} @ ₹{price:.2f} x {quantity} = ₹{total_cost:.2f}")
        return True

    def execute_sell(
        self,
        symbol: str,
        price: float,
        timestamp: pd.Timestamp,
        reason: str = "",
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0
    ) -> bool:
        """
        Execute a sell order (close position).

        Args:
            symbol: Stock symbol
            price: Sell price
            timestamp: Execution timestamp
            reason: Reason for selling
            commission_pct: Commission percentage
            slippage_pct: Slippage percentage

        Returns:
            True if executed successfully
        """
        # Check if we have the position
        if not self.position_tracker.has_position(symbol):
            logger.warning(f"No position to sell: {symbol}")
            return False

        position = self.position_tracker.get_position(symbol)
        quantity = position.quantity

        # Calculate proceeds
        base_proceeds = price * quantity
        commission = base_proceeds * (commission_pct / 100)
        slippage_cost = (price * (slippage_pct / 100)) * quantity
        total_proceeds = base_proceeds - commission - slippage_cost

        # Add cash
        self.cash += total_proceeds

        # Close position
        closed_position = self.position_tracker.close_position(
            symbol=symbol,
            exit_price=price,
            timestamp=timestamp,
            reason=reason,
            commission=commission,
            slippage=slippage_cost / quantity if quantity > 0 else 0
        )

        # Record trade
        if closed_position:
            trade_record = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'SELL',
                'price': price,
                'quantity': quantity,
                'value': base_proceeds,
                'commission': commission,
                'slippage': slippage_cost,
                'total_proceeds': total_proceeds,
                'pnl': closed_position.realized_pnl,
                'pnl_pct': closed_position.realized_pnl_pct,
                'reason': reason,
                'strategy': closed_position.strategy,
                'days_held': closed_position.days_held
            }
            self.trade_history.append(trade_record)

            logger.info(f"SELL: {symbol} @ ₹{price:.2f} x {quantity} = ₹{total_proceeds:.2f}, P&L: ₹{closed_position.realized_pnl:.2f} ({closed_position.realized_pnl_pct:.2f}%)")

        return True

    @property
    def total_value(self) -> float:
        """Get current total portfolio value."""
        if len(self.value_history) > 0:
            return self.value_history[-1].total_value
        return self.initial_capital

    @property
    def total_return(self) -> float:
        """Total return since inception (%)."""
        if self.initial_capital == 0:
            return 0.0
        return ((self.total_value - self.initial_capital) / self.initial_capital) * 100

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak (%)."""
        if self.peak_value == 0:
            return 0.0
        return ((self.total_value - self.peak_value) / self.peak_value) * 100

    @property
    def available_cash_pct(self) -> float:
        """Available cash as percentage of total value."""
        if self.total_value == 0:
            return 0.0
        return (self.cash / self.total_value) * 100

    def get_allocation(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Get current portfolio allocation.

        Args:
            prices: Current prices

        Returns:
            Dictionary of {symbol: allocation_pct}
        """
        allocation = {}
        total = self.total_value

        if total == 0:
            return allocation

        # Cash allocation
        allocation['CASH'] = (self.cash / total) * 100

        # Position allocations
        for symbol, position in self.position_tracker.open_positions.items():
            if symbol in prices:
                position_value = position.quantity * prices[symbol]
                allocation[symbol] = (position_value / total) * 100

        return allocation

    def can_take_position(
        self,
        position_value: float,
        prices: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Check if we can take a new position.

        Validates against:
        - Maximum concurrent positions
        - Maximum total exposure
        - Minimum cash reserve

        Args:
            position_value: Value of proposed position
            prices: Current prices

        Returns:
            (can_take, reason)
        """
        # Check position count
        if self.position_tracker.get_position_count() >= self.config.risk.max_concurrent_positions:
            return False, f"Max positions ({self.config.risk.max_concurrent_positions}) reached"

        # Check if we have enough cash
        if position_value > self.cash:
            return False, f"Insufficient cash: need ₹{position_value:.0f}, have ₹{self.cash:.0f}"

        # Check total exposure limit
        current_exposure = self.position_tracker.get_total_exposure(prices)
        new_exposure = current_exposure + position_value
        max_exposure = self.total_value * (self.config.risk.max_total_exposure_pct / 100)

        if new_exposure > max_exposure:
            return False, f"Would exceed max exposure: {(new_exposure/self.total_value)*100:.1f}% > {self.config.risk.max_total_exposure_pct}%"

        # Check cash reserve
        cash_after = self.cash - position_value
        min_cash_reserve = self.total_value * (self.config.capital.min_cash_reserve_pct / 100)

        if cash_after < min_cash_reserve:
            return False, f"Would violate cash reserve: ₹{cash_after:.0f} < ₹{min_cash_reserve:.0f}"

        return True, "OK"

    def get_value_series(self) -> pd.Series:
        """Get historical portfolio values as pandas Series."""
        if not self.value_history:
            return pd.Series()

        return pd.Series(
            [state.total_value for state in self.value_history],
            index=[state.timestamp for state in self.value_history]
        )

    def get_returns_series(self) -> pd.Series:
        """Get daily returns as pandas Series."""
        value_series = self.get_value_series()
        return value_series.pct_change().fillna(0)

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get all trades as DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()

        return pd.DataFrame(self.trade_history)

    def get_summary(self, prices: Dict[str, float]) -> Dict:
        """
        Get portfolio summary.

        Args:
            prices: Current prices

        Returns:
            Summary dictionary
        """
        return {
            'initial_capital': self.initial_capital,
            'current_value': self.total_value,
            'cash': self.cash,
            'cash_pct': self.available_cash_pct,
            'positions_value': self.position_tracker.get_total_exposure(prices),
            'positions_count': self.position_tracker.get_position_count(),
            'total_return_pct': self.total_return,
            'peak_value': self.peak_value,
            'current_drawdown_pct': self.current_drawdown,
            'total_trades': len(self.trade_history),
            'total_realized_pnl': self.position_tracker.get_total_realized_pnl(),
            'total_unrealized_pnl': self.position_tracker.get_total_unrealized_pnl(prices)
        }

    def __repr__(self) -> str:
        return f"Portfolio(value=₹{self.total_value:,.0f}, cash=₹{self.cash:,.0f}, positions={self.position_tracker.get_position_count()}, return={self.total_return:.2f}%)"
