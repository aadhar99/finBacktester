"""
Position tracking module.

Tracks individual positions including:
- Entry/exit prices and timestamps
- P&L calculation (realized and unrealized)
- Position metrics (days held, max drawdown, etc.)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Individual trade record."""
    trade_id: str
    symbol: str
    timestamp: pd.Timestamp
    action: str  # 'buy' or 'sell'
    quantity: int
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    reason: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def total_value(self) -> float:
        """Total trade value including costs."""
        base_value = self.quantity * self.price
        costs = self.commission + (self.slippage * self.quantity)

        if self.action == 'buy':
            return base_value + costs  # Pay costs when buying
        else:
            return base_value - costs  # Costs reduce proceeds when selling


@dataclass
class Position:
    """
    Represents an open or closed position.

    Tracks all trades, calculates P&L, and monitors position metrics.
    """
    symbol: str
    entry_timestamp: pd.Timestamp
    entry_price: float
    quantity: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = "unknown"

    # Position state
    is_open: bool = True
    exit_timestamp: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""

    # Costs
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0

    # Metrics
    peak_price: Optional[float] = None  # Highest price reached
    trough_price: Optional[float] = None  # Lowest price reached
    trades: List[Trade] = field(default_factory=list)

    def __post_init__(self):
        """Initialize tracking metrics."""
        if self.peak_price is None:
            self.peak_price = self.entry_price
        if self.trough_price is None:
            self.trough_price = self.entry_price

    def update_price(self, current_price: float, timestamp: pd.Timestamp):
        """
        Update position with current market price.

        Args:
            current_price: Current market price
            timestamp: Current timestamp
        """
        if not self.is_open:
            return

        # Update peak and trough
        if self.peak_price is None or current_price > self.peak_price:
            self.peak_price = current_price

        if self.trough_price is None or current_price < self.trough_price:
            self.trough_price = current_price

    def close(
        self,
        exit_price: float,
        exit_timestamp: pd.Timestamp,
        reason: str = "",
        commission: float = 0.0,
        slippage: float = 0.0
    ):
        """
        Close the position.

        Args:
            exit_price: Exit price
            exit_timestamp: Exit timestamp
            reason: Reason for exit
            commission: Exit commission
            slippage: Exit slippage
        """
        self.is_open = False
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.exit_reason = reason
        self.exit_commission = commission
        self.exit_slippage = slippage

        logger.info(f"Closed position: {self.symbol} @ {exit_price:.2f}, P&L: ₹{self.realized_pnl:.2f}")

    @property
    def days_held(self) -> int:
        """Number of days position has been held."""
        if self.exit_timestamp:
            end_time = self.exit_timestamp
        else:
            end_time = pd.Timestamp.now()

        return (end_time - self.entry_timestamp).days

    @property
    def entry_value(self) -> float:
        """Total entry value including costs."""
        base_value = self.quantity * self.entry_price
        costs = self.entry_commission + (self.entry_slippage * self.quantity)
        return base_value + costs

    @property
    def exit_value(self) -> float:
        """Total exit value after costs."""
        if not self.exit_price:
            return 0.0

        base_value = self.quantity * self.exit_price
        costs = self.exit_commission + (self.exit_slippage * self.quantity)
        return base_value - costs

    @property
    def realized_pnl(self) -> float:
        """Realized P&L (only for closed positions)."""
        if not self.is_open and self.exit_price:
            return self.exit_value - self.entry_value
        return 0.0

    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        if not self.is_open:
            return 0.0

        current_value = self.quantity * current_price
        # Account for entry costs but not exit costs (not realized yet)
        return current_value - self.entry_value

    @property
    def realized_pnl_pct(self) -> float:
        """Realized P&L as percentage of entry value."""
        if self.entry_value == 0:
            return 0.0
        return (self.realized_pnl / self.entry_value) * 100

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Unrealized P&L as percentage."""
        if self.entry_value == 0:
            return 0.0
        return (self.unrealized_pnl(current_price) / self.entry_value) * 100

    @property
    def max_favorable_excursion(self) -> float:
        """Maximum profit reached (MFE)."""
        if self.peak_price and self.entry_price:
            return ((self.peak_price - self.entry_price) / self.entry_price) * 100
        return 0.0

    @property
    def max_adverse_excursion(self) -> float:
        """Maximum drawdown from entry (MAE)."""
        if self.trough_price and self.entry_price:
            return ((self.trough_price - self.entry_price) / self.entry_price) * 100
        return 0.0

    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop loss is hit.

        Args:
            current_price: Current price

        Returns:
            True if stop loss hit
        """
        if not self.is_open or not self.stop_loss:
            return False

        # Long position: stop loss below entry
        if self.quantity > 0:
            return current_price <= self.stop_loss

        # Short position: stop loss above entry
        return current_price >= self.stop_loss

    def check_take_profit(self, current_price: float) -> bool:
        """
        Check if take profit is hit.

        Args:
            current_price: Current price

        Returns:
            True if take profit hit
        """
        if not self.is_open or not self.take_profit:
            return False

        # Long position: take profit above entry
        if self.quantity > 0:
            return current_price >= self.take_profit

        # Short position: take profit below entry
        return current_price <= self.take_profit

    def to_dict(self) -> dict:
        """Convert position to dictionary for logging/analysis."""
        return {
            'symbol': self.symbol,
            'entry_timestamp': self.entry_timestamp,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'strategy': self.strategy,
            'is_open': self.is_open,
            'exit_timestamp': self.exit_timestamp,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'days_held': self.days_held,
            'entry_value': self.entry_value,
            'exit_value': self.exit_value,
            'realized_pnl': self.realized_pnl,
            'realized_pnl_pct': self.realized_pnl_pct,
            'peak_price': self.peak_price,
            'trough_price': self.trough_price,
            'mfe': self.max_favorable_excursion,
            'mae': self.max_adverse_excursion,
        }

    def __repr__(self) -> str:
        status = "OPEN" if self.is_open else "CLOSED"
        pnl = self.realized_pnl if not self.is_open else "N/A"
        return f"Position({self.symbol}, {status}, entry={self.entry_price:.2f}, qty={self.quantity}, P&L={pnl})"


class PositionTracker:
    """Track all positions (open and closed)."""

    def __init__(self):
        """Initialize the position tracker."""
        self.open_positions: dict[str, Position] = {}  # symbol -> Position
        self.closed_positions: List[Position] = []
        self.all_trades: List[Trade] = []

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        timestamp: pd.Timestamp,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: str = "unknown",
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> Position:
        """
        Open a new position.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            quantity: Quantity
            timestamp: Entry timestamp
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy: Strategy name
            commission: Entry commission
            slippage: Entry slippage

        Returns:
            Created position
        """
        position = Position(
            symbol=symbol,
            entry_timestamp=timestamp,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy,
            entry_commission=commission,
            entry_slippage=slippage
        )

        self.open_positions[symbol] = position

        # Record trade
        trade = Trade(
            trade_id=f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol,
            timestamp=timestamp,
            action='buy',
            quantity=quantity,
            price=entry_price,
            commission=commission,
            slippage=slippage,
            reason=f"Open position ({strategy})"
        )
        self.all_trades.append(trade)

        logger.info(f"Opened position: {symbol} @ ₹{entry_price:.2f} x {quantity} shares")
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: pd.Timestamp,
        reason: str = "",
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> Optional[Position]:
        """
        Close an existing position.

        Args:
            symbol: Stock symbol
            exit_price: Exit price
            timestamp: Exit timestamp
            reason: Exit reason
            commission: Exit commission
            slippage: Exit slippage

        Returns:
            Closed position or None if not found
        """
        if symbol not in self.open_positions:
            logger.warning(f"Cannot close position: {symbol} not found")
            return None

        position = self.open_positions.pop(symbol)
        position.close(exit_price, timestamp, reason, commission, slippage)
        self.closed_positions.append(position)

        # Record trade
        trade = Trade(
            trade_id=f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol,
            timestamp=timestamp,
            action='sell',
            quantity=position.quantity,
            price=exit_price,
            commission=commission,
            slippage=slippage,
            reason=reason
        )
        self.all_trades.append(trade)

        return position

    def update_prices(self, prices: dict[str, float], timestamp: pd.Timestamp):
        """
        Update all open positions with current prices.

        Args:
            prices: Dictionary of {symbol: current_price}
            timestamp: Current timestamp
        """
        for symbol, position in self.open_positions.items():
            if symbol in prices:
                position.update_price(prices[symbol], timestamp)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.open_positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position."""
        return symbol in self.open_positions

    def get_total_exposure(self, prices: dict[str, float]) -> float:
        """Calculate total exposure across all open positions."""
        total = 0.0
        for symbol, position in self.open_positions.items():
            if symbol in prices:
                total += position.quantity * prices[symbol]
        return total

    def get_total_unrealized_pnl(self, prices: dict[str, float]) -> float:
        """Calculate total unrealized P&L."""
        total_pnl = 0.0
        for symbol, position in self.open_positions.items():
            if symbol in prices:
                total_pnl += position.unrealized_pnl(prices[symbol])
        return total_pnl

    def get_total_realized_pnl(self) -> float:
        """Calculate total realized P&L from closed positions."""
        return sum(pos.realized_pnl for pos in self.closed_positions)

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.open_positions)

    def get_positions_summary(self, prices: dict[str, float]) -> pd.DataFrame:
        """
        Get summary of all open positions.

        Args:
            prices: Current prices

        Returns:
            DataFrame with position summaries
        """
        if not self.open_positions:
            return pd.DataFrame()

        summaries = []
        for symbol, position in self.open_positions.items():
            current_price = prices.get(symbol, position.entry_price)
            summary = position.to_dict()
            summary['current_price'] = current_price
            summary['unrealized_pnl'] = position.unrealized_pnl(current_price)
            summary['unrealized_pnl_pct'] = position.unrealized_pnl_pct(current_price)
            summaries.append(summary)

        return pd.DataFrame(summaries)
