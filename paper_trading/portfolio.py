"""
Paper Trading Portfolio - Simulate Real Trading Without Risk

Tracks simulated trades based on real smart money signals:
- Executes trades based on daily signals
- Tracks positions and P&L
- Updates with real market prices
- Generates performance reports

This validates the system before risking real money.

Usage:
    portfolio = PaperTradingPortfolio(initial_capital=1000000)
    portfolio.execute_signal(signal, current_price)
    portfolio.update_positions(current_prices)
    report = portfolio.get_performance_report()
"""

import logging
import json
from datetime import datetime, date
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """A simulated position in paper trading."""
    symbol: str
    entry_date: date
    entry_price: float
    quantity: int
    signal_type: str  # BUY or SELL
    confidence: float
    stop_loss: float
    take_profit: float
    pattern_type: str

    # Current state
    current_price: float = 0.0
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    # P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0

    # Tracking
    days_held: int = 0
    max_gain: float = 0.0
    max_loss: float = 0.0

    def update_current_price(self, price: float):
        """Update position with current market price."""
        self.current_price = price

        # Calculate unrealized P&L
        if self.signal_type == 'BUY':
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

        self.unrealized_pnl_pct = (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100

        # Track max gain/loss
        self.max_gain = max(self.max_gain, self.unrealized_pnl)
        self.max_loss = min(self.max_loss, self.unrealized_pnl)

        # Update days held
        self.days_held = (date.today() - self.entry_date).days

    def close(self, exit_price: float, reason: str):
        """Close the position."""
        self.exit_price = exit_price
        self.exit_date = date.today()
        self.exit_reason = reason

        # Calculate realized P&L
        if self.signal_type == 'BUY':
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity

        self.realized_pnl_pct = (self.realized_pnl / (self.entry_price * self.quantity)) * 100

    def check_stop_loss(self) -> bool:
        """Check if stop loss is hit."""
        if self.signal_type == 'BUY':
            return self.current_price <= self.stop_loss
        else:  # SHORT
            return self.current_price >= self.stop_loss

    def check_take_profit(self) -> bool:
        """Check if take profit is hit."""
        if self.signal_type == 'BUY':
            return self.current_price >= self.take_profit
        else:  # SHORT
            return self.current_price <= self.take_profit


@dataclass
class PaperTradingPortfolio:
    """
    Paper trading portfolio that simulates real trading.

    Executes trades based on smart money signals and tracks performance
    without risking real capital.
    """

    initial_capital: float
    cash: float = field(init=False)
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    closed_positions: List[PaperPosition] = field(default_factory=list)
    trade_log: List[dict] = field(default_factory=list)

    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_portfolio_value: float = 0.0

    def __post_init__(self):
        self.cash = self.initial_capital
        self.peak_portfolio_value = self.initial_capital
        logger.info(f"✅ Paper trading portfolio initialized: ₹{self.initial_capital:,.0f}")

    def execute_signal(
        self,
        symbol: str,
        signal_type: str,
        price: float,
        quantity: int,
        confidence: float,
        stop_loss: float,
        take_profit: float,
        pattern_type: str
    ) -> bool:
        """
        Execute a trading signal (paper trade).

        Args:
            symbol: Stock symbol
            signal_type: BUY or SELL
            price: Entry price
            quantity: Number of shares
            confidence: Signal confidence (0-100)
            stop_loss: Stop loss price
            take_profit: Take profit price
            pattern_type: Pattern that generated signal

        Returns:
            True if trade executed, False otherwise
        """
        # Check if we already have a position
        if symbol in self.positions:
            logger.warning(f"  ⚠️  {symbol}: Already have position, skipping")
            return False

        # Calculate position value
        position_value = price * quantity

        # Check if we have enough cash
        if position_value > self.cash:
            logger.warning(f"  ⚠️  {symbol}: Insufficient cash (need ₹{position_value:,.0f}, have ₹{self.cash:,.0f})")
            return False

        # Execute trade
        position = PaperPosition(
            symbol=symbol,
            entry_date=date.today(),
            entry_price=price,
            quantity=quantity,
            signal_type=signal_type,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pattern_type=pattern_type,
            current_price=price
        )

        self.positions[symbol] = position
        self.cash -= position_value
        self.total_trades += 1

        # Log trade
        self._log_trade('ENTRY', position)

        logger.info(f"  ✅ PAPER TRADE: {signal_type} {symbol}")
        logger.info(f"      Qty: {quantity} @ ₹{price:.2f} = ₹{position_value:,.0f}")
        logger.info(f"      SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f}")
        logger.info(f"      Confidence: {confidence:.0f}% | Pattern: {pattern_type}")
        logger.info(f"      Cash remaining: ₹{self.cash:,.0f}")

        return True

    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update all positions with current market prices.

        Args:
            current_prices: Dict of {symbol: current_price}
        """
        for symbol, position in list(self.positions.items()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.update_current_price(current_price)

                # Check stop loss
                if position.check_stop_loss():
                    self._close_position(symbol, current_price, "Stop Loss Hit")

                # Check take profit
                elif position.check_take_profit():
                    self._close_position(symbol, current_price, "Take Profit Hit")

                # Check max holding period (30 days)
                elif position.days_held >= 30:
                    self._close_position(symbol, current_price, "Max Holding Period")

        # Update drawdown
        self._update_drawdown()

    def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        position.close(exit_price, reason)

        # Update cash
        exit_value = exit_price * position.quantity
        self.cash += exit_value

        # Update P&L
        self.total_pnl += position.realized_pnl

        if position.realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]

        # Log trade
        self._log_trade('EXIT', position)

        logger.info(f"  🔒 CLOSED: {symbol}")
        logger.info(f"      Entry: ₹{position.entry_price:.2f} → Exit: ₹{exit_price:.2f}")
        logger.info(f"      P&L: ₹{position.realized_pnl:,.0f} ({position.realized_pnl_pct:+.2f}%)")
        logger.info(f"      Reason: {reason}")
        logger.info(f"      Days held: {position.days_held}")

    def _update_drawdown(self):
        """Update max drawdown calculation."""
        current_value = self.get_portfolio_value()

        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value

        drawdown = (current_value - self.peak_portfolio_value) / self.peak_portfolio_value * 100
        self.max_drawdown = min(self.max_drawdown, drawdown)

    def get_portfolio_value(self) -> float:
        """Get current total portfolio value."""
        positions_value = sum(
            pos.current_price * pos.quantity
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def get_total_return(self) -> float:
        """Get total return percentage."""
        current_value = self.get_portfolio_value()
        return ((current_value - self.initial_capital) / self.initial_capital) * 100

    def get_win_rate(self) -> float:
        """Get win rate percentage."""
        total_closed = len(self.closed_positions)
        if total_closed == 0:
            return 0.0
        return (self.winning_trades / total_closed) * 100

    def _log_trade(self, action: str, position: PaperPosition):
        """Log trade to trade log."""
        self.trade_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': position.symbol,
            'price': position.exit_price if action == 'EXIT' else position.entry_price,
            'quantity': position.quantity,
            'signal_type': position.signal_type,
            'confidence': position.confidence,
            'pattern': position.pattern_type,
            'pnl': position.realized_pnl if action == 'EXIT' else 0,
            'pnl_pct': position.realized_pnl_pct if action == 'EXIT' else 0,
            'reason': position.exit_reason if action == 'EXIT' else None
        })

    def get_performance_report(self) -> dict:
        """Generate comprehensive performance report."""
        current_value = self.get_portfolio_value()
        total_return = self.get_total_return()
        win_rate = self.get_win_rate()

        # Calculate average win/loss
        if self.winning_trades > 0:
            avg_win = sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0) / self.winning_trades
        else:
            avg_win = 0

        if self.losing_trades > 0:
            avg_loss = sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0) / self.losing_trades
        else:
            avg_loss = 0

        return {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'cash': self.cash,
            'total_return_pct': total_return,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown_pct': self.max_drawdown,
            'positions': [asdict(p) for p in self.positions.values()],
            'closed_trades': [asdict(p) for p in self.closed_positions]
        }

    def save_to_file(self, filename: str = None):
        """Save portfolio state to JSON file."""
        if filename is None:
            filename = f"paper_trading/portfolio_{date.today()}.json"

        report = self.get_performance_report()

        # Convert dates to strings for JSON serialization
        def date_converter(obj):
            if isinstance(obj, date):
                return obj.isoformat()
            return obj

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=date_converter)

        logger.info(f"  💾 Portfolio saved to {filename}")

    def print_summary(self):
        """Print portfolio summary."""
        print("\n" + "=" * 70)
        print("PAPER TRADING PORTFOLIO SUMMARY")
        print("=" * 70)

        current_value = self.get_portfolio_value()
        total_return = self.get_total_return()

        print(f"\n💰 Portfolio Value:")
        print(f"   Initial: ₹{self.initial_capital:,.0f}")
        print(f"   Current: ₹{current_value:,.0f}")
        print(f"   Cash: ₹{self.cash:,.0f}")
        print(f"   Total Return: {total_return:+.2f}%")

        print(f"\n📊 Trading Performance:")
        print(f"   Total Trades: {self.total_trades}")
        print(f"   Open Positions: {len(self.positions)}")
        print(f"   Closed Positions: {len(self.closed_positions)}")
        print(f"   Winners: {self.winning_trades}")
        print(f"   Losers: {self.losing_trades}")
        print(f"   Win Rate: {self.get_win_rate():.1f}%")

        if self.closed_positions:
            avg_win = sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0) / max(self.winning_trades, 1)
            avg_loss = sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0) / max(self.losing_trades, 1)

            print(f"\n💵 P&L:")
            print(f"   Total: ₹{self.total_pnl:,.0f}")
            print(f"   Avg Win: ₹{avg_win:,.0f}")
            print(f"   Avg Loss: ₹{avg_loss:,.0f}")
            print(f"   Max Drawdown: {self.max_drawdown:.2f}%")

        if self.positions:
            print(f"\n📈 Open Positions:")
            for pos in self.positions.values():
                print(f"   {pos.symbol}: {pos.quantity} shares @ ₹{pos.entry_price:.2f}")
                print(f"      Current: ₹{pos.current_price:.2f} | P&L: ₹{pos.unrealized_pnl:,.0f} ({pos.unrealized_pnl_pct:+.2f}%)")
                print(f"      Days held: {pos.days_held} | Confidence: {pos.confidence:.0f}%")

        print("\n" + "=" * 70)
