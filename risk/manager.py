"""
Risk Management Module.

Implements risk checks and controls:
- Position size limits
- Loss limits (per trade, daily, total drawdown)
- Exposure limits
- Correlation checks (future enhancement)
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import logging

from config import get_config
from execution.portfolio import Portfolio
from agents.base_agent import Signal

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk manager enforcing trading limits and risk controls.

    All trades must pass risk checks before execution.
    """

    def __init__(self, portfolio: Portfolio):
        """
        Initialize risk manager.

        Args:
            portfolio: Portfolio instance to monitor
        """
        self.portfolio = portfolio
        self.config = get_config()

        # Daily tracking
        self.daily_start_value: Optional[float] = None
        self.daily_losses = 0.0
        self.daily_date: Optional[pd.Timestamp] = None

        # Circuit breakers
        self.trading_halted = False
        self.halt_reason = ""

        logger.info("Risk manager initialized")

    def check_signal(
        self,
        signal: Signal,
        current_prices: Dict[str, float],
        current_date: pd.Timestamp
    ) -> Tuple[bool, str]:
        """
        Check if a signal passes all risk checks.

        Args:
            signal: Trading signal to validate
            current_prices: Current market prices
            current_date: Current date

        Returns:
            (is_approved, reason)
        """
        # Check if trading is halted
        if self.trading_halted:
            return False, f"Trading halted: {self.halt_reason}"

        # Update daily tracking
        self._update_daily_tracking(current_date)

        # Run all risk checks
        checks = [
            self._check_position_size(signal, current_prices),
            self._check_exposure_limit(signal, current_prices),
            self._check_daily_loss_limit(),
            self._check_max_drawdown(),
            self._check_position_count(),
            self._check_cash_reserve(signal, current_prices)
        ]

        # Find first failed check
        for passed, reason in checks:
            if not passed:
                logger.warning(f"Risk check failed: {reason}")
                return False, reason

        return True, "All risk checks passed"

    def _check_position_size(
        self,
        signal: Signal,
        current_prices: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Check if position size is within limits."""
        position_value = signal.price * signal.size
        max_position_value = self.portfolio.total_value * (self.config.risk.max_position_size_pct / 100)

        if position_value > max_position_value:
            return False, f"Position size ₹{position_value:.0f} exceeds max ₹{max_position_value:.0f} ({self.config.risk.max_position_size_pct}%)"

        # Check minimum position size
        if position_value < self.config.risk.min_position_value:
            return False, f"Position size ₹{position_value:.0f} below minimum ₹{self.config.risk.min_position_value:.0f}"

        return True, "Position size OK"

    def _check_exposure_limit(
        self,
        signal: Signal,
        current_prices: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Check if adding this position would exceed exposure limit."""
        current_exposure = self.portfolio.position_tracker.get_total_exposure(current_prices)
        new_position_value = signal.price * signal.size
        total_exposure = current_exposure + new_position_value

        max_exposure = self.portfolio.total_value * (self.config.risk.max_total_exposure_pct / 100)

        if total_exposure > max_exposure:
            exposure_pct = (total_exposure / self.portfolio.total_value) * 100
            return False, f"Total exposure {exposure_pct:.1f}% would exceed limit {self.config.risk.max_total_exposure_pct}%"

        return True, "Exposure OK"

    def _check_daily_loss_limit(self) -> Tuple[bool, str]:
        """Check if daily loss limit has been hit."""
        if self.daily_start_value is None:
            return True, "Daily loss check N/A"

        daily_loss_pct = ((self.portfolio.total_value - self.daily_start_value) / self.daily_start_value) * 100

        if daily_loss_pct < -self.config.risk.max_daily_loss_pct:
            self._halt_trading(f"Daily loss limit hit: {daily_loss_pct:.2f}%")
            return False, f"Daily loss {daily_loss_pct:.2f}% exceeds limit {self.config.risk.max_daily_loss_pct}%"

        return True, "Daily loss OK"

    def _check_max_drawdown(self) -> Tuple[bool, str]:
        """Check if maximum drawdown has been exceeded."""
        current_dd_pct = abs(self.portfolio.current_drawdown)

        if current_dd_pct > self.config.risk.max_drawdown_pct:
            self._halt_trading(f"Max drawdown exceeded: {current_dd_pct:.2f}%")
            return False, f"Drawdown {current_dd_pct:.2f}% exceeds limit {self.config.risk.max_drawdown_pct}%"

        return True, "Drawdown OK"

    def _check_position_count(self) -> Tuple[bool, str]:
        """Check if maximum position count would be exceeded."""
        current_count = self.portfolio.position_tracker.get_position_count()

        if current_count >= self.config.risk.max_concurrent_positions:
            return False, f"Max positions {self.config.risk.max_concurrent_positions} reached"

        return True, "Position count OK"

    def _check_cash_reserve(
        self,
        signal: Signal,
        current_prices: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Check if minimum cash reserve would be maintained."""
        position_cost = signal.price * signal.size * 1.004  # Include estimated costs
        cash_after = self.portfolio.cash - position_cost
        min_reserve = self.portfolio.total_value * (self.config.capital.min_cash_reserve_pct / 100)

        if cash_after < min_reserve:
            return False, f"Would violate cash reserve: ₹{cash_after:.0f} < ₹{min_reserve:.0f}"

        return True, "Cash reserve OK"

    def _update_daily_tracking(self, current_date: pd.Timestamp):
        """Update daily loss tracking."""
        if self.daily_date is None or current_date.date() != self.daily_date.date():
            # New trading day
            self.daily_start_value = self.portfolio.total_value
            self.daily_losses = 0.0
            self.daily_date = current_date

            # Reset halt if it was daily-loss related
            if self.trading_halted and "daily loss" in self.halt_reason.lower():
                self._resume_trading()

    def _halt_trading(self, reason: str):
        """Halt all trading."""
        self.trading_halted = True
        self.halt_reason = reason
        logger.error(f"TRADING HALTED: {reason}")

    def _resume_trading(self):
        """Resume trading after halt."""
        self.trading_halted = False
        self.halt_reason = ""
        logger.info("Trading resumed")

    def check_position_exit(
        self,
        symbol: str,
        current_price: float,
        current_date: pd.Timestamp
    ) -> Tuple[bool, str]:
        """
        Check if a position should be force-exited due to risk limits.

        Args:
            symbol: Stock symbol
            current_price: Current price
            current_date: Current date

        Returns:
            (should_exit, reason)
        """
        position = self.portfolio.position_tracker.get_position(symbol)
        if not position:
            return False, "No position"

        # Check stop loss
        if position.check_stop_loss(current_price):
            return True, "Stop loss hit"

        # Check max loss per trade
        unrealized_pnl_pct = position.unrealized_pnl_pct(current_price)
        if unrealized_pnl_pct < -self.config.risk.max_loss_per_trade_pct:
            return True, f"Max loss per trade exceeded: {unrealized_pnl_pct:.2f}%"

        return False, ""

    def get_risk_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """
        Get current risk metrics.

        Args:
            current_prices: Current prices

        Returns:
            Dictionary of risk metrics
        """
        total_value = self.portfolio.total_value
        current_exposure = self.portfolio.position_tracker.get_total_exposure(current_prices)

        metrics = {
            'total_value': total_value,
            'cash': self.portfolio.cash,
            'cash_pct': self.portfolio.available_cash_pct,
            'exposure': current_exposure,
            'exposure_pct': (current_exposure / total_value * 100) if total_value > 0 else 0,
            'position_count': self.portfolio.position_tracker.get_position_count(),
            'max_positions': self.config.risk.max_concurrent_positions,
            'current_drawdown_pct': abs(self.portfolio.current_drawdown),
            'max_drawdown_pct': self.config.risk.max_drawdown_pct,
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason
        }

        # Daily metrics
        if self.daily_start_value:
            daily_pnl = total_value - self.daily_start_value
            daily_pnl_pct = (daily_pnl / self.daily_start_value) * 100
            metrics['daily_pnl'] = daily_pnl
            metrics['daily_pnl_pct'] = daily_pnl_pct
            metrics['daily_loss_limit_pct'] = self.config.risk.max_daily_loss_pct

        return metrics

    def __repr__(self) -> str:
        status = "HALTED" if self.trading_halted else "ACTIVE"
        return f"RiskManager(status={status}, positions={self.portfolio.position_tracker.get_position_count()}/{self.config.risk.max_concurrent_positions})"
