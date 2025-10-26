"""
Performance Metrics Calculator.

Calculates all 20+ metrics for evaluating trading system performance:
- Returns (total, annualized, monthly)
- Risk metrics (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Win/Loss statistics
- Trade analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""

    # Returns
    total_return_pct: float
    annualized_return_pct: float
    monthly_return_pct: float
    daily_return_pct: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Drawdown
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    avg_drawdown_pct: float
    current_drawdown_pct: float

    # Volatility
    annualized_volatility_pct: float
    downside_deviation_pct: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    avg_trade_duration_days: float

    # Position metrics
    avg_positions_held: float
    max_positions_held: int

    # P&L
    total_pnl: float
    gross_profit: float
    gross_loss: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self.__dict__


class MetricsCalculator:
    """Calculate performance metrics from backtest results."""

    def __init__(self, risk_free_rate: float = 0.065):
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 6.5% for India)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(
        self,
        portfolio_values: pd.Series,
        trades: pd.DataFrame,
        closed_positions: List,
        initial_capital: float
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            portfolio_values: Time series of portfolio values
            trades: DataFrame of all trades
            closed_positions: List of closed Position objects
            initial_capital: Starting capital

        Returns:
            PerformanceMetrics object
        """
        # Calculate returns
        returns = portfolio_values.pct_change().fillna(0)

        # Return metrics
        total_return_pct = self._total_return(portfolio_values, initial_capital)
        annualized_return_pct = self._annualized_return(returns)
        monthly_return_pct = annualized_return_pct / 12
        daily_return_pct = returns.mean() * 100

        # Risk-adjusted metrics
        sharpe_ratio = self._sharpe_ratio(returns)
        sortino_ratio = self._sortino_ratio(returns)
        calmar_ratio = self._calmar_ratio(returns, portfolio_values, initial_capital)
        omega_ratio = self._omega_ratio(returns)

        # Drawdown metrics
        dd_metrics = self._drawdown_metrics(portfolio_values)

        # Volatility
        annualized_vol = returns.std() * np.sqrt(252) * 100
        downside_dev = self._downside_deviation(returns)

        # Trade metrics
        trade_metrics = self._trade_metrics(trades, closed_positions)

        # Position metrics
        if len(portfolio_values) > 0:
            # This is simplified - in real implementation would track from backtest
            avg_positions = 2.0  # Placeholder
            max_positions = 6  # Placeholder
        else:
            avg_positions = 0.0
            max_positions = 0

        # P&L metrics
        pnl_metrics = self._pnl_metrics(closed_positions)

        return PerformanceMetrics(
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            monthly_return_pct=monthly_return_pct,
            daily_return_pct=daily_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            max_drawdown_pct=dd_metrics['max_drawdown_pct'],
            max_drawdown_duration_days=dd_metrics['max_dd_duration'],
            avg_drawdown_pct=dd_metrics['avg_drawdown_pct'],
            current_drawdown_pct=dd_metrics['current_drawdown_pct'],
            annualized_volatility_pct=annualized_vol,
            downside_deviation_pct=downside_dev,
            total_trades=trade_metrics['total_trades'],
            winning_trades=trade_metrics['winning_trades'],
            losing_trades=trade_metrics['losing_trades'],
            win_rate_pct=trade_metrics['win_rate_pct'],
            profit_factor=trade_metrics['profit_factor'],
            avg_win_pct=trade_metrics['avg_win_pct'],
            avg_loss_pct=trade_metrics['avg_loss_pct'],
            largest_win_pct=trade_metrics['largest_win_pct'],
            largest_loss_pct=trade_metrics['largest_loss_pct'],
            avg_trade_duration_days=trade_metrics['avg_duration_days'],
            avg_positions_held=avg_positions,
            max_positions_held=max_positions,
            total_pnl=pnl_metrics['total_pnl'],
            gross_profit=pnl_metrics['gross_profit'],
            gross_loss=pnl_metrics['gross_loss']
        )

    def _total_return(self, portfolio_values: pd.Series, initial_capital: float) -> float:
        """Calculate total return percentage."""
        if len(portfolio_values) == 0 or initial_capital == 0:
            return 0.0
        final_value = portfolio_values.iloc[-1]
        return ((final_value - initial_capital) / initial_capital) * 100

    def _annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0

        # Compound annual growth rate
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252  # 252 trading days per year

        if n_years == 0:
            return 0.0

        annualized = (1 + total_return) ** (1 / n_years) - 1
        return annualized * 100

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio.

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        mean_excess = excess_returns.mean()
        std_excess = returns.std()

        if std_excess == 0:
            return 0.0

        sharpe = (mean_excess / std_excess) * np.sqrt(252)  # Annualized
        return sharpe

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sortino ratio.

        Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 252)
        mean_excess = excess_returns.mean()
        downside_dev = self._downside_deviation(returns) / 100  # Convert to decimal

        if downside_dev == 0:
            return 0.0

        sortino = (mean_excess / downside_dev) * np.sqrt(252)
        return sortino

    def _calmar_ratio(self, returns: pd.Series, portfolio_values: pd.Series, initial_capital: float) -> float:
        """
        Calculate Calmar ratio.

        Calmar = Annualized Return / Max Drawdown
        """
        ann_return = self._annualized_return(returns)
        dd_metrics = self._drawdown_metrics(portfolio_values)
        max_dd = dd_metrics['max_drawdown_pct']

        if max_dd == 0:
            return 0.0

        return ann_return / abs(max_dd)

    def _omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.

        Omega = Probability-weighted gains / Probability-weighted losses
        """
        if len(returns) == 0:
            return 0.0

        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())

        if losses == 0:
            return float('inf') if gains > 0 else 0.0

        return gains / losses

    def _drawdown_metrics(self, portfolio_values: pd.Series) -> Dict:
        """Calculate all drawdown-related metrics."""
        if len(portfolio_values) == 0:
            return {
                'max_drawdown_pct': 0.0,
                'max_dd_duration': 0,
                'avg_drawdown_pct': 0.0,
                'current_drawdown_pct': 0.0
            }

        # Calculate running maximum
        running_max = portfolio_values.expanding().max()

        # Calculate drawdown series
        drawdown = (portfolio_values - running_max) / running_max * 100

        # Max drawdown
        max_dd = drawdown.min()

        # Max drawdown duration
        in_dd = drawdown < 0
        dd_durations = []
        current_duration = 0

        for is_dd in in_dd:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    dd_durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            dd_durations.append(current_duration)

        max_dd_duration = max(dd_durations) if dd_durations else 0

        # Average drawdown
        avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0

        # Current drawdown
        current_dd = drawdown.iloc[-1]

        return {
            'max_drawdown_pct': max_dd,
            'max_dd_duration': max_dd_duration,
            'avg_drawdown_pct': avg_dd,
            'current_drawdown_pct': current_dd
        }

    def _downside_deviation(self, returns: pd.Series, target: float = 0.0) -> float:
        """
        Calculate downside deviation.

        Only considers returns below target (typically 0).
        """
        downside_returns = returns[returns < target]

        if len(downside_returns) == 0:
            return 0.0

        downside_dev = downside_returns.std() * np.sqrt(252) * 100  # Annualized
        return downside_dev

    def _trade_metrics(self, trades: pd.DataFrame, closed_positions: List) -> Dict:
        """Calculate trade-related metrics."""
        if len(closed_positions) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate_pct': 0.0,
                'profit_factor': 0.0,
                'avg_win_pct': 0.0,
                'avg_loss_pct': 0.0,
                'largest_win_pct': 0.0,
                'largest_loss_pct': 0.0,
                'avg_duration_days': 0.0
            }

        # Extract P&L from closed positions
        pnls = [pos.realized_pnl_pct for pos in closed_positions]
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]

        total_trades = len(closed_positions)
        winning_trades = len(winning_pnls)
        losing_trades = len(losing_pnls)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = sum(winning_pnls) if winning_pnls else 0.0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

        # Average win/loss
        avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0

        # Largest win/loss
        largest_win = max(winning_pnls) if winning_pnls else 0.0
        largest_loss = min(losing_pnls) if losing_pnls else 0.0

        # Average duration
        durations = [pos.days_held for pos in closed_positions]
        avg_duration = np.mean(durations) if durations else 0.0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'largest_win_pct': largest_win,
            'largest_loss_pct': largest_loss,
            'avg_duration_days': avg_duration
        }

    def _pnl_metrics(self, closed_positions: List) -> Dict:
        """Calculate P&L metrics."""
        if len(closed_positions) == 0:
            return {
                'total_pnl': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0
            }

        pnls = [pos.realized_pnl for pos in closed_positions]
        total_pnl = sum(pnls)
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = sum(p for p in pnls if p < 0)

        return {
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }

    def print_metrics(self, metrics: PerformanceMetrics):
        """Print formatted metrics report."""
        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS REPORT")
        print("=" * 70)

        print("\n📊 RETURNS")
        print(f"  Total Return:              {metrics.total_return_pct:>10.2f}%")
        print(f"  Annualized Return:         {metrics.annualized_return_pct:>10.2f}%")
        print(f"  Monthly Return (avg):      {metrics.monthly_return_pct:>10.2f}%")

        print("\n📈 RISK-ADJUSTED RETURNS")
        print(f"  Sharpe Ratio:              {metrics.sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:             {metrics.sortino_ratio:>10.2f}")
        print(f"  Calmar Ratio:              {metrics.calmar_ratio:>10.2f}")
        print(f"  Omega Ratio:               {metrics.omega_ratio:>10.2f}")

        print("\n📉 DRAWDOWN")
        print(f"  Max Drawdown:              {metrics.max_drawdown_pct:>10.2f}%")
        print(f"  Max DD Duration:           {metrics.max_drawdown_duration_days:>10} days")
        print(f"  Average Drawdown:          {metrics.avg_drawdown_pct:>10.2f}%")

        print("\n📊 VOLATILITY")
        print(f"  Annualized Volatility:     {metrics.annualized_volatility_pct:>10.2f}%")
        print(f"  Downside Deviation:        {metrics.downside_deviation_pct:>10.2f}%")

        print("\n💰 TRADE STATISTICS")
        print(f"  Total Trades:              {metrics.total_trades:>10}")
        print(f"  Winning Trades:            {metrics.winning_trades:>10}")
        print(f"  Losing Trades:             {metrics.losing_trades:>10}")
        print(f"  Win Rate:                  {metrics.win_rate_pct:>10.2f}%")
        print(f"  Profit Factor:             {metrics.profit_factor:>10.2f}")
        print(f"  Average Win:               {metrics.avg_win_pct:>10.2f}%")
        print(f"  Average Loss:              {metrics.avg_loss_pct:>10.2f}%")
        print(f"  Largest Win:               {metrics.largest_win_pct:>10.2f}%")
        print(f"  Largest Loss:              {metrics.largest_loss_pct:>10.2f}%")
        print(f"  Avg Trade Duration:        {metrics.avg_trade_duration_days:>10.1f} days")

        print("\n💵 P&L")
        print(f"  Total P&L:                 ₹{metrics.total_pnl:>10,.0f}")
        print(f"  Gross Profit:              ₹{metrics.gross_profit:>10,.0f}")
        print(f"  Gross Loss:                ₹{metrics.gross_loss:>10,.0f}")

        print("\n" + "=" * 70)
