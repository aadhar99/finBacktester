"""
Intraday backtest engine for 15-min candle strategies.

Processes candle-by-candle within each trading day, tracks positions,
and logs every evaluation to the SQLite audit store.
"""

from typing import Optional, Dict, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import logging

from config import get_config
from agents.nifty_short_agent import NiftyShortAgent
from utils.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass
class IntradayTrade:
    """Record of a single intraday trade."""
    date: str
    entry_time: str
    entry_price: float
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    direction: str = "SHORT"
    pnl_points: float = 0.0
    pnl_rupees: float = 0.0
    exit_reason: str = ""
    lot_size: int = 25
    brokerage: float = 0.0
    swing_high: Optional[float] = None
    candle_1_low: Optional[float] = None
    candle_1_high: Optional[float] = None


@dataclass
class IntradayBacktestResult:
    """Results of an intraday backtest."""
    trades: List[IntradayTrade] = field(default_factory=list)
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    total_pnl_points: float = 0.0
    total_pnl_rupees: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win_points: float = 0.0
    avg_loss_points: float = 0.0
    max_win_points: float = 0.0
    max_loss_points: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_points: float = 0.0
    sharpe_ratio: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    equity_dates: List[str] = field(default_factory=list)
    run_id: Optional[int] = None


class IntradayBacktestEngine:
    """
    Intraday backtest engine for candle-based strategies.

    Processes 15-min candles day by day:
    1. Get previous day's daily OHLC
    2. Walk through today's candles chronologically
    3. Check entry/exit signals via the agent
    4. Force-close at session end
    5. Log everything to SQLite audit store
    """

    def __init__(
        self,
        agent: NiftyShortAgent,
        store: Optional[SQLiteStore] = None,
        initial_capital: float = 100_000.0,
    ):
        self.config = get_config()
        self.agent = agent
        self.store = store or SQLiteStore()
        self.initial_capital = initial_capital
        self.lot_size = agent.lot_size
        self.brokerage_per_order = self.config.intraday.brokerage_per_order
        self.force_close_time = self.config.intraday.force_close_time

    def run(
        self,
        intraday_data: pd.DataFrame,
        daily_data: Optional[pd.DataFrame] = None,
        symbol: str = "NIFTY50"
    ) -> IntradayBacktestResult:
        """
        Run the intraday backtest.

        Args:
            intraday_data: 15-min candle DataFrame (datetime index, OHLCV columns)
            daily_data: Daily OHLC for previous-day lookups. If None, derived from intraday.
            symbol: Symbol name for logging

        Returns:
            IntradayBacktestResult with all trades and metrics
        """
        # Group intraday candles by date
        intraday_data = intraday_data.sort_index()
        intraday_data['_date'] = intraday_data.index.date

        if daily_data is None:
            daily_data = self._derive_daily_from_intraday(intraday_data)

        trading_days = sorted(intraday_data['_date'].unique())
        logger.info(f"Running intraday backtest: {len(trading_days)} trading days, symbol={symbol}")

        # Create audit run
        run_id = self.store.create_run(
            strategy_name=self.agent.name,
            params=self.agent.get_params(),
            start_date=str(trading_days[0]),
            end_date=str(trading_days[-1])
        )

        result = IntradayBacktestResult(run_id=run_id)
        equity = self.initial_capital
        result.equity_curve.append(equity)
        result.equity_dates.append(str(trading_days[0]))
        peak_equity = equity

        for day in trading_days:
            day_candles = intraday_data[intraday_data['_date'] == day].copy()
            if len(day_candles) == 0:
                continue

            # Get previous day's daily candle
            prev_day = self._get_prev_daily(daily_data, day)
            if prev_day is None:
                continue

            trade = self._process_day(
                day=day,
                day_candles=day_candles,
                prev_day=prev_day,
                run_id=run_id
            )

            day_pnl_rupees = 0.0
            if trade is not None:
                result.trades.append(trade)
                day_pnl_rupees = trade.pnl_rupees

            result.daily_pnl[str(day)] = day_pnl_rupees
            equity += day_pnl_rupees
            result.equity_curve.append(equity)
            result.equity_dates.append(str(day))

            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            dd = peak_equity - equity
            if dd > result.max_drawdown_points * self.lot_size:
                result.max_drawdown_points = dd / self.lot_size if self.lot_size else 0

        # Calculate summary metrics
        self._calculate_summary(result)

        # Update run in DB
        self.store.update_run_metrics(
            run_id=run_id,
            total_pnl_points=result.total_pnl_points,
            total_pnl_rupees=result.total_pnl_rupees,
            total_trades=result.total_trades,
            win_rate=result.win_rate,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown_points
        )

        logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"P&L={result.total_pnl_points:.0f} pts (Rs.{result.total_pnl_rupees:,.0f}), "
            f"Win rate={result.win_rate:.1f}%"
        )

        return result

    def _process_day(
        self,
        day,
        day_candles: pd.DataFrame,
        prev_day: Dict[str, float],
        run_id: int
    ) -> Optional[IntradayTrade]:
        """Process a single trading day candle by candle."""
        self.agent.reset_day()

        in_position = False
        trade: Optional[IntradayTrade] = None
        entry_price = 0.0
        audit_batch = []

        force_close_time = pd.Timestamp(f"{day} {self.force_close_time}")

        for i, (ts, candle) in enumerate(day_candles.iterrows()):
            ohlc = {
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close']
            }

            condition_met = False
            signal_generated = False
            notes = ""

            if not in_position:
                # Entry cutoff check
                past_cutoff = False
                if self.agent.entry_cutoff_time != "00:00":
                    cutoff = pd.Timestamp(f"{day} {self.agent.entry_cutoff_time}")
                    if ts >= cutoff:
                        past_cutoff = True
                        notes = f"Past entry cutoff {self.agent.entry_cutoff_time}"

                if not past_cutoff:
                    # Check entry
                    should_enter, reason = self.agent.check_entry_conditions(
                        today_candles=day_candles.iloc[:i + 1],
                        prev_day_open=prev_day['open'],
                        prev_day_close=prev_day['close'],
                        candle_index=i
                    )
                    notes = reason

                    if should_enter:
                        condition_met = True
                        signal_generated = True
                        entry_price = candle['close']
                        in_position = True
                        trade = IntradayTrade(
                            date=str(day),
                            entry_time=str(ts),
                            entry_price=entry_price,
                            direction="SHORT",
                            lot_size=self.lot_size,
                            candle_1_low=self.agent._candle_1_low,
                            candle_1_high=self.agent._candle_1_high
                        )
            else:
                # Exit checks: stop loss -> swing high -> force close
                should_exit = False
                exit_reason = ""
                swing_high = None
                exit_price_override = None

                # 1. Stop loss check
                if self.agent.stop_loss_points > 0:
                    sl_price = entry_price + self.agent.stop_loss_points  # Short: SL above entry
                    if candle['high'] >= sl_price:
                        should_exit = True
                        exit_reason = f"Stop loss hit at {sl_price:.0f}"
                        exit_price_override = sl_price

                # 2. Swing high exit check
                if not should_exit:
                    should_exit, exit_reason, swing_high = self.agent.check_exit_conditions(
                        candles_so_far=day_candles.iloc[:i + 1],
                        entry_price=entry_price,
                        current_candle_idx=i
                    )

                # 3. Force close check
                if not should_exit and ts >= force_close_time:
                    should_exit = True
                    exit_reason = "Force close at session end"

                if should_exit:
                    condition_met = True
                    notes = exit_reason
                    exit_price = exit_price_override if exit_price_override is not None else candle['close']
                    pnl_points = entry_price - exit_price  # Short: profit = entry - exit
                    brokerage = self.brokerage_per_order * 2  # Entry + exit
                    pnl_rupees = (pnl_points * self.lot_size) - brokerage

                    trade.exit_time = str(ts)
                    trade.exit_price = exit_price
                    trade.pnl_points = pnl_points
                    trade.pnl_rupees = pnl_rupees
                    trade.exit_reason = exit_reason
                    trade.brokerage = brokerage
                    trade.swing_high = swing_high
                    in_position = False

            # Collect audit record
            audit_batch.append((
                run_id, str(day), i + 1, str(ts),
                ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'],
                prev_day['open'], prev_day['close'], prev_day.get('high'), prev_day.get('low'),
                int(condition_met), int(signal_generated), notes
            ))

        # If still in position at end of day (shouldn't happen due to force close, but safety net)
        if in_position and trade is not None and trade.exit_time is None:
            last_candle = day_candles.iloc[-1]
            exit_price = last_candle['close']
            pnl_points = entry_price - exit_price
            brokerage = self.brokerage_per_order * 2
            pnl_rupees = (pnl_points * self.lot_size) - brokerage

            trade.exit_time = str(day_candles.index[-1])
            trade.exit_price = exit_price
            trade.pnl_points = pnl_points
            trade.pnl_rupees = pnl_rupees
            trade.exit_reason = "End of day close"
            trade.brokerage = brokerage

        # Batch insert audit records
        self.store.insert_candle_audit_batch(audit_batch)

        # Insert trade to DB
        if trade is not None and trade.exit_time is not None:
            self.store.insert_trade(
                run_id=run_id,
                date=trade.date,
                entry_time=trade.entry_time,
                entry_price=trade.entry_price,
                exit_time=trade.exit_time,
                exit_price=trade.exit_price,
                direction=trade.direction,
                pnl_points=trade.pnl_points,
                pnl_rupees=trade.pnl_rupees,
                exit_reason=trade.exit_reason,
                lot_size=trade.lot_size,
                brokerage=trade.brokerage,
                swing_high=trade.swing_high,
                candle_1_low=trade.candle_1_low,
                candle_1_high=trade.candle_1_high
            )

        return trade if (trade is not None and trade.exit_time is not None) else None

    def _get_prev_daily(self, daily_data: pd.DataFrame, current_day) -> Optional[Dict[str, float]]:
        """Get the previous trading day's daily OHLC."""
        daily_dates = daily_data.index
        # Find dates strictly before current_day
        prev_dates = daily_dates[daily_dates < pd.Timestamp(current_day)]
        if len(prev_dates) == 0:
            return None

        prev_date = prev_dates[-1]
        row = daily_data.loc[prev_date]
        return {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        }

    def _derive_daily_from_intraday(self, intraday_data: pd.DataFrame) -> pd.DataFrame:
        """Derive daily OHLC from intraday candles."""
        daily = intraday_data.groupby('_date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        daily.index = pd.DatetimeIndex(daily.index)
        daily.index.name = 'date'
        return daily

    def _calculate_summary(self, result: IntradayBacktestResult):
        """Calculate summary metrics from trade list."""
        trades = result.trades
        result.total_trades = len(trades)

        if result.total_trades == 0:
            return

        pnl_list = [t.pnl_points for t in trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]

        result.total_pnl_points = sum(pnl_list)
        result.total_pnl_rupees = sum(t.pnl_rupees for t in trades)
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = (len(wins) / len(trades)) * 100 if trades else 0

        result.avg_win_points = np.mean(wins) if wins else 0.0
        result.avg_loss_points = np.mean(losses) if losses else 0.0
        result.max_win_points = max(wins) if wins else 0.0
        result.max_loss_points = min(losses) if losses else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        # Sharpe ratio from daily P&L
        daily_pnl_values = list(result.daily_pnl.values())
        if len(daily_pnl_values) > 1:
            daily_returns = pd.Series(daily_pnl_values)
            mean_ret = daily_returns.mean()
            std_ret = daily_returns.std()
            if std_ret > 0:
                result.sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252)
