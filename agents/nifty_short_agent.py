"""
Nifty Short Intraday Strategy Agent.

Strategy logic:
  Entry: Short Nifty when today's open is inside previous day's body
         AND 3rd 15-min candle closes below 1st candle's low.
         Optional: 1st candle range >= min_first_candle_range points.
  Exit:  Price crosses above most recent swing high.
  Force close at 15:15 if still in position.

P&L modeled as Nifty points x lot size (futures proxy).
"""

from typing import Dict, Optional, Tuple, Any, List
import pandas as pd
import logging

from agents.base_agent import BaseAgent, Signal, SignalType

logger = logging.getLogger(__name__)


class NiftyShortAgent(BaseAgent):
    """Intraday Nifty short strategy based on opening range breakdown."""

    def __init__(
        self,
        min_first_candle_range: float = 75.0,
        entry_candle_index: int = 3,
        swing_high_lookback: int = 5,
        lot_size: int = 25,
        entry_cutoff_time: str = "14:00",
        stop_loss_points: float = 0.0,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            min_first_candle_range: Minimum range of 1st candle in points (0 to disable)
            entry_candle_index: Which candle triggers entry (1-indexed, default 3rd)
            swing_high_lookback: Number of candles to look back for swing high exit
            lot_size: Nifty lot size for position sizing
            entry_cutoff_time: No new entries after this time, HH:MM format ("00:00" to disable)
            stop_loss_points: Fixed stop loss in points above entry (0 to disable)
            config: Additional config dict
        """
        super().__init__(name="NiftyShortAgent", config=config)
        self.min_first_candle_range = min_first_candle_range
        self.entry_candle_index = entry_candle_index
        self.swing_high_lookback = swing_high_lookback
        self.lot_size = lot_size
        self.entry_cutoff_time = entry_cutoff_time
        self.stop_loss_points = stop_loss_points

        # State tracked per-day by the engine
        self._candle_1_high: Optional[float] = None
        self._candle_1_low: Optional[float] = None
        self._entry_price: Optional[float] = None
        self._swing_high: Optional[float] = None

    def get_params(self) -> Dict[str, Any]:
        """Return current strategy parameters as a dict."""
        return {
            'min_first_candle_range': self.min_first_candle_range,
            'entry_candle_index': self.entry_candle_index,
            'swing_high_lookback': self.swing_high_lookback,
            'lot_size': self.lot_size,
            'entry_cutoff_time': self.entry_cutoff_time,
            'stop_loss_points': self.stop_loss_points,
        }

    def reset_day(self):
        """Reset per-day state. Called by the engine at the start of each day."""
        self._candle_1_high = None
        self._candle_1_low = None
        self._entry_price = None
        self._swing_high = None

    def check_entry_conditions(
        self,
        today_candles: pd.DataFrame,
        prev_day_open: float,
        prev_day_close: float,
        candle_index: int
    ) -> Tuple[bool, str]:
        """
        Check if entry conditions are met at the given candle.

        Entry is checked on candle entry_candle_index and every candle after it.
        The first candle (>= entry_candle_index) that closes below candle 1's low
        triggers the short.

        Args:
            today_candles: DataFrame of today's 15-min candles processed so far
                           (index 0 = 1st candle, 1 = 2nd, etc.)
            prev_day_open: Previous day's open price
            prev_day_close: Previous day's close price
            candle_index: Current candle number (0-indexed)

        Returns:
            (should_enter, reason)
        """
        # Need at least entry_candle_index candles before we start checking
        if candle_index < self.entry_candle_index - 1:
            return False, "Not enough candles yet"

        candle_1 = today_candles.iloc[0]
        self._candle_1_high = candle_1['high']
        self._candle_1_low = candle_1['low']

        # Condition 1: Today's open inside previous day's body
        today_open = candle_1['open']
        prev_body_high = max(prev_day_open, prev_day_close)
        prev_body_low = min(prev_day_open, prev_day_close)

        if not (prev_body_low <= today_open <= prev_body_high):
            return False, f"Open {today_open:.0f} not inside prev body [{prev_body_low:.0f}, {prev_body_high:.0f}]"

        # Condition 2 (optional): 1st candle range check
        candle_1_range = candle_1['high'] - candle_1['low']
        if self.min_first_candle_range > 0 and candle_1_range < self.min_first_candle_range:
            return False, f"1st candle range {candle_1_range:.0f} < min {self.min_first_candle_range:.0f}"

        # Condition 3: Current candle closes below 1st candle's low
        current_candle = today_candles.iloc[candle_index]
        if current_candle['close'] >= candle_1['low']:
            return False, f"Candle {candle_index + 1} close {current_candle['close']:.0f} >= candle 1 low {candle_1['low']:.0f}"

        return True, (
            f"SHORT: open {today_open:.0f} in prev body [{prev_body_low:.0f},{prev_body_high:.0f}], "
            f"candle {candle_index + 1} close {current_candle['close']:.0f} < candle 1 low {candle_1['low']:.0f}"
        )

    def check_exit_conditions(
        self,
        candles_so_far: pd.DataFrame,
        entry_price: float,
        current_candle_idx: int
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Check if exit conditions are met.

        Args:
            candles_so_far: All candles processed today up to current
            entry_price: Short entry price
            current_candle_idx: Current candle index (0-indexed)

        Returns:
            (should_exit, reason, swing_high_used)
        """
        if current_candle_idx < 1:
            return False, "", None

        # Calculate swing high from recent candles (excluding current)
        lookback_start = max(0, current_candle_idx - self.swing_high_lookback)
        recent_candles = candles_so_far.iloc[lookback_start:current_candle_idx]

        if len(recent_candles) == 0:
            return False, "", None

        swing_high = recent_candles['high'].max()
        self._swing_high = swing_high

        current_candle = candles_so_far.iloc[current_candle_idx]

        # Exit if price crosses above swing high
        if current_candle['high'] > swing_high:
            return True, f"Price high {current_candle['high']:.0f} crossed swing high {swing_high:.0f}", swing_high

        return False, "", swing_high

    # ── BaseAgent interface (used by ensemble/daily engine, adapted for intraday) ──

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_positions: Dict[str, int],
        portfolio_value: float,
        market_regime: Optional[str] = None
    ) -> list[Signal]:
        """
        Generate signals — called by the intraday engine with today's candles.

        For intraday use, the engine calls check_entry_conditions / check_exit_conditions
        directly. This method provides compatibility with the BaseAgent interface.
        """
        return []

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        volatility: float,
        max_position_pct: float = 5.0
    ) -> int:
        """Fixed lot size for Nifty futures."""
        return self.lot_size

    def should_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        current_data: pd.Series,
        days_held: int
    ) -> Tuple[bool, str]:
        """
        BaseAgent interface for exit check.

        The intraday engine uses check_exit_conditions() directly for
        candle-level exit logic. This provides a fallback.
        """
        if self._swing_high is not None and current_price > self._swing_high:
            return True, f"Price {current_price:.0f} > swing high {self._swing_high:.0f}"
        return False, ""
