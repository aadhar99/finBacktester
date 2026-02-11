"""
Transaction Cost Model - India-specific (Zerodha)

CRITICAL: ALL backtests MUST include transaction costs
Lesson from research: "Strategies showing 2% monthly returns potentially lose money
after accounting for 0.05% slippage and 0.1% transaction costs"

This module calculates realistic transaction costs for NSE trading via Zerodha.
"""

import logging
from typing import Dict, Tuple
from decimal import Decimal

logger = logging.getLogger(__name__)


class TransactionCostModel:
    """
    Zerodha pricing model for NSE trading (as of 2026).

    Based on: https://zerodha.com/charges

    All costs in INR (₹)
    """

    # Zerodha Brokerage
    BROKERAGE_PER_ORDER = 20.00  # ₹20 flat per order (equity delivery)
    BROKERAGE_PER_ORDER_INTRADAY = 20.00  # ₹20 or 0.03% (whichever is lower)
    BROKERAGE_MAX_INTRADAY_PCT = 0.03  # 0.03% cap for intraday

    # Government Taxes
    STT_DELIVERY_BUY = 0.0  # No STT on buy side for delivery
    STT_DELIVERY_SELL = 0.025  # 0.025% on sell side for delivery
    STT_INTRADAY = 0.025  # 0.025% on sell side for intraday

    # Exchange Transaction Charges
    EXCHANGE_TXN_CHARGE_NSE = 0.00325  # NSE: 0.00325% (₹325 per lakh)

    # SEBI Charges
    SEBI_CHARGES = 0.0001  # ₹10 per crore turnover

    # Stamp Duty (Maharashtra - adjust per state)
    STAMP_DUTY_BUY = 0.015  # 0.015% on buy side
    STAMP_DUTY_SELL = 0.003  # 0.003% on sell side

    # GST
    GST_RATE = 0.18  # 18% on brokerage + exchange fees

    # Market Impact / Slippage
    SLIPPAGE_BPS = 5  # 5 basis points = 0.05%
    SLIPPAGE_PCT = 0.05

    # Safety Margin (for unexpected costs)
    SAFETY_MARGIN_PCT = 0.02  # 2% buffer

    def __init__(self, is_intraday: bool = True):
        """
        Initialize transaction cost model.

        Args:
            is_intraday: True for intraday, False for delivery
        """
        self.is_intraday = is_intraday
        self.total_trades_calculated = 0
        self.total_costs_calculated = Decimal('0.0')

    def calculate_buy_cost(
        self,
        quantity: int,
        price: float,
        include_slippage: bool = True
    ) -> Dict[str, float]:
        """
        Calculate total cost for BUY order.

        Args:
            quantity: Number of shares
            price: Price per share (₹)
            include_slippage: Include market impact/slippage

        Returns:
            Dict with breakdown of all costs
        """
        trade_value = quantity * price

        # Brokerage
        if self.is_intraday:
            brokerage = min(
                self.BROKERAGE_PER_ORDER_INTRADAY,
                trade_value * self.BROKERAGE_MAX_INTRADAY_PCT / 100
            )
        else:
            brokerage = self.BROKERAGE_PER_ORDER

        # STT (no STT on buy for delivery, 0.025% for intraday)
        if self.is_intraday:
            stt = trade_value * self.STT_INTRADAY / 100
        else:
            stt = 0.0  # No STT on buy side for delivery

        # Exchange transaction charges
        exchange_txn = trade_value * self.EXCHANGE_TXN_CHARGE_NSE / 100

        # SEBI charges
        sebi_charges = trade_value * self.SEBI_CHARGES / 100

        # Stamp duty (on buy side)
        stamp_duty = trade_value * self.STAMP_DUTY_BUY / 100

        # GST (on brokerage + exchange fees)
        gst_base = brokerage + exchange_txn
        gst = gst_base * self.GST_RATE

        # Slippage (market impact - you pay slightly higher)
        if include_slippage:
            slippage = trade_value * self.SLIPPAGE_PCT / 100
        else:
            slippage = 0.0

        # Total cost
        total_cost = (
            brokerage + stt + exchange_txn + sebi_charges +
            stamp_duty + gst + slippage
        )

        return {
            'trade_value': round(trade_value, 2),
            'brokerage': round(brokerage, 2),
            'stt': round(stt, 2),
            'exchange_txn': round(exchange_txn, 2),
            'sebi_charges': round(sebi_charges, 4),
            'stamp_duty': round(stamp_duty, 2),
            'gst': round(gst, 2),
            'slippage': round(slippage, 2),
            'total_cost': round(total_cost, 2),
            'cost_pct': round((total_cost / trade_value) * 100, 4)
        }

    def calculate_sell_cost(
        self,
        quantity: int,
        price: float,
        include_slippage: bool = True
    ) -> Dict[str, float]:
        """
        Calculate total cost for SELL order.

        Args:
            quantity: Number of shares
            price: Price per share (₹)
            include_slippage: Include market impact/slippage

        Returns:
            Dict with breakdown of all costs
        """
        trade_value = quantity * price

        # Brokerage
        if self.is_intraday:
            brokerage = min(
                self.BROKERAGE_PER_ORDER_INTRADAY,
                trade_value * self.BROKERAGE_MAX_INTRADAY_PCT / 100
            )
        else:
            brokerage = self.BROKERAGE_PER_ORDER

        # STT (0.025% on sell side)
        stt = trade_value * self.STT_DELIVERY_SELL / 100

        # Exchange transaction charges
        exchange_txn = trade_value * self.EXCHANGE_TXN_CHARGE_NSE / 100

        # SEBI charges
        sebi_charges = trade_value * self.SEBI_CHARGES / 100

        # Stamp duty (on sell side)
        stamp_duty = trade_value * self.STAMP_DUTY_SELL / 100

        # GST (on brokerage + exchange fees)
        gst_base = brokerage + exchange_txn
        gst = gst_base * self.GST_RATE

        # Slippage (market impact - you receive slightly lower)
        if include_slippage:
            slippage = trade_value * self.SLIPPAGE_PCT / 100
        else:
            slippage = 0.0

        # Total cost
        total_cost = (
            brokerage + stt + exchange_txn + sebi_charges +
            stamp_duty + gst + slippage
        )

        return {
            'trade_value': round(trade_value, 2),
            'brokerage': round(brokerage, 2),
            'stt': round(stt, 2),
            'exchange_txn': round(exchange_txn, 2),
            'sebi_charges': round(sebi_charges, 4),
            'stamp_duty': round(stamp_duty, 2),
            'gst': round(gst, 2),
            'slippage': round(slippage, 2),
            'total_cost': round(total_cost, 2),
            'cost_pct': round((total_cost / trade_value) * 100, 4)
        }

    def calculate_round_trip_cost(
        self,
        quantity: int,
        buy_price: float,
        sell_price: float,
        include_slippage: bool = True
    ) -> Dict[str, float]:
        """
        Calculate total cost for complete round-trip (buy + sell).

        Args:
            quantity: Number of shares
            buy_price: Buy price per share (₹)
            sell_price: Sell price per share (₹)
            include_slippage: Include market impact/slippage

        Returns:
            Dict with complete cost breakdown
        """
        buy_costs = self.calculate_buy_cost(quantity, buy_price, include_slippage)
        sell_costs = self.calculate_sell_cost(quantity, sell_price, include_slippage)

        total_cost = buy_costs['total_cost'] + sell_costs['total_cost']
        avg_trade_value = (buy_costs['trade_value'] + sell_costs['trade_value']) / 2

        # Calculate breakeven price movement needed
        cost_to_recover = total_cost
        breakeven_move_pct = (cost_to_recover / buy_costs['trade_value']) * 100

        # Track statistics
        self.total_trades_calculated += 1
        self.total_costs_calculated += Decimal(str(total_cost))

        return {
            'buy_costs': buy_costs,
            'sell_costs': sell_costs,
            'total_cost': round(total_cost, 2),
            'total_cost_pct': round((total_cost / avg_trade_value) * 100, 4),
            'breakeven_move_pct': round(breakeven_move_pct, 4),
            'gross_pnl': round((sell_price - buy_price) * quantity, 2),
            'net_pnl': round((sell_price - buy_price) * quantity - total_cost, 2)
        }

    def get_minimum_profit_target(
        self,
        quantity: int,
        buy_price: float,
        include_safety_margin: bool = True
    ) -> Tuple[float, float]:
        """
        Calculate minimum profit target to break even after costs.

        Args:
            quantity: Number of shares
            buy_price: Buy price per share (₹)
            include_safety_margin: Include 2% safety buffer

        Returns:
            (minimum_sell_price, minimum_profit_pct)
        """
        # Calculate round-trip costs assuming same buy/sell price
        costs = self.calculate_round_trip_cost(quantity, buy_price, buy_price)
        total_cost = costs['total_cost']

        # Calculate minimum price movement to cover costs
        min_move = total_cost / quantity

        # Add safety margin if requested
        if include_safety_margin:
            min_move *= (1 + self.SAFETY_MARGIN_PCT)

        min_sell_price = buy_price + min_move
        min_profit_pct = (min_move / buy_price) * 100

        return (round(min_sell_price, 2), round(min_profit_pct, 4))

    def validate_strategy_profitability(
        self,
        avg_profit_pct: float,
        avg_trade_size: float
    ) -> Dict[str, any]:
        """
        Validate if a strategy can be profitable after costs.

        Args:
            avg_profit_pct: Average profit % per trade (before costs)
            avg_trade_size: Average trade size in ₹

        Returns:
            Validation result with recommendations
        """
        # Estimate typical costs
        typical_quantity = int(avg_trade_size / 100)  # Assume ₹100 stock
        costs = self.calculate_round_trip_cost(typical_quantity, 100, 100)
        cost_pct = costs['total_cost_pct']

        # Net profit after costs
        net_profit_pct = avg_profit_pct - cost_pct

        # Validation
        is_profitable = net_profit_pct > 0
        is_recommended = net_profit_pct > 0.3  # At least 0.3% net profit

        recommendation = ""
        if net_profit_pct < 0:
            recommendation = "❌ REJECT: Strategy loses money after costs"
        elif net_profit_pct < 0.3:
            recommendation = "⚠️  CAUTION: Low margin, high risk of losses"
        elif net_profit_pct < 0.5:
            recommendation = "✅ ACCEPTABLE: Positive but slim margin"
        else:
            recommendation = "✅ GOOD: Healthy profit margin after costs"

        return {
            'gross_profit_pct': round(avg_profit_pct, 4),
            'cost_pct': round(cost_pct, 4),
            'net_profit_pct': round(net_profit_pct, 4),
            'is_profitable': is_profitable,
            'is_recommended': is_recommended,
            'recommendation': recommendation,
            'min_profit_needed_pct': round(cost_pct + 0.3, 4)  # Costs + 0.3% buffer
        }

    def get_statistics(self) -> Dict[str, any]:
        """Get cost calculation statistics."""
        return {
            'total_trades_calculated': self.total_trades_calculated,
            'total_costs_calculated': float(self.total_costs_calculated),
            'avg_cost_per_trade': (
                float(self.total_costs_calculated) / self.total_trades_calculated
                if self.total_trades_calculated > 0 else 0.0
            )
        }


# ============================================================================
# EXAMPLE USAGE AND TESTS
# ============================================================================

def example_usage():
    """Example usage of transaction cost model."""

    print("=" * 70)
    print("TRANSACTION COST MODEL - Zerodha India (NSE)")
    print("=" * 70)

    # Example 1: RELIANCE trade
    print("\n📊 Example 1: RELIANCE (₹2,500) - Intraday")
    print("-" * 70)

    model = TransactionCostModel(is_intraday=True)

    quantity = 20  # 20 shares
    buy_price = 2500.00
    sell_price = 2525.00  # +1% move

    result = model.calculate_round_trip_cost(quantity, buy_price, sell_price)

    print(f"Trade: Buy {quantity} @ ₹{buy_price}, Sell @ ₹{sell_price}")
    print(f"Gross P&L: ₹{result['gross_pnl']}")
    print(f"Total Costs: ₹{result['total_cost']} ({result['total_cost_pct']}%)")
    print(f"Net P&L: ₹{result['net_pnl']}")
    print(f"Breakeven: Need {result['breakeven_move_pct']}% move to cover costs")

    # Example 2: Minimum profit target
    print("\n📊 Example 2: Minimum Profit Target")
    print("-" * 70)

    min_price, min_pct = model.get_minimum_profit_target(quantity, buy_price)
    print(f"Buy Price: ₹{buy_price}")
    print(f"Minimum Sell Price: ₹{min_price} (+{min_pct}%)")
    print(f"Anything below ₹{min_price} results in net loss!")

    # Example 3: Strategy validation
    print("\n📊 Example 3: Strategy Profitability Check")
    print("-" * 70)

    strategies = [
        ("High-frequency (0.3% avg)", 0.3, 50000),
        ("Swing trading (1.5% avg)", 1.5, 50000),
        ("Position trading (3% avg)", 3.0, 50000),
    ]

    for name, profit_pct, trade_size in strategies:
        validation = model.validate_strategy_profitability(profit_pct, trade_size)
        print(f"\n{name}:")
        print(f"  Gross: {validation['gross_profit_pct']}%")
        print(f"  Costs: {validation['cost_pct']}%")
        print(f"  Net: {validation['net_profit_pct']}%")
        print(f"  {validation['recommendation']}")

    print("\n" + "=" * 70)
    print("✅ Transaction cost model ready for use in backtesting")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
