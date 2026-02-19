"""
Paper Trading Runner - Execute Trades Based on Daily Signals

Runs daily to:
1. Get signals from daily scan
2. Execute paper trades
3. Update positions with current prices
4. Track performance
5. Generate reports

Usage:
    python3 scripts/run_paper_trading.py
    python3 scripts/run_paper_trading.py --initial-capital 2000000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import asyncio
import argparse
import os
import json
from datetime import datetime, date
import pandas as pd

from paper_trading.portfolio import PaperTradingPortfolio
from agents.smart_money_agent import SmartMoneyTradingAgent
from utils.smart_money_sqlite import SmartMoneySQLite
from utils.llm import LLMManager, LLMConfig, LLMProvider
from data.fetcher import DataFetcher
from utils.price_cache import PriceCache
from utils.alert_manager import AlertManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingRunner:
    """
    Automated paper trading based on smart money signals.
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        portfolio_file: str = "paper_trading/portfolio_state.json"
    ):
        """
        Initialize paper trading runner.

        Args:
            initial_capital: Starting capital for new portfolio
            portfolio_file: File to save/load portfolio state
        """
        self.portfolio_file = portfolio_file
        self.data_fetcher = DataFetcher()
        self.price_cache = PriceCache(ttl_seconds=300)  # 5 minute cache
        self.alerts = AlertManager()  # Alert notifications

        # Load or create portfolio
        if os.path.exists(portfolio_file):
            logger.info(f"📂 Loading existing portfolio from {portfolio_file}")
            self.portfolio = self._load_portfolio(portfolio_file)
        else:
            logger.info(f"🆕 Creating new portfolio: ₹{initial_capital:,.0f}")
            self.portfolio = PaperTradingPortfolio(initial_capital=initial_capital)

        # Initialize components
        self.db = SmartMoneySQLite()

        # LLM (optional, using without for speed)
        self.llm = None

        self.trading_agent = SmartMoneyTradingAgent(
            self.db,
            self.llm,
            use_llm=False  # Disabled for speed
        )

        logger.info("✅ Paper trading runner initialized")

    def _load_portfolio(self, filename: str) -> PaperTradingPortfolio:
        """Load portfolio from file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        portfolio = PaperTradingPortfolio(initial_capital=data['initial_capital'])
        portfolio.cash = data['cash']
        portfolio.total_trades = data['total_trades']
        portfolio.winning_trades = data['winning_trades']
        portfolio.losing_trades = data['losing_trades']
        portfolio.total_pnl = data['total_pnl']
        portfolio.max_drawdown = data['max_drawdown_pct']
        portfolio.peak_portfolio_value = data.get('current_value', portfolio.initial_capital)

        # Reconstruct positions
        from paper_trading.portfolio import PaperPosition
        for pos_data in data.get('positions', []):
            pos = PaperPosition(
                symbol=pos_data['symbol'],
                entry_date=date.fromisoformat(pos_data['entry_date']),
                entry_price=pos_data['entry_price'],
                quantity=pos_data['quantity'],
                signal_type=pos_data['signal_type'],
                confidence=pos_data['confidence'],
                stop_loss=pos_data['stop_loss'],
                take_profit=pos_data['take_profit'],
                pattern_type=pos_data['pattern_type']
            )
            pos.current_price = pos_data['current_price']
            pos.days_held = pos_data['days_held']
            portfolio.positions[pos.symbol] = pos

        # Reconstruct closed positions
        for pos_data in data.get('closed_trades', []):
            pos = PaperPosition(
                symbol=pos_data['symbol'],
                entry_date=date.fromisoformat(pos_data['entry_date']),
                entry_price=pos_data['entry_price'],
                quantity=pos_data['quantity'],
                signal_type=pos_data['signal_type'],
                confidence=pos_data['confidence'],
                stop_loss=pos_data['stop_loss'],
                take_profit=pos_data['take_profit'],
                pattern_type=pos_data['pattern_type']
            )
            if pos_data['exit_date']:
                pos.exit_date = date.fromisoformat(pos_data['exit_date'])
                pos.exit_price = pos_data['exit_price']
                pos.exit_reason = pos_data['exit_reason']
                pos.realized_pnl = pos_data['realized_pnl']
                pos.realized_pnl_pct = pos_data['realized_pnl_pct']
            portfolio.closed_positions.append(pos)

        logger.info(f"  ✅ Loaded portfolio: ₹{portfolio.get_portfolio_value():,.0f}")
        logger.info(f"     Open positions: {len(portfolio.positions)}")
        logger.info(f"     Closed trades: {len(portfolio.closed_positions)}")

        return portfolio

    async def run_daily_update(self):
        """
        Run daily paper trading update.

        1. Get new signals
        2. Execute new trades
        3. Update existing positions
        4. Check exits
        5. Save state
        """
        logger.info("=" * 70)
        logger.info(f"PAPER TRADING - Daily Update - {date.today()}")
        logger.info("=" * 70)

        # Step 1: Update existing positions with current prices
        logger.info("\n📈 Updating positions with current prices...")
        await self._update_positions()

        # Step 2: Get new signals
        logger.info("\n🔍 Getting new trading signals...")
        signals = await self._get_new_signals()

        # Step 3: Execute new signals
        if signals:
            logger.info(f"\n💼 Executing {len(signals)} new signals...")
            self._execute_signals(signals)
        else:
            logger.info("\n  ℹ️  No new signals to execute")

        # Step 4: Save portfolio state
        logger.info("\n💾 Saving portfolio state...")
        self.portfolio.save_to_file(self.portfolio_file)

        # Step 5: Print summary
        logger.info("\n")
        self.portfolio.print_summary()

        # Step 6: Generate daily report
        self._generate_daily_report()

        # Step 7: Send daily summary alert
        logger.info("\n📧 Sending daily summary...")
        num_signals = len(signals) if signals else 0
        self.alerts.send_daily_summary(
            portfolio_value=self.portfolio.get_portfolio_value(),
            total_return_pct=self.portfolio.get_total_return(),
            open_positions=len(self.portfolio.positions),
            total_trades=self.portfolio.total_trades,
            win_rate=self.portfolio.get_win_rate(),
            new_signals=num_signals
        )

        logger.info("\n✅ Daily paper trading update complete!")
        logger.info("=" * 70)

    async def _update_positions(self):
        """Update all positions with current market prices."""
        if not self.portfolio.positions:
            logger.info("  ℹ️  No open positions to update")
            return

        # Track closed positions count before update
        closed_count_before = len(self.portfolio.closed_positions)

        symbols = list(self.portfolio.positions.keys())
        current_prices = {}

        for symbol in symbols:
            try:
                # Check cache first
                cached_price = self.price_cache.get_price(symbol)
                if cached_price is not None:
                    current_prices[symbol] = cached_price['close']
                    logger.info(f"  {symbol}: ₹{cached_price['close']:.2f} (cached)")
                    continue

                # Fetch current price from API
                yf_symbol = f"{symbol}.NS"
                df = self.data_fetcher.fetch_data(
                    symbol=yf_symbol,
                    start_date=(date.today().replace(day=1)).strftime('%Y-%m-%d'),
                    end_date=date.today().strftime('%Y-%m-%d'),
                    interval='1d'
                )

                if df is not None and not df.empty:
                    current_price = float(df['close'].iloc[-1])
                    volume = float(df['volume'].iloc[-1]) if df['volume'].iloc[-1] > 0 else 1000000
                    current_prices[symbol] = current_price

                    # Cache the price
                    self.price_cache.set_price(symbol, current_price, volume)

                    logger.info(f"  {symbol}: ₹{current_price:.2f}")
                else:
                    logger.warning(f"  ⚠️  {symbol}: No price data")

            except Exception as e:
                logger.warning(f"  ⚠️  {symbol}: Error fetching price - {e}")

        # Update positions
        if current_prices:
            self.portfolio.update_positions(current_prices)
            logger.info(f"  ✅ Updated {len(current_prices)} positions")

            # Check for new closed positions and send alerts
            closed_count_after = len(self.portfolio.closed_positions)
            if closed_count_after > closed_count_before:
                # New positions were closed, send alerts
                newly_closed = self.portfolio.closed_positions[closed_count_before:]
                for position in newly_closed:
                    self.alerts.send_position_exit(
                        symbol=position.symbol,
                        entry_price=position.entry_price,
                        exit_price=position.exit_price,
                        pnl=position.realized_pnl,
                        pnl_pct=position.realized_pnl_pct,
                        reason=position.exit_reason,
                        days_held=position.days_held
                    )

    async def _get_new_signals(self) -> list:
        """Get new trading signals from smart money agent."""
        # Get opportunities from tracker
        report = await self.trading_agent.get_opportunities()

        if not report.opportunities:
            return []

        # Get current prices for each symbol
        symbols = [opp.symbol for opp in report.opportunities]

        # Fetch real prices from yfinance (with caching)
        price_data = {}
        for symbol in symbols:
            try:
                # Check cache first
                cached_price = self.price_cache.get_price(symbol)
                if cached_price is not None:
                    price_data[symbol] = cached_price
                    logger.info(f"  {symbol}: ₹{cached_price['close']:.2f} (cached)")
                    continue

                # Fetch from API
                yf_symbol = f"{symbol}.NS"
                df = self.data_fetcher.fetch_data(
                    symbol=yf_symbol,
                    start_date=(date.today().replace(day=1)).strftime('%Y-%m-%d'),
                    end_date=date.today().strftime('%Y-%m-%d'),
                    interval='1d'
                )

                if df is not None and not df.empty:
                    close_price = float(df['close'].iloc[-1])
                    volume = float(df['volume'].iloc[-1]) if df['volume'].iloc[-1] > 0 else 1000000

                    price_data[symbol] = {'close': close_price, 'volume': volume}

                    # Cache the price
                    self.price_cache.set_price(symbol, close_price, volume)

                    logger.info(f"  {symbol}: ₹{close_price:.2f}")
                else:
                    logger.warning(f"  ⚠️  {symbol}: No price data, skipping")
            except Exception as e:
                logger.warning(f"  ⚠️  {symbol}: Error fetching price - {e}")

        if not price_data:
            logger.warning("  ⚠️  No valid price data fetched, skipping signal generation")
            return []

        # Build DataFrame from real prices
        market_data = pd.DataFrame([
            {'symbol': symbol, 'close': data['close'], 'volume': data['volume']}
            for symbol, data in price_data.items()
        ]).set_index('symbol')

        # Generate signals
        signals = self.trading_agent.generate_signals(
            data=market_data,
            current_positions=self.portfolio.positions,
            portfolio_value=self.portfolio.get_portfolio_value()
        )

        logger.info(f"  ✅ Generated {len(signals)} new signals")

        return signals

    def _execute_signals(self, signals: list):
        """Execute trading signals as paper trades."""
        for signal in signals:
            # Execute trade
            success = self.portfolio.execute_signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type.name,
                price=signal.price,
                quantity=signal.size,
                confidence=signal.confidence,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                pattern_type=signal.metadata.get('pattern', 'UNKNOWN')
            )

            # Send alert if trade executed
            if success:
                self.alerts.send_position_opened(
                    symbol=signal.symbol,
                    signal_type=signal.signal_type.name,
                    price=signal.price,
                    quantity=signal.size,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    confidence=signal.confidence
                )

    def _generate_daily_report(self):
        """Generate and save daily performance report."""
        report = self.portfolio.get_performance_report()

        filename = f"paper_trading/reports/daily_{date.today()}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Convert dates to strings for JSON serialization
        def date_converter(obj):
            if isinstance(obj, date):
                return obj.isoformat()
            return obj

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=date_converter)

        logger.info(f"  📊 Daily report saved to {filename}")


async def main():
    """Run paper trading from command line."""
    parser = argparse.ArgumentParser(description='Paper Trading Runner')
    parser.add_argument('--initial-capital', type=float, default=1000000,
                        help='Initial capital (default: 1000000)')
    parser.add_argument('--portfolio-file', type=str, default='paper_trading/portfolio_state.json',
                        help='Portfolio state file')
    parser.add_argument('--show-only', action='store_true',
                        help='Only show current portfolio, do not update')

    args = parser.parse_args()

    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    os.makedirs('paper_trading', exist_ok=True)

    # Initialize runner
    runner = PaperTradingRunner(
        initial_capital=args.initial_capital,
        portfolio_file=args.portfolio_file
    )

    if args.show_only:
        # Just show current state
        runner.portfolio.print_summary()
    else:
        # Run daily update
        await runner.run_daily_update()


if __name__ == "__main__":
    asyncio.run(main())
