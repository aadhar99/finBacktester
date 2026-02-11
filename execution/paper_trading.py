"""
Paper Trading Engine - Simulates live trading with real market data.

Connects to Zerodha WebSocket for live prices and executes trades
in simulation mode (no real money).
"""

import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time
import signal
import sys

from execution.portfolio import Portfolio
from execution.position import Position
from risk.manager import RiskManager
from agents.base_agent import BaseAgent, Signal, SignalType
from regime.filter import RegimeFilter
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from utils.alerts import create_alerter, TradeAlert
from config import get_config

logger = logging.getLogger(__name__)

# Try to import KiteTicker (optional dependency)
try:
    from kiteconnect import KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    logger.warning("KiteTicker not available. Install with: pip install kiteconnect")


class PaperTradingEngine:
    """
    Paper trading engine using live or simulated market data.

    Features:
    - Real-time price updates (via WebSocket or polling)
    - Virtual portfolio execution
    - All agents and risk management active
    - Performance tracking
    - Alert notifications
    """

    def __init__(
        self,
        initial_capital: float,
        agents: List[BaseAgent],
        symbols: List[str],
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        use_websocket: bool = False,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None
    ):
        """
        Initialize paper trading engine.

        Args:
            initial_capital: Starting capital
            agents: List of trading agents
            symbols: List of symbols to trade
            api_key: Zerodha API key (optional)
            access_token: Zerodha access token (optional)
            use_websocket: Use WebSocket for live data (requires Zerodha)
            telegram_bot_token: Telegram bot token for alerts
            telegram_chat_id: Telegram chat ID for alerts
        """
        self.config = get_config()
        self.initial_capital = initial_capital
        self.agents = agents
        self.symbols = symbols
        self.use_websocket = use_websocket and KITE_AVAILABLE

        # Trading components
        self.portfolio = Portfolio(initial_capital)
        self.risk_manager = RiskManager(self.portfolio)
        self.regime_filter = RegimeFilter()
        self.data_fetcher = DataFetcher(api_key, access_token)
        self.data_preprocessor = DataPreprocessor()

        # Alert system
        self.alerter = create_alerter(telegram_bot_token, telegram_chat_id)

        # Market data
        self.current_prices: Dict[str, float] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.nifty_data: Optional[pd.DataFrame] = None
        self.vix_data: Optional[pd.DataFrame] = None

        # WebSocket (if available)
        self.kws: Optional[KiteTicker] = None
        self.instrument_tokens: Dict[str, int] = {}

        # State
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.trades_today = 0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Paper trading engine initialized")
        logger.info(f"  Capital: ₹{initial_capital:,.0f}")
        logger.info(f"  Symbols: {len(symbols)}")
        logger.info(f"  Agents: {[a.name for a in agents]}")
        logger.info(f"  WebSocket: {'Enabled' if self.use_websocket else 'Disabled (using polling)'}")

    def load_historical_data(self, days: int = 100):
        """
        Load historical data for technical indicators.

        Args:
            days: Number of days of historical data to load
        """
        logger.info(f"Loading {days} days of historical data...")

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")

        # Load symbol data
        for symbol in self.symbols:
            try:
                df = self.data_fetcher.fetch_historical_data(symbol, start_date, end_date)
                df = self.data_preprocessor.prepare_for_backtest(df)
                self.historical_data[symbol] = df
                logger.info(f"✓ Loaded {len(df)} days for {symbol}")
            except Exception as e:
                logger.error(f"✗ Failed to load {symbol}: {e}")

        # Load regime data
        try:
            self.nifty_data = self.data_fetcher.fetch_nifty_data(start_date, end_date)
            self.vix_data = self.data_fetcher.fetch_vix_data(start_date, end_date)
            logger.info(f"✓ Loaded regime data")
        except Exception as e:
            logger.warning(f"Failed to load regime data: {e}")

        logger.info("Historical data loading complete")

    def start(self):
        """Start paper trading."""
        logger.info("=" * 70)
        logger.info("STARTING PAPER TRADING")
        logger.info("=" * 70)

        self.is_running = True
        self.start_time = datetime.now()

        # Load historical data
        self.load_historical_data()

        # Send startup alert
        self.alerter.send_system_alert(
            f"Paper trading started\n"
            f"Capital: ₹{self.initial_capital:,.0f}\n"
            f"Symbols: {', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}"
        )

        if self.use_websocket:
            self._start_websocket_mode()
        else:
            self._start_polling_mode()

    def _start_websocket_mode(self):
        """Start in WebSocket mode (real-time)."""
        if not KITE_AVAILABLE:
            logger.error("KiteTicker not available. Falling back to polling mode.")
            self._start_polling_mode()
            return

        logger.info("Starting WebSocket mode (real-time data)...")

        # Setup WebSocket (implementation would require Kite API key)
        # This is a placeholder - full implementation requires Zerodha credentials
        logger.warning("WebSocket mode requires valid Zerodha API credentials")
        logger.info("Falling back to polling mode...")
        self._start_polling_mode()

    def _start_polling_mode(self):
        """Start in polling mode (fetch prices periodically)."""
        logger.info("Starting polling mode (checking every 60 seconds)...")

        poll_interval = 60  # Check every 60 seconds

        try:
            while self.is_running:
                try:
                    self._poll_and_process()
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")

                # Sleep until next poll
                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("\nShutdown requested...")
            self.stop()

    def _poll_and_process(self):
        """Poll prices and process trading logic."""
        logger.info(f"Polling prices at {datetime.now().strftime('%H:%M:%S')}...")

        # Update current prices (using latest available data)
        for symbol in self.symbols:
            if symbol in self.historical_data:
                # In real implementation, fetch latest price from API
                # For now, use last available price from historical data
                self.current_prices[symbol] = self.historical_data[symbol]['close'].iloc[-1]

        if not self.current_prices:
            logger.warning("No prices available")
            return

        # Update portfolio valuation
        self.portfolio.update_portfolio_value(
            self.current_prices,
            pd.Timestamp.now()
        )

        # Detect market regime
        current_regime = None
        if self.nifty_data is not None:
            try:
                regime, metrics = self.regime_filter.detect_regime(
                    self.nifty_data,
                    self.vix_data,
                    pd.Timestamp.now()
                )
                current_regime = regime
                logger.info(f"Market regime: {regime.value}")
            except Exception as e:
                logger.debug(f"Regime detection failed: {e}")

        # Check for exits
        self._check_position_exits()

        # Check for entries
        self._check_position_entries(current_regime)

        # Log status
        self._log_status()

    def _check_position_exits(self):
        """Check if any positions should be exited."""
        positions_to_exit = []

        for symbol, position in self.portfolio.position_tracker.open_positions.items():
            if symbol not in self.current_prices:
                continue

            current_price = self.current_prices[symbol]

            # Check risk manager
            should_exit, reason = self.risk_manager.check_position_exit(
                symbol, current_price, pd.Timestamp.now()
            )

            if should_exit:
                positions_to_exit.append((symbol, current_price, reason))
                continue

            # Check stop loss / take profit
            if position.check_stop_loss(current_price):
                positions_to_exit.append((symbol, current_price, "Stop loss"))
            elif position.check_take_profit(current_price):
                positions_to_exit.append((symbol, current_price, "Take profit"))
            else:
                # Check agent exit logic
                if symbol in self.historical_data:
                    current_data = self.historical_data[symbol].iloc[-1]

                    for agent in self.agents:
                        should_exit, reason = agent.should_exit(
                            symbol=symbol,
                            entry_price=position.entry_price,
                            current_price=current_price,
                            current_data=current_data,
                            days_held=position.days_held
                        )

                        if should_exit:
                            positions_to_exit.append((symbol, current_price, reason))
                            break

        # Execute exits
        for symbol, price, reason in positions_to_exit:
            self._execute_sell(symbol, price, reason)

    def _check_position_entries(self, current_regime):
        """Check for new entry signals."""
        for symbol in self.symbols:
            # Skip if already have position
            if self.portfolio.position_tracker.has_position(symbol):
                continue

            # Skip if no historical data
            if symbol not in self.historical_data:
                continue

            # Generate signals from agents
            for agent in self.agents:
                # Check if agent enabled for regime
                if current_regime and not agent.is_enabled_for_regime(current_regime.value):
                    continue

                try:
                    signals = agent.generate_signals(
                        data=self.historical_data[symbol],
                        current_positions=self.portfolio.position_tracker.open_positions,
                        portfolio_value=self.portfolio.total_value,
                        market_regime=current_regime.value if current_regime else None
                    )

                    # Execute BUY signals
                    for signal in signals:
                        if signal.signal_type == SignalType.BUY:
                            self._execute_buy(signal)

                except Exception as e:
                    logger.error(f"Error generating signals from {agent.name} for {symbol}: {e}")

    def _execute_buy(self, signal: Signal):
        """Execute buy signal."""
        # Risk check
        approved, reason = self.risk_manager.check_signal(
            signal, self.current_prices, pd.Timestamp.now()
        )

        if not approved:
            logger.debug(f"Signal rejected: {signal.symbol} - {reason}")
            return

        # Apply slippage
        execution_price = signal.price * (1 + self.config.zerodha.slippage_bps / 10000)

        # Execute
        success = self.portfolio.execute_buy(
            symbol=signal.symbol,
            price=execution_price,
            quantity=signal.size,
            timestamp=pd.Timestamp.now(),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=signal.metadata.get('strategy', 'unknown'),
            commission_pct=self.config.zerodha.effective_cost_per_round_trip / 2,
            slippage_pct=self.config.zerodha.slippage_bps / 100
        )

        if success:
            self.trades_today += 1
            logger.info(f"📈 BUY: {signal.symbol} @ ₹{execution_price:.2f} x {signal.size}")

            # Send alert
            trade = TradeAlert(
                action="BUY",
                symbol=signal.symbol,
                price=execution_price,
                quantity=signal.size,
                reason=signal.reason
            )
            self.alerter.send_trade_alert(trade)

    def _execute_sell(self, symbol: str, price: float, reason: str):
        """Execute sell order."""
        execution_price = price * (1 - self.config.zerodha.slippage_bps / 10000)

        # Get position for P&L
        position = self.portfolio.position_tracker.get_position(symbol)

        success = self.portfolio.execute_sell(
            symbol=symbol,
            price=execution_price,
            timestamp=pd.Timestamp.now(),
            reason=reason,
            commission_pct=self.config.zerodha.effective_cost_per_round_trip / 2,
            slippage_pct=self.config.zerodha.slippage_bps / 100
        )

        if success:
            self.trades_today += 1

            # Get closed position for P&L
            closed_positions = [p for p in self.portfolio.position_tracker.closed_positions if p.symbol == symbol]
            if closed_positions:
                closed = closed_positions[-1]
                pnl = closed.realized_pnl
                pnl_pct = closed.realized_pnl_pct
            else:
                pnl = None
                pnl_pct = None

            logger.info(f"📉 SELL: {symbol} @ ₹{execution_price:.2f} - {reason}")

            # Send alert
            trade = TradeAlert(
                action="SELL",
                symbol=symbol,
                price=execution_price,
                quantity=position.quantity if position else 0,
                reason=reason,
                pnl=pnl,
                pnl_pct=pnl_pct
            )
            self.alerter.send_trade_alert(trade)

    def _log_status(self):
        """Log current status."""
        summary = self.portfolio.get_summary(self.current_prices)

        logger.info(f"Portfolio: ₹{summary['current_value']:,.0f} ({summary['total_return_pct']:+.2f}%)")
        logger.info(f"Positions: {summary['positions_count']}, Cash: ₹{summary['cash']:,.0f} ({summary['cash_pct']:.1f}%)")

    def stop(self):
        """Stop paper trading."""
        self.is_running = False

        logger.info("\n" + "=" * 70)
        logger.info("STOPPING PAPER TRADING")
        logger.info("=" * 70)

        # Send daily summary
        summary = self.portfolio.get_summary(self.current_prices)
        self.alerter.send_daily_summary(
            portfolio_value=summary['current_value'],
            daily_pnl=summary['current_value'] - self.initial_capital,
            daily_return_pct=summary['total_return_pct'],
            open_positions=summary['positions_count'],
            trades_today=self.trades_today
        )

        # Print final summary
        self._print_summary()

        # Send shutdown alert
        self.alerter.send_system_alert("Paper trading stopped")

    def _print_summary(self):
        """Print trading session summary."""
        summary = self.portfolio.get_summary(self.current_prices)

        logger.info(f"\nInitial Capital:  ₹{self.initial_capital:,.0f}")
        logger.info(f"Final Value:      ₹{summary['current_value']:,.0f}")
        logger.info(f"Total Return:     {summary['total_return_pct']:.2f}%")
        logger.info(f"Total Trades:     {summary['total_trades']}")
        logger.info(f"Open Positions:   {summary['positions_count']}")
        logger.info(f"Realized P&L:     ₹{summary['total_realized_pnl']:,.2f}")

        if self.start_time:
            duration = datetime.now() - self.start_time
            logger.info(f"Duration:         {duration}")

        logger.info("=" * 70)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("\nReceived shutdown signal...")
        self.stop()
        sys.exit(0)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from agents import MomentumAgent, ReversionAgent

    # Initialize engine
    engine = PaperTradingEngine(
        initial_capital=100_000,
        agents=[MomentumAgent(), ReversionAgent()],
        symbols=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
        use_websocket=False,  # Use polling mode
        # telegram_bot_token="YOUR_BOT_TOKEN",
        # telegram_chat_id="YOUR_CHAT_ID"
    )

    # Start trading
    try:
        engine.start()
    except KeyboardInterrupt:
        engine.stop()
