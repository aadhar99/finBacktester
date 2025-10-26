"""
Event-driven backtest engine.

Simulates realistic trading by:
- Processing data day by day (no look-ahead bias)
- Applying transaction costs and slippage
- Enforcing risk management rules
- Tracking all positions and P&L
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

from config import get_config, MarketRegime
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from agents.base_agent import BaseAgent, Signal, SignalType
from execution.portfolio import Portfolio
from execution.position import Position
from risk.manager import RiskManager
from regime.filter import RegimeFilter
from metrics.calculator import MetricsCalculator, PerformanceMetrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtest engine.

    Processes market data chronologically, generating signals,
    executing trades, and tracking performance.
    """

    def __init__(
        self,
        initial_capital: float,
        agents: List[BaseAgent],
        start_date: str,
        end_date: str,
        symbols: List[str],
        enable_regime_filter: bool = True
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            agents: List of trading agents
            start_date: Backtest start date
            end_date: Backtest end date
            symbols: List of symbols to trade
            enable_regime_filter: Enable market regime filtering
        """
        self.config = get_config()
        self.initial_capital = initial_capital
        self.agents = agents
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.enable_regime_filter = enable_regime_filter

        # Initialize components
        self.portfolio = Portfolio(initial_capital)
        self.risk_manager = RiskManager(self.portfolio)
        self.regime_filter = RegimeFilter() if enable_regime_filter else None
        self.metrics_calculator = MetricsCalculator()

        # Data
        self.data_fetcher = DataFetcher()
        self.data_preprocessor = DataPreprocessor()
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.nifty_data: Optional[pd.DataFrame] = None
        self.vix_data: Optional[pd.DataFrame] = None

        # State tracking
        self.current_date: Optional[pd.Timestamp] = None
        self.current_regime: Optional[MarketRegime] = None

        logger.info(f"Initialized backtest: {start_date} to {end_date}, {len(symbols)} symbols, ₹{initial_capital:,.0f}")

    def load_data(self):
        """Load and preprocess all market data."""
        logger.info("Loading market data...")

        # Load symbol data
        for symbol in self.symbols:
            try:
                df = self.data_fetcher.fetch_historical_data(symbol, self.start_date, self.end_date)
                df = self.data_preprocessor.prepare_for_backtest(df)
                df['symbol'] = symbol
                self.market_data[symbol] = df
                logger.info(f"Loaded {len(df)} days for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")

        # Load Nifty and VIX for regime detection
        if self.enable_regime_filter:
            try:
                self.nifty_data = self.data_fetcher.fetch_nifty_data(self.start_date, self.end_date)
                self.vix_data = self.data_fetcher.fetch_vix_data(self.start_date, self.end_date)
                logger.info(f"Loaded regime data: Nifty ({len(self.nifty_data)}), VIX ({len(self.vix_data)})")
            except Exception as e:
                logger.warning(f"Failed to load regime data: {e}")

        logger.info(f"Data loading complete: {len(self.market_data)} symbols")

    def run(self) -> PerformanceMetrics:
        """
        Run the backtest.

        Returns:
            PerformanceMetrics object with all results
        """
        logger.info("=" * 70)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 70)

        # Load data
        self.load_data()

        if not self.market_data:
            raise ValueError("No market data loaded")

        # Get common date range across all symbols
        all_dates = set()
        for df in self.market_data.values():
            all_dates.update(df.index)

        trading_dates = sorted(list(all_dates))

        if not trading_dates:
            raise ValueError("No trading dates available. Data may have been filtered out during preprocessing. Try using a longer date range (recommended: 1+ year)")

        logger.info(f"Trading period: {trading_dates[0].date()} to {trading_dates[-1].date()} ({len(trading_dates)} days)")

        # Run event loop
        for date in tqdm(trading_dates, desc="Backtesting"):
            self.current_date = date
            self._process_trading_day(date)

        # Calculate final metrics
        logger.info("\nBacktest complete. Calculating metrics...")
        metrics = self._calculate_final_metrics()

        return metrics

    def _process_trading_day(self, date: pd.Timestamp):
        """
        Process a single trading day.

        Args:
            date: Trading date to process
        """
        # 1. Get current prices for all symbols
        current_prices = self._get_current_prices(date)

        if not current_prices:
            return

        # 2. Update portfolio valuation
        self.portfolio.update_portfolio_value(current_prices, date)

        # 3. Detect market regime
        if self.enable_regime_filter and self.nifty_data is not None:
            try:
                regime, metrics = self.regime_filter.detect_regime(
                    self.nifty_data,
                    self.vix_data,
                    date
                )
                self.current_regime = regime
            except Exception as e:
                logger.debug(f"Regime detection failed: {e}")
                self.current_regime = None

        # 4. Check existing positions for exits
        self._check_position_exits(current_prices, date)

        # 5. Generate new signals from agents
        signals = self._generate_signals(date, current_prices)

        # 6. Execute approved signals
        self._execute_signals(signals, current_prices, date)

    def _get_current_prices(self, date: pd.Timestamp) -> Dict[str, float]:
        """Get current prices for all symbols."""
        prices = {}

        for symbol, df in self.market_data.items():
            if date in df.index:
                prices[symbol] = df.loc[date, 'close']

        return prices

    def _check_position_exits(self, current_prices: Dict[str, float], date: pd.Timestamp):
        """Check if any positions should be exited."""
        positions_to_exit = []

        for symbol, position in self.portfolio.position_tracker.open_positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            # Check risk manager forced exits
            should_exit, reason = self.risk_manager.check_position_exit(symbol, current_price, date)

            if should_exit:
                positions_to_exit.append((symbol, current_price, reason))
                continue

            # Check stop loss and take profit
            if position.check_stop_loss(current_price):
                positions_to_exit.append((symbol, current_price, "Stop loss hit"))
            elif position.check_take_profit(current_price):
                positions_to_exit.append((symbol, current_price, "Take profit hit"))
            else:
                # Check agent-specific exit logic
                if symbol in self.market_data:
                    df = self.market_data[symbol]
                    data_up_to_date = df[df.index <= date]

                    if len(data_up_to_date) > 0:
                        current_data = data_up_to_date.iloc[-1]

                        # Ask agent if should exit
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
            self._execute_exit(symbol, price, date, reason)

    def _generate_signals(self, date: pd.Timestamp, current_prices: Dict[str, float]) -> List[Signal]:
        """Generate trading signals from all agents."""
        all_signals = []

        for symbol in self.symbols:
            # Skip if we already have a position
            if self.portfolio.position_tracker.has_position(symbol):
                continue

            # Skip if no data available
            if symbol not in self.market_data:
                continue

            # Get data up to current date (no look-ahead)
            df = self.market_data[symbol]
            data_up_to_date = df[df.index <= date]

            if len(data_up_to_date) < 100:  # Need minimum data for indicators
                continue

            # Generate signals from each agent
            for agent in self.agents:
                # Check if agent is enabled for current regime
                if self.current_regime and not agent.is_enabled_for_regime(self.current_regime.value):
                    continue

                try:
                    signals = agent.generate_signals(
                        data=data_up_to_date,
                        current_positions=self.portfolio.position_tracker.open_positions,
                        portfolio_value=self.portfolio.total_value,
                        market_regime=self.current_regime.value if self.current_regime else None
                    )

                    all_signals.extend(signals)

                except Exception as e:
                    logger.error(f"Error generating signals from {agent.name} for {symbol}: {e}")

        return all_signals

    def _execute_signals(self, signals: List[Signal], current_prices: Dict[str, float], date: pd.Timestamp):
        """Execute approved signals."""
        for signal in signals:
            if signal.signal_type == SignalType.BUY:
                self._execute_buy(signal, current_prices, date)
            elif signal.signal_type == SignalType.SELL:
                self._execute_sell(signal, current_prices, date)

    def _execute_buy(self, signal: Signal, current_prices: Dict[str, float], date: pd.Timestamp):
        """Execute buy signal."""
        # Risk check
        approved, reason = self.risk_manager.check_signal(signal, current_prices, date)

        if not approved:
            logger.debug(f"Signal rejected: {signal.symbol} - {reason}")
            return

        # Apply slippage to execution price
        execution_price = signal.price * (1 + self.config.zerodha.slippage_bps / 10000)

        # Execute
        success = self.portfolio.execute_buy(
            symbol=signal.symbol,
            price=execution_price,
            quantity=signal.size,
            timestamp=date,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=signal.metadata.get('strategy', 'unknown'),
            commission_pct=self.config.zerodha.effective_cost_per_round_trip / 2,  # Half for buy
            slippage_pct=self.config.zerodha.slippage_bps / 100
        )

        if success:
            logger.info(f"BUY executed: {signal.symbol} @ ₹{execution_price:.2f} x {signal.size}")

    def _execute_sell(self, signal: Signal, current_prices: Dict[str, float], date: pd.Timestamp):
        """Execute sell signal."""
        # Apply slippage
        execution_price = signal.price * (1 - self.config.zerodha.slippage_bps / 10000)

        success = self.portfolio.execute_sell(
            symbol=signal.symbol,
            price=execution_price,
            timestamp=date,
            reason=signal.reason,
            commission_pct=self.config.zerodha.effective_cost_per_round_trip / 2,  # Half for sell
            slippage_pct=self.config.zerodha.slippage_bps / 100
        )

        if success:
            logger.info(f"SELL executed: {signal.symbol} @ ₹{execution_price:.2f}")

    def _execute_exit(self, symbol: str, price: float, date: pd.Timestamp, reason: str):
        """Exit a position."""
        execution_price = price * (1 - self.config.zerodha.slippage_bps / 10000)

        self.portfolio.execute_sell(
            symbol=symbol,
            price=execution_price,
            timestamp=date,
            reason=reason,
            commission_pct=self.config.zerodha.effective_cost_per_round_trip / 2,
            slippage_pct=self.config.zerodha.slippage_bps / 100
        )

    def _calculate_final_metrics(self) -> PerformanceMetrics:
        """Calculate final performance metrics."""
        portfolio_values = self.portfolio.get_value_series()
        trades = self.portfolio.get_trades_dataframe()
        closed_positions = self.portfolio.position_tracker.closed_positions

        metrics = self.metrics_calculator.calculate_all_metrics(
            portfolio_values=portfolio_values,
            trades=trades,
            closed_positions=closed_positions,
            initial_capital=self.initial_capital
        )

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Initial Capital:     ₹{self.initial_capital:,.0f}")
        logger.info(f"Final Value:         ₹{self.portfolio.total_value:,.0f}")
        logger.info(f"Total Return:        {metrics.total_return_pct:.2f}%")
        logger.info(f"Annualized Return:   {metrics.annualized_return_pct:.2f}%")
        logger.info(f"Sharpe Ratio:        {metrics.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown:        {metrics.max_drawdown_pct:.2f}%")
        logger.info(f"Total Trades:        {metrics.total_trades}")
        logger.info(f"Win Rate:            {metrics.win_rate_pct:.2f}%")
        logger.info("=" * 70)

        return metrics
