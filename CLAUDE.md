# Project: Quantitative Trading System for NSE

## Architecture
- **Config**: `config/settings.py` — All dataclass configs (ZerodhaConfig, RiskConfig, IntradayConfig, etc.) with `SystemConfig` master. Global instance via `get_config()`.
- **Agents**: `agents/` — BaseAgent ABC with `generate_signals()`, `should_exit()`, `calculate_position_size()`. Implementations: MomentumAgent, ReversionAgent, NiftyShortAgent, EnsembleAgent.
- **Execution**: `execution/` — Portfolio (cash/positions/P&L), PositionTracker, BacktestEngine (daily), IntradayBacktestEngine (15-min candles).
- **Data**: `data/fetcher.py` — DataFetcher with yfinance (primary), Kite API, CSV, synthetic fallback. Intraday uses `period=` param (not start/end) for yfinance sub-daily intervals.
- **Metrics**: `metrics/calculator.py` — MetricsCalculator with 20+ metrics (Sharpe, Sortino, drawdown, etc.).
- **Risk**: `risk/manager.py` — RiskManager with position size, exposure, daily loss, drawdown limits.
- **Dashboard**: `dashboard/app.py` — Streamlit app reading from SQLite audit store. 4 views: Strategy Results, Trade Analysis, Strategy Comparison, Audit Log.
- **Audit**: `utils/sqlite_store.py` — SQLite with backtest_runs, trades, candle_audit tables.

## Intraday Strategy (MVP)
- **NiftyShortAgent**: Short when open inside prev day body AND 3rd 15-min candle closes below 1st candle low. Exit on swing high cross. Force close at 15:15.
- **CLI**: `python scripts/run_intraday.py` with args: `--min-range`, `--entry-candle`, `--swing-lookback`, `--csv`, `--dashboard`
- **P&L model**: Nifty points x lot size (25) minus Rs.40/order brokerage (futures proxy).

## Key Interfaces (avoid re-reading these files)

### BaseAgent (`agents/base_agent.py`)
- `generate_signals(data: DataFrame, current_positions: Dict[str, int], portfolio_value: float, market_regime: Optional[str]) -> list[Signal]`
- `should_exit(symbol: str, entry_price: float, current_price: float, current_data: Series, days_held: int) -> Tuple[bool, str]`
- `calculate_position_size(symbol: str, price: float, portfolio_value: float, volatility: float, max_position_pct: float) -> int`
- Signal dataclass: `signal_type, symbol, timestamp, price, size, confidence, reason, stop_loss, take_profit, metadata`
- SignalType enum: BUY, SELL, HOLD, EXIT_LONG, EXIT_SHORT

### Portfolio (`execution/portfolio.py`)
- `__init__(initial_capital: float)`
- `execute_buy(symbol, price, quantity, timestamp, stop_loss=None, take_profit=None, strategy="unknown", commission_pct=0, slippage_pct=0) -> bool`
- `execute_sell(symbol, price, timestamp, reason="", commission_pct=0, slippage_pct=0) -> bool`
- `update_portfolio_value(prices: Dict[str, float], timestamp)`
- Properties: `total_value`, `total_return`, `current_drawdown`, `available_cash_pct`, `cash`
- `get_value_series() -> Series`, `get_returns_series() -> Series`, `get_trades_dataframe() -> DataFrame`

### PositionTracker (`execution/position.py`)
- `open_position(symbol, entry_price, quantity, timestamp, stop_loss, take_profit, strategy, commission, slippage) -> Position`
- `close_position(symbol, exit_price, timestamp, reason, commission, slippage) -> Optional[Position]`
- `has_position(symbol) -> bool`, `get_position(symbol) -> Optional[Position]`
- `get_total_exposure(prices) -> float`, `get_position_count() -> int`
- Position: `realized_pnl`, `realized_pnl_pct`, `unrealized_pnl(price)`, `check_stop_loss(price)`, `check_take_profit(price)`, `days_held`

### MetricsCalculator (`metrics/calculator.py`)
- `calculate_all_metrics(portfolio_values: Series, trades: DataFrame, closed_positions: List, initial_capital: float) -> PerformanceMetrics`
- `print_metrics(metrics: PerformanceMetrics)`
- PerformanceMetrics fields: total_return_pct, sharpe_ratio, sortino_ratio, max_drawdown_pct, win_rate_pct, profit_factor, total_trades, total_pnl, etc.

### RiskManager (`risk/manager.py`)
- `__init__(portfolio: Portfolio)`
- `check_signal(signal: Signal, current_prices: Dict, current_date) -> Tuple[bool, str]`
- `check_position_exit(symbol, current_price, current_date) -> Tuple[bool, str]`
- Checks: position size, exposure limit, daily loss, max drawdown, position count, cash reserve

### SQLiteStore (`utils/sqlite_store.py`)
- `create_run(strategy_name, params, start_date, end_date) -> int`
- `update_run_metrics(run_id, total_pnl_points, total_pnl_rupees, total_trades, win_rate, sharpe_ratio, max_drawdown)`
- `insert_trade(run_id, date, entry_time, entry_price, exit_time, exit_price, direction, pnl_points, pnl_rupees, exit_reason, lot_size, ...)`
- `insert_candle_audit_batch(records: List[tuple])`
- `get_runs() -> DataFrame`, `get_trades(run_id) -> DataFrame`, `get_candle_audit(run_id, date?) -> DataFrame`

### IntradayBacktestEngine (`execution/intraday_engine.py`)
- `__init__(agent: NiftyShortAgent, store: SQLiteStore, initial_capital: float)`
- `run(intraday_data: DataFrame, daily_data: Optional[DataFrame], symbol: str) -> IntradayBacktestResult`
- IntradayBacktestResult: trades, daily_pnl, total_pnl_points/rupees, win_rate, sharpe_ratio, equity_curve, run_id

### NiftyShortAgent (`agents/nifty_short_agent.py`)
- `__init__(min_first_candle_range=75.0, entry_candle_index=3, swing_high_lookback=5, lot_size=25)`
- `check_entry_conditions(today_candles, prev_day_open, prev_day_close, candle_index) -> Tuple[bool, str]`
- `check_exit_conditions(candles_so_far, entry_price, current_candle_idx) -> Tuple[bool, str, Optional[float]]`
- `reset_day()`, `get_params() -> Dict`

## yfinance Gotchas
- NSE symbol mapping: NIFTY50 -> ^NSEI, BANKNIFTY -> ^NSEBANK, stocks -> SYMBOL.NS
- **15m/5m/1h data**: Must use `period='60d'` NOT `start/end` dates — yfinance rejects date ranges for sub-daily intervals
- Max lookback for 15m: ~60 days
- `auto_adjust=False` to get raw prices
- Index data (^NSEI) volume is often 0 — that's normal
- Data comes with timezone (IST), we strip it with `tz_localize(None)`

## Key Patterns
- All configs are dataclasses with defaults, nested under SystemConfig
- New agents extend BaseAgent, new strategies plug into IntradayBacktestEngine
- SQLite DB at `data/backtest_audit.db` — no external DB needed for MVP

## Environment
- Use `python3` (not `python`) — `python` is not aliased on this machine

## Commands
- `python3 scripts/run_intraday.py` — Run intraday backtest
- `python3 scripts/run_intraday.py --min-range 0 --dashboard` — No filter + launch dashboard
- `streamlit run dashboard/app.py` — View results
- `python3 main.py` — Menu (option 5 for intraday)
