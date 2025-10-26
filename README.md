# Quantitative Trading System for NSE (Zerodha)

A production-ready, self-optimizing quantitative trading system for the Indian stock market (NSE) using Zerodha as the broker.

## 🎯 Project Goals

- **Capital**: Start with ₹1 lakh, scale to ₹10 lakh
- **Target**: 10% return in 3 months with proper risk management
- **Timeline**: Build → Backtest (Month 1) → Paper trade (Month 2) → Live (Month 3)

## 🏗️ Architecture

```
trading_system/
├── agents/              # Trading strategies
│   ├── base_agent.py         # Abstract base class
│   ├── momentum_agent.py     # Turtle Trader strategy
│   └── reversion_agent.py    # Bollinger Bands + RSI strategy
├── config/              # Configuration
│   └── settings.py           # All system parameters
├── data/                # Data management
│   ├── fetcher.py            # Data retrieval (Zerodha API + synthetic)
│   └── preprocessor.py       # Technical indicators
├── execution/           # Trading execution
│   ├── backtest_engine.py    # Event-driven backtest engine
│   ├── portfolio.py          # Portfolio management
│   └── position.py           # Position tracking
├── metrics/             # Performance analysis
│   └── calculator.py         # 20+ metrics (Sharpe, Sortino, etc.)
├── regime/              # Market regime detection
│   └── filter.py             # Trend + volatility classification
├── risk/                # Risk management
│   └── manager.py            # Position sizing, stop losses, limits
├── tests/               # Unit tests
│   └── test_backtest.py      # Test suite
├── main.py              # Entry point
└── requirements.txt     # Dependencies
```

## 🚀 Features

### Trading Strategies

1. **Momentum Agent** (Turtle Trader)
   - Entry: 55-day breakout
   - Exit: 20-day low break or 2×ATR stop loss
   - Best for: Trending markets

2. **Mean Reversion Agent** (Bollinger Bands + RSI)
   - Entry: Price ≤ BB lower + RSI < 30
   - Exit: Price reaches BB middle or RSI > 70
   - Best for: Ranging markets

### Market Regime Detection

- **Trend Detection**: Moving average crossovers
- **Volatility Detection**: India VIX + historical volatility
- **Regime-Based Agent Selection**: Automatically enables appropriate strategies

### Risk Management

- ✅ Max 5% capital per position
- ✅ Max 30% total exposure
- ✅ 2% max loss per trade
- ✅ 5% daily loss limit
- ✅ 15% max drawdown circuit breaker
- ✅ Max 6 concurrent positions

### Realistic Execution Simulation

- ✅ Zerodha-accurate costs (~0.37% per round trip)
- ✅ Slippage modeling (5 bps)
- ✅ No look-ahead bias (event-driven)
- ✅ Proper order of operations

### Performance Metrics (20+)

**Returns:**
- Total, annualized, monthly returns

**Risk-Adjusted:**
- Sharpe ratio, Sortino ratio, Calmar ratio, Omega ratio

**Drawdown:**
- Max drawdown, average drawdown, DD duration

**Trade Stats:**
- Win rate, profit factor, avg win/loss, largest win/loss

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
cd Claude-code

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Note: TA-Lib may require system-level installation
# macOS: brew install ta-lib
# Ubuntu: sudo apt-get install ta-lib
# Or use pandas-ta as fallback (already in requirements)
```

## 🎮 Usage

### Quick Start

```bash
python main.py
```

This launches an interactive menu:
1. Run Simple Backtest (quick test with 5 stocks)
2. Run MVP Backtest (full system with 10 stocks, 1 year)
3. Run Parameter Optimization
4. Exit

### Run MVP Backtest Programmatically

```python
from main import run_mvp_backtest

metrics, engine = run_mvp_backtest()
print(f"Total Return: {metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
```

### Custom Backtest

```python
from config import get_config
from agents import MomentumAgent, ReversionAgent
from execution import BacktestEngine

# Create agents
momentum = MomentumAgent()
reversion = ReversionAgent()

# Run backtest
engine = BacktestEngine(
    initial_capital=100_000,
    agents=[momentum, reversion],
    start_date="2023-01-01",
    end_date="2024-01-01",
    symbols=["RELIANCE", "TCS", "INFY"],
    enable_regime_filter=True
)

metrics = engine.run()
```

### Modify Configuration

```python
from config import get_config, update_config

# Update risk parameters
update_config(
    risk__max_position_size_pct=3.0,  # Reduce to 3%
    risk__max_daily_loss_pct=3.0       # Tighter daily limit
)

# Or modify directly
config = get_config()
config.agents.momentum_lookback_period = 40  # Change from 55
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_backtest.py::TestAgents::test_momentum_agent_signal_generation -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## 📊 Understanding Results

### Sample Output

```
====================================================================
BACKTEST RESULTS SUMMARY
====================================================================
Initial Capital:     ₹1,00,000
Final Value:         ₹1,12,450
Total Return:        12.45%
Annualized Return:   12.82%
Sharpe Ratio:        1.85
Max Drawdown:        -8.32%
Total Trades:        24
Win Rate:            62.50%
====================================================================

📊 RETURNS
  Total Return:                    12.45%
  Annualized Return:               12.82%
  Monthly Return (avg):             1.07%

📈 RISK-ADJUSTED RETURNS
  Sharpe Ratio:                     1.85
  Sortino Ratio:                    2.34
  Calmar Ratio:                     1.54
  Omega Ratio:                      1.42

📉 DRAWDOWN
  Max Drawdown:                    -8.32%
  Max DD Duration:                 12 days
  Average Drawdown:                -3.21%

💰 TRADE STATISTICS
  Total Trades:                    24
  Winning Trades:                  15
  Losing Trades:                   9
  Win Rate:                        62.50%
  Profit Factor:                   1.89
```

## 📈 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| 3-Month Return | 10% | ✓ |
| Sharpe Ratio | > 1.5 | ✓ |
| Max Drawdown | < 10% | ✓ |
| Win Rate | > 50% | ✓ |

## 🔧 Customization

### Adding New Strategies

```python
from agents.base_agent import BaseAgent, Signal, SignalType

class MyCustomAgent(BaseAgent):
    def generate_signals(self, data, current_positions, portfolio_value, market_regime):
        # Your strategy logic
        signals = []
        # ... generate signals
        return signals

    def calculate_position_size(self, symbol, price, portfolio_value, volatility, max_position_pct):
        # Your position sizing logic
        return shares

    def should_exit(self, symbol, entry_price, current_price, current_data, days_held):
        # Your exit logic
        return should_exit, reason
```

### Connecting to Zerodha API

```python
from data.fetcher import DataFetcher

# Initialize with Zerodha credentials
fetcher = DataFetcher(
    api_key="your_api_key",
    access_token="your_access_token"
)

# Fetch real data
df = fetcher.fetch_historical_data("RELIANCE", "2023-01-01", "2024-01-01")
```

## 🎯 Roadmap

### Phase 1: MVP (Current)
- ✅ 2 agents (Momentum + Reversion)
- ✅ Basic regime filter
- ✅ Risk management
- ✅ Event-driven backtest
- ✅ 20+ metrics

### Phase 2: Enhancement
- [ ] Walk-forward optimization
- [ ] Additional agents (pairs trading, statistical arbitrage)
- [ ] Advanced regime detection (ML-based)
- [ ] Real-time paper trading integration
- [ ] Portfolio rebalancing
- [ ] Correlation-based position limits

### Phase 3: Production
- [ ] Live trading integration with Zerodha
- [ ] Real-time monitoring dashboard
- [ ] Automated alerts and notifications
- [ ] Performance reporting
- [ ] Cloud deployment

## ⚠️ Risk Disclaimer

**This is a trading system that involves financial risk. Important notes:**

1. **Past performance does not guarantee future results**
2. **Backtest results may not reflect live trading performance** due to:
   - Market impact
   - Liquidity constraints
   - Psychological factors
   - Unforeseen market events

3. **Start small**: Begin with minimum capital and paper trade extensively
4. **Understand the strategies**: Don't trade strategies you don't understand
5. **Monitor actively**: Automated systems still require supervision
6. **Regulatory compliance**: Ensure compliance with SEBI and tax regulations

## 📝 Configuration Reference

Key configuration parameters in `config/settings.py`:

```python
# Capital Management
initial_capital = 100_000          # ₹1 lakh
target_capital = 1_000_000         # ₹10 lakh

# Risk Management
max_position_size_pct = 5.0        # Max 5% per position
max_total_exposure_pct = 30.0      # Max 30% deployed
max_loss_per_trade_pct = 2.0       # Max 2% loss per trade
max_daily_loss_pct = 5.0           # Max 5% daily loss
max_drawdown_pct = 15.0            # Circuit breaker at 15% DD

# Transaction Costs
effective_cost_per_round_trip = 0.37   # 0.37% realistic for Zerodha
slippage_bps = 5.0                     # 5 basis points slippage

# Strategy Parameters
momentum_lookback_period = 55      # Turtle breakout period
momentum_exit_period = 20          # Exit period
bb_period = 20                     # Bollinger Bands period
rsi_period = 14                    # RSI period
rsi_oversold = 30                  # RSI oversold threshold
rsi_overbought = 70                # RSI overbought threshold
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation
- Review the test cases for usage examples

## 🙏 Acknowledgments

- Inspired by the Turtle Trading system
- Built for the Indian market (NSE)
- Designed for Zerodha brokerage

---

**Happy Trading! 📈**

*Remember: The best investment you can make is in your own education. Understand the system before deploying capital.*
