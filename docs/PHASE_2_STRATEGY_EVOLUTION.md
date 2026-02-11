# 📘 PHASE 2: STRATEGY EVOLUTION PIPELINE (Month 3-5)

## 🎯 **Objective**
Build the AI-powered strategy evolution system: AI invents strategies → Validates via backtesting → Converts to heuristic code → Optimizes parameters.

**Timeline**: 12 weeks
**Risk Level**: Medium (testing new strategies in paper trading)
**Success Criteria**: AI generates 1+ profitable strategy, converted to heuristic, beating existing strategies

---

## 🔄 **The Strategy Evolution Flow**

```
┌────────────────────────────────────────────────────────────────┐
│         STRATEGY LIFECYCLE (Continuous Evolution)             │
└────────────────────────────────────────────────────────────────┘

Week 1: INVENTION
┌──────────────┐
│ Strategy     │  "What if we trade gap-ups with volume surge?"
│ Inventor     │  (LLM generates hypothesis)
│ Agent        │
└──────┬───────┘
       │
       ↓
Week 2-3: VALIDATION
┌──────────────┐
│ Strategy     │  Backtest on 2 years of data
│ Evaluator    │  → Sharpe: 1.8, Max DD: -6.2%, Win Rate: 64%
│ Agent        │  VERDICT: PROMISING ✅
└──────┬───────┘
       │
       ↓
Week 4: CONVERSION (IF score > threshold)
┌──────────────┐
│ Heuristic    │  Convert LLM logic → Python class
│ Converter    │  class GapUpStrategy(BaseAgent): ...
│              │  Now executes in 10ms instead of 3s!
└──────┬───────┘
       │
       ↓
Week 5-8: OPTIMIZATION
┌──────────────┐
│ Parameter    │  Tune parameters: gap_threshold, volume_multiple
│ Tuner Agent  │  Find optimal: 2.5% gap, 1.8x volume
│              │  → Sharpe improves: 1.8 → 2.1 ✅
└──────┬───────┘
       │
       ↓
Week 9+: DEPLOYMENT & MONITORING
┌──────────────┐
│ Live Paper   │  Deploy in paper trading
│ Trading      │  Monitor for 4 weeks
│              │  If real performance > 80% of backtest → GO LIVE
└──────────────┘
```

---

## 🧠 **Agent 1: Strategy Inventor** (LLM-based, Creative)

### **Purpose**
Generate entirely new trading strategy ideas that don't exist in your system.

**When It Runs**:
- Weekly (every Sunday)
- On-demand when you request new ideas
- When existing strategies underperform

**How It Works**:
```python
class StrategyInventorAgent:
    """
    AI that invents new trading strategies.

    Analyzes:
    - Market patterns not exploited by current strategies
    - Recent profitable trades (what worked)
    - Recent losses (what to avoid)
    - Academic research, industry trends
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    async def invent_new_strategy(self, context):
        """
        Generate a completely new trading strategy idea.

        Args:
            context: {
                "current_strategies": [...],  # What we already have
                "market_patterns": [...],     # Recent observations
                "performance_gaps": [...],    # Where we're weak
                "inspiration": "..."          # Optional: "focus on intraday"
            }

        Returns:
            {
                "strategy_name": "GapUpMomentum",
                "hypothesis": "Stocks gapping up >2% on high volume tend to continue",
                "entry_rules": [...],
                "exit_rules": [...],
                "position_sizing": "...",
                "expected_sharpe": 1.5,
                "market_conditions": "Works best in trending markets",
                "risks": [...]
            }
        """

        prompt = f"""
You are a quantitative strategy researcher specializing in Indian stock markets (NSE).

CURRENT SITUATION:
We have these strategies:
{json.dumps(context['current_strategies'], indent=2)}

Recent performance analysis shows:
{json.dumps(context['performance_gaps'], indent=2)}

TASK:
Invent a NEW trading strategy that:
1. Doesn't duplicate existing strategies
2. Exploits patterns we're not currently trading
3. Suitable for intraday/swing trading (2-3 hour holding)
4. Works on NSE stocks (consider India-specific factors)

REQUIREMENTS:
- Clear entry rules (when to buy/sell)
- Clear exit rules (when to close position)
- Risk management (stop loss, position sizing)
- Expected performance (Sharpe ratio estimate)
- Market regime suitability

INSPIRATION (optional):
{context.get('inspiration', 'Be creative, but practical')}

OUTPUT FORMAT (JSON):
{{
  "strategy_name": "descriptive name",
  "category": "momentum|mean_reversion|breakout|other",
  "hypothesis": "Why this should work",

  "entry_rules": [
    "Rule 1: Price condition",
    "Rule 2: Volume condition",
    "Rule 3: Indicator condition"
  ],

  "exit_rules": [
    "Exit 1: Target hit",
    "Exit 2: Stop loss",
    "Exit 3: Time-based"
  ],

  "position_sizing": "How to calculate position size",

  "indicators_needed": ["RSI", "Volume", "ATR"],

  "expected_metrics": {{
    "sharpe_ratio": 1.5,
    "win_rate": 0.60,
    "avg_trade_duration_hours": 4
  }},

  "market_conditions": "When this works best",

  "risks": ["Risk 1", "Risk 2"],

  "inspiration_source": "Based on X research/observation"
}}

Be innovative but realistic. Indian markets have unique characteristics (circuit breakers, lunch break, etc).
"""

        response = await self.llm.complete(prompt, max_tokens=2000)
        strategy = json.loads(response)

        # Validate strategy structure
        self._validate_strategy(strategy)

        return strategy

    def _validate_strategy(self, strategy):
        """Ensure strategy has all required fields and makes sense."""
        required_fields = [
            'strategy_name', 'hypothesis', 'entry_rules',
            'exit_rules', 'position_sizing', 'expected_metrics'
        ]

        for field in required_fields:
            if field not in strategy:
                raise ValueError(f"Strategy missing required field: {field}")

        # Sanity checks
        if len(strategy['entry_rules']) < 2:
            raise ValueError("Strategy needs at least 2 entry rules")

        if strategy['expected_metrics']['sharpe_ratio'] > 4:
            raise ValueError("Expected Sharpe > 4 is unrealistic, AI may be hallucinating")
```

**Example Output**:
```json
{
  "strategy_name": "OpeningRangeBreakout",
  "category": "breakout",
  "hypothesis": "Stocks breaking first 15min high/low often continue trending",

  "entry_rules": [
    "Wait for first 15 minutes (9:15-9:30 AM)",
    "Identify 15min high and low",
    "BUY if price breaks above 15min high with volume > 1.5x avg",
    "SHORT if price breaks below 15min low with volume > 1.5x avg"
  ],

  "exit_rules": [
    "Target: 2x the 15min range",
    "Stop loss: Opposite end of 15min range",
    "Time stop: Exit at 3:15 PM if still holding"
  ],

  "position_sizing": "Risk 1% of capital, position size = capital * 0.01 / (entry - stop_loss)",

  "indicators_needed": ["Volume", "High", "Low"],

  "expected_metrics": {
    "sharpe_ratio": 1.7,
    "win_rate": 0.58,
    "avg_trade_duration_hours": 3
  },

  "market_conditions": "Works best on volatile days (VIX > 18)",

  "risks": [
    "Whipsaws in ranging markets",
    "Requires quick execution (minutes matter)",
    "India-specific: lunch break (12:30-1:30) can reverse trends"
  ]
}
```

**Acceptance Criteria**:
- ✅ Generates 3-5 distinct strategy ideas weekly
- ✅ At least 1 idea novel (not just variations of existing)
- ✅ Strategies are implementable (clear rules)
- ✅ Expected metrics realistic (Sharpe < 3)

---

## 🧪 **Agent 2: Strategy Evaluator** (Backtesting, Validation)

### **Purpose**
Test if AI-invented strategies actually work on historical data.

**Files to Create**:
- `ai/agents/strategy_evaluator.py` - Core evaluator
- `ai/strategy_compiler.py` - Convert LLM strategy to executable code

**How It Works**:
```python
class StrategyEvaluatorAgent:
    """
    Validates new strategy ideas via rigorous backtesting.
    """

    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
        self.compiler = StrategyCompiler()

    async def evaluate_strategy(self, strategy_idea, test_period_years=2):
        """
        Backtest a strategy idea.

        Process:
        1. Convert LLM description to executable code
        2. Run backtest on historical data
        3. Calculate performance metrics
        4. Compare vs benchmark (buy-and-hold Nifty)
        5. Check for overfitting (walk-forward analysis)

        Returns:
            {
                "verdict": "EXCELLENT|GOOD|MEDIOCRE|POOR",
                "metrics": {...},
                "recommendation": "DEPLOY|OPTIMIZE|REJECT",
                "issues": [...]
            }
        """

        # Step 1: Compile strategy to code
        logger.info(f"Compiling strategy: {strategy_idea['strategy_name']}")
        strategy_code = self.compiler.compile(strategy_idea)

        # Step 2: Run backtest
        logger.info("Running backtest...")
        results = await self._run_backtest(
            strategy_code=strategy_code,
            start_date=self._get_start_date(years_back=test_period_years),
            end_date=datetime.now().strftime("%Y-%m-%d")
        )

        # Step 3: Analyze results
        analysis = self._analyze_results(results, strategy_idea)

        # Step 4: Check for overfitting
        overfitting_score = self._check_overfitting(strategy_code)

        # Step 5: Make recommendation
        verdict = self._determine_verdict(analysis, overfitting_score)

        return {
            "verdict": verdict,
            "metrics": analysis['metrics'],
            "recommendation": verdict['recommendation'],
            "issues": analysis['issues'],
            "backtest_results": results
        }

    def _analyze_results(self, results, expected_metrics):
        """
        Deep analysis of backtest results.

        Checks:
        - Performance metrics (Sharpe, Sortino, Calmar)
        - Drawdown characteristics
        - Win rate, profit factor
        - Trade frequency
        - Compared to expected metrics from AI
        """

        actual_sharpe = results.metrics.sharpe_ratio
        expected_sharpe = expected_metrics['expected_metrics']['sharpe_ratio']

        analysis = {
            "metrics": {
                "sharpe_ratio": actual_sharpe,
                "sortino_ratio": results.metrics.sortino_ratio,
                "max_drawdown": results.metrics.max_drawdown_pct,
                "win_rate": results.metrics.win_rate_pct,
                "total_trades": results.metrics.total_trades,
                "avg_trade_duration": results.metrics.avg_trade_duration_days
            },
            "vs_expected": {
                "sharpe_diff": actual_sharpe - expected_sharpe,
                "sharpe_accuracy": actual_sharpe / expected_sharpe if expected_sharpe > 0 else 0
            },
            "issues": []
        }

        # Flag issues
        if actual_sharpe < 1.0:
            analysis['issues'].append("Sharpe ratio < 1.0 (not good enough)")

        if results.metrics.max_drawdown_pct < -15:
            analysis['issues'].append(f"Max drawdown {results.metrics.max_drawdown_pct}% too high")

        if results.metrics.total_trades < 20:
            analysis['issues'].append(f"Only {results.metrics.total_trades} trades (not enough data)")

        if analysis['vs_expected']['sharpe_diff'] < -0.5:
            analysis['issues'].append(f"AI overestimated performance by {abs(analysis['vs_expected']['sharpe_diff']):.1f} Sharpe points")

        return analysis

    def _determine_verdict(self, analysis, overfitting_score):
        """
        Decide if strategy is worth deploying.

        Criteria:
        - EXCELLENT: Sharpe > 2.0, Max DD < -10%, Win rate > 60%, No major issues
        - GOOD: Sharpe > 1.5, Max DD < -12%, Win rate > 55%
        - MEDIOCRE: Sharpe > 1.0, but has issues
        - POOR: Sharpe < 1.0 or major issues
        """

        sharpe = analysis['metrics']['sharpe_ratio']
        max_dd = abs(analysis['metrics']['max_drawdown'])
        win_rate = analysis['metrics']['win_rate']
        issues = len(analysis['issues'])

        if sharpe > 2.0 and max_dd < 10 and win_rate > 60 and issues == 0:
            return {
                "verdict": "EXCELLENT",
                "recommendation": "DEPLOY",
                "reason": "Outstanding metrics, no issues detected"
            }

        elif sharpe > 1.5 and max_dd < 12 and win_rate > 55 and issues <= 1:
            return {
                "verdict": "GOOD",
                "recommendation": "DEPLOY" if overfitting_score < 0.3 else "OPTIMIZE",
                "reason": "Good metrics, minor issues can be addressed"
            }

        elif sharpe > 1.0 and max_dd < 15:
            return {
                "verdict": "MEDIOCRE",
                "recommendation": "OPTIMIZE",
                "reason": "Acceptable but needs improvement"
            }

        else:
            return {
                "verdict": "POOR",
                "recommendation": "REJECT",
                "reason": f"Underperforms: Sharpe {sharpe:.2f}, issues: {analysis['issues']}"
            }
```

**Walk-Forward Analysis** (Prevent Overfitting):
```python
def _check_overfitting(self, strategy_code):
    """
    Walk-forward analysis to detect overfitting.

    Method:
    - Split data into 6 periods (6 months each)
    - Train on first 5, test on 6th
    - Roll forward, repeat
    - Compare in-sample vs out-of-sample performance

    Overfitting Score:
    0.0 = Perfect (out-of-sample matches in-sample)
    0.5 = Moderate overfitting
    1.0 = Severe overfitting (useless strategy)
    """

    periods = self._create_walk_forward_periods(num_periods=6, period_length_months=6)

    in_sample_sharpes = []
    out_of_sample_sharpes = []

    for i, (train_period, test_period) in enumerate(periods):
        # Backtest on training period
        train_results = self.backtest_engine.run(
            strategy=strategy_code,
            start_date=train_period['start'],
            end_date=train_period['end']
        )
        in_sample_sharpes.append(train_results.sharpe_ratio)

        # Backtest on test period
        test_results = self.backtest_engine.run(
            strategy=strategy_code,
            start_date=test_period['start'],
            end_date=test_period['end']
        )
        out_of_sample_sharpes.append(test_results.sharpe_ratio)

    # Calculate overfitting score
    avg_in_sample = np.mean(in_sample_sharpes)
    avg_out_of_sample = np.mean(out_of_sample_sharpes)

    if avg_in_sample <= 0:
        return 1.0  # Strategy doesn't work even in-sample

    performance_degradation = (avg_in_sample - avg_out_of_sample) / avg_in_sample

    return max(0, min(1, performance_degradation))
```

**Acceptance Criteria**:
- ✅ Can backtest a strategy in < 5 minutes
- ✅ Walk-forward analysis detects overfitting
- ✅ Metrics match manual backtest (within 5%)
- ✅ Clear verdict + reasoning

---

## 🔧 **Agent 3: Heuristic Converter** (Code Generation)

### **Purpose**
Convert LLM-generated strategies (slow) into Python code (fast).

**Why This Matters**:
- LLM decision: ~3-5 seconds per call
- Heuristic code: ~10 milliseconds
- **300x speed improvement!**

**Files to Create**:
- `ai/strategy_compiler.py` - Strategy compiler
- `ai/code_templates/` - Code generation templates

**How It Works**:
```python
class HeuristicConverter:
    """
    Converts AI strategy descriptions into executable Python classes.

    Process:
    1. Parse strategy rules from natural language
    2. Map to technical indicators
    3. Generate Python class inheriting from BaseAgent
    4. Add safety checks and validation
    5. Return executable code
    """

    def convert_to_heuristic(self, strategy_dict, backtest_results):
        """
        Convert validated strategy to Python code.

        Args:
            strategy_dict: AI-generated strategy description
            backtest_results: Validation results from evaluator

        Returns:
            Python code as string (can be saved to .py file)
        """

        # Build class skeleton
        code = self._generate_class_template(strategy_dict['strategy_name'])

        # Add entry logic
        code += self._generate_entry_logic(strategy_dict['entry_rules'])

        # Add exit logic
        code += self._generate_exit_logic(strategy_dict['exit_rules'])

        # Add position sizing
        code += self._generate_position_sizing(strategy_dict['position_sizing'])

        # Add regime filtering
        code += self._generate_regime_filter(strategy_dict['market_conditions'])

        # Validate generated code
        self._validate_code(code)

        return code

    def _generate_class_template(self, strategy_name):
        """Generate class skeleton."""
        return f"""
class {strategy_name}Agent(BaseAgent):
    \"\"\"
    Auto-generated strategy from AI.

    Generated: {datetime.now()}
    Performance (backtest): Sharpe {backtest_results.sharpe_ratio:.2f}
    \"\"\"

    def __init__(self, name="{strategy_name}"):
        super().__init__(name)
        self.config_obj = get_config()
"""

    def _generate_entry_logic(self, entry_rules):
        """
        Convert natural language entry rules to Python code.

        Example:
        Input: ["Price breaks above 15min high", "Volume > 1.5x average"]
        Output: Python if conditions checking these
        """

        code = """
    def generate_signals(self, data, current_positions, portfolio_value, market_regime):
        signals = []

        if len(data) < 50:  # Need minimum data
            return signals

        current = data.iloc[-1]
        symbol = current.get('symbol', 'UNKNOWN')

        # Entry conditions
"""

        for rule in entry_rules:
            condition_code = self._parse_rule_to_code(rule)
            code += f"        {condition_code}\n"

        code += """
        # If all conditions met, generate signal
        if all_conditions_met:
            size = self.calculate_position_size(...)
            signal = Signal(...)
            signals.append(signal)

        return signals
"""

        return code
```

**Example Conversion**:

**Input** (AI Strategy):
```json
{
  "entry_rules": [
    "Price breaks above 15min high",
    "Volume > 1.5x average volume",
    "RSI between 50-70"
  ]
}
```

**Output** (Python Code):
```python
class OpeningRangeBreakoutAgent(BaseAgent):
    def generate_signals(self, data, current_positions, portfolio_value, market_regime):
        signals = []

        # Calculate 15min high (first 15 min of trading day)
        opening_period = data.between_time('09:15', '09:30')
        if len(opening_period) == 0:
            return signals

        range_high = opening_period['high'].max()
        range_low = opening_period['low'].min()

        # Current price
        current_price = data.iloc[-1]['close']

        # Entry condition 1: Break above 15min high
        breakout = current_price > range_high

        # Entry condition 2: Volume > 1.5x average
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        current_volume = data.iloc[-1]['volume']
        volume_surge = current_volume > (avg_volume * 1.5)

        # Entry condition 3: RSI 50-70
        rsi = data.iloc[-1]['rsi']
        rsi_ok = 50 <= rsi <= 70

        # Generate signal if all conditions met
        if breakout and volume_surge and rsi_ok:
            size = self.calculate_position_size(...)
            signal = Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current_price,
                size=size,
                reason="Opening range breakout with volume"
            )
            signals.append(signal)

        return signals
```

**Acceptance Criteria**:
- ✅ Generated code is syntactically valid Python
- ✅ Inherits from BaseAgent properly
- ✅ Backtest results match LLM version (within 10%)
- ✅ Executes 100x faster than LLM calls
- ✅ Includes comments explaining logic

---

## ⚙️ **Agent 4: Parameter Tuner** (Optimization, Fast)

### **Purpose**
Optimize parameters of heuristic strategies WITHOUT using LLM.

**Why Separate from Strategy Inventor**:
- Strategy Inventor: Creates new ideas (rare, expensive LLM)
- Parameter Tuner: Tweaks existing strategies (frequent, fast math)

**Optimization Methods**:
1. Grid search
2. Bayesian optimization
3. Genetic algorithms

**Files to Create**:
- `ai/agents/parameter_tuner.py` - Core tuner
- `ai/optimizers/grid_search.py` - Grid search
- `ai/optimizers/bayesian_optimizer.py` - Smart optimization

**How It Works**:
```python
class ParameterTunerAgent:
    """
    Optimizes strategy parameters using math (no LLM needed).

    Runs daily/weekly to keep strategies adapted to changing markets.
    """

    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
        self.optimizer = BayesianOptimizer()  # Smart search

    def optimize_strategy(self, strategy, parameter_ranges, optimization_target="sharpe_ratio"):
        """
        Find optimal parameters for a strategy.

        Args:
            strategy: Strategy class to optimize
            parameter_ranges: {
                "lookback_period": [20, 40, 60, 80, 100],
                "rsi_threshold": [25, 30, 35, 40],
                "stop_loss_pct": [2.0, 2.5, 3.0]
            }
            optimization_target: "sharpe_ratio" | "sortino_ratio" | "profit_factor"

        Returns:
            {
                "optimal_params": {...},
                "improvement": "+0.3 Sharpe",
                "recommendation": "DEPLOY" | "TEST_MORE"
            }
        """

        logger.info(f"Optimizing {strategy.name}")
        logger.info(f"Parameter space: {len(list(self._generate_combinations(parameter_ranges)))} combinations")

        # Bayesian optimization (smart search, not exhaustive)
        best_params = None
        best_score = -999

        for iteration in range(50):  # Try 50 combinations
            # Bayesian optimizer suggests next params to try
            params = self.optimizer.suggest_next(
                tried_params=self.tried_params,
                tried_scores=self.tried_scores
            )

            # Backtest with these params
            result = self._backtest_with_params(strategy, params)
            score = getattr(result.metrics, optimization_target)

            # Track this attempt
            self.tried_params.append(params)
            self.tried_scores.append(score)

            # Update best
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"New best: {params} → {score:.2f}")

        # Compare to current params
        current_result = self._backtest_with_params(strategy, strategy.current_params)
        current_score = getattr(current_result.metrics, optimization_target)

        improvement = best_score - current_score

        return {
            "optimal_params": best_params,
            "current_score": current_score,
            "optimized_score": best_score,
            "improvement": improvement,
            "recommendation": "DEPLOY" if improvement > 0.2 else "TEST_MORE"
        }
```

**Example Usage**:
```python
# Optimize MomentumAgent parameters
tuner = ParameterTunerAgent(backtest_engine)

results = tuner.optimize_strategy(
    strategy=MomentumAgent,
    parameter_ranges={
        "lookback_period": [30, 40, 50, 55, 60, 70],
        "exit_period": [10, 15, 20, 25],
        "atr_multiplier": [1.5, 2.0, 2.5, 3.0]
    },
    optimization_target="sharpe_ratio"
)

# Output:
# {
#   "optimal_params": {
#     "lookback_period": 50,
#     "exit_period": 15,
#     "atr_multiplier": 2.5
#   },
#   "improvement": 0.35,  # +0.35 Sharpe improvement
#   "recommendation": "DEPLOY"
# }
```

**Acceptance Criteria**:
- ✅ Finds better parameters in 80%+ of cases
- ✅ Optimization completes in < 30 minutes
- ✅ Improvement validated on out-of-sample data
- ✅ No overfitting (walk-forward validation)

---

## 🗓️ **Week-by-Week Plan (Phase 2)**

### **Week 1-3: Strategy Inventor**
- [ ] Build Strategy Inventor agent
- [ ] Test with 5 strategy ideas
- [ ] Validate ideas make sense (manual review)
- [ ] Document top 3 ideas

### **Week 4-6: Strategy Evaluator**
- [ ] Build backtest compiler
- [ ] Implement walk-forward analysis
- [ ] Test on 3 AI strategies
- [ ] Compare vs manual backtest

### **Week 7-8: Heuristic Converter**
- [ ] Build code generator
- [ ] Convert 2 strategies to heuristics
- [ ] Verify performance matches
- [ ] Test execution speed

### **Week 9-10: Parameter Tuner**
- [ ] Implement Bayesian optimizer
- [ ] Optimize existing Momentum/Reversion agents
- [ ] Validate improvements
- [ ] Deploy optimized params

### **Week 11-12: Integration & Testing**
- [ ] Full pipeline test: Invent → Evaluate → Convert → Optimize
- [ ] Deploy 1 new AI strategy in paper trading
- [ ] Monitor for 2 weeks
- [ ] Measure vs existing strategies

---

## 📊 **Success Metrics (Phase 2)**

- ✅ AI generates 10+ strategy ideas (3+ novel)
- ✅ At least 1 strategy passes evaluation (Sharpe > 1.5)
- ✅ Heuristic version performs within 10% of LLM version
- ✅ Parameter tuning improves existing strategies by 0.2+ Sharpe
- ✅ Full pipeline runs end-to-end successfully
- ✅ New strategy profitable in paper trading (2+ weeks)

---

## 💰 **Estimated Costs (Phase 2)**

| Item | Monthly Cost | Notes |
|------|--------------|-------|
| LLM (Strategy Invention) | ₹1,500-2,500 | Weekly strategy generation |
| Compute (Backtesting) | ₹500-1,000 | Cloud compute for optimization |
| Total | ₹2,000-3,500 | Still very affordable |

---

**Next**: [PHASE 3: Advanced Intelligence](./PHASE_3_ADVANCED_INTELLIGENCE.md)
