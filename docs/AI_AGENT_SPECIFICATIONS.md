# 🤖 AI AGENT SPECIFICATIONS

## 📋 **Overview**

This document provides detailed technical specifications for all AI agents in the Hybrid Trading System. Each agent has defined responsibilities, inputs, outputs, LLM prompts, performance targets, and integration points.

---

## 🏗️ **Agent Architecture Principles**

### **Design Principles**
1. **Single Responsibility**: Each agent has one clear purpose
2. **Loose Coupling**: Agents communicate via standardized interfaces
3. **Fail-Safe**: Graceful degradation if agent fails
4. **Explainable**: All decisions include natural language reasoning
5. **Auditable**: All agent actions logged in audit trail
6. **Cost-Conscious**: Minimize unnecessary LLM calls

### **Agent Communication**
```
AI Orchestrator (Master)
    ↓
Sends context/requests to specialized agents
    ↓
Agents process and return decisions
    ↓
Orchestrator aggregates and makes final decision
    ↓
Logs to audit trail + executes (if approved)
```

### **Standard Agent Interface**
```python
class BaseAIAgent:
    """Base class for all AI agents."""

    def __init__(self, llm_client: LLMClient, db: Database):
        self.llm = llm_client
        self.db = db
        self.name = self.__class__.__name__

    async def process(self, context: dict) -> dict:
        """
        Main processing method.

        Args:
            context: Input data relevant to this agent

        Returns:
            {
                'decision': str,           # Agent's decision
                'reasoning': str,          # Natural language explanation
                'confidence': float,       # 0.0 - 1.0
                'metadata': dict,          # Additional data
                'cost': float              # LLM cost for this call
            }
        """
        raise NotImplementedError

    def _log_decision(self, context: dict, result: dict):
        """Log agent decision to audit trail."""
        self.db.log_agent_decision(
            agent_name=self.name,
            context=context,
            result=result,
            timestamp=datetime.now()
        )
```

---

## 🎯 **CORE AGENTS** (Always Active)

---

### **1. AI Orchestrator** 🎭

**Role**: Master coordinator for all AI agents and final decision maker

**Phase**: 1 (Foundation)

**Complexity**: High

**LLM Usage**: Medium (per trade decision)

#### **Responsibilities**
1. Receive signal from quantitative agents (Momentum, Mean Reversion, etc.)
2. Query Market Intelligence Agent for current context
3. Query Risk Sentinel for portfolio state
4. Aggregate all inputs into unified decision
5. Calculate risk score (0-100)
6. Determine if human approval needed
7. Generate explainable reasoning
8. Execute (if approved) or route to dashboard

#### **Inputs**
```python
context = {
    'signal': {
        'agent': 'MomentumAgent',
        'action': 'BUY',
        'symbol': 'RELIANCE',
        'confidence': 0.75,
        'indicators': {'rsi': 65, 'macd_signal': 'bullish'}
    },
    'current_price': 2500.50,
    'portfolio': {
        'cash': 85000,
        'positions': [...],
        'total_exposure_pct': 15.0
    },
    'market_context': {  # From Market Intelligence Agent
        'regime': 'TRENDING_UP',
        'vix': 14.5,
        'sector_sentiment': 'NEUTRAL',
        'recent_news': [...]
    },
    'risk_context': {  # From Risk Sentinel
        'daily_pnl_pct': 1.2,
        'sector_exposure': {'Energy': 12, 'IT': 8},
        'alerts': []
    }
}
```

#### **LLM Prompt Template**
```python
ORCHESTRATOR_PROMPT = """
You are the AI Orchestrator for a quantitative trading system.

SIGNAL RECEIVED:
Agent: {agent_name}
Action: {action} {symbol}
Confidence: {agent_confidence}
Reason: {agent_reason}
Technical Indicators: {indicators}

MARKET CONTEXT:
Regime: {market_regime}
VIX: {vix}
Sector Sentiment ({symbol_sector}): {sector_sentiment}
Recent News Headlines: {news_summary}

PORTFOLIO STATE:
Cash: ₹{cash:,.0f}
Open Positions: {num_positions}
Total Exposure: {exposure_pct}%
Today's P&L: {daily_pnl_pct:+.2f}%
Sector Concentration: {sector_exposure}

RISK CHECKS:
- Position size would be: {proposed_position_pct}% of portfolio
- Sector exposure after trade: {sector_exposure_after}%
- VIX level: {vix_assessment}
- Daily loss limit remaining: {loss_headroom}%

DECISION REQUIRED:
Should we execute this {action} on {symbol}?

ANALYSIS:
1. Does the signal align with current market regime?
2. Are there any contradicting signals from news/sentiment?
3. Does the portfolio have capacity for this trade?
4. What are the key risks?
5. What is your confidence level?

OUTPUT (JSON):
{{
  "decision": "APPROVE" or "REJECT" or "MODIFY",
  "reasoning": "2-3 sentence explanation in plain language",
  "risk_score": 0-100,  // 0=very safe, 100=extreme risk
  "confidence": 0.0-1.0,
  "modifications": {{  // If decision is MODIFY
    "suggested_quantity": X,
    "suggested_stop_loss": Y
  }},
  "key_factors": ["factor1", "factor2", "factor3"]  // Max 5
}}

IMPORTANT:
- Be conservative. When in doubt, REJECT or reduce position size.
- Risk score should consider: signal confidence, market volatility, portfolio concentration, news sentiment.
- High risk score (>70) should trigger human approval.
"""
```

#### **Output**
```python
{
    'decision': 'APPROVE',  # or 'REJECT' or 'MODIFY'
    'reasoning': 'Strong momentum signal in trending market with positive sector sentiment. Portfolio has capacity. Risk is moderate.',
    'risk_score': 45,  # 0-100
    'confidence': 0.78,  # 0.0-1.0
    'requires_human_approval': False,  # Based on autonomy level + risk score
    'modifications': {
        'quantity': 10,
        'stop_loss': 2450.00,
        'take_profit': 2575.00
    },
    'key_factors': [
        'Momentum breakout confirmed',
        'Market regime favorable (TRENDING_UP)',
        'Sector sentiment neutral-positive',
        'Portfolio capacity available'
    ],
    'cost': 0.05  # LLM cost in ₹
}
```

#### **Performance Targets**
- Latency: < 3 seconds per decision
- Cost per decision: < ₹0.10
- Uptime: > 99%

#### **Error Handling**
- If Market Intel Agent unavailable → Use cached market regime
- If Risk Sentinel unavailable → Default to conservative risk assessment
- If LLM fails → Route to human approval (100% safety)

---

### **2. Market Intelligence Agent** 📰

**Role**: Analyze news, sentiment, and market conditions

**Phase**: 1 (Foundation)

**Complexity**: Medium-High

**LLM Usage**: Medium (1-2 times per hour)

#### **Responsibilities**
1. Fetch and analyze news headlines (Economic Times, Moneycontrol, NSE announcements)
2. Determine market regime (TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE)
3. Analyze sector rotation and sentiment
4. Monitor RBI policy announcements
5. Detect unusual market conditions
6. Cache results to avoid redundant LLM calls

#### **Inputs**
```python
{
    'vix_current': 14.5,
    'vix_20d_avg': 15.2,
    'nifty_50_change_1d': 0.8,
    'nifty_50_change_5d': -1.2,
    'sector_indices': {
        'NIFTY_IT': {'change_1d': 1.5, 'change_5d': 2.3},
        'NIFTY_BANK': {'change_1d': -0.5, 'change_5d': -1.8},
        'NIFTY_AUTO': {'change_1d': 0.3, 'change_5d': 0.1}
    },
    'recent_news': [
        {'headline': 'RBI keeps repo rate unchanged at 6.5%', 'source': 'ET', 'time': '2026-02-06 14:00'},
        {'headline': 'IT sector sees strong Q4 earnings', 'source': 'MC', 'time': '2026-02-06 10:30'},
        # ... more headlines
    ],
    'timestamp': '2026-02-07 09:30:00'
}
```

#### **News Fetching** (Python Implementation)
```python
import requests
from bs4 import BeautifulSoup

class NewsFetcher:
    """Fetch financial news from public sources."""

    SOURCES = [
        {
            'name': 'Economic Times',
            'url': 'https://economictimes.indiatimes.com/markets',
            'selector': '.eachStory'
        },
        {
            'name': 'Moneycontrol',
            'url': 'https://www.moneycontrol.com/news/business/markets/',
            'selector': '.clearfix'
        }
    ]

    def fetch_headlines(self, max_per_source=10):
        """Fetch recent headlines from all sources."""
        all_headlines = []

        for source in self.SOURCES:
            try:
                response = requests.get(source['url'], timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')

                stories = soup.select(source['selector'])[:max_per_source]

                for story in stories:
                    headline = story.get_text(strip=True)
                    all_headlines.append({
                        'headline': headline,
                        'source': source['name'],
                        'time': datetime.now()
                    })

            except Exception as e:
                logger.error(f"Failed to fetch from {source['name']}: {e}")

        return all_headlines
```

#### **LLM Prompt Template**
```python
MARKET_INTEL_PROMPT = """
You are a market intelligence analyst for NSE trading.

MARKET DATA (last update: {timestamp}):
- VIX: {vix_current} (20-day avg: {vix_20d_avg})
- Nifty 50: {nifty_change_1d:+.2f}% (1D), {nifty_change_5d:+.2f}% (5D)

SECTOR PERFORMANCE (1D / 5D):
{sector_table}

RECENT NEWS HEADLINES (last 24 hours):
{news_headlines}

ANALYSIS REQUIRED:
1. Market Regime Classification:
   - TRENDING_UP: Clear uptrend, VIX normal, bullish sentiment
   - TRENDING_DOWN: Clear downtrend, risk-off sentiment
   - RANGING: Sideways, low volatility, no clear direction
   - VOLATILE: High VIX (>20), choppy, uncertain

2. Sector Rotation:
   - Which sectors are showing strength?
   - Which sectors are weakening?
   - Any sector-specific news?

3. Sentiment Analysis:
   - Overall market sentiment (BULLISH/NEUTRAL/BEARISH)
   - Any major risk events on horizon?
   - Policy impacts (RBI, government)

4. Trading Implications:
   - Favorable for momentum strategies?
   - Favorable for mean reversion?
   - Recommended caution areas?

OUTPUT (JSON):
{{
  "market_regime": "TRENDING_UP|TRENDING_DOWN|RANGING|VOLATILE",
  "regime_confidence": 0.0-1.0,
  "overall_sentiment": "BULLISH|NEUTRAL|BEARISH",
  "sector_analysis": {{
    "strong_sectors": ["IT", "Banking"],
    "weak_sectors": ["Auto", "Pharma"],
    "neutral_sectors": ["Energy"]
  }},
  "key_events": [
    "RBI kept rates unchanged - neutral for equities",
    "IT sector Q4 earnings beat estimates - bullish for tech"
  ],
  "trading_recommendation": {{
    "momentum_favorable": true,
    "mean_reversion_favorable": false,
    "recommended_caution": "Banking sector showing weakness"
  }},
  "summary": "2-3 sentence market summary"
}}
"""
```

#### **Output**
```python
{
    'market_regime': 'TRENDING_UP',
    'regime_confidence': 0.82,
    'overall_sentiment': 'BULLISH',
    'sector_analysis': {
        'strong_sectors': ['IT', 'Pharma'],
        'weak_sectors': ['Banking', 'Auto'],
        'neutral_sectors': ['Energy', 'FMCG']
    },
    'key_events': [
        'RBI maintains status quo - markets stable',
        'IT sector Q4 results exceed expectations',
        'FII buying continues for 3rd consecutive day'
    ],
    'trading_recommendation': {
        'momentum_favorable': True,
        'mean_reversion_favorable': False,
        'recommended_caution': 'Avoid banking stocks amid sector weakness'
    },
    'summary': 'Market in uptrend with strong IT sector performance. VIX stable. Momentum strategies favored.',
    'timestamp': '2026-02-07 09:30:00',
    'cache_until': '2026-02-07 10:30:00',  # Cache for 1 hour
    'cost': 0.08
}
```

#### **Caching Strategy**
```python
class MarketIntelligenceAgent(BaseAIAgent):
    """Market intelligence with smart caching."""

    def __init__(self, llm_client, db):
        super().__init__(llm_client, db)
        self.cache = {}
        self.cache_duration = timedelta(hours=1)  # Refresh hourly

    async def get_market_context(self):
        """Get market context with caching."""
        now = datetime.now()

        # Check if cache is still valid
        if 'market_context' in self.cache:
            cached_data, cached_time = self.cache['market_context']
            if now - cached_time < self.cache_duration:
                logger.info("Using cached market context (age: {:.1f}m)".format(
                    (now - cached_time).total_seconds() / 60
                ))
                return cached_data

        # Cache expired or doesn't exist - fetch fresh
        logger.info("Fetching fresh market intelligence...")

        # Fetch news
        news = NewsFetcher().fetch_headlines()

        # Fetch market data
        market_data = self._fetch_market_data()

        # Analyze with LLM
        context = await self._analyze_with_llm(news, market_data)

        # Cache result
        self.cache['market_context'] = (context, now)

        return context
```

#### **Performance Targets**
- Latency: < 5 seconds (first call), < 100ms (cached)
- Cost: < ₹30/day (~1 LLM call/hour during market hours)
- Regime accuracy: > 70% (validated against actual market movements)

---

### **3. Risk Sentinel Agent** 🛡️

**Role**: Monitor portfolio risk in real-time and alert on violations

**Phase**: 1 (Foundation)

**Complexity**: Low-Medium

**LLM Usage**: Low (only when risk alert triggered)

#### **Responsibilities**
1. Monitor portfolio exposure continuously
2. Track daily P&L and drawdown
3. Check position concentration limits
4. Detect circuit breaker conditions
5. Alert on risk limit violations
6. Recommend position reductions if needed

#### **Inputs**
```python
{
    'portfolio': {
        'cash': 85000,
        'positions': [
            {'symbol': 'RELIANCE', 'quantity': 10, 'avg_price': 2480, 'current_price': 2500, 'pnl': 200, 'sector': 'Energy'},
            {'symbol': 'TCS', 'quantity': 5, 'avg_price': 3600, 'current_price': 3580, 'pnl': -100, 'sector': 'IT'}
        ],
        'total_value': 100000,
        'daily_pnl': 100,
        'daily_pnl_pct': 0.1
    },
    'risk_limits': {
        'max_position_size_pct': 5.0,
        'max_total_exposure_pct': 30.0,
        'max_sector_concentration_pct': 20.0,
        'max_daily_loss_pct': 5.0,
        'max_drawdown_pct': 15.0,
        'max_concurrent_positions': 6
    }
}
```

#### **Processing Logic** (Mostly Rule-Based, Minimal LLM)
```python
class RiskSentinelAgent(BaseAIAgent):
    """Portfolio risk monitoring agent."""

    def check_risk_violations(self, portfolio, risk_limits):
        """Check for risk limit violations."""
        violations = []

        # 1. Position size check
        for position in portfolio['positions']:
            position_pct = (position['quantity'] * position['current_price']) / portfolio['total_value'] * 100
            if position_pct > risk_limits['max_position_size_pct']:
                violations.append({
                    'type': 'POSITION_SIZE',
                    'severity': 'MEDIUM',
                    'message': f"{position['symbol']} position is {position_pct:.1f}% (limit: {risk_limits['max_position_size_pct']}%)",
                    'recommendation': f"Reduce {position['symbol']} by {position_pct - risk_limits['max_position_size_pct']:.1f}%"
                })

        # 2. Total exposure check
        total_exposure = sum(
            (p['quantity'] * p['current_price']) for p in portfolio['positions']
        ) / portfolio['total_value'] * 100

        if total_exposure > risk_limits['max_total_exposure_pct']:
            violations.append({
                'type': 'TOTAL_EXPOSURE',
                'severity': 'HIGH',
                'message': f"Total exposure {total_exposure:.1f}% exceeds limit {risk_limits['max_total_exposure_pct']}%",
                'recommendation': 'Reduce overall exposure or halt new trades'
            })

        # 3. Sector concentration check
        sector_exposure = {}
        for position in portfolio['positions']:
            sector = position['sector']
            exposure = (position['quantity'] * position['current_price']) / portfolio['total_value'] * 100
            sector_exposure[sector] = sector_exposure.get(sector, 0) + exposure

        for sector, exposure in sector_exposure.items():
            if exposure > risk_limits['max_sector_concentration_pct']:
                violations.append({
                    'type': 'SECTOR_CONCENTRATION',
                    'severity': 'MEDIUM',
                    'message': f"{sector} sector exposure {exposure:.1f}% exceeds limit {risk_limits['max_sector_concentration_pct']}%",
                    'recommendation': f"Avoid new {sector} positions"
                })

        # 4. Daily loss check (CIRCUIT BREAKER)
        if portfolio['daily_pnl_pct'] < -risk_limits['max_daily_loss_pct']:
            violations.append({
                'type': 'DAILY_LOSS_LIMIT',
                'severity': 'CRITICAL',
                'message': f"Daily loss {portfolio['daily_pnl_pct']:.2f}% exceeds limit {-risk_limits['max_daily_loss_pct']}%",
                'recommendation': 'HALT ALL TRADING - Circuit breaker triggered'
            })

        # 5. Concurrent positions check
        if len(portfolio['positions']) >= risk_limits['max_concurrent_positions']:
            violations.append({
                'type': 'MAX_POSITIONS',
                'severity': 'LOW',
                'message': f"{len(portfolio['positions'])} positions (limit: {risk_limits['max_concurrent_positions']})",
                'recommendation': 'Exit a position before opening new trades'
            })

        return violations

    async def process(self, context):
        """Process risk check and generate alert if needed."""
        violations = self.check_risk_violations(
            context['portfolio'],
            context['risk_limits']
        )

        if not violations:
            return {
                'status': 'OK',
                'violations': [],
                'reasoning': 'All risk limits within acceptable ranges',
                'confidence': 1.0,
                'cost': 0.0  # No LLM call needed
            }

        # If violations exist, use LLM for natural language alert
        critical_violations = [v for v in violations if v['severity'] == 'CRITICAL']

        if critical_violations:
            # Use LLM to generate clear alert message
            alert = await self._generate_alert(violations, context['portfolio'])

            return {
                'status': 'VIOLATION',
                'violations': violations,
                'reasoning': alert['message'],
                'confidence': 1.0,
                'halt_trading': True if critical_violations else False,
                'cost': alert['cost']
            }
        else:
            # Minor violations - just return structured data
            return {
                'status': 'WARNING',
                'violations': violations,
                'reasoning': f"{len(violations)} risk warning(s) detected",
                'confidence': 1.0,
                'halt_trading': False,
                'cost': 0.0
            }
```

#### **LLM Prompt (Only for Critical Alerts)**
```python
RISK_ALERT_PROMPT = """
CRITICAL RISK ALERT

Portfolio State:
- Total Value: ₹{total_value:,.0f}
- Daily P&L: ₹{daily_pnl:,.0f} ({daily_pnl_pct:+.2f}%)
- Open Positions: {num_positions}

Risk Violations Detected:
{violations_list}

Generate a clear, actionable alert message for the human trader.

OUTPUT (JSON):
{{
  "alert_title": "Clear, urgent title",
  "message": "2-3 sentence explanation of the risk situation",
  "immediate_actions": ["action1", "action2"],
  "severity": "CRITICAL|HIGH|MEDIUM"
}}
"""
```

#### **Output**
```python
{
    'status': 'VIOLATION',
    'violations': [
        {
            'type': 'DAILY_LOSS_LIMIT',
            'severity': 'CRITICAL',
            'message': 'Daily loss -5.2% exceeds limit -5.0%',
            'recommendation': 'HALT ALL TRADING - Circuit breaker triggered'
        }
    ],
    'reasoning': '⛔ CIRCUIT BREAKER TRIGGERED: Daily loss limit exceeded. All trading halted to prevent further losses.',
    'halt_trading': True,
    'confidence': 1.0,
    'cost': 0.03
}
```

#### **Performance Targets**
- Latency: < 100ms (rule-based checks), < 2s (with LLM alert)
- Cost: < ₹2/day (only on alerts)
- False positive rate: < 5%

---

### **4. Portfolio Optimizer Agent** 📊

**Role**: Optimize position sizing and portfolio allocation

**Phase**: 1 (Foundation)

**Complexity**: Medium

**LLM Usage**: Low (daily optimization run)

#### **Responsibilities**
1. Suggest optimal position sizes based on Kelly Criterion
2. Recommend portfolio rebalancing
3. Detect overconcentration in sectors/stocks
4. Suggest diversification opportunities
5. Consider correlation between positions

#### **Inputs**
```python
{
    'portfolio': {
        'cash': 85000,
        'positions': [...],
        'total_value': 100000
    },
    'proposed_trade': {
        'action': 'BUY',
        'symbol': 'INFY',
        'sector': 'IT',
        'confidence': 0.75,
        'expected_return': 3.0,  # %
        'expected_risk': 2.0     # % (stop-loss distance)
    },
    'historical_performance': {
        'win_rate': 0.62,
        'avg_win': 2.8,   # %
        'avg_loss': -1.5  # %
    }
}
```

#### **Kelly Criterion Implementation**
```python
def calculate_kelly_position_size(win_rate, avg_win, avg_loss, max_kelly=0.25):
    """
    Calculate optimal position size using Kelly Criterion.

    Kelly % = (W * R - L) / R
    Where:
        W = Win rate
        R = Avg win / Avg loss
        L = Loss rate (1 - W)

    Args:
        win_rate: Historical win rate (0.0 - 1.0)
        avg_win: Average winning trade %
        avg_loss: Average losing trade % (positive number)
        max_kelly: Maximum Kelly fraction (0.25 = quarter Kelly for safety)

    Returns:
        Recommended position size as % of portfolio
    """
    if avg_loss <= 0:
        return max_kelly * 100  # Default to conservative if no loss data

    R = avg_win / avg_loss
    L = 1 - win_rate

    kelly_pct = (win_rate * R - L) / R

    # Apply fractional Kelly for safety (full Kelly is too aggressive)
    kelly_pct = min(kelly_pct, max_kelly)

    # Convert to position size %
    position_size_pct = max(0, kelly_pct * 100)

    # Cap at 5% (our max position limit)
    return min(position_size_pct, 5.0)
```

#### **LLM Prompt**
```python
PORTFOLIO_OPTIMIZER_PROMPT = """
You are a portfolio optimizer for an NSE trading system.

CURRENT PORTFOLIO:
Cash: ₹{cash:,.0f}
Positions: {positions_summary}
Sector Allocation: {sector_allocation}

PROPOSED TRADE:
{action} {symbol} ({sector})
Confidence: {confidence}
Expected Return: {expected_return}%
Expected Risk: {expected_risk}%

HISTORICAL PERFORMANCE:
Win Rate: {win_rate:.1%}
Avg Win: {avg_win:.2f}%
Avg Loss: {avg_loss:.2f}%

KELLY CALCULATION:
Recommended Position Size: {kelly_position_size:.2f}% of portfolio

ANALYSIS REQUIRED:
1. Is the Kelly position size appropriate given current portfolio?
2. Would this trade improve or worsen diversification?
3. Any correlation concerns with existing positions?
4. Suggested position size (in ₹)?

OUTPUT (JSON):
{{
  "recommended_position_size_pct": 0-5,  // % of portfolio
  "recommended_quantity": X,             // Number of shares
  "recommended_capital": X,              // Rupees to allocate
  "reasoning": "Brief explanation",
  "diversification_impact": "POSITIVE|NEUTRAL|NEGATIVE",
  "correlation_warning": "Any concerns about correlation with existing positions"
}}
"""
```

#### **Output**
```python
{
    'recommended_position_size_pct': 3.5,
    'recommended_quantity': 12,
    'recommended_capital': 3500,
    'reasoning': 'Kelly suggests 3.5% position. Portfolio has room for IT exposure. No correlation concerns.',
    'diversification_impact': 'POSITIVE',
    'correlation_warning': None,
    'confidence': 0.80,
    'cost': 0.04
}
```

#### **Performance Targets**
- Latency: < 2 seconds
- Cost: < ₹5/day (only on new trades)
- Position sizing accuracy: Within 10% of optimal in backtests

---

## 🔄 **STRATEGY EVOLUTION AGENTS** (Periodic)

---

### **5. Strategy Inventor Agent** 💡

**Role**: Generate entirely new trading strategy ideas using LLM creativity

**Phase**: 2 (Strategy Evolution)

**Complexity**: High

**LLM Usage**: High (weekly strategy invention)

#### **Responsibilities**
1. Analyze current strategy portfolio and identify gaps
2. Research market patterns not currently exploited
3. Generate 1-2 new strategy ideas per week
4. Provide detailed specification for each strategy
5. Justify why the strategy should work

#### **Trigger**: Weekly (every Sunday)

#### **Inputs**
```python
{
    'current_strategies': [
        {
            'name': 'MomentumAgent',
            'type': 'Trend-following',
            'sharpe_ratio': 1.8,
            'win_rate': 0.65,
            'typical_hold_time': '2-3 hours',
            'strengths': 'Works well in trending markets',
            'weaknesses': 'Whipsawed in ranging markets'
        },
        {
            'name': 'MeanReversionAgent',
            'type': 'Counter-trend',
            'sharpe_ratio': 1.4,
            'win_rate': 0.58,
            'typical_hold_time': '1-2 hours',
            'strengths': 'Profits from oversold bounces',
            'weaknesses': 'Loses in strong trends'
        }
    ],
    'performance_gaps': [
        'No strategies for volatile/choppy markets (VIX > 20)',
        'Limited coverage of banking sector',
        'No intraday reversal patterns'
    ],
    'recent_market_patterns': [
        'Morning gap-ups followed by afternoon fade (5 times in last 10 days)',
        'Banking stocks showing mean reversion around 10:30 AM',
        'Energy sector momentum continues across days'
    ]
}
```

#### **LLM Prompt**
```python
STRATEGY_INVENTOR_PROMPT = """
You are a quantitative strategy researcher for NSE intraday trading.

CURRENT STRATEGY PORTFOLIO:
{current_strategies_summary}

IDENTIFIED GAPS:
{performance_gaps}

RECENT MARKET PATTERNS (not currently exploited):
{recent_patterns}

TASK: Invent a NEW trading strategy that:
1. Does NOT duplicate existing strategies
2. Exploits a market pattern we're not currently trading
3. Suitable for intraday (2-3 hour holding period maximum)
4. Has clear, testable entry and exit rules
5. Works specifically in NSE market conditions

Be creative but realistic. The strategy should be based on sound trading principles.

OUTPUT (JSON):
{{
  "strategy_name": "Descriptive name",
  "strategy_type": "Momentum|MeanReversion|Breakout|Fading|Pairs|Other",
  "hypothesis": "Why this strategy should work (2-3 sentences)",
  "target_market_conditions": "When this strategy performs best",

  "entry_rules": [
    "Specific condition 1",
    "Specific condition 2",
    "..."
  ],

  "exit_rules": [
    "Take profit condition",
    "Stop loss condition",
    "Time-based exit"
  ],

  "indicators_needed": ["RSI", "VWAP", "..."],

  "expected_performance": {{
    "sharpe_ratio": 1.5,
    "win_rate": 0.60,
    "avg_hold_time_hours": 2.5
  }},

  "risk_considerations": [
    "Risk 1",
    "Risk 2"
  ],

  "uniqueness": "How this differs from existing strategies"
}}

EXAMPLES OF GOOD STRATEGIES:
- "Gap Fade Strategy": Fade large morning gaps (>2%) when VIX < 15, expecting mean reversion by noon
- "Sector Rotation Momentum": Enter strongest sector stock when sector index breaks 5-day high
- "Volatility Breakout": Enter on low-volatility compression followed by volume spike

EXAMPLES OF BAD STRATEGIES:
- "Buy when RSI < 30" (too simplistic, already covered by mean reversion)
- "Hold overnight" (not intraday)
- "Trade based on moon phases" (not rational)

Now invent a strategy:
"""
```

#### **Example Output**
```python
{
    'strategy_name': 'Morning Reversal Scalp',
    'strategy_type': 'MeanReversion',
    'hypothesis': 'NSE stocks often show exaggerated moves in first 30 minutes due to retail emotion. When move exceeds 2 standard deviations from 5-day average and volume is below average, expect reversal by 11 AM.',
    'target_market_conditions': 'Normal volatility (VIX < 18), liquid large-caps, non-news days',

    'entry_rules': [
        'Stock moves >2% in first 30 minutes (9:15-9:45)',
        'Move is >2 standard deviations from 5-day average opening range',
        'Volume in first 30 min is BELOW 5-day average (low conviction move)',
        'No major news on stock or sector',
        'Enter counter-trend at 9:50 AM'
    ],

    'exit_rules': [
        'Take profit: 50% of opening move retraced (e.g., if +2%, exit at +1%)',
        'Stop loss: Move extends another 0.5% against us',
        'Time exit: 11:30 AM regardless of P&L'
    ],

    'indicators_needed': ['5-day opening range std dev', '5-day avg volume', 'VWAP'],

    'expected_performance': {
        'sharpe_ratio': 1.6,
        'win_rate': 0.68,
        'avg_hold_time_hours': 1.5,
        'avg_win': 1.2,
        'avg_loss': 0.6
    },

    'risk_considerations': [
        'Fails if morning move is justified (earnings, news) - need news filter',
        'Low volume days may not provide enough liquidity',
        'Works best on large-caps (RELIANCE, TCS, INFY, ICICI)'
    ],

    'uniqueness': 'Specifically targets irrational morning moves, unlike our existing mean reversion which trades oversold bounces anytime. Time-specific entry (9:50 AM) is novel.',

    'confidence': 0.72,
    'cost': 0.25
}
```

#### **Post-Processing**
After LLM generates strategy:
1. Parse JSON response
2. Validate strategy is sufficiently different from existing
3. Check that indicators are feasible to calculate
4. Forward to Strategy Evaluator Agent for backtesting

#### **Performance Targets**
- Generate 1-2 viable strategies per week
- At least 50% of strategies pass initial evaluation
- At least 20% become production strategies

---

### **6. Strategy Evaluator Agent** ✅

**Role**: Rigorously backtest and validate new strategies

**Phase**: 2 (Strategy Evolution)

**Complexity**: High

**LLM Usage**: None (pure computational backtesting)

#### **Responsibilities**
1. Convert strategy specification to backtest code
2. Run walk-forward analysis (prevent overfitting)
3. Calculate comprehensive performance metrics
4. Determine verdict: EXCELLENT / GOOD / MEDIOCRE / POOR
5. Generate detailed evaluation report

#### **Inputs**
```python
{
    'strategy_spec': {  # From Strategy Inventor
        'strategy_name': 'Morning Reversal Scalp',
        'entry_rules': [...],
        'exit_rules': [...],
        'indicators_needed': [...]
    },
    'backtest_params': {
        'start_date': '2024-01-01',
        'end_date': '2026-01-31',
        'symbols': ['RELIANCE', 'TCS', 'INFY', 'HDFC BANK', 'ICICIBANK'],
        'initial_capital': 100000,
        'walk_forward_windows': [
            {'train': '2024-01-01 to 2024-06-30', 'test': '2024-07-01 to 2024-09-30'},
            {'train': '2024-04-01 to 2024-09-30', 'test': '2024-10-01 to 2024-12-31'},
            {'train': '2024-07-01 to 2024-12-31', 'test': '2025-01-01 to 2025-03-31'},
            # ... rolling windows
        ]
    }
}
```

#### **Implementation** (No LLM, Pure Python)
```python
class StrategyEvaluatorAgent:
    """Backtest and validate trading strategies."""

    def evaluate_strategy(self, strategy_spec, backtest_params):
        """
        Comprehensive strategy evaluation.

        Returns verdict: EXCELLENT / GOOD / MEDIOCRE / POOR
        """

        # 1. Convert strategy to executable code
        strategy_class = self._generate_strategy_code(strategy_spec)

        # 2. Run walk-forward backtests
        wf_results = []
        for window in backtest_params['walk_forward_windows']:
            result = self._run_backtest(
                strategy_class,
                train_period=window['train'],
                test_period=window['test'],
                symbols=backtest_params['symbols']
            )
            wf_results.append(result)

        # 3. Calculate aggregate metrics
        metrics = self._calculate_metrics(wf_results)

        # 4. Determine verdict
        verdict = self._determine_verdict(metrics)

        # 5. Generate report
        report = self._generate_report(strategy_spec, metrics, verdict, wf_results)

        return report

    def _determine_verdict(self, metrics):
        """
        Determine strategy quality based on metrics.

        EXCELLENT: Sharpe > 2.0, Win rate > 65%, All walk-forward tests profitable
        GOOD: Sharpe > 1.5, Win rate > 60%, Most walk-forward tests profitable
        MEDIOCRE: Sharpe > 1.0, Win rate > 55%, Some walk-forward tests profitable
        POOR: Below MEDIOCRE thresholds
        """

        sharpe = metrics['sharpe_ratio']
        win_rate = metrics['win_rate']
        wf_profitable_pct = metrics['walk_forward_profitable_pct']

        if (sharpe > 2.0 and win_rate > 0.65 and wf_profitable_pct == 1.0):
            return 'EXCELLENT'
        elif (sharpe > 1.5 and win_rate > 0.60 and wf_profitable_pct > 0.75):
            return 'GOOD'
        elif (sharpe > 1.0 and win_rate > 0.55 and wf_profitable_pct > 0.50):
            return 'MEDIOCRE'
        else:
            return 'POOR'
```

#### **Output**
```python
{
    'strategy_name': 'Morning Reversal Scalp',
    'verdict': 'GOOD',

    'overall_metrics': {
        'sharpe_ratio': 1.68,
        'win_rate': 0.63,
        'total_trades': 127,
        'avg_win': 1.18,
        'avg_loss': -0.58,
        'max_drawdown': -8.3,
        'profit_factor': 2.1
    },

    'walk_forward_results': [
        {
            'test_period': '2024-07-01 to 2024-09-30',
            'sharpe': 1.72,
            'win_rate': 0.65,
            'total_return': 8.2,
            'profitable': True
        },
        {
            'test_period': '2024-10-01 to 2024-12-31',
            'sharpe': 1.55,
            'win_rate': 0.61,
            'total_return': 6.8,
            'profitable': True
        },
        # ... more windows
    ],

    'walk_forward_profitable_pct': 0.83,  # 5 out of 6 windows profitable

    'robustness_analysis': {
        'sharpe_std_dev': 0.15,  # Low variance = robust
        'win_rate_std_dev': 0.04,
        'consistent_across_symbols': True,
        'consistent_across_regimes': False  # Weaker in volatile markets
    },

    'recommendation': 'CONVERT_TO_HEURISTIC',  # or 'NEEDS_TUNING' or 'REJECT'

    'improvement_suggestions': [
        'Consider VIX filter (skip when VIX > 20)',
        'May need tighter stop-loss in volatile periods',
        'Works best on large-cap IT and Banking stocks'
    ],

    'cost': 0.0  # No LLM cost
}
```

#### **Performance Targets**
- Backtest completion: < 2 minutes per strategy
- False positive rate (marks bad strategy as good): < 10%
- Walk-forward windows: Minimum 4 out-of-sample periods

---

### **7. Heuristic Converter Agent** ⚡

**Role**: Convert LLM-based strategies to fast Python code

**Phase**: 2 (Strategy Evolution)

**Complexity**: High

**LLM Usage**: High (code generation, one-time per strategy)

#### **Responsibilities**
1. Take natural language strategy specification
2. Generate Python class inheriting from BaseAgent
3. Implement entry/exit logic as pure Python (no LLM calls)
4. Validate generated code syntax
5. Backtest heuristic to verify it matches LLM version
6. Deploy as production-ready agent

#### **Key Innovation**: 300x Speed Improvement
- **LLM Strategy**: 3-5 seconds per decision (LLM call required)
- **Heuristic Strategy**: 10 milliseconds per decision (pure Python)

#### **Inputs**
```python
{
    'strategy_spec': {  # From Strategy Inventor
        'strategy_name': 'Morning Reversal Scalp',
        'entry_rules': [
            'Stock moves >2% in first 30 minutes',
            'Move is >2 standard deviations from 5-day average opening range',
            'Volume in first 30 min is BELOW 5-day average',
            'No major news',
            'Enter counter-trend at 9:50 AM'
        ],
        'exit_rules': [
            'Take profit: 50% retracement of opening move',
            'Stop loss: Move extends 0.5% against us',
            'Time exit: 11:30 AM'
        ],
        'indicators_needed': ['5-day opening range std dev', '5-day avg volume', 'VWAP']
    },
    'backtest_validation': {  # From Strategy Evaluator
        'expected_sharpe': 1.68,
        'expected_win_rate': 0.63,
        'expected_total_trades': 127
    }
}
```

#### **LLM Prompt** (Code Generation)
```python
HEURISTIC_CONVERTER_PROMPT = """
You are a Python code generator for trading strategies.

Convert the following strategy specification into a Python class that inherits from BaseAgent.

STRATEGY SPECIFICATION:
Name: {strategy_name}

Entry Rules:
{entry_rules}

Exit Rules:
{exit_rules}

Indicators Needed:
{indicators_needed}

TEMPLATE:
```python
from agents.base import BaseAgent
import pandas as pd
import numpy as np

class {ClassName}(BaseAgent):
    \"\"\"
    {strategy_description}

    Entry: {entry_summary}
    Exit: {exit_summary}
    \"\"\"

    def __init__(self, params=None):
        super().__init__(name="{strategy_name}")

        # Strategy parameters
        self.opening_move_threshold = params.get('opening_move_threshold', 2.0)  # %
        self.std_dev_threshold = params.get('std_dev_threshold', 2.0)
        self.entry_time = params.get('entry_time', '09:50')
        self.exit_time = params.get('exit_time', '11:30')
        self.take_profit_pct = params.get('take_profit_pct', 50)  # % of retracement
        self.stop_loss_extension = params.get('stop_loss_extension', 0.5)  # %

        # State tracking
        self.opening_move = None
        self.entry_price = None

    def _calculate_indicators(self, data):
        \"\"\"Calculate required indicators.\"\"\"
        # Implement indicator calculations here
        # Example:
        data['opening_range_std'] = data['open'].rolling(5).std()
        data['avg_volume'] = data['volume'].rolling(5).mean()
        data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()
        return data

    def should_enter(self, symbol, current_time, data):
        \"\"\"Check if entry conditions are met.\"\"\"

        # Must be entry time (9:50 AM)
        if current_time.strftime('%H:%M') != self.entry_time:
            return False, None

        # Get today's data
        today_open = data['open'].iloc[-1]
        current_price = data['close'].iloc[-1]

        # Calculate opening move
        self.opening_move = ((current_price - today_open) / today_open) * 100

        # Check if move is significant (>2%)
        if abs(self.opening_move) < self.opening_move_threshold:
            return False, None

        # Check if move is >2 std dev from 5-day average
        opening_range_std = data['opening_range_std'].iloc[-1]
        if abs(self.opening_move) < (self.std_dev_threshold * opening_range_std):
            return False, None

        # Check volume (should be BELOW average)
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['avg_volume'].iloc[-1]
        if current_volume > avg_volume:
            return False, None

        # All conditions met - enter COUNTER to the move
        action = 'SELL' if self.opening_move > 0 else 'BUY'

        confidence = 0.70  # Base confidence

        # Boost confidence if move is very exaggerated
        if abs(self.opening_move) > 3.0:
            confidence += 0.10

        self.entry_price = current_price

        return True, {
            'action': action,
            'confidence': confidence,
            'reason': f"Exaggerated opening move {self.opening_move:+.2f}% on low volume - expecting reversal"
        }

    def should_exit(self, symbol, current_time, data, position):
        \"\"\"Check if exit conditions are met.\"\"\"

        current_price = data['close'].iloc[-1]

        # Time-based exit (11:30 AM)
        if current_time.strftime('%H:%M') >= self.exit_time:
            return True, "Time exit (11:30 AM)"

        # Calculate P&L
        if position['action'] == 'BUY':
            pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # SELL (short)
            pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100

        # Take profit: 50% retracement of opening move
        target_retracement = abs(self.opening_move) * (self.take_profit_pct / 100)
        if pnl_pct >= target_retracement:
            return True, f"Take profit: {pnl_pct:.2f}% ({self.take_profit_pct}% retracement achieved)"

        # Stop loss: Move extends 0.5% against us
        if pnl_pct < -self.stop_loss_extension:
            return True, f"Stop loss: {pnl_pct:.2f}% loss"

        return False, None
```

IMPORTANT:
1. Generate syntactically correct Python code
2. Inherit from BaseAgent
3. Implement all logic as pure Python (NO LLM calls inside the class)
4. Add parameters for tuning (with sensible defaults)
5. Include clear comments
6. Handle edge cases (missing data, etc.)

OUTPUT: Complete Python class code (no explanations, just code)
"""
```

#### **Output**
```python
{
    'strategy_name': 'Morning Reversal Scalp',
    'heuristic_class_name': 'MorningReversalScalpAgent',
    'generated_code': '... [full Python class code] ...',
    'code_validation': {
        'syntax_valid': True,
        'imports_valid': True,
        'inherits_from_base': True,
        'no_llm_calls': True
    },
    'backtest_match': {
        'sharpe_diff': 0.03,  # Heuristic: 1.65, LLM version: 1.68
        'win_rate_diff': 0.01,
        'total_trades_diff': 2,
        'match_quality': 'EXCELLENT'  # <5% difference
    },
    'performance_improvement': {
        'llm_latency_ms': 3500,
        'heuristic_latency_ms': 12,
        'speedup_factor': 291
    },
    'recommendation': 'DEPLOY_TO_PRODUCTION',
    'cost': 0.35  # One-time LLM cost for code generation
}
```

#### **Deployment Process**
```python
def deploy_heuristic_strategy(self, heuristic_code, strategy_name):
    """Deploy heuristic as production agent."""

    # 1. Write code to file
    file_path = f"agents/{strategy_name.lower().replace(' ', '_')}.py"
    with open(file_path, 'w') as f:
        f.write(heuristic_code)

    # 2. Import and instantiate
    module = importlib.import_module(f"agents.{strategy_name.lower().replace(' ', '_')}")
    agent_class = getattr(module, f"{strategy_name.replace(' ', '')}Agent")

    # 3. Add to active agents
    self.active_agents.append(agent_class())

    logger.info(f"✅ Deployed {strategy_name} as heuristic agent")
```

#### **Performance Targets**
- Code generation success rate: > 90%
- Backtest match quality: < 5% difference from LLM version
- Speedup factor: > 200x
- Code generation time: < 30 seconds

---

### **8. Parameter Tuner Agent** 🎯

**Role**: Optimize heuristic strategy parameters without LLM

**Phase**: 2 (Strategy Evolution)

**Complexity**: Medium

**LLM Usage**: None (pure Bayesian optimization)

#### **Responsibilities**
1. Take deployed heuristic strategy
2. Identify tunable parameters
3. Run Bayesian optimization to find optimal values
4. A/B test parameter sets
5. Update strategy with best parameters

#### **Key Advantage**: No LLM calls needed, runs frequently (daily/weekly)

#### **Inputs**
```python
{
    'strategy': MorningReversalScalpAgent,
    'tunable_params': {
        'opening_move_threshold': {
            'current': 2.0,
            'range': [1.5, 3.0],
            'type': 'float'
        },
        'std_dev_threshold': {
            'current': 2.0,
            'range': [1.5, 2.5],
            'type': 'float'
        },
        'take_profit_pct': {
            'current': 50,
            'range': [40, 70],
            'type': 'int'
        },
        'stop_loss_extension': {
            'current': 0.5,
            'range': [0.3, 0.8],
            'type': 'float'
        }
    },
    'optimization_window': {
        'train': '2025-07-01 to 2025-12-31',
        'test': '2026-01-01 to 2026-02-07'
    },
    'objective': 'sharpe_ratio'  # or 'win_rate' or 'profit_factor'
}
```

#### **Bayesian Optimization Implementation**
```python
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor

class ParameterTunerAgent:
    """Optimize strategy parameters using Bayesian optimization."""

    def optimize_parameters(self, strategy, tunable_params, optimization_window, n_iterations=50):
        """
        Find optimal parameters using Bayesian optimization.

        Args:
            strategy: Heuristic agent class
            tunable_params: Dict of parameters to tune
            optimization_window: Train/test split
            n_iterations: Number of optimization iterations

        Returns:
            Best parameters and performance
        """

        # 1. Define parameter search space
        param_bounds = [
            (param['range'][0], param['range'][1])
            for param in tunable_params.values()
        ]

        # 2. Define objective function (backtest with parameters)
        def objective(params_array):
            # Convert array to parameter dict
            params_dict = {
                name: value
                for name, value in zip(tunable_params.keys(), params_array)
            }

            # Instantiate strategy with these parameters
            strategy_instance = strategy(params=params_dict)

            # Backtest on training window
            result = self._run_backtest(
                strategy_instance,
                start_date=optimization_window['train'].split(' to ')[0],
                end_date=optimization_window['train'].split(' to ')[1]
            )

            # Return negative Sharpe (we're minimizing)
            return -result['sharpe_ratio']

        # 3. Run Bayesian optimization
        from bayes_opt import BayesianOptimization

        optimizer = BayesianOptimization(
            f=objective,
            pbounds={name: tuple(param['range']) for name, param in tunable_params.items()},
            random_state=42
        )

        optimizer.maximize(
            init_points=10,  # Random exploration
            n_iter=n_iterations  # Bayesian optimization
        )

        # 4. Get best parameters
        best_params = optimizer.max['params']

        # 5. Validate on test window
        strategy_optimized = strategy(params=best_params)
        test_result = self._run_backtest(
            strategy_optimized,
            start_date=optimization_window['test'].split(' to ')[0],
            end_date=optimization_window['test'].split(' to ')[1]
        )

        return {
            'best_params': best_params,
            'train_sharpe': -optimizer.max['target'],
            'test_sharpe': test_result['sharpe_ratio'],
            'improvement_pct': ((test_result['sharpe_ratio'] / tunable_params['sharpe_current']) - 1) * 100
        }
```

#### **Output**
```python
{
    'strategy_name': 'Morning Reversal Scalp',
    'optimization_result': {
        'original_params': {
            'opening_move_threshold': 2.0,
            'std_dev_threshold': 2.0,
            'take_profit_pct': 50,
            'stop_loss_extension': 0.5
        },
        'optimized_params': {
            'opening_move_threshold': 1.8,  # Tightened
            'std_dev_threshold': 2.2,       # Increased
            'take_profit_pct': 55,          # Increased
            'stop_loss_extension': 0.4      # Tightened
        },
        'performance_comparison': {
            'original_sharpe': 1.68,
            'optimized_sharpe': 1.89,
            'improvement_pct': 12.5
        },
        'test_window_validation': {
            'sharpe': 1.92,
            'win_rate': 0.66,
            'total_trades': 31
        },
        'recommendation': 'DEPLOY_OPTIMIZED_PARAMS',
        'confidence': 0.85
    },
    'cost': 0.0  # No LLM cost
}
```

#### **Deployment Strategy**
```python
# A/B Testing: Run both parameter sets in parallel
def ab_test_parameters(self, strategy, original_params, optimized_params, duration_days=7):
    """
    A/B test new parameters against original.

    Run both in paper trading for 7 days, compare results.
    """

    results_original = []
    results_optimized = []

    for day in range(duration_days):
        # Run both strategies
        result_original = self.paper_trade(strategy(params=original_params))
        result_optimized = self.paper_trade(strategy(params=optimized_params))

        results_original.append(result_original)
        results_optimized.append(result_optimized)

    # Compare
    if mean(results_optimized) > mean(results_original):
        logger.info("✅ Optimized parameters perform better - deploying")
        return optimized_params
    else:
        logger.info("⚠️ Original parameters still better - keeping original")
        return original_params
```

#### **Performance Targets**
- Optimization time: < 5 minutes for 50 iterations
- Sharpe improvement: > 10% on average
- Robustness: Optimized params should work in test window

---

## 🧠 **LEARNING AGENTS** (Post-Trade)

---

### **9. Post-Trade Analyzer Agent** 📝

**Role**: Analyze every closed trade to extract learnings

**Phase**: 3 (Continuous Learning)

**Complexity**: Medium

**LLM Usage**: Medium (per closed trade)

#### **Trigger**: Immediately after any position is closed (win or loss)

#### **Inputs**
```python
{
    'closed_trade': {
        'id': 'TRADE_20260207_001',
        'symbol': 'RELIANCE',
        'action': 'BUY',
        'entry_time': '2026-02-07 10:15:00',
        'entry_price': 2485.00,
        'exit_time': '2026-02-07 12:45:00',
        'exit_price': 2510.50,
        'quantity': 10,
        'pnl': 255.00,  # After costs
        'pnl_pct': 1.03,
        'holding_hours': 2.5,

        # AI decision data
        'ai_reasoning': 'Strong momentum breakout with bullish MACD crossover. Market regime favorable.',
        'expected_outcome': 'Target: +2.5%, Stop: -1.5%',
        'confidence': 0.78,
        'risk_score': 42,
        'agent': 'MomentumAgent',

        # Market context at entry
        'market_regime': 'TRENDING_UP',
        'vix': 14.2,
        'sector_sentiment': 'BULLISH',
        'news_at_entry': ['Oil prices rise on OPEC+ supply cut', ...]
    },
    'market_data': {  # Historical data during the trade
        'price_action': [...],  # Minute-by-minute prices
        'volume': [...],
        'indicators': {'rsi': [...], 'macd': [...]}
    }
}
```

#### **LLM Prompt**
```python
POST_TRADE_ANALYSIS_PROMPT = """
Analyze this completed trade to extract learnings.

TRADE DETAILS:
Symbol: {symbol}
Action: {action}
Entry: ₹{entry_price} at {entry_time}
Exit: ₹{exit_price} at {exit_time}
Holding Period: {holding_hours:.1f} hours

OUTCOME:
Result: {result}  (WIN or LOSS)
P&L: ₹{pnl:,.2f} ({pnl_pct:+.2f}%)

AI'S PREDICTION AT ENTRY:
Reasoning: {ai_reasoning}
Expected: {expected_outcome}
Confidence: {confidence}
Risk Score: {risk_score}

MARKET CONTEXT AT ENTRY:
Regime: {market_regime}
VIX: {vix}
Sector Sentiment: {sector_sentiment}
News: {news_summary}

WHAT ACTUALLY HAPPENED:
{price_action_narrative}

QUESTIONS TO ANSWER:
1. Was the AI's prediction accurate? Why or why not?
2. What signals did we miss (if loss) or correctly identify (if win)?
3. Was the entry timing optimal?
4. Was the exit timing optimal?
5. What could we do differently next time?
6. Is there an emerging pattern (similar to past trades)?

OUTPUT (JSON):
{{
  "prediction_accuracy": "ACCURATE|PARTIALLY_ACCURATE|INACCURATE",

  "what_went_right": [
    "Positive factor 1",
    "Positive factor 2"
  ],

  "what_went_wrong": [
    "Negative factor 1",  // Empty if win
    "Negative factor 2"
  ],

  "missed_signals": [
    "Signal we should have noticed",
    "..."
  ],

  "lessons_learned": [
    "Actionable lesson 1",
    "Actionable lesson 2"
  ],

  "pattern_detected": {{
    "pattern_name": "Name if pattern recognized, else null",
    "description": "Description of pattern",
    "confidence": 0.0-1.0
  }},

  "recommendation_for_future": "Brief actionable recommendation",

  "confidence_in_analysis": 0.0-1.0
}}

Be honest and critical. If we got lucky, say so. If we made mistakes, identify them clearly.
"""
```

#### **Example Output (Winning Trade)**
```python
{
    'trade_id': 'TRADE_20260207_001',
    'result': 'WIN',
    'analysis': {
        'prediction_accuracy': 'ACCURATE',

        'what_went_right': [
            'Momentum breakout signal was confirmed by strong volume',
            'Energy sector sentiment aligned with overall market trend',
            'Entry timing at 10:15 AM caught the start of upward move'
        ],

        'what_went_wrong': [],

        'missed_signals': [
            'Could have held longer - stock continued upward to ₹2,525 (another +1%)',
            'Volume spike at 12:30 PM indicated more momentum, but we exited at 12:45'
        ],

        'lessons_learned': [
            'In TRENDING_UP regime with sector tailwinds, consider wider profit targets',
            'Volume spike during trade = strength continuation signal (hold longer)',
            'RELIANCE responds well to oil price news - build this into model'
        ],

        'pattern_detected': {
            'pattern_name': 'Energy Sector + Oil News Momentum',
            'description': 'RELIANCE shows strong momentum when oil prices rise and energy sector is bullish',
            'confidence': 0.85,
            'similar_trades': ['TRADE_20260125_003', 'TRADE_20260130_007']
        },

        'recommendation_for_future': 'For energy stocks during oil price rallies, use 3% profit target instead of 2.5%',

        'confidence_in_analysis': 0.90
    },
    'cost': 0.06
}
```

#### **Example Output (Losing Trade)**
```python
{
    'trade_id': 'TRADE_20260205_012',
    'result': 'LOSS',
    'analysis': {
        'prediction_accuracy': 'INACCURATE',

        'what_went_right': [
            'Risk management worked - stop loss at -1.5% prevented bigger loss'
        ],

        'what_went_wrong': [
            'Entered momentum trade just as market regime shifted to RANGING',
            'Ignored news about RBI governor speech at 11 AM - caused volatility',
            'RSI was at 72 (overbought) but we entered anyway due to MACD signal'
        ],

        'missed_signals': [
            'VIX started rising 30 minutes before entry (13.8 → 15.2) - volatility warning',
            'Banking sector (correlated with RELIANCE) showed weakness (-0.8%) - sector divergence',
            'News scanner should have flagged RBI event risk'
        ],

        'lessons_learned': [
            'CRITICAL: Always check news calendar for major events (RBI, policy announcements)',
            'When RSI > 70 in RANGING regime, wait for pullback - not a breakout',
            'Cross-check sector performance before entry - divergence is red flag',
            'VIX rising during entry = abort trade'
        ],

        'pattern_detected': {
            'pattern_name': 'Event Risk Ignore Pattern',
            'description': 'We tend to ignore upcoming scheduled events (RBI, earnings) and get caught',
            'confidence': 0.78,
            'similar_trades': ['TRADE_20260128_005', 'TRADE_20260201_009']
        },

        'recommendation_for_future': 'Add event calendar check to pre-trade checklist. Auto-reject trades 1 hour before major events.',

        'confidence_in_analysis': 0.95
    },
    'cost': 0.07
}
```

#### **Learning Database Schema**
```sql
CREATE TABLE trade_learnings (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(50) REFERENCES trade_decisions(decision_id),
    analysis_timestamp TIMESTAMP NOT NULL,

    prediction_accuracy VARCHAR(20),
    what_went_right TEXT[],
    what_went_wrong TEXT[],
    missed_signals TEXT[],
    lessons_learned TEXT[],

    pattern_name VARCHAR(100),
    pattern_description TEXT,
    pattern_confidence DECIMAL(3,2),

    recommendation TEXT,
    analysis_confidence DECIMAL(3,2),

    -- Metadata
    llm_cost DECIMAL(6,4),
    analyst_agent VARCHAR(50)
);

CREATE INDEX idx_trade_learnings_pattern ON trade_learnings(pattern_name);
CREATE INDEX idx_trade_learnings_trade_id ON trade_learnings(trade_id);
```

#### **Actionable Learning Notification**
```python
def notify_actionable_learning(self, analysis):
    """
    If high-confidence learning, notify human immediately.
    """
    if analysis['confidence_in_analysis'] > 0.85:
        alert = f"""
🧠 ACTIONABLE LEARNING DETECTED

Trade: {analysis['trade_id']}
Pattern: {analysis['pattern_detected']['pattern_name']}

Key Lesson:
{analysis['lessons_learned'][0]}

Recommendation:
{analysis['recommendation_for_future']}

Review and consider updating strategy rules.
        """

        self.alerter.send_message(alert)
```

#### **Performance Targets**
- Analysis latency: < 5 seconds per trade
- Cost: < ₹0.10 per trade
- Learning coverage: 100% of closed trades
- Actionable learnings: > 20% of analyses

---

### **10. Pattern Recognizer Agent** 🔍

**Role**: Identify profitable trading patterns from historical data

**Phase**: 3 (Continuous Learning)

**Complexity**: Medium

**LLM Usage**: Low (weekly pattern analysis)

#### **Trigger**: Weekly (every Sunday)

#### **Responsibilities**
1. Query winning trades from last 30-90 days
2. Find common patterns across profitable trades
3. Identify loss patterns to avoid
4. Update strategy confidence boosters
5. Generate pattern report

#### **SQL Queries** (Pattern Mining)
```sql
-- Find profitable patterns (>2% return, at least 5 occurrences)
SELECT
    market_regime,
    symbol_sector,
    EXTRACT(HOUR FROM entry_time) as entry_hour,
    COUNT(*) as num_trades,
    AVG(realized_pnl_pct) as avg_return,
    STDDEV(realized_pnl_pct) as return_volatility,
    AVG(confidence) as avg_ai_confidence
FROM trade_decisions
WHERE outcome = 'WIN'
  AND realized_pnl_pct > 2.0
  AND timestamp > NOW() - INTERVAL '90 days'
GROUP BY market_regime, symbol_sector, entry_hour
HAVING COUNT(*) >= 5
ORDER BY avg_return DESC
LIMIT 10;

-- Example Output:
-- TRENDING_UP, IT, 10, 12 trades, 4.2% avg return, 1.1% volatility, 0.76 confidence
-- RANGING, Banking, 14, 8 trades, 3.8% avg return, 0.9% volatility, 0.72 confidence
```

```sql
-- Find loss patterns to avoid
SELECT
    market_regime,
    symbol_sector,
    CASE
        WHEN risk_score < 30 THEN 'LOW_RISK'
        WHEN risk_score < 60 THEN 'MEDIUM_RISK'
        ELSE 'HIGH_RISK'
    END as risk_category,
    COUNT(*) as num_losses,
    AVG(realized_pnl_pct) as avg_loss,
    ARRAY_AGG(DISTINCT ai_reasoning) as common_reasoning
FROM trade_decisions
WHERE outcome = 'LOSS'
  AND realized_pnl_pct < -1.5
  AND timestamp > NOW() - INTERVAL '90 days'
GROUP BY market_regime, symbol_sector, risk_category
HAVING COUNT(*) >= 3
ORDER BY avg_loss ASC
LIMIT 10;

-- Example Output:
-- VOLATILE, Auto, HIGH_RISK, 5 losses, -2.3% avg loss, ["Momentum in volatile regime", "High VIX ignored"]
```

#### **LLM Prompt** (Pattern Interpretation)
```python
PATTERN_RECOGNIZER_PROMPT = """
Analyze these patterns from our trading history (last 90 days).

PROFITABLE PATTERNS (>2% return, 5+ occurrences):
{profitable_patterns_table}

LOSS PATTERNS (>-1.5% loss, 3+ occurrences):
{loss_patterns_table}

QUESTIONS:
1. What are the most reliable profitable patterns?
2. What conditions lead to losses?
3. Are there any surprising insights?
4. How should we adjust our trading rules?

OUTPUT (JSON):
{{
  "top_patterns": [
    {{
      "pattern_name": "Descriptive name",
      "description": "When this pattern occurs",
      "avg_return": X.X,
      "frequency": X,
      "reliability": "HIGH|MEDIUM|LOW",
      "recommendation": "How to exploit this pattern"
    }},
    ...
  ],

  "patterns_to_avoid": [
    {{
      "pattern_name": "Descriptive name",
      "description": "When this anti-pattern occurs",
      "avg_loss": -X.X,
      "recommendation": "How to avoid"
    }},
    ...
  ],

  "strategic_adjustments": [
    "Adjustment 1: Increase confidence when...",
    "Adjustment 2: Avoid trades when...",
    "Adjustment 3: ..."
  ],

  "insights": "2-3 sentence summary of key findings"
}}
"""
```

#### **Output**
```python
{
    'analysis_period': '2025-11-07 to 2026-02-07',
    'total_trades_analyzed': 127,

    'top_patterns': [
        {
            'pattern_name': 'IT Sector Morning Momentum',
            'description': 'IT stocks (TCS, INFY, WIPRO) at 10-11 AM in TRENDING_UP regime',
            'conditions': {
                'regime': 'TRENDING_UP',
                'sector': 'IT',
                'entry_hour': 10,
                'vix_range': [12, 16]
            },
            'stats': {
                'num_trades': 12,
                'avg_return': 4.2,
                'win_rate': 0.92,
                'avg_confidence': 0.76
            },
            'reliability': 'HIGH',
            'recommendation': 'Boost confidence by +15% for IT stocks in this window. Consider increasing position size to 5% (max).'
        },
        {
            'pattern_name': 'Banking Afternoon Mean Reversion',
            'description': 'Banking stocks (HDFC, ICICI) at 2-3 PM in RANGING regime after morning dip',
            'conditions': {
                'regime': 'RANGING',
                'sector': 'Banking',
                'entry_hour': 14,
                'morning_move': 'negative (< -1%)'
            },
            'stats': {
                'num_trades': 8,
                'avg_return': 3.8,
                'win_rate': 0.875,
                'avg_confidence': 0.72
            },
            'reliability': 'HIGH',
            'recommendation': 'Add specific rule: If banking stock down >1% by noon in RANGING regime, flag for mean reversion at 2 PM.'
        }
    ],

    'patterns_to_avoid': [
        {
            'pattern_name': 'High-VIX Auto Sector Trades',
            'description': 'Auto stocks when VIX > 18',
            'conditions': {
                'sector': 'Auto',
                'vix': '> 18'
            },
            'stats': {
                'num_losses': 5,
                'avg_loss': -2.3,
                'win_rate': 0.20
            },
            'recommendation': 'Auto-reject Auto sector trades when VIX > 18. Pattern shows 80% loss rate.'
        },
        {
            'pattern_name': 'Momentum in Volatile Regime',
            'description': 'Momentum agent trades during VOLATILE regime',
            'conditions': {
                'regime': 'VOLATILE',
                'agent': 'MomentumAgent'
            },
            'stats': {
                'num_losses': 7,
                'avg_loss': -1.9,
                'win_rate': 0.30
            },
            'recommendation': 'Disable MomentumAgent when regime is VOLATILE. Use Mean Reversion instead.'
        }
    ],

    'strategic_adjustments': [
        'Increase IT sector confidence by +15% during 10-11 AM in TRENDING_UP regime',
        'Add VIX-based sector filter: Reject Auto when VIX > 18',
        'Disable MomentumAgent in VOLATILE regime, favor Mean Reversion',
        'Add time-based mean reversion rule for banking stocks at 2 PM after morning weakness'
    ],

    'insights': 'IT sector shows strong morning momentum (10-11 AM) with 92% win rate. Auto sector is VIX-sensitive (avoid when volatile). Momentum strategies fail in VOLATILE regime - switch to mean reversion.',

    'confidence': 0.88,
    'cost': 0.09
}
```

#### **Apply Patterns to AI Orchestrator**
```python
class PatternApplier:
    """Apply discovered patterns to boost/reduce confidence."""

    def apply_pattern_adjustments(self, decision, patterns):
        """
        Adjust decision confidence based on discovered patterns.
        """
        original_confidence = decision['confidence']
        confidence_adjustment = 0.0

        # Check if current decision matches profitable pattern
        for pattern in patterns['top_patterns']:
            if self._matches_pattern(decision, pattern['conditions']):
                confidence_adjustment += 0.15  # Boost
                decision['pattern_match'] = pattern['pattern_name']
                logger.info(f"✅ Profitable pattern detected: {pattern['pattern_name']} (+15% confidence)")

        # Check if current decision matches loss pattern
        for pattern in patterns['patterns_to_avoid']:
            if self._matches_pattern(decision, pattern['conditions']):
                confidence_adjustment -= 0.30  # Penalize heavily
                decision['anti_pattern_match'] = pattern['pattern_name']
                logger.warning(f"⚠️ Loss pattern detected: {pattern['pattern_name']} (-30% confidence)")

        # Apply adjustment
        decision['confidence'] = max(0.0, min(1.0, original_confidence + confidence_adjustment))
        decision['confidence_adjustment'] = confidence_adjustment

        return decision
```

#### **Performance Targets**
- Weekly analysis time: < 10 minutes
- Cost: < ₹5/week
- Pattern accuracy: > 75% (patterns should predict future outcomes)
- Actionable patterns: > 3 per week

---

## 📊 **Summary Table: All Agents**

| Agent | Phase | LLM Usage | Latency | Cost/Call | Trigger | Key Output |
|-------|-------|-----------|---------|-----------|---------|------------|
| **AI Orchestrator** | 1 | Medium | 3s | ₹0.05 | Per trade | Final decision + risk score |
| **Market Intelligence** | 1 | Medium | 5s (100ms cached) | ₹0.08 | Hourly | Market regime + sentiment |
| **Risk Sentinel** | 1 | Low | 100ms | ₹0.02 | Continuous | Risk violations |
| **Portfolio Optimizer** | 1 | Low | 2s | ₹0.04 | Per trade | Optimal position size |
| **Strategy Inventor** | 2 | High | 10s | ₹0.25 | Weekly | New strategy ideas |
| **Strategy Evaluator** | 2 | None | 120s | ₹0.00 | On-demand | Backtest verdict |
| **Heuristic Converter** | 2 | High | 30s | ₹0.35 | One-time | Python code (300x faster) |
| **Parameter Tuner** | 2 | None | 300s | ₹0.00 | Weekly | Optimized parameters |
| **Post-Trade Analyzer** | 3 | Medium | 5s | ₹0.06 | Per closed trade | Learnings + patterns |
| **Pattern Recognizer** | 3 | Low | 10s | ₹0.09 | Weekly | Profitable patterns |

**Total Monthly Cost Estimate**: ₹1,500 - ₹3,500 (depending on trading frequency and phase)

---

## 🔌 **Integration Points**

### **With Existing System**
- **Data Fetcher**: Agents consume historical data for indicators
- **Quantitative Agents**: Agents receive signals from Momentum/Mean Reversion agents
- **Risk Management**: Agents respect position limits and circuit breakers
- **Paper Trading**: All agents tested in paper trading before live
- **Dashboard**: Agent decisions visualized in Streamlit dashboard
- **Alerts**: Agent actions trigger Telegram notifications

### **Agent-to-Agent Communication**
```
User/Market → Quant Agents (Momentum, Mean Reversion)
                    ↓
                Signal Generated
                    ↓
            AI Orchestrator
                ↓       ↓       ↓
    Market Intel   Risk Sentinel   Portfolio Optimizer
                ↓
        Aggregated Decision
                ↓
        Risk Score Calculated
                ↓
    Autonomy Check (auto vs. approval)
                ↓
        Execute or Route to Dashboard
                ↓
        Log to Audit Trail
                ↓
    (After trade closes) → Post-Trade Analyzer
                ↓
        Learning Database Updated
                ↓
    (Weekly) → Pattern Recognizer
                ↓
        Confidence Adjustments Applied
```

---

## ✅ **Implementation Checklist**

### **Phase 1 Agents**
- [ ] AI Orchestrator
- [ ] Market Intelligence Agent
- [ ] Risk Sentinel Agent
- [ ] Portfolio Optimizer Agent

### **Phase 2 Agents**
- [ ] Strategy Inventor Agent
- [ ] Strategy Evaluator Agent
- [ ] Heuristic Converter Agent
- [ ] Parameter Tuner Agent

### **Phase 3 Agents**
- [ ] Post-Trade Analyzer Agent
- [ ] Pattern Recognizer Agent

---

**Complete Roadmap**: [MASTER_ROADMAP.md](./MASTER_ROADMAP.md)

**Phase Implementation Details**:
- [Phase 1: Foundation](./PHASE_1_FOUNDATION.md)
- [Phase 2: Strategy Evolution](./PHASE_2_STRATEGY_EVOLUTION.md)
- [Phase 3-4: Learning & Autonomy](./PHASE_3_LEARNING_AUTONOMY.md)
