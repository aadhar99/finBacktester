# 📘 PHASE 1: FOUNDATION & INFRASTRUCTURE (Month 1-2)

## 🎯 **Objective**
Build the foundational infrastructure for hybrid AI trading system while continuing paper trading with existing strategies.

**Timeline**: 8 weeks
**Risk Level**: Low (no live trading yet)
**Success Criteria**: Infrastructure ready, audit trails working, AI can make basic decisions

---

## 📋 **Deliverables**

### **1. Audit Trail Database** ✅ Critical for SEBI + AI Learning

**What to Build**:
```sql
-- Complete PostgreSQL/SQLite schema for trade decision logging
-- Every decision, approval, execution, outcome tracked
-- SEBI-compliant audit trail structure
```

**Files to Create**:
- `database/schema.sql` - Database schema
- `database/audit_logger.py` - Audit trail manager
- `database/models.py` - ORM models (SQLAlchemy)

**Implementation**:
```python
class AuditTrailLogger:
    """
    Logs every trading decision for regulatory compliance and AI learning.
    """
    def log_decision(self, decision_data):
        """Log a trading decision with full context."""
        pass

    def log_execution(self, execution_data):
        """Log trade execution details."""
        pass

    def log_outcome(self, outcome_data):
        """Log trade outcome for learning."""
        pass

    def get_sebi_report(self, start_date, end_date):
        """Generate SEBI-compliant audit report."""
        pass
```

**Acceptance Criteria**:
- ✅ Can log 1000+ decisions without performance issues
- ✅ Query response < 100ms for reporting
- ✅ SEBI-compliant export (CSV/JSON)
- ✅ Automatic backups enabled

---

### **2. LLM Integration Layer** 🤖 Connect to Claude/GPT

**What to Build**:
- Unified interface for LLM providers (Claude, GPT-4, fallbacks)
- Prompt template management
- Rate limiting and cost tracking
- Response validation

**Files to Create**:
- `ai/llm_client.py` - LLM client wrapper
- `ai/prompt_templates.py` - Reusable prompts
- `ai/response_validator.py` - Validate LLM outputs

**Implementation**:
```python
class LLMClient:
    """
    Unified client for LLM providers with fallbacks.
    """
    def __init__(self, primary="claude", fallback="gpt4"):
        self.primary = self._init_provider(primary)
        self.fallback = self._init_provider(fallback)
        self.cost_tracker = CostTracker()

    async def complete(self, prompt, max_tokens=1000, temperature=0.7):
        """
        Get completion from LLM with automatic fallback.
        """
        try:
            response = await self.primary.complete(prompt, max_tokens, temperature)
            self.cost_tracker.log(provider="primary", tokens=response.tokens)
            return response.text
        except Exception as e:
            logger.warning(f"Primary LLM failed: {e}, using fallback")
            response = await self.fallback.complete(prompt, max_tokens, temperature)
            self.cost_tracker.log(provider="fallback", tokens=response.tokens)
            return response.text

    def get_cost_summary(self):
        """Get daily/monthly LLM cost breakdown."""
        return self.cost_tracker.get_summary()


class PromptTemplates:
    """
    Centralized prompt template management.
    """
    @staticmethod
    def market_intelligence(market_data):
        return f"""
You are a market analyst for Indian stock markets (NSE).

Current Data:
{json.dumps(market_data, indent=2)}

Analyze and provide:
1. Market regime classification
2. Sector outlook
3. Key risk factors
4. Trading opportunities

Output JSON format.
"""

    @staticmethod
    def trade_decision(context):
        return f"""
You are a trading orchestrator.

Context:
{json.dumps(context, indent=2)}

Decide: APPROVE, REJECT, or MODIFY this trade.

Output JSON with risk_score, confidence, reasoning.
"""
```

**Acceptance Criteria**:
- ✅ Can switch between Claude/GPT-4 seamlessly
- ✅ Automatic fallback if primary fails
- ✅ Cost tracking per agent/decision
- ✅ Response time < 5 seconds (p95)
- ✅ Validation catches malformed responses

---

### **3. Market Intelligence Agent** 📰 First AI Agent

**What to Build**:
- News sentiment analysis (RSS feeds, financial news)
- RBI policy tracking
- Sector rotation detection
- Simple market regime classification (augments existing)

**Files to Create**:
- `ai/agents/market_intelligence.py` - Core agent
- `ai/agents/news_fetcher.py` - News collection
- `ai/agents/sentiment_analyzer.py` - Sentiment scoring

**Data Sources** (Free):
- Moneycontrol RSS feeds
- Economic Times RSS
- RBI press releases
- NSE announcements
- Twitter FinTwit (optional)

**Implementation**:
```python
class MarketIntelligenceAgent:
    """
    Analyzes market context beyond technical indicators.
    """
    def __init__(self, llm_client):
        self.llm = llm_client
        self.news_fetcher = NewsFetcher()

    async def analyze_market(self, current_date):
        """
        Comprehensive market analysis.

        Returns:
            {
                "regime": "TRENDING_UP|DOWN|RANGING",
                "confidence": 0.78,
                "sector_outlook": {...},
                "risk_factors": [...],
                "opportunities": [...]
            }
        """
        # Fetch recent news
        news = self.news_fetcher.get_recent_news(days=3)

        # Get market data
        market_data = self.get_market_context(current_date)

        # Build prompt
        prompt = PromptTemplates.market_intelligence({
            "date": current_date,
            "news": news[:10],  # Top 10 headlines
            "market_data": market_data
        })

        # Get LLM analysis
        response = await self.llm.complete(prompt)

        # Validate and return
        analysis = json.loads(response)
        return self.validate_analysis(analysis)

    def get_market_context(self, date):
        """Get current market metrics."""
        return {
            "nifty_level": self.get_nifty_level(date),
            "vix": self.get_vix(date),
            "advance_decline": self.get_breadth(date),
            "sector_performance": self.get_sector_returns(date)
        }


class NewsFetcher:
    """Fetch financial news from RSS feeds."""

    RSS_FEEDS = [
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        # Add more
    ]

    def get_recent_news(self, days=3):
        """Fetch news from last N days."""
        news_items = []
        for feed_url in self.RSS_FEEDS:
            items = self.parse_rss(feed_url, days)
            news_items.extend(items)

        return sorted(news_items, key=lambda x: x['date'], reverse=True)
```

**Acceptance Criteria**:
- ✅ Fetches news from 3+ sources
- ✅ Sentiment scoring works (positive/negative/neutral)
- ✅ Market regime classification 70%+ accuracy vs manual labeling
- ✅ Analysis completes in < 10 seconds
- ✅ Cached results (don't re-analyze same data)

---

### **4. Explainability System** 💡 Make AI Decisions Transparent

**What to Build**:
- Natural language explanation generator
- Decision breakdown visualization
- Human-readable reasoning chains

**Files to Create**:
- `ai/explainability/explainer.py` - Core explainer
- `ai/explainability/templates.py` - Explanation templates

**Implementation**:
```python
class DecisionExplainer:
    """
    Converts AI decisions into human-readable explanations.
    """
    def explain_trade_decision(self, decision_data):
        """
        Generate clear explanation of why AI made this decision.

        Args:
            decision_data: Full decision context from AI

        Returns:
            {
                "simple_summary": "2-sentence explanation",
                "detailed_reasoning": "Paragraph explanation",
                "key_factors": ["factor1", "factor2", ...],
                "risk_assessment": "What could go wrong",
                "alternative_considered": "Why not do X instead"
            }
        """
        template = f"""
**Trade Decision: {decision_data.action} {decision_data.symbol}**

**Quick Summary:**
{self._generate_simple_summary(decision_data)}

**Detailed Reasoning:**
{self._generate_detailed_reasoning(decision_data)}

**Key Factors:**
{self._list_key_factors(decision_data)}

**Risk Assessment:**
{self._assess_risks(decision_data)}

**Why This Over Alternatives:**
{self._explain_alternatives(decision_data)}
"""
        return template

    def _generate_simple_summary(self, data):
        """2-sentence summary for busy humans."""
        if data.action == "BUY":
            return f"""
AI recommends buying {data.quantity} shares of {data.symbol} at ₹{data.price}.
{data.primary_reason} with {data.confidence*100:.0f}% confidence.
"""
```

**Acceptance Criteria**:
- ✅ Every decision has explanation < 200 words
- ✅ Non-technical person can understand
- ✅ Shows both pros and cons
- ✅ Includes "what could go wrong"

---

### **5. Human Approval Dashboard** 🖥️ Review & Approve Trades

**What to Build**:
- Real-time approval interface (extend existing Streamlit dashboard)
- Shows AI reasoning, risk score, historical performance
- One-click approve/reject
- Override capability

**Files to Create**:
- `dashboard/approval_page.py` - Approval interface
- `dashboard/components/decision_card.py` - Decision display component

**UI Design**:
```
┌──────────────────────────────────────────────────────┐
│  🔔 PENDING APPROVAL                    Risk: 72/100  │
├──────────────────────────────────────────────────────┤
│                                                       │
│  BUY RELIANCE @ ₹2,500  (10 shares = ₹25,000)       │
│                                                       │
│  AI Reasoning:                                        │
│  Momentum breakout with volume confirmation.         │
│  Market regime supportive. Low correlation with      │
│  existing positions.                                  │
│                                                       │
│  Risk Factors:                                        │
│  • Position size near 5% limit                        │
│  • Already holding energy positions                   │
│                                                       │
│  Historical Similar: 3 trades, 2 wins (67%)          │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ APPROVE  │  │ REJECT   │  │ MODIFY   │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└──────────────────────────────────────────────────────┘
```

**Implementation**:
```python
import streamlit as st

def approval_dashboard():
    """Main approval interface."""
    st.title("🔔 Trade Approvals")

    # Get pending decisions
    pending = db.get_pending_approvals()

    if not pending:
        st.success("No pending approvals!")
        return

    # Show each decision
    for decision in pending:
        with st.container():
            render_decision_card(decision)

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("✅ Approve", key=f"approve_{decision.id}"):
                    approve_trade(decision.id)
                    st.rerun()

            with col2:
                if st.button("❌ Reject", key=f"reject_{decision.id}"):
                    reject_trade(decision.id)
                    st.rerun()

            with col3:
                if st.button("✏️ Modify", key=f"modify_{decision.id}"):
                    show_modify_form(decision)

            st.markdown("---")


def render_decision_card(decision):
    """Render a single decision for approval."""
    st.subheader(f"{decision.action} {decision.symbol}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Quantity", f"{decision.quantity} shares")
        st.metric("Value", f"₹{decision.value:,.0f}")

    with col2:
        st.metric("Risk Score", f"{decision.risk_score}/100")
        st.metric("Confidence", f"{decision.confidence*100:.0f}%")

    # AI Reasoning
    st.markdown("**AI Reasoning:**")
    st.info(decision.reasoning)

    # Risk factors
    if decision.risk_factors:
        st.warning("**Risk Factors:**\n" + "\n".join(f"• {r}" for r in decision.risk_factors))

    # Historical performance
    similar_trades = db.get_similar_trades(decision.symbol, decision.strategy)
    if similar_trades:
        win_rate = len([t for t in similar_trades if t.pnl > 0]) / len(similar_trades)
        st.caption(f"Historical: {len(similar_trades)} similar trades, {win_rate*100:.0f}% win rate")
```

**Acceptance Criteria**:
- ✅ Shows all pending approvals
- ✅ Can approve/reject in < 3 clicks
- ✅ Explanation clearly visible
- ✅ Historical context shown
- ✅ Mobile-friendly (can approve on phone)
- ✅ Auto-refreshes when new decisions arrive

---

### **6. Basic AI Orchestrator** 🧠 Simple Decision Making

**What to Build**:
- Combines Market Intelligence + Existing Strategies
- Makes simple BUY/SELL/HOLD decisions
- Generates reasoning
- Calculates risk scores

**Files to Create**:
- `ai/orchestrator.py` - Main orchestrator
- `ai/risk_scorer.py` - Risk score calculator

**Implementation**:
```python
class AIOrchestrator:
    """
    Master AI that coordinates all agents and makes final decisions.
    """
    def __init__(self, llm_client, market_intel_agent):
        self.llm = llm_client
        self.market_intel = market_intel_agent
        self.risk_scorer = RiskScorer()

    async def evaluate_trade_signal(self, signal, portfolio_state):
        """
        Evaluate a trade signal from traditional strategies.

        Args:
            signal: Signal from MomentumAgent/ReversionAgent
            portfolio_state: Current portfolio state

        Returns:
            {
                "decision": "APPROVE|REJECT|MODIFY",
                "risk_score": 0-100,
                "confidence": 0-1,
                "reasoning": "explanation",
                "modified_size": optional
            }
        """
        # Get market intelligence
        market_context = await self.market_intel.analyze_market(datetime.now())

        # Build decision prompt
        prompt = self._build_decision_prompt(
            signal=signal,
            market_context=market_context,
            portfolio=portfolio_state
        )

        # Get AI decision
        response = await self.llm.complete(prompt)
        decision = json.loads(response)

        # Calculate risk score
        decision['risk_score'] = self.risk_scorer.calculate(
            decision=decision,
            signal=signal,
            portfolio=portfolio_state,
            market=market_context
        )

        # Validate
        is_valid, issues = self.validate_decision(decision)
        if not is_valid:
            logger.error(f"Invalid AI decision: {issues}")
            return self._safe_reject(signal, issues)

        return decision

    def _build_decision_prompt(self, signal, market_context, portfolio):
        """Build the orchestrator decision prompt."""
        return f"""
You are a trading orchestrator evaluating a trade signal.

SIGNAL:
Strategy: {signal.strategy}
Action: {signal.action}
Symbol: {signal.symbol}
Price: ₹{signal.price}
Quantity: {signal.size}
Reason: {signal.reason}

MARKET CONTEXT:
{json.dumps(market_context, indent=2)}

PORTFOLIO STATE:
Current Value: ₹{portfolio.total_value:,.0f}
Cash: ₹{portfolio.cash:,.0f}
Positions: {portfolio.position_count}
Today's P&L: {portfolio.daily_pnl:+.2f}%

DECISION FRAMEWORK:
1. Is market context favorable for this trade?
2. Does signal make sense given current conditions?
3. What's the risk-reward ratio?
4. How does this affect portfolio diversification?
5. What could go wrong?

OUTPUT (JSON):
{{
  "decision": "APPROVE" | "REJECT" | "MODIFY",
  "confidence": 0-1,
  "reasoning": "2-3 sentence explanation",
  "key_factors": ["factor1", "factor2", "factor3"],
  "risk_factors": ["risk1", "risk2"],
  "modified_size": optional_integer_if_modifying
}}

Be conservative. When uncertain, prefer REJECT or reduce size.
"""
```

**Acceptance Criteria**:
- ✅ Can evaluate 10 signals in < 60 seconds
- ✅ Decisions match human judgment 70%+ of time (initial validation)
- ✅ All decisions have clear reasoning
- ✅ Risk scores correlate with actual risk

---

## 🗓️ **Week-by-Week Plan**

### **Week 1-2: Database & Audit Trail**
- [ ] Setup PostgreSQL/SQLite
- [ ] Implement audit logger
- [ ] Test with 1000 sample records
- [ ] Build SEBI export function

### **Week 3-4: LLM Integration**
- [ ] Setup Claude API (Anthropic)
- [ ] Build LLM client wrapper
- [ ] Create prompt templates
- [ ] Add cost tracking
- [ ] Test response validation

### **Week 5-6: Market Intelligence Agent**
- [ ] Setup news fetchers (RSS)
- [ ] Build sentiment analyzer
- [ ] Integrate with LLM
- [ ] Test market regime detection
- [ ] Validate against manual labels

### **Week 7: Explainability + Approval UI**
- [ ] Build decision explainer
- [ ] Extend Streamlit dashboard
- [ ] Add approval interface
- [ ] Test end-to-end approval flow

### **Week 8: AI Orchestrator**
- [ ] Build basic orchestrator
- [ ] Integrate market intel
- [ ] Test with paper trading
- [ ] Tune prompts for accuracy
- [ ] Full system integration test

---

## 📊 **Success Metrics**

### **Technical Metrics**:
- ✅ Audit trail captures 100% of decisions
- ✅ LLM response time < 5s (p95)
- ✅ Market intel accuracy > 70%
- ✅ AI approval rate matches human 70%+ of time
- ✅ Zero data loss in audit logs

### **Operational Metrics**:
- ✅ Can review and approve trade in < 30 seconds
- ✅ LLM costs < ₹500/day during paper trading
- ✅ Dashboard loads in < 2 seconds
- ✅ System uptime > 99%

### **Learning Metrics**:
- ✅ All trades logged for future learning
- ✅ Can query "why did we lose on X trade?" in < 5s
- ✅ Historical pattern detection works

---

## 💰 **Estimated Costs (Phase 1)**

| Item | Monthly Cost | Notes |
|------|--------------|-------|
| Claude API | ₹500-1,000 | Paper trading, low volume |
| Database (Cloud) | ₹200-500 | Or free (SQLite local) |
| News APIs | ₹0 | Using free RSS feeds |
| Total | ₹700-1,500 | Very affordable for learning |

---

## ⚠️ **Risks & Mitigation**

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM API downtime | Can't make AI decisions | Fallback to traditional strategies only |
| Poor AI accuracy | Wrong decisions | Keep human approval 100% in Phase 1 |
| High LLM costs | Budget overrun | Rate limiting, caching, cost alerts |
| Slow response time | Missed trades | Async processing, pre-fetch market intel |

---

## ✅ **Phase 1 Completion Checklist**

Before moving to Phase 2:

- [ ] Audit trail logging 100 paper trades successfully
- [ ] Market Intelligence Agent running daily
- [ ] AI Orchestrator making decisions with >70% human agreement
- [ ] Approval dashboard functional and fast
- [ ] All code in version control (Git)
- [ ] Documentation complete
- [ ] Cost tracking showing < ₹1,500/month
- [ ] Zero critical bugs in 1 week of continuous operation

---

## 📁 **File Structure After Phase 1**

```
Claude-code/
├── ai/
│   ├── llm_client.py              # ✅ NEW
│   ├── prompt_templates.py        # ✅ NEW
│   ├── response_validator.py      # ✅ NEW
│   ├── orchestrator.py            # ✅ NEW
│   ├── risk_scorer.py             # ✅ NEW
│   ├── agents/
│   │   ├── market_intelligence.py # ✅ NEW
│   │   ├── news_fetcher.py        # ✅ NEW
│   │   └── sentiment_analyzer.py  # ✅ NEW
│   └── explainability/
│       ├── explainer.py           # ✅ NEW
│       └── templates.py           # ✅ NEW
├── database/
│   ├── schema.sql                 # ✅ NEW
│   ├── audit_logger.py            # ✅ NEW
│   └── models.py                  # ✅ NEW
├── dashboard/
│   ├── approval_page.py           # ✅ NEW
│   └── components/
│       └── decision_card.py       # ✅ NEW
└── tests/
    └── test_ai_agents.py          # ✅ NEW
```

---

**Next**: [PHASE 2: Strategy Evolution Pipeline](./PHASE_2_STRATEGY_EVOLUTION.md)
