# 🗺️ MASTER ROADMAP: Hybrid AI Trading System

## 📊 **Executive Summary**

**Objective**: Transform a basic quantitative trading system into an AI-augmented system that learns from every trade and gradually achieves 80% autonomous execution.

**Timeline**: 12 months (3 months minimum before live trading)
**Initial Capital**: ₹1,00,000 (start with ₹50,000)
**Target Returns**: Sharpe Ratio > 1.5, consistent profitability
**Risk Profile**: Conservative → Medium (gradual increase)

---

## 🎯 **Current State vs. Target State**

### **Current State** (After Thread 1 - ✅ Complete)

```
✅ Basic quantitative agents (Momentum, Mean Reversion)
✅ Data fetching from yfinance (5x faster with parallelization)
✅ Risk management framework (position sizing, exposure limits)
✅ Backtesting engine
✅ Paper trading infrastructure
✅ Streamlit monitoring dashboard
✅ Telegram alert system
✅ Security fixes (path traversal prevention)
✅ Rate limiting (no API bans)
```

**Decision Flow**: Rule-based agents → Risk checks → Human approval (100%)

### **Target State** (After 12 months)

```
🎯 AI-augmented decision making (LLM + Quant hybrid)
🎯 Market intelligence from news/sentiment
🎯 Self-evolving strategies (AI creates + validates new strategies)
🎯 Heuristic optimization (300x faster than pure LLM)
🎯 Continuous learning from every trade
🎯 80% autonomous execution (only high-risk needs approval)
🎯 Complete audit trail (SEBI-ready)
🎯 Explainable AI decisions
🎯 Performance-based autonomy gates
```

**Decision Flow**: Market Intel → AI Orchestrator → Strategy Selection → Risk Scoring → Auto-execute (if low-risk) OR Human approval (if high-risk) → Continuous learning

---

## 📅 **12-Month Timeline Overview**

| Phase | Months | Focus | Autonomy Level | Key Deliverables |
|-------|--------|-------|----------------|------------------|
| **Phase 1** | 1-2 | Foundation & Infrastructure | 0% (100% approval) | Audit trail, LLM client, Market Intel Agent, Approval dashboard |
| **Phase 2** | 3-5 | Strategy Evolution Pipeline | 0-10% (testing only) | Strategy Inventor, Evaluator, Heuristic Converter, Parameter Tuner |
| **Phase 3** | 6-8 | Continuous Learning | 10-20% (low-risk auto) | Post-trade analysis, Pattern recognition, Adaptive risk scoring |
| **Phase 4** | 9-12 | Gradual Autonomy | 20% → 80% | Performance-based autonomy increase, Production readiness |

---

## 📋 **Phase-by-Phase Breakdown**

### **PHASE 1: FOUNDATION (Month 1-2)** 📘

**Objective**: Build core AI infrastructure without changing decision logic yet

**Budget**: ₹1,500/month (LLM costs)

**Key Components**:

1. **Audit Trail Database** (Week 1-2)
   - PostgreSQL/SQLite schema
   - Captures every decision (AI + Human)
   - SEBI-compliant exports
   - Enables AI learning from outcomes

2. **LLM Integration Layer** (Week 2-3)
   - Unified client (Claude primary, GPT-4 fallback)
   - Cost tracking (<₹1,500/month target)
   - Rate limiting, retry logic
   - Prompt templates

3. **Market Intelligence Agent** (Week 3-5)
   - News sentiment analysis (Economic Times, Moneycontrol)
   - Sector rotation detection
   - Market regime classification (TRENDING_UP/DOWN, RANGING, VOLATILE)
   - RBI policy impact analysis

4. **Explainability System** (Week 5-6)
   - Natural language reasoning for all AI decisions
   - Human-readable justifications
   - SEBI audit-ready explanations

5. **Human Approval Dashboard** (Week 6-8)
   - Real-time decision review
   - Accept/Reject/Modify workflow
   - Performance analytics
   - Integration with existing Streamlit dashboard

6. **AI Orchestrator (Basic)** (Week 7-8)
   - Coordinator for all AI agents
   - Decision aggregation logic
   - Risk score calculation (0-100 scale)
   - Initial approval routing (all trades → human)

**Success Criteria**:
- ✅ All trades logged in audit database
- ✅ Market Intelligence Agent operational
- ✅ LLM costs < ₹1,500/month
- ✅ Approval dashboard functional
- ✅ No system downtime > 1 hour

**Reference**: [PHASE_1_FOUNDATION.md](./PHASE_1_FOUNDATION.md)

---

### **PHASE 2: STRATEGY EVOLUTION (Month 3-5)** 📗

**Objective**: Build AI pipeline that creates, validates, and optimizes trading strategies

**Budget**: ₹2,500-₹3,500/month (higher LLM usage)

**Key Agents**:

1. **Strategy Inventor Agent** (Week 9-11)
   - LLM-powered creative strategy generation
   - Analyzes gaps in current strategy coverage
   - Proposes new trading ideas weekly
   - Output: Natural language strategy descriptions

2. **Strategy Evaluator Agent** (Week 11-13)
   - Rigorous backtesting of new strategies
   - Walk-forward analysis (prevent overfitting)
   - Statistical validation (Sharpe, win rate, max drawdown)
   - Verdict: EXCELLENT/GOOD/MEDIOCRE/POOR

3. **Heuristic Converter** (Week 13-16)
   - **Key Innovation**: Convert LLM strategies → Python code
   - **Performance**: 3-5 seconds → 10 milliseconds (300x faster)
   - Generates BaseAgent subclasses
   - Validates code syntax and backtests match
   - Deploys as production-ready heuristic

4. **Parameter Tuner Agent** (Week 16-20)
   - Optimizes heuristic strategies (no LLM needed)
   - Bayesian optimization for parameter search
   - Daily/weekly tuning runs
   - A/B testing of parameter sets

**Strategy Evolution Flow**:
```
Inventor (creates idea) → Evaluator (validates) → Converter (codes it) → Tuner (optimizes)
     ↓
If EXCELLENT: Convert to heuristic → Deploy to production
If GOOD: Further tuning needed
If MEDIOCRE/POOR: Archive for later review
```

**Success Criteria**:
- ✅ At least 2 new strategies invented
- ✅ At least 1 strategy converted to heuristic
- ✅ Heuristic execution < 10ms per decision
- ✅ Parameter tuner improves Sharpe by >10%
- ✅ All strategies pass walk-forward validation

**Reference**: [PHASE_2_STRATEGY_EVOLUTION.md](./PHASE_2_STRATEGY_EVOLUTION.md)

---

### **PHASE 3: CONTINUOUS LEARNING (Month 6-8)** 📙

**Objective**: Build learning loops so AI improves from every trade

**Budget**: ₹2,000-₹2,500/month

**Key Systems**:

1. **Post-Trade Analysis** (Week 21-23)
   - Analyze every closed trade (win or loss)
   - Compare AI prediction vs. actual outcome
   - Extract learnings: "What did we miss?"
   - Store in learning database

2. **Weekly Performance Review** (Week 23-24)
   - AI reviews its own performance every Sunday
   - Identifies patterns in wins vs. losses
   - Suggests strategy adjustments
   - Detects repeated mistakes

3. **Pattern Recognition** (Week 24-26)
   - SQL queries to find profitable patterns
   - Example: "IT stocks at 10 AM in trending markets → 4.2% avg return"
   - AI uses these patterns to boost confidence
   - Reduces risk scores for proven patterns

4. **Adaptive Risk Scoring** (Week 26-30)
   - Initial: Rule-based risk scoring
   - After 100 trades: ML model trained on outcomes
   - Risk scores improve over time
   - Accuracy target: >75%

5. **Mistake Database** (Week 30-32)
   - Track all AI mistakes with lessons
   - Check new decisions against past mistakes
   - Prevent repeated errors
   - Alert human if similar mistake detected

**Learning Loop**:
```
Trade Executed → Outcome Observed → Analysis (LLM) → Learnings Extracted →
Model Updated → Better Future Decisions
```

**Success Criteria**:
- ✅ Post-trade analysis on 100% of closed trades
- ✅ AI accuracy improves by 10%+ after 100 trades
- ✅ Risk scoring accuracy > 75%
- ✅ At least 5 actionable patterns identified
- ✅ Zero repeated mistakes from database

**Reference**: [PHASE_3_LEARNING_AUTONOMY.md](./PHASE_3_LEARNING_AUTONOMY.md) (Months 6-8)

---

### **PHASE 4: GRADUAL AUTONOMY (Month 9-12)** 📕

**Objective**: Gradually increase AI autonomy from 20% → 80% based on performance

**Budget**: ₹1,500-₹2,000/month (lower LLM usage as heuristics dominate)

**Autonomy Progression Schedule**:

| Month | Autonomy Level | What Needs Approval | Auto-Execute | Performance Gate |
|-------|----------------|---------------------|--------------|------------------|
| **9** | Level 2 (20%) | Medium + High risk | Low risk only (<30 risk score) | Win rate > 55%, Sharpe > 1.3 |
| **10** | Level 2.5 (50%) | High risk only | Low + Some medium (<50) | Win rate > 58%, Sharpe > 1.4 |
| **11** | Level 3 (80%) | High risk only | Low + Medium (<70) | Win rate > 60%, Sharpe > 1.5 |
| **12** | Level 3+ (90%) | Very high risk only | Almost everything (<90) | Win rate > 60%, Sharpe > 1.5, 3 months profitable |

**Auto-Approval Logic**:
```python
def should_auto_execute(decision):
    risk_score = decision.risk_score
    ai_accuracy = last_30_day_accuracy
    autonomy_level = current_autonomy_level

    # NEVER auto-execute if:
    if risk_score > 90:  # Extreme risk
        return False
    if ai_accuracy < 0.60:  # AI performing poorly
        return False
    if market_vix > 30:  # Extreme volatility
        return False

    # Autonomy-based thresholds
    if autonomy_level == 2:  # 20%
        return risk_score < 30
    elif autonomy_level == 2.5:  # 50%
        return risk_score < 50
    elif autonomy_level == 3:  # 80%
        return risk_score < 70
    elif autonomy_level >= 3.5:  # 90%
        return risk_score < 90

    return False  # Default: require approval
```

**Performance-Based Gates**:
- **Increase Autonomy**: Win rate > 60%, Sharpe > 1.5, No major losses (>5%), 20+ trades at current level
- **Decrease Autonomy**: Win rate < 50%, 3 consecutive big losses, Sharpe < 1.0

**Emergency Stop Conditions**:
1. Daily loss > -3%
2. 4 consecutive losses
3. VIX spikes > 35
4. AI confidence drops below 0.50
5. Human manually pauses

**Success Criteria**:
- ✅ Autonomy reaches 80% by Month 12
- ✅ Auto-executed trades have >60% win rate
- ✅ Human approval time < 2 hours/week
- ✅ No autonomy reductions in last 2 months
- ✅ 3 consecutive months profitable
- ✅ All circuit breakers tested and working

**Reference**: [PHASE_3_LEARNING_AUTONOMY.md](./PHASE_3_LEARNING_AUTONOMY.md) (Months 9-12)

---

## 🤖 **AI Agent Architecture Summary**

### **Core Agents** (Always Active)

| Agent | Role | LLM Usage | Speed | Phase |
|-------|------|-----------|-------|-------|
| **Market Intelligence** | News, sentiment, regime detection | Medium (daily) | ~5s/analysis | Phase 1 |
| **AI Orchestrator** | Master decision coordinator | Low (per decision) | ~2s | Phase 1 |
| **Risk Sentinel** | Portfolio monitoring, alerts | Low (when triggered) | Real-time | Phase 1 |
| **Portfolio Optimizer** | Allocation, diversification | Medium (daily) | ~3s | Phase 1 |

### **Strategy Evolution Agents** (Periodic)

| Agent | Role | LLM Usage | Speed | Phase |
|-------|------|-----------|-------|-------|
| **Strategy Inventor** | Create new strategies | High (weekly) | ~10s | Phase 2 |
| **Strategy Evaluator** | Backtest validation | None (pure compute) | ~30s | Phase 2 |
| **Heuristic Converter** | Code generation | High (one-time per strategy) | ~15s | Phase 2 |
| **Parameter Tuner** | Optimize heuristics | None (Bayesian opt) | ~60s | Phase 2 |

### **Learning Agents** (Post-Trade)

| Agent | Role | LLM Usage | Speed | Phase |
|-------|------|-----------|-------|-------|
| **Post-Trade Analyzer** | Analyze closed trades | Medium (per trade) | ~5s | Phase 3 |
| **Pattern Recognizer** | Find profitable patterns | Low (weekly) | ~10s | Phase 3 |
| **Mistake Tracker** | Prevent repeated errors | Low (on-demand) | ~2s | Phase 3 |

**Total LLM Cost Estimate**: ₹1,500-₹3,500/month (peaks in Phase 2, drops in Phase 4)

---

## 🛡️ **Risk Management Framework**

### **Multi-Layer Safety**

1. **Position-Level Limits**
   - Max 5% per position
   - Stop-loss at -2% per trade
   - Take-profit at +3% per trade

2. **Portfolio-Level Limits**
   - Max 30% total exposure
   - Max 6 concurrent positions
   - Sector concentration < 20%

3. **Daily Circuit Breakers**
   - Daily loss > -3% → Halt trading
   - 4 consecutive losses → Reduce autonomy
   - VIX > 35 → Human approval for all trades

4. **AI-Specific Safeguards**
   - Confidence < 0.50 → Reject trade
   - Risk score > 90 → Always require approval
   - Repeated mistake detected → Alert human
   - Performance degradation → Auto-reduce autonomy

5. **Emergency Stops**
   - Manual pause button (human override)
   - Drawdown > -15% → Trading halt
   - System error detection → Safe mode

---

## 📊 **Success Metrics by Phase**

### **Phase 1 (Foundation)**
| Metric | Target | Purpose |
|--------|--------|---------|
| Audit trail coverage | 100% | SEBI compliance + Learning |
| LLM API costs | < ₹1,500/month | Budget control |
| Market Intel accuracy | > 70% | Regime detection quality |
| System uptime | > 99% | Reliability |
| Dashboard response time | < 2s | User experience |

### **Phase 2 (Strategy Evolution)**
| Metric | Target | Purpose |
|--------|--------|---------|
| New strategies invented | ≥ 2 | Innovation pipeline |
| Strategies converted to heuristics | ≥ 1 | Performance optimization |
| Heuristic execution time | < 10ms | Real-time capability |
| Parameter tuner Sharpe improvement | > +10% | Optimization effectiveness |
| Walk-forward test pass rate | 100% | Prevent overfitting |

### **Phase 3 (Learning)**
| Metric | Target | Purpose |
|--------|--------|---------|
| Post-trade analysis coverage | 100% | Complete learning |
| AI accuracy improvement | > +10% | Learning effectiveness |
| Risk scoring accuracy | > 75% | Better risk assessment |
| Profitable patterns identified | ≥ 5 | Pattern exploitation |
| Repeated mistakes | 0 | Mistake prevention |

### **Phase 4 (Autonomy)**
| Metric | Target | Purpose |
|--------|--------|---------|
| Autonomy level | 80% | Reduced human workload |
| Auto-trade win rate | > 60% | Autonomous reliability |
| Human approval time | < 2 hrs/week | Efficiency |
| Consecutive profitable months | 3 | Consistency |
| Sharpe ratio | > 1.5 | Risk-adjusted returns |

---

## 💰 **Budget Breakdown**

### **Monthly Costs**

| Phase | LLM API | Infrastructure | Total | Notes |
|-------|---------|----------------|-------|-------|
| **Phase 1** | ₹1,200 | ₹300 | ₹1,500 | Low LLM usage, setup costs |
| **Phase 2** | ₹3,000 | ₹500 | ₹3,500 | Peak LLM (strategy generation) |
| **Phase 3** | ₹2,000 | ₹500 | ₹2,500 | Medium LLM (learning analysis) |
| **Phase 4** | ₹1,500 | ₹500 | ₹2,000 | Lower LLM (heuristics dominate) |

**Average**: ~₹2,400/month

### **One-Time Costs**
- Development time: Self (no cost)
- SEBI compliance (if needed): ₹50,000 (deferred, find workarounds)
- Static IP (if needed): ₹500/month (get when moving to live trading)

### **Cost Optimization Strategies**
1. Use Claude Haiku for simple tasks (10x cheaper than Sonnet)
2. Cache repetitive prompts (50% cost reduction)
3. Convert validated strategies to heuristics (300x faster, ₹0 LLM cost)
4. Batch analysis tasks (fewer API calls)
5. Use synthetic data for backtests (no API costs)

---

## 🚦 **Go/No-Go Decision Points**

### **Go Live Decision (After Month 12)**

**Must Have ALL of These**:
- ✅ 3+ consecutive months paper trading profitable (Sharpe > 1.5)
- ✅ AI autonomy stable at 70%+ for 2 months
- ✅ No critical bugs in 30 days
- ✅ SEBI compliance ready (audit trail complete)
- ✅ Emergency stop procedures tested
- ✅ Comfortable risking capital (start with ₹50k, not ₹1L)

**Start Live Trading Strategy**:
1. Begin with ₹50,000 capital (50% of target)
2. Keep autonomy at 70% (not 100%)
3. Daily review for first month
4. Scale to ₹1,00,000 only after 1 month success
5. Monitor for 1 week before increasing autonomy

### **Phase Gate Checklist**

**After Phase 1** (Month 2):
- [ ] Audit trail database functional
- [ ] Market Intelligence Agent operational
- [ ] LLM costs within budget
- [ ] Approval dashboard working
- [ ] All trades logged correctly

**After Phase 2** (Month 5):
- [ ] At least 1 new strategy created and validated
- [ ] Heuristic converter working (< 10ms execution)
- [ ] Parameter tuner improves performance
- [ ] Walk-forward tests passing

**After Phase 3** (Month 8):
- [ ] Learning system operational
- [ ] AI accuracy improved by 10%+
- [ ] Pattern recognition working
- [ ] Mistake prevention active

**After Phase 4** (Month 12):
- [ ] Autonomy at 80%
- [ ] 3 months profitable
- [ ] All circuit breakers tested
- [ ] Ready for live trading decision

---

## 🔄 **Integration with Existing System**

### **What Stays the Same**
- ✅ Data fetcher (already optimized in Thread 1)
- ✅ Risk management limits (position sizing, exposure)
- ✅ Paper trading engine
- ✅ Streamlit dashboard (will be enhanced)
- ✅ Telegram alerts

### **What Changes**
- **Decision Flow**: Rule-based → AI-augmented
- **Strategy Creation**: Manual → AI-invented + AI-optimized
- **Learning**: None → Continuous from every trade
- **Autonomy**: 0% → 80%
- **Explainability**: None → Natural language reasoning
- **Audit Trail**: Basic logging → SEBI-compliant database

### **Migration Strategy**
1. **Phase 1**: Add AI agents alongside existing system (no changes to execution)
2. **Phase 2**: AI suggests strategies, but don't deploy yet (validation only)
3. **Phase 3**: Deploy 1-2 AI strategies, monitor performance
4. **Phase 4**: Gradually increase autonomy as confidence grows

**Rollback Plan**: At any point, can revert to 100% human approval or pause AI agents entirely.

---

## 📚 **Documentation Structure**

```
docs/
├── MASTER_ROADMAP.md              # This file (executive summary)
├── PHASE_1_FOUNDATION.md          # Month 1-2 implementation details
├── PHASE_2_STRATEGY_EVOLUTION.md  # Month 3-5 implementation details
├── PHASE_3_LEARNING_AUTONOMY.md   # Month 6-12 implementation details
└── AI_AGENT_SPECIFICATIONS.md     # Detailed agent specs (to be created)

Root level:
├── THREAD1_IMPROVEMENTS.md        # Critical fixes & paper trading (complete)
└── README.md                       # Project overview
```

---

## 🎯 **Immediate Next Steps**

### **Week 1-2: Setup & Planning**
1. Review and approve this roadmap
2. Choose LLM provider (Claude Sonnet recommended)
3. Get API keys (Anthropic or OpenAI)
4. Choose database (PostgreSQL or SQLite)
5. Setup development environment

### **Week 3-4: Begin Phase 1**
1. Create audit trail database schema
2. Build LLM client wrapper
3. Test LLM integration with sample prompts
4. Start Market Intelligence Agent

### **Month 2: Complete Phase 1**
1. Finish all Phase 1 components
2. Test approval dashboard workflow
3. Verify audit trail completeness
4. Validate LLM costs within budget

### **Month 3-5: Phase 2 Execution**
1. Build Strategy Inventor Agent
2. Create Strategy Evaluator
3. Develop Heuristic Converter
4. Deploy Parameter Tuner

---

## 🎓 **Learning Resources**

### **SEBI Compliance**
- SEBI Algorithmic Trading Framework (2026)
- Audit trail requirements
- Algo ID registration process
- Research Analyst registration (if needed)

### **LLM Integration**
- Anthropic Claude API documentation
- OpenAI GPT-4 API documentation
- Prompt engineering best practices
- Cost optimization strategies

### **Trading Strategy Development**
- Quantitative trading fundamentals
- Backtesting best practices (walk-forward analysis)
- Risk management frameworks
- Performance metrics (Sharpe, Sortino, max drawdown)

### **AI/ML for Trading**
- Reinforcement learning for trading
- Sentiment analysis techniques
- Pattern recognition algorithms
- Bayesian optimization

---

## ⚠️ **Risk Disclosures**

### **Technical Risks**
1. **LLM Hallucinations**: AI may generate plausible-sounding but incorrect strategies
   - **Mitigation**: Rigorous backtesting, human validation, confidence thresholds

2. **Data Quality**: Garbage in, garbage out
   - **Mitigation**: Multiple data sources (yfinance primary, Zerodha backup), data validation

3. **Overfitting**: Strategies may work in backtests but fail in live trading
   - **Mitigation**: Walk-forward analysis, out-of-sample testing, paper trading

4. **System Failures**: Bugs, API outages, connectivity issues
   - **Mitigation**: Error handling, fallbacks, circuit breakers, manual override

### **Market Risks**
1. **Black Swan Events**: Unexpected market crashes
   - **Mitigation**: Position limits, stop-losses, VIX-based trading halts

2. **Regime Changes**: Strategies may stop working when market changes
   - **Mitigation**: Continuous learning, regime detection, strategy rotation

3. **Slippage**: Paper trading results may not match live execution
   - **Mitigation**: Conservative estimates, start small (₹50k), scale gradually

### **Regulatory Risks**
1. **SEBI Compliance**: Algo trading regulations evolving
   - **Mitigation**: Complete audit trails, stay updated on regulations, legal review

2. **Tax Implications**: Intraday trading has different tax treatment
   - **Mitigation**: Consult tax advisor, maintain records

---

## ✅ **Success Definition**

This project is **successful** if after 12 months:

1. ✅ **Profitability**: 3+ consecutive months profitable (Sharpe > 1.5)
2. ✅ **Autonomy**: AI handles 80% of decisions reliably
3. ✅ **Learning**: System demonstrably improves over time
4. ✅ **Reliability**: <1% system downtime, zero critical bugs
5. ✅ **Compliance**: SEBI audit-ready, complete trails
6. ✅ **Efficiency**: Human approval time < 2 hours/week
7. ✅ **Capital Preservation**: No single loss > -5%, max drawdown < -15%

**Ultimate Goal**: A trading system that combines the best of human judgment (strategy direction, risk appetite) with AI capabilities (pattern recognition, continuous learning, tireless execution).

---

## 📞 **Support & Maintenance**

### **Ongoing Tasks**
- **Daily**: Monitor dashboard, review alerts, approve high-risk trades
- **Weekly**: Review AI performance, check for repeated mistakes
- **Monthly**: Evaluate autonomy level, review strategy performance, budget check
- **Quarterly**: SEBI compliance review, tax planning, system audit

### **Escalation**
- **Level 1**: Circuit breakers (auto-pause)
- **Level 2**: Manual pause (human decision)
- **Level 3**: Rollback to pure human control
- **Level 4**: Emergency liquidation (if necessary)

---

## 🚀 **Vision: Beyond 12 Months**

Once the system is stable and profitable:

1. **Scale Capital**: ₹50k → ₹1L → ₹5L (based on consistent performance)
2. **Add Asset Classes**: Equities → Futures → Options
3. **Multi-Timeframe**: Intraday → Swing trading → Long-term
4. **Advanced AI**: Reinforcement learning, multi-agent collaboration
5. **Community**: Share learnings (without revealing exact strategies)

**Long-Term Vision**: A fully autonomous, continuously learning trading system that adapts to any market condition and requires minimal human oversight.

---

**Next Document**: [AI_AGENT_SPECIFICATIONS.md](./AI_AGENT_SPECIFICATIONS.md) - Detailed technical specs for each agent

**Complete Phase Details**:
- [Phase 1 Details](./PHASE_1_FOUNDATION.md)
- [Phase 2 Details](./PHASE_2_STRATEGY_EVOLUTION.md)
- [Phase 3-4 Details](./PHASE_3_LEARNING_AUTONOMY.md)
