# Gas Town Architecture for Trading Bot

## Overview

This document describes how to leverage Gas Town Hall to accelerate development of the quantitative trading system, splitting work between **Claude** (strategic/infrastructure roles) and **Gemini** (parallel worker roles) to minimize token usage while maximizing throughput.

---

## 1. Architecture Layout

### Town Structure

```
~/gt/                                     # Gas Town HQ
├── CLAUDE.md                             # Mayor context (Claude)
├── mayor/                                # Mayor agent home
│   ├── town.json                         # Town config
│   └── .claude/settings.json
├── deacon/                               # Deacon agent home (Claude)
│   ├── .claude/settings.json
│   └── dogs/
│       └── boot/                         # Boot watchdog (Claude Haiku)
├── .beads/                               # Town-level beads (hq-* prefix)
│   ├── routes.jsonl                      # Beads routing
│   └── formulas/                         # Workflow templates
│       ├── mol-scraper-build.toml        # Scraper development formula
│       ├── mol-strategy-backtest.toml    # Strategy backtesting formula
│       ├── mol-feature-standard.toml     # General feature formula
│       └── mol-bug-fix.toml             # Bug fix formula
├── settings/
│   ├── config.json                       # Town defaults
│   ├── agents.json                       # Agent runtime configs
│   └── escalation.json                   # Escalation routing
└── tradingbot/                           # Rig: the trading bot project
    ├── config.json                       # Rig identity (prefix: tb)
    ├── .beads/                           # Rig-level issue tracking
    ├── .repo.git/                        # Bare repo (shared by worktrees)
    ├── mayor/rig/                        # Mayor's clone (canonical beads)
    ├── witness/                           # Witness (Claude Haiku)
    │   └── .claude/settings.json
    ├── refinery/                          # Refinery (Claude Haiku)
    │   ├── .claude/settings.json
    │   └── rig/                          # Worktree on main
    ├── crew/                              # Persistent human workspaces
    │   ├── .claude/settings.json         # Shared crew settings
    │   └── architect/rig/                # Long-running architecture work
    └── polecats/                          # Ephemeral workers (Gemini)
        ├── .claude/settings.json         # Shared polecat settings (overridden for gemini)
        ├── Toast/rig/                    # Worker 1
        ├── Shadow/rig/                   # Worker 2
        ├── Copper/rig/                   # Worker 3
        └── Ash/rig/                      # Worker 4
```

### Role-to-Model Mapping

| Role | Model | Rationale |
|------|-------|-----------|
| **Mayor** | Claude Opus/Sonnet | Strategic coordination, cross-rig decisions, escalation handling. Low volume, high reasoning. |
| **Deacon** | Claude Haiku | Background supervision. Patrol cycles are formulaic; doesn't need deep reasoning. |
| **Boot** | Claude Haiku | Single triage decision per daemon tick. Minimal tokens. |
| **Witness** | Claude Haiku | Monitors polecat health, nudges stuck workers. Pattern-matching, not creative. |
| **Refinery** | Claude Haiku | Merge queue processing, rebase conflicts. Mechanical. |
| **Crew** | Claude Sonnet | Persistent human-directed workspace for architecture/exploration. |
| **Polecats** | **Gemini** | Ephemeral parallel workers. High volume, well-defined tasks. Token-efficient. |

**Token savings**: The high-volume role (Polecats) runs on Gemini, which handles the bulk of actual code-writing work. Claude handles the low-volume strategic/infrastructure roles (Mayor, Deacon, Witness, Refinery) where its superior reasoning matters but token consumption is naturally low due to infrequent, structured patrol cycles.

---

## 2. Setup Guide

### Prerequisites

```bash
# Install Gas Town + Beads
go install github.com/steveyegge/gastown/cmd/gt@latest
go install github.com/steveyegge/beads/cmd/bd@latest

# Verify
gt version
bd version

# Ensure tmux is installed (for full stack mode)
brew install tmux   # macOS
```

### Step 1: Create the Town

```bash
# Create Gas Town workspace
gt install ~/gt --name trading-hq --shell

# This creates:
# ~/gt/
# ├── CLAUDE.md
# ├── mayor/
# └── .beads/
```

### Step 2: Add the Trading Bot as a Rig

```bash
# Add the existing trading bot repo
gt rig add tradingbot https://github.com/aadhar99/gtTradingBot.git

# This clones the repo and sets up:
# ~/gt/tradingbot/
# ├── .beads/
# ├── mayor/rig/
# ├── refinery/rig/
# ├── witness/
# └── polecats/
```

### Step 3: Configure Agents (Claude + Gemini Split)

```bash
# List built-in agents
gt config agent list

# Set up agent aliases
# Claude for infrastructure (mayor, deacon, witness, refinery, crew)
gt config agent set claude-infra "claude --model haiku --dangerously-skip-permissions"
gt config agent set claude-mayor "claude --model sonnet"
gt config agent set claude-crew "claude --model sonnet"

# Gemini for polecats (the parallel workers)
gt config agent set gemini-worker "gemini"

# Set town default agent (infrastructure roles)
gt config default-agent claude-infra
```

To override the agent for polecats at the rig level, edit `~/gt/tradingbot/settings/config.json`:

```json
{
  "type": "rig-settings",
  "version": 1,
  "agent": "gemini-worker",
  "agents": {
    "gemini-worker": {
      "command": "gemini",
      "args": []
    }
  }
}
```

### Step 4: Configure Escalation Routing

Create `~/gt/settings/escalation.json`:

```json
{
  "routes": {
    "critical": ["bead", "mail:mayor"],
    "high": ["bead", "mail:mayor"],
    "medium": ["bead"],
    "low": ["bead"]
  },
  "stale_threshold": "4h",
  "max_reescalations": 2
}
```

### Step 5: Start Gas Town

```bash
# Start infrastructure (Daemon + Deacon + Mayor)
gt start

# Start witness and refinery for the tradingbot rig
gt rig boot tradingbot

# Or start everything at once
gt up
```

### Step 6: Verify Health

```bash
gt doctor              # Run health checks
gt doctor --fix        # Auto-repair common issues
gt status              # Show workspace status
```

---

## 3. Workflow Patterns for the Trading Bot

### Pattern A: Parallel Feature Development (Convoy + Polecats)

Use this when you have multiple independent tasks to complete.

```bash
# 1. Create issues for the work
bd create --title "Add MACD crossover signal to MomentumAgent" --type task
# Returns: tb-abc

bd create --title "Add Fibonacci retracement levels to DataPreprocessor" --type task
# Returns: tb-def

bd create --title "Add portfolio rebalancing logic" --type task
# Returns: tb-ghi

# 2. Create a convoy tracking all issues
gt convoy create "Technical Indicators v2" tb-abc tb-def tb-ghi --notify human

# 3. Sling work to polecats (each gets its own Gemini worker)
gt sling tb-abc tradingbot --agent gemini-worker
gt sling tb-def tradingbot --agent gemini-worker
gt sling tb-ghi tradingbot --agent gemini-worker

# 4. Monitor progress
gt convoy list                    # Dashboard view
gt convoy status hq-cv-xyz       # Detailed progress
gt feed                           # Real-time activity feed

# 5. Convoy auto-closes when all issues complete
# Refinery merges each polecat's work to main
```

### Pattern B: Sequential Workflow with Molecules

Use this for multi-step work that needs a specific order.

```bash
# Create a formula for strategy development
# File: ~/gt/.beads/formulas/mol-strategy-dev.formula.toml

# Then cook and pour:
bd cook mol-strategy-dev
bd mol pour mol-strategy-dev --var strategy=momentum_v2

# Sling the molecule to a worker
gt sling mol-strategy-dev tradingbot
```

### Pattern C: Architecture Work (Crew on Claude)

Use this for exploratory, long-running work that needs deep reasoning.

```bash
# Start a crew workspace on Claude
gt start crew tradingbot/architect --agent claude-crew

# Attach to the session
gt crew at architect

# Work happens interactively with Claude Sonnet
# The crew workspace persists across sessions
```

### Pattern D: Bug Fix (Quick Sling)

```bash
# Create issue and sling immediately
bd create --title "Fix NSE bulk deals parser for new API format" --type bug
# Returns: tb-xyz

# Auto-creates convoy, spawns polecat, starts work
gt sling tb-xyz tradingbot --agent gemini-worker
```

---

## 4. Trading Bot Module Mapping to Gas Town Work Units

### Work Categories

| Category | Beads Type | Typical Worker | Example |
|----------|-----------|----------------|---------|
| **New Scraper** | `task` | Polecat (Gemini) | Add NSE options chain scraper |
| **Strategy Development** | `task` + molecule | Crew (Claude) | Design new mean-reversion strategy |
| **Bug Fix** | `bug` | Polecat (Gemini) | Fix date parsing in corporate actions |
| **Backtest** | `task` | Polecat (Gemini) | Run 6-month backtest on momentum strategy |
| **Infrastructure** | `task` | Polecat (Gemini) | Add Redis caching layer |
| **Architecture Decision** | `task` | Crew (Claude) | Evaluate async vs sync execution engine |
| **Data Pipeline** | `task` + molecule | Polecat (Gemini) | Build daily data collection pipeline |
| **Risk Model** | `task` | Crew (Claude) | Design new circuit breaker triggers |
| **Dashboard** | `task` | Polecat (Gemini) | Add P&L chart to Streamlit dashboard |

### Project Structure to Rig Mapping

The entire trading bot repo is a single rig (`tradingbot`), with beads prefix `tb`.

```
tradingbot/                 # Rig root
├── agents/                 # Trading agents (MomentumAgent, ReversionAgent)
├── ai/agents/scrapers/     # NSE data scrapers
├── config/                 # System configuration
├── data/                   # Data fetching and preprocessing
├── execution/              # Portfolio, kill switch
├── risk/                   # Risk management, circuit breakers
├── scripts/                # Operational scripts
├── dashboard/              # Streamlit dashboard
├── validation/             # Complexity enforcer
└── tests/                  # Test suite
```

---

## 5. Migration Steps

### Phase 1: Setup (Day 1)

1. **Install Gas Town and Beads**
   ```bash
   go install github.com/steveyegge/gastown/cmd/gt@latest
   go install github.com/steveyegge/beads/cmd/bd@latest
   ```

2. **Create the Town**
   ```bash
   gt install ~/gt --name trading-hq --shell --git
   ```

3. **Register the trading bot repo as a rig**
   ```bash
   gt rig add tradingbot https://github.com/YOUR_USER/Claude-code.git
   ```

4. **Configure agent aliases** (Claude for infra, Gemini for polecats)
   ```bash
   gt config agent set claude-mayor "claude --model sonnet"
   gt config agent set claude-infra "claude --model haiku --dangerously-skip-permissions"
   gt config agent set gemini-worker "gemini"
   gt config default-agent claude-infra
   ```

5. **Verify installation**
   ```bash
   gt doctor --fix
   gt status
   ```

### Phase 2: First Convoy (Day 1-2)

1. **Start Gas Town services**
   ```bash
   gt up
   ```

2. **Create a test issue and sling it**
   ```bash
   bd create --title "Add unit tests for BulkDealsScraper.parse()" --type task
   gt sling tb-xxx tradingbot --agent gemini-worker
   ```

3. **Monitor the polecat working**
   ```bash
   gt peek tradingbot/Toast       # See what the Gemini worker is doing
   gt convoy list                  # Check convoy status
   gt feed                         # Real-time activity
   ```

4. **Verify the Refinery merges the work**
   ```bash
   gt mq list tradingbot           # Check merge queue
   ```

### Phase 3: Create Workflow Formulas (Day 2-3)

Create reusable formulas for common trading bot tasks:

**`~/gt/.beads/formulas/mol-scraper-build.formula.toml`**:
```toml
formula = "scraper-build"
type = "workflow"
version = 1
description = "Build a new NSE data scraper"

[vars.endpoint]
description = "NSE API endpoint URL"
required = true

[vars.data_name]
description = "Name of the data being scraped (e.g., options_chain)"
required = true

[[steps]]
id = "design"
title = "Design scraper for {{data_name}}"
description = "Study the NSE API endpoint, document fields, plan dataclass and parser"

[[steps]]
id = "implement"
title = "Implement {{data_name}} scraper"
description = "Create scraper class extending BaseScraper with fetch/parse/store methods"
needs = ["design"]

[[steps]]
id = "test"
title = "Test {{data_name}} scraper"
description = "Run scraper against live NSE endpoint, verify parsing and storage"
needs = ["implement"]

[[steps]]
id = "integrate"
title = "Integrate into daily_collector.py"
description = "Add scraper to daily collection pipeline and backfill script"
needs = ["test"]
```

**`~/gt/.beads/formulas/mol-strategy-backtest.formula.toml`**:
```toml
formula = "strategy-backtest"
type = "workflow"
version = 1
description = "Develop and backtest a trading strategy"

[vars.strategy_name]
description = "Strategy name"
required = true

[[steps]]
id = "research"
title = "Research {{strategy_name}} approach"
description = "Review academic papers, define entry/exit signals, set parameters"

[[steps]]
id = "implement"
title = "Implement {{strategy_name}} agent"
description = "Create agent class extending BaseAgent with generate_signals()"
needs = ["research"]

[[steps]]
id = "backtest"
title = "Backtest {{strategy_name}}"
description = "Run against 6 months of historical data, evaluate Sharpe/drawdown/win rate"
needs = ["implement"]

[[steps]]
id = "risk-validate"
title = "Validate risk parameters"
description = "Ensure strategy passes RiskManager, TransactionCostModel, and ComplexityEnforcer"
needs = ["backtest"]
```

Cook the formulas:
```bash
bd cook scraper-build
bd cook strategy-backtest
```

### Phase 4: Parallel Development Sprint (Day 3+)

With formulas in place, dispatch work at scale:

```bash
# Create issues for next development phase
bd create --title "Options chain scraper" --type task           # tb-001
bd create --title "Insider trading scraper" --type task          # tb-002
bd create --title "VWAP strategy agent" --type task             # tb-003
bd create --title "Streamlit P&L dashboard" --type task         # tb-004

# Create convoy
gt convoy create "Week 3 Sprint" tb-001 tb-002 tb-003 tb-004 --notify human

# Sling with formulas where applicable
gt sling scraper-build --on tb-001 tradingbot --var endpoint="/api/option-chain" --var data_name="options_chain"
gt sling scraper-build --on tb-002 tradingbot --var endpoint="/api/insider-trading" --var data_name="insider_trading"
gt sling strategy-backtest --on tb-003 tradingbot --var strategy_name="VWAP"
gt sling tb-004 tradingbot   # Simple task, no formula needed

# All 4 polecats work in parallel on Gemini
# Refinery merges completed work to main
# Convoy auto-closes when all 4 land
```

### Phase 5: Crew Workspace for Architecture (Ongoing)

Keep a persistent Claude crew workspace for strategic decisions:

```bash
# Create and start crew workspace
gt start crew tradingbot/architect --agent claude-crew

# Use for:
# - Reviewing polecat work quality
# - Making architecture decisions
# - Designing new risk models
# - Evaluating strategy performance
# - Handling escalations from polecats
```

---

## 6. Operational Commands Cheat Sheet

### Daily Operations

```bash
gt up                                    # Start all services
gt convoy list                           # See active work
gt feed                                  # Real-time activity stream
gt status                                # Overall health
gt down                                  # Pause everything (saves tokens)
```

### Work Management

```bash
bd create --title "..." --type task      # Create work item
gt sling tb-xxx tradingbot               # Dispatch to polecat
gt convoy create "Name" tb-a tb-b        # Track batch
gt convoy status hq-cv-xxx              # Check progress
gt ready                                 # See unblocked work
```

### Monitoring

```bash
gt peek tradingbot/Toast                 # See polecat output
gt nudge tradingbot/Toast "status?"      # Ask for status
gt mq list tradingbot                    # Merge queue
gt audit --actor=tradingbot/polecats/*   # Work history
gt doctor                                # Health check
```

### Emergency

```bash
gt down --all                            # Stop everything
gt shutdown                              # Stop + cleanup polecats
gt polecat nuke Toast                    # Kill specific polecat
gt escalate "..." --severity critical    # Escalate to Mayor
```

---

## 7. Token Optimization Strategy

| Strategy | Impact |
|----------|--------|
| **Polecats on Gemini** | ~70% of code-writing tokens shifted off Claude |
| **Infrastructure on Haiku** | Witness/Refinery/Deacon use cheapest Claude tier |
| **Crew on Sonnet (not Opus)** | Good reasoning at 5x less cost than Opus |
| **Mayor on Sonnet** | Infrequent usage; Sonnet sufficient for coordination |
| **Formulas** | Structured work reduces per-session context building |
| **gt down when idle** | Zero token consumption when not actively developing |
| **Wisps for patrols** | Ephemeral molecules avoid accumulating bead history |

### Estimated Token Distribution

```
Before Gas Town:  Claude handles 100% of work (all in one session)

After Gas Town:
  Gemini (Polecats):     ~65%  ← bulk code writing, tests, scrapers
  Claude Haiku (Infra):  ~20%  ← patrol cycles, merge queue, monitoring
  Claude Sonnet (Crew):  ~10%  ← architecture, design, reviews
  Claude Sonnet (Mayor): ~5%   ← coordination, escalations
```

---

## 8. Key Concepts to Remember

1. **Propulsion Principle**: When work is on a hook, the agent executes immediately. No waiting for confirmation.

2. **Polecats are ephemeral**: Spawned for one task, nuked when done. No idle state.

3. **Convoys track batched work**: Create a convoy, sling issues to polecats, monitor on the dashboard.

4. **Refinery serializes merges**: Polecats push branches; Refinery rebases and merges to main.

5. **Molecules define workflows**: Multi-step processes (design -> implement -> test -> integrate) are tracked as molecules.

6. **Escalation flows upward**: Polecat -> Witness -> Deacon -> Mayor -> Human.

7. **Attribution is universal**: Every commit, every bead, every event is attributed to the agent that performed it.
