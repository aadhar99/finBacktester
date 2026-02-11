# 🏗️ SETUP & ARCHITECTURE

## 📋 **Overview**

This document covers the production-ready setup for the Hybrid AI Trading System with:
- **Plug-and-play LLM providers** (Claude, Gemini, finance-specific models)
- **Scalable database architecture** (SQL for active data, time-series for audit trails)
- **Cloud deployment** recommendations (availability, reliability, ease of use)
- **Cost optimization** strategies

---

## 🤖 **LLM Provider Architecture**

### **Unified LLM Interface Design**

```python
# utils/llm/base.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

class LLMProvider(Enum):
    """Supported LLM providers."""
    CLAUDE = "claude"
    GEMINI = "gemini"
    GPT4 = "gpt4"
    FИНГPT = "fingpt"
    CUSTOM = "custom"

@dataclass
class LLMResponse:
    """Standardized response from any LLM."""
    text: str
    model: str
    tokens_used: int
    cost: float  # In INR
    latency_ms: int
    provider: LLMProvider

@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: LLMProvider
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30  # seconds
    rate_limit: Optional[int] = None  # calls per minute

class BaseLLMClient(ABC):
    """Base class for all LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.total_cost = 0.0
        self.total_calls = 0

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion from prompt."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @abstractmethod
    def get_cost_per_1k_tokens(self) -> Dict[str, float]:
        """Get pricing (input and output per 1k tokens)."""
        pass

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in INR."""
        pricing = self.get_cost_per_1k_tokens()
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        return input_cost + output_cost
```

---

### **1. Claude Sonnet (Primary - Recommended)**

**Why Claude?**
- ✅ Best reasoning and analysis capabilities
- ✅ 200k context window (great for market data)
- ✅ Strong financial/trading understanding
- ✅ Excellent at structured JSON outputs
- ✅ Good cost/performance ratio

**Pricing** (Feb 2026):
- Input: ~₹0.30 per 1K tokens
- Output: ~₹1.50 per 1K tokens
- Average decision: ~2K tokens = ₹0.05-0.10

```python
# utils/llm/claude.py
import anthropic
import asyncio
from .base import BaseLLMClient, LLMResponse, LLMProvider

class ClaudeClient(BaseLLMClient):
    """Anthropic Claude integration."""

    def __init__(self, config):
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)

        # Model selection
        self.model_map = {
            'sonnet': 'claude-sonnet-4-5-20250929',  # Latest
            'haiku': 'claude-haiku-4-20250311',      # Fast & cheap
            'opus': 'claude-opus-4-20250514'         # Most capable
        }
        self.model = self.model_map.get(config.model, config.model)

    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion."""
        import time
        start_time = time.time()

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self.calculate_cost(input_tokens, output_tokens)

            self.total_cost += cost
            self.total_calls += 1

            return LLMResponse(
                text=response.content[0].text,
                model=self.model,
                tokens_used=input_tokens + output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                provider=LLMProvider.CLAUDE
            )

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Approximate token count (Claude uses similar to GPT-4)."""
        return len(text) // 4  # Rough estimate

    def get_cost_per_1k_tokens(self) -> Dict[str, float]:
        """Pricing in INR (1 USD = ~83 INR as of Feb 2026)."""
        if 'haiku' in self.model:
            return {'input': 0.025, 'output': 0.125}  # 10x cheaper
        elif 'opus' in self.model:
            return {'input': 1.50, 'output': 7.50}    # 5x more expensive
        else:  # Sonnet (default)
            return {'input': 0.30, 'output': 1.50}
```

---

### **2. Google Gemini (Alternative - Cost-Effective)**

**Why Gemini?**
- ✅ **FREE tier**: 15 calls/min, 1M tokens/day (perfect for paper trading!)
- ✅ Native GCP integration (if deploying on GCP)
- ✅ Good multimodal capabilities (future: chart analysis)
- ✅ 1M token context window (2M for Pro)
- ✅ Competitive performance

**Pricing** (Feb 2026):
- **Free tier**: Up to 1M tokens/day (covers most paper trading)
- **Paid**: Input: ₹0.05/1K, Output: ₹0.15/1K (5x cheaper than Claude!)

**Trade-off**: Slightly less sophisticated reasoning than Claude, but excellent value

```python
# utils/llm/gemini.py
import google.generativeai as genai
from .base import BaseLLMClient, LLMResponse, LLMProvider

class GeminiClient(BaseLLMClient):
    """Google Gemini integration."""

    def __init__(self, config):
        super().__init__(config)
        genai.configure(api_key=config.api_key)

        # Model selection
        self.model_map = {
            'flash': 'gemini-2.0-flash-exp',      # Fastest, free tier
            'pro': 'gemini-2.0-pro',              # Most capable
            'flash-thinking': 'gemini-2.0-flash-thinking-exp'  # Reasoning
        }
        self.model_name = self.model_map.get(config.model, config.model)
        self.model = genai.GenerativeModel(self.model_name)

    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion."""
        import time
        start_time = time.time()

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    'temperature': kwargs.get('temperature', self.config.temperature),
                    'max_output_tokens': kwargs.get('max_tokens', self.config.max_tokens),
                }
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Calculate tokens and cost
            input_tokens = self.count_tokens(prompt)
            output_tokens = self.count_tokens(response.text)
            cost = self.calculate_cost(input_tokens, output_tokens)

            self.total_cost += cost
            self.total_calls += 1

            return LLMResponse(
                text=response.text,
                model=self.model_name,
                tokens_used=input_tokens + output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                provider=LLMProvider.GEMINI
            )

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's tokenizer."""
        return self.model.count_tokens(text).total_tokens

    def get_cost_per_1k_tokens(self) -> Dict[str, float]:
        """Pricing in INR."""
        if 'flash' in self.model_name:
            # Free tier up to 1M tokens/day, then:
            return {'input': 0.0, 'output': 0.0}  # Assume free tier
        else:  # Pro
            return {'input': 0.05, 'output': 0.15}
```

---

### **3. GPT-4 (Fallback Option)**

**Why GPT-4?**
- ✅ Widely used and tested
- ✅ Good JSON mode support
- ✅ Strong reasoning
- ⚠️ More expensive than Gemini
- ⚠️ Shorter context (128k)

**Pricing** (Feb 2026):
- Input: ₹0.50/1K tokens
- Output: ₹2.50/1K tokens

```python
# utils/llm/openai.py
from openai import AsyncOpenAI
from .base import BaseLLMClient, LLMResponse, LLMProvider

class OpenAIClient(BaseLLMClient):
    """OpenAI GPT-4 integration."""

    def __init__(self, config):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=config.api_key)

        self.model_map = {
            'gpt4': 'gpt-4-turbo-2024-04-09',
            'gpt4o': 'gpt-4o',
            'gpt4o-mini': 'gpt-4o-mini'  # Cheaper option
        }
        self.model = self.model_map.get(config.model, config.model)

    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion."""
        import time
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                response_format={"type": "json_object"} if kwargs.get('json_mode') else None
            )

            latency_ms = int((time.time() - start_time) * 1000)

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self.calculate_cost(input_tokens, output_tokens)

            self.total_cost += cost
            self.total_calls += 1

            return LLMResponse(
                text=response.choices[0].message.content,
                model=self.model,
                tokens_used=input_tokens + output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                provider=LLMProvider.GPT4
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def get_cost_per_1k_tokens(self) -> Dict[str, float]:
        """Pricing in INR."""
        if 'mini' in self.model:
            return {'input': 0.015, 'output': 0.075}  # Much cheaper
        else:
            return {'input': 0.50, 'output': 2.50}
```

---

### **4. FinGPT (Finance-Specific Open Source)**

**Why FinGPT?**
- ✅ Trained on financial data (news, earnings, SEC filings)
- ✅ Open source (can self-host)
- ✅ Zero API costs (if self-hosted)
- ⚠️ Smaller model (not as capable as Claude/GPT-4)
- ⚠️ Requires GPU for inference

**Recommendation**: Use for specific tasks like sentiment analysis, not main decision-making

**Deployment Options**:
1. **Hugging Face Inference API** (₹200-500/month for small usage)
2. **Self-hosted on GCP/AWS** (₹2,000-3,000/month for GPU instance)
3. **RunPod/Vast.ai** (₹500-1,000/month for spot GPU)

```python
# utils/llm/fingpt.py
import requests
from .base import BaseLLMClient, LLMResponse, LLMProvider

class FinGPTClient(BaseLLMClient):
    """FinGPT integration (Hugging Face hosted)."""

    def __init__(self, config):
        super().__init__(config)
        self.api_url = "https://api-inference.huggingface.co/models/FinGPT/fingpt-forecaster"
        self.headers = {"Authorization": f"Bearer {config.api_key}"}

    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion."""
        import time
        import aiohttp

        start_time = time.time()

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "temperature": kwargs.get('temperature', self.config.temperature)
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                result = await response.json()

        latency_ms = int((time.time() - start_time) * 1000)

        # HuggingFace inference API pricing: ~₹0.02 per call
        cost = 0.02

        return LLMResponse(
            text=result[0]['generated_text'],
            model='fingpt-forecaster',
            tokens_used=0,  # Not tracked by HF API
            cost=cost,
            latency_ms=latency_ms,
            provider=LLMProvider.FИНГPT
        )

    def get_cost_per_1k_tokens(self) -> Dict[str, float]:
        """Fixed cost per call."""
        return {'input': 0.0, 'output': 0.0}  # Flat rate
```

---

### **Unified LLM Manager with Auto-Fallback**

```python
# utils/llm/manager.py
from typing import List, Optional
from .base import LLMProvider, LLMConfig, LLMResponse
from .claude import ClaudeClient
from .gemini import GeminiClient
from .openai import OpenAIClient
from .fingpt import FinGPTClient

class LLMManager:
    """
    Unified LLM manager with automatic fallback.

    Features:
    - Plug-and-play provider switching
    - Automatic fallback on errors
    - Cost tracking across all providers
    - Smart routing based on task type
    """

    def __init__(self, configs: List[LLMConfig]):
        """
        Initialize with multiple providers (in priority order).

        Example:
            configs = [
                LLMConfig(provider=LLMProvider.CLAUDE, model='sonnet', api_key='...'),
                LLMConfig(provider=LLMProvider.GEMINI, model='flash', api_key='...'),
                LLMConfig(provider=LLMProvider.GPT4, model='gpt4o-mini', api_key='...')
            ]
        """
        self.providers = []

        for config in configs:
            if config.provider == LLMProvider.CLAUDE:
                self.providers.append(ClaudeClient(config))
            elif config.provider == LLMProvider.GEMINI:
                self.providers.append(GeminiClient(config))
            elif config.provider == LLMProvider.GPT4:
                self.providers.append(OpenAIClient(config))
            elif config.provider == LLMProvider.FИНГPT:
                self.providers.append(FinGPTClient(config))

        if not self.providers:
            raise ValueError("At least one LLM provider must be configured")

        self.primary = self.providers[0]
        self.fallbacks = self.providers[1:] if len(self.providers) > 1 else []

        logger.info(f"✅ LLM Manager initialized with {len(self.providers)} provider(s)")
        logger.info(f"   Primary: {self.primary.config.provider.value}")
        if self.fallbacks:
            logger.info(f"   Fallbacks: {[p.config.provider.value for p in self.fallbacks]}")

    async def complete(self, prompt: str, task_type: str = 'general', **kwargs) -> LLMResponse:
        """
        Generate completion with automatic fallback.

        Args:
            prompt: Input prompt
            task_type: Type of task (used for smart routing)
                - 'general': Use primary provider
                - 'sentiment': Use FinGPT if available, else primary
                - 'fast': Use cheapest/fastest provider
                - 'critical': Use most capable provider

        Returns:
            LLMResponse with result
        """

        # Smart routing based on task type
        provider = self._select_provider(task_type)

        # Try primary provider
        try:
            response = await provider.complete(prompt, **kwargs)
            logger.info(f"✅ {provider.config.provider.value}: {response.latency_ms}ms, ₹{response.cost:.4f}")
            return response

        except Exception as e:
            logger.warning(f"⚠️ {provider.config.provider.value} failed: {e}")

            # Try fallbacks
            for fallback in self.fallbacks:
                try:
                    logger.info(f"🔄 Trying fallback: {fallback.config.provider.value}")
                    response = await fallback.complete(prompt, **kwargs)
                    logger.info(f"✅ {fallback.config.provider.value}: {response.latency_ms}ms, ₹{response.cost:.4f}")
                    return response

                except Exception as fallback_error:
                    logger.warning(f"⚠️ {fallback.config.provider.value} failed: {fallback_error}")
                    continue

            # All providers failed
            raise Exception(f"All LLM providers failed. Last error: {e}")

    def _select_provider(self, task_type: str):
        """Select optimal provider based on task type."""

        if task_type == 'sentiment' and any(p.config.provider == LLMProvider.FИНГPT for p in self.providers):
            # Use FinGPT for sentiment if available
            return next(p for p in self.providers if p.config.provider == LLMProvider.FИНГPT)

        elif task_type == 'fast' and any(p.config.provider == LLMProvider.GEMINI for p in self.providers):
            # Use Gemini Flash for speed
            return next(p for p in self.providers if p.config.provider == LLMProvider.GEMINI)

        elif task_type == 'critical':
            # Use most capable (Claude Opus or GPT-4)
            opus_provider = next((p for p in self.providers if 'opus' in p.model), None)
            if opus_provider:
                return opus_provider

        # Default: use primary
        return self.primary

    def get_total_cost(self) -> float:
        """Get total cost across all providers."""
        return sum(p.total_cost for p in self.providers)

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            'total_cost': self.get_total_cost(),
            'total_calls': sum(p.total_calls for p in self.providers),
            'by_provider': {
                p.config.provider.value: {
                    'calls': p.total_calls,
                    'cost': p.total_cost
                }
                for p in self.providers
            }
        }
```

---

### **Configuration (Environment Variables)**

```bash
# .env
# LLM Provider Configuration

# Claude (Primary - Recommended)
CLAUDE_API_KEY=sk-ant-xxx
CLAUDE_MODEL=sonnet  # or 'haiku' for cheaper, 'opus' for best

# Gemini (Free tier - Excellent for paper trading)
GEMINI_API_KEY=AIzaSyxxx
GEMINI_MODEL=flash  # or 'pro' for more capability

# OpenAI (Fallback)
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt4o-mini  # or 'gpt4' for full capability

# FinGPT (Optional - for sentiment analysis)
FINGPT_API_KEY=hf_xxx  # Hugging Face API key
```

```python
# config/llm_config.py
import os
from utils.llm.base import LLMConfig, LLMProvider

def get_llm_configs():
    """Load LLM configurations from environment."""
    configs = []

    # Primary: Claude (if configured)
    if os.getenv('CLAUDE_API_KEY'):
        configs.append(LLMConfig(
            provider=LLMProvider.CLAUDE,
            model=os.getenv('CLAUDE_MODEL', 'sonnet'),
            api_key=os.getenv('CLAUDE_API_KEY'),
            temperature=0.7,
            max_tokens=2000
        ))

    # Fallback 1: Gemini (if configured)
    if os.getenv('GEMINI_API_KEY'):
        configs.append(LLMConfig(
            provider=LLMProvider.GEMINI,
            model=os.getenv('GEMINI_MODEL', 'flash'),
            api_key=os.getenv('GEMINI_API_KEY'),
            temperature=0.7,
            max_tokens=2000
        ))

    # Fallback 2: OpenAI (if configured)
    if os.getenv('OPENAI_API_KEY'):
        configs.append(LLMConfig(
            provider=LLMProvider.GPT4,
            model=os.getenv('OPENAI_MODEL', 'gpt4o-mini'),
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0.7,
            max_tokens=2000
        ))

    if not configs:
        raise ValueError("No LLM providers configured! Set at least one API key in .env")

    return configs
```

---

## 🗄️ **DATABASE ARCHITECTURE**

### **Problem**: Audit trails will overwhelm SQL

You're absolutely right! Let's use a **hybrid approach**:

1. **PostgreSQL**: Active trades, portfolio state, summaries
2. **TimescaleDB** (or ClickHouse): High-volume time-series audit trails
3. **S3/GCS**: Archive cold data (>90 days)

---

### **Architecture Overview**

```
┌─────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                      │
└─────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                   ↓
┌───────────────┐  ┌──────────────┐  ┌────────────────┐
│  PostgreSQL   │  │ TimescaleDB  │  │ Cloud Storage  │
│  (Hot Data)   │  │ (Warm Data)  │  │ (Cold Archive) │
└───────────────┘  └──────────────┘  └────────────────┘
│                  │                  │
│ Active trades    │ Audit trails     │ Historical data
│ Portfolio state  │ Decision logs    │ >90 days old
│ User config      │ Performance      │ Parquet files
│ Summaries        │ Time-series      │ Queryable via
│ < 1 GB           │ < 10 GB          │ Athena/BigQuery
└──────────────────┴──────────────────┴────────────────┘
```

---

### **1. PostgreSQL (Hot Data - Active Trading)**

**Schema** (Core tables only):

```sql
-- config/schema/postgres.sql

-- Active Trades (small table, fast queries)
CREATE TABLE active_trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,  -- BUY/SELL
    entry_time TIMESTAMP NOT NULL DEFAULT NOW(),
    entry_price DECIMAL(10,2) NOT NULL,
    quantity INTEGER NOT NULL,
    stop_loss DECIMAL(10,2),
    take_profit DECIMAL(10,2),
    current_pnl DECIMAL(10,2),
    current_pnl_pct DECIMAL(5,2),
    agent VARCHAR(50) NOT NULL,

    -- Quick lookup indexes
    INDEX idx_active_trades_symbol (symbol),
    INDEX idx_active_trades_entry_time (entry_time)
);

-- Portfolio State (single row, frequently updated)
CREATE TABLE portfolio_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    cash DECIMAL(12,2) NOT NULL,
    total_value DECIMAL(12,2) NOT NULL,
    daily_pnl DECIMAL(10,2) NOT NULL DEFAULT 0,
    daily_pnl_pct DECIMAL(5,2) NOT NULL DEFAULT 0,
    total_exposure_pct DECIMAL(5,2) NOT NULL DEFAULT 0,
    num_positions INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Ensure only one row
    CHECK (id = 1)
);

-- Daily Summary (one row per day)
CREATE TABLE daily_summary (
    date DATE PRIMARY KEY,
    starting_capital DECIMAL(12,2) NOT NULL,
    ending_capital DECIMAL(12,2) NOT NULL,
    pnl DECIMAL(10,2) NOT NULL,
    pnl_pct DECIMAL(5,2) NOT NULL,
    num_trades INTEGER NOT NULL,
    num_wins INTEGER NOT NULL,
    num_losses INTEGER NOT NULL,
    win_rate DECIMAL(4,3),
    best_trade_pnl DECIMAL(10,2),
    worst_trade_pnl DECIMAL(10,2),
    sharpe_ratio DECIMAL(5,2),
    max_drawdown_pct DECIMAL(5,2),

    INDEX idx_daily_summary_date (date DESC)
);

-- User Configuration
CREATE TABLE user_config (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    last_updated TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Strategy Performance (aggregate metrics)
CREATE TABLE strategy_performance (
    strategy_name VARCHAR(100) PRIMARY KEY,
    total_trades INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    total_pnl DECIMAL(10,2) NOT NULL DEFAULT 0,
    avg_win_pct DECIMAL(5,2),
    avg_loss_pct DECIMAL(5,2),
    sharpe_ratio DECIMAL(5,2),
    last_trade_time TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    INDEX idx_strategy_performance_sharpe (sharpe_ratio DESC)
);

-- Total rows: ~100-1,000 (very light)
```

**Estimated Size**: < 1 GB (even after 1 year)

---

### **2. TimescaleDB (Warm Data - Audit Trails & Time-Series)**

**Why TimescaleDB?**
- ✅ PostgreSQL extension (familiar SQL interface)
- ✅ Optimized for time-series data (10-100x better compression)
- ✅ Automatic data retention policies
- ✅ Fast aggregations on time ranges
- ✅ Free and open source

**Installation**:
```bash
# Add to existing PostgreSQL
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

**Schema** (Audit trails):

```sql
-- config/schema/timescale.sql

-- Trade Decisions (HYPERTABLE - automatic partitioning by time)
CREATE TABLE trade_decisions (
    id BIGSERIAL,
    decision_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Trade details
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,
    quantity INTEGER,
    entry_price DECIMAL(10,2),
    exit_price DECIMAL(10,2),

    -- AI decision data
    agent VARCHAR(50) NOT NULL,
    ai_reasoning TEXT NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    risk_score INTEGER NOT NULL,

    -- Market context
    market_regime VARCHAR(20),
    vix DECIMAL(5,2),
    sector_sentiment VARCHAR(20),

    -- Outcome (filled after trade closes)
    realized_pnl DECIMAL(10,2),
    realized_pnl_pct DECIMAL(5,2),
    outcome VARCHAR(20),  -- WIN/LOSS/PENDING

    -- Human interaction
    required_human_approval BOOLEAN NOT NULL,
    human_decision VARCHAR(20),  -- APPROVED/REJECTED/MODIFIED
    human_decision_time TIMESTAMPTZ,

    -- SEBI compliance
    algo_id VARCHAR(50),
    client_id VARCHAR(50),

    PRIMARY KEY (timestamp, decision_id)
);

-- Convert to hypertable (automatic partitioning by time)
SELECT create_hypertable('trade_decisions', 'timestamp');

-- Automatic data retention (move to S3 after 90 days)
SELECT add_retention_policy('trade_decisions', INTERVAL '90 days');

-- Continuous aggregates (pre-computed for fast queries)
CREATE MATERIALIZED VIEW trade_decisions_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS hour,
    agent,
    COUNT(*) as num_decisions,
    AVG(confidence) as avg_confidence,
    AVG(risk_score) as avg_risk_score,
    COUNT(*) FILTER (WHERE outcome = 'WIN') as num_wins,
    COUNT(*) FILTER (WHERE outcome = 'LOSS') as num_losses
FROM trade_decisions
GROUP BY hour, agent;

-- Refresh policy (update every hour)
SELECT add_continuous_aggregate_policy('trade_decisions_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

```sql
-- Agent Actions Log (high-volume, append-only)
CREATE TABLE agent_actions (
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_name VARCHAR(100) NOT NULL,
    action_type VARCHAR(50) NOT NULL,  -- ANALYSIS/DECISION/LEARNING
    context JSONB,
    result JSONB,
    latency_ms INTEGER,
    llm_cost DECIMAL(8,4),

    PRIMARY KEY (timestamp, agent_name)
);

SELECT create_hypertable('agent_actions', 'timestamp');
SELECT add_retention_policy('agent_actions', INTERVAL '90 days');
```

```sql
-- Market Data Snapshots (optional - if storing historical prices)
CREATE TABLE market_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    vwap DECIMAL(10,2),

    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('market_snapshots', 'timestamp');
SELECT add_retention_policy('market_snapshots', INTERVAL '180 days');
```

**Compression** (Automatic):
```sql
-- Enable compression (10-100x space savings)
ALTER TABLE trade_decisions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'agent',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Compress data older than 7 days
SELECT add_compression_policy('trade_decisions', INTERVAL '7 days');
```

**Estimated Size**: 5-10 GB (after 1 year, with compression)

---

### **3. Cloud Object Storage (Cold Archive)**

**Schema**: Parquet files (columnar format, highly compressed)

```
s3://trading-system-audit-archive/
├── trade_decisions/
│   ├── year=2026/
│   │   ├── month=01/
│   │   │   ├── trade_decisions_20260101.parquet
│   │   │   ├── trade_decisions_20260102.parquet
│   │   │   └── ...
│   │   ├── month=02/
│   │   └── ...
│   └── ...
├── agent_actions/
│   └── ...
└── market_snapshots/
    └── ...
```

**Automated Archival** (Python script):

```python
# utils/archival/archive_to_s3.py
import boto3
import pandas as pd
from datetime import datetime, timedelta

class AuditArchiver:
    """Archive old audit data to S3."""

    def __init__(self, bucket_name='trading-system-audit-archive'):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name

    def archive_old_data(self, table_name, days_old=90):
        """
        Archive data older than X days to S3 as Parquet.
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)

        # Query old data
        query = f"""
        SELECT * FROM {table_name}
        WHERE timestamp < %s
        ORDER BY timestamp
        """
        df = pd.read_sql(query, self.db_conn, params=[cutoff_date])

        if df.empty:
            logger.info(f"No data to archive for {table_name}")
            return

        # Group by month
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month

        for (year, month), group in df.groupby(['year', 'month']):
            # Convert to Parquet
            parquet_file = f"/tmp/{table_name}_{year}{month:02d}.parquet"
            group.to_parquet(parquet_file, compression='snappy')

            # Upload to S3
            s3_key = f"{table_name}/year={year}/month={month:02d}/{table_name}_{year}{month:02d}.parquet"
            self.s3.upload_file(parquet_file, self.bucket, s3_key)

            logger.info(f"✅ Archived {len(group)} rows to s3://{self.bucket}/{s3_key}")

        # Delete archived data from TimescaleDB
        delete_query = f"DELETE FROM {table_name} WHERE timestamp < %s"
        self.db_conn.execute(delete_query, [cutoff_date])
        logger.info(f"🗑️ Deleted {len(df)} archived rows from {table_name}")
```

**Query Archived Data** (via AWS Athena or GCP BigQuery):

```sql
-- Query via Athena (serverless)
SELECT
    agent,
    DATE_TRUNC('day', timestamp) as date,
    COUNT(*) as num_trades,
    AVG(realized_pnl_pct) as avg_return
FROM "s3://trading-system-audit-archive/trade_decisions/"
WHERE year = 2026 AND month = 1
GROUP BY agent, date
ORDER BY date;
```

**Cost**: ~₹200-500/month for 100 GB of archived data (S3/GCS)

---

## ☁️ **CLOUD DEPLOYMENT RECOMMENDATIONS**

### **Evaluation Criteria**

| Criteria | AWS | GCP | Azure | Railway/Render |
|----------|-----|-----|-------|----------------|
| **Availability** | 99.99% | 99.95% | 99.95% | 99.9% |
| **Reliability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Ease of Use** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Maintenance** | High (complex) | Medium | Medium | Low (managed) |
| **Cost (₹/month)** | ₹3,000-5,000 | ₹2,000-4,000 | ₹3,500-5,500 | ₹1,500-2,500 |
| **Free Tier** | Limited (12 months) | Generous (always-free) | Limited (12 months) | Generous |
| **LLM Integration** | Bedrock (complex) | **Gemini native** | OpenAI native | Manual |
| **India Region** | ✅ Mumbai | ✅ Mumbai | ✅ Pune | ❌ Singapore |

---

### **RECOMMENDATION: Google Cloud Platform (GCP)** ✅

**Why GCP?**

1. **Native Gemini Integration** → Free tier (1M tokens/day) = ₹0 LLM costs during paper trading
2. **Excellent Free Tier** → ₹23,000 credits for 90 days, then always-free tier
3. **Ease of Use** → Simpler than AWS, less complex pricing
4. **Good India Presence** → Mumbai region (low latency)
5. **Cloud SQL for PostgreSQL** → Fully managed (includes TimescaleDB support)
6. **Cloud Storage** → Cheap archival (₹0.15/GB/month)
7. **BigQuery** → Query archived Parquet files (serverless)

**Cost Estimate** (after free tier expires):
```
Cloud SQL (PostgreSQL + TimescaleDB): ₹1,500/month (db-f1-micro)
Cloud Run (API server):               ₹500/month (minimal usage)
Cloud Storage (100 GB archive):        ₹150/month
Gemini API (free tier):                ₹0/month
VPC + Networking:                      ₹200/month
────────────────────────────────────────────────────
TOTAL:                                 ₹2,350/month
```

**Setup Steps**:

```bash
# 1. Create GCP project
gcloud projects create trading-system-prod --name="Trading System"

# 2. Enable required APIs
gcloud services enable sqladmin.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com

# 3. Create Cloud SQL instance (PostgreSQL 15 + TimescaleDB)
gcloud sql instances create trading-db \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=asia-south1 \  # Mumbai
    --database-flags=cloudsql.enable_pgaudit=on,shared_preload_libraries=timescaledb

# 4. Create database
gcloud sql databases create trading_system --instance=trading-db

# 5. Create storage bucket
gsutil mb -l asia-south1 gs://trading-system-audit-archive

# 6. Deploy application (Cloud Run)
gcloud run deploy trading-api \
    --source . \
    --region=asia-south1 \
    --allow-unauthenticated \
    --set-env-vars="DATABASE_URL=postgresql://...,GEMINI_API_KEY=..."
```

---

### **Alternative: Railway (Easiest, Budget-Friendly)** 🚂

**Why Railway?**

1. **Extreme Simplicity** → Git push to deploy
2. **Managed PostgreSQL** → Includes TimescaleDB
3. **Low Cost** → ₹1,500-2,000/month
4. **Great Developer Experience** → Best UI, auto-scaling
5. **Generous Free Tier** → $5/month free

**Limitations**:
- ❌ No India region (Singapore closest)
- ❌ Less enterprise features
- ❌ Not suitable for very high scale

**Cost Estimate**:
```
PostgreSQL (1 GB RAM):     ₹800/month
Web Service (512 MB):      ₹600/month
Egress (10 GB):            ₹100/month
────────────────────────────────────
TOTAL:                     ₹1,500/month
```

**Setup** (incredibly simple):

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Initialize project
railway init

# 4. Add PostgreSQL
railway add --database postgresql

# 5. Deploy
railway up

# Done! Railway automatically:
# - Builds your app
# - Provisions database
# - Sets environment variables
# - Gives you a URL
```

---

### **Hybrid Approach (Recommended for Production)**

**Phase 1-2 (Paper Trading)**: Railway
- Low cost (₹1,500/month)
- Easy setup
- Perfect for testing

**Phase 3-4 (Live Trading)**: GCP
- More reliable (99.95% SLA)
- Better observability
- Gemini free tier
- India region

**Migration**: Dump PostgreSQL → Restore on GCP Cloud SQL (1 hour)

---

## 📦 **FINAL ARCHITECTURE DIAGRAM**

```
┌────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  ┌──────────────────┐  ┌───────────────┐  ┌─────────────────┐ │
│  │ Streamlit        │  │ Telegram Bot  │  │ API (FastAPI)   │ │
│  │ Dashboard        │  │ Alerts        │  │ /health /status │ │
│  └──────────────────┘  └───────────────┘  └─────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ LLM Manager (Plug & Play)                                 │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │ │
│  │  │ Claude   │  │ Gemini   │  │ GPT-4    │  │ FinGPT  │ │ │
│  │  │ (Primary)│  │ (Fallback│  │ (Fallback│  │ (Sentiment│ │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ AI Agents                                                 │ │
│  │  • Orchestrator  • Market Intel  • Risk Sentinel         │ │
│  │  • Portfolio Optimizer  • Strategy Inventor/Evaluator    │ │
│  │  • Post-Trade Analyzer  • Pattern Recognizer             │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │ PostgreSQL      │  │ TimescaleDB     │  │ GCS/S3         │ │
│  │ (Hot)           │  │ (Warm)          │  │ (Cold Archive) │ │
│  ├─────────────────┤  ├─────────────────┤  ├────────────────┤ │
│  │ • active_trades │  │ • trade_        │  │ • Parquet files│ │
│  │ • portfolio_    │  │   decisions     │  │ • >90 days old │ │
│  │   state         │  │ • agent_actions │  │ • Queryable via│ │
│  │ • daily_summary │  │ • market_       │  │   BigQuery/    │ │
│  │ • config        │  │   snapshots     │  │   Athena       │ │
│  │                 │  │ • Auto-compress │  │                │ │
│  │ < 1 GB          │  │ < 10 GB         │  │ < 100 GB       │ │
│  └─────────────────┘  └─────────────────┘  └────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                    EXTERNAL DATA SOURCES                        │
│  • yfinance (market data)  • News APIs  • NSE/BSE feeds        │
└────────────────────────────────────────────────────────────────┘
```

---

## ✅ **SETUP CHECKLIST**

### **Week 1: LLM Setup**
- [ ] Create Claude API account → Get API key
- [ ] Create Google Cloud account → Get Gemini API key (free tier)
- [ ] (Optional) Create OpenAI account → Get API key
- [ ] Create `.env` file with API keys
- [ ] Test LLM Manager with sample prompts
- [ ] Verify fallback mechanism works

### **Week 2: Database Setup**
- [ ] Choose cloud provider (GCP recommended, or Railway for simplicity)
- [ ] Provision PostgreSQL instance
- [ ] Install TimescaleDB extension
- [ ] Run schema migrations (`postgres.sql`, `timescale.sql`)
- [ ] Test database connections
- [ ] Setup automated backups

### **Week 3: Cloud Deployment**
- [ ] Deploy application to cloud
- [ ] Configure environment variables
- [ ] Setup Cloud Storage bucket for archives
- [ ] Test archival pipeline
- [ ] Configure monitoring/alerting
- [ ] Setup SSL/HTTPS

### **Week 4: Testing & Validation**
- [ ] Test LLM provider switching
- [ ] Test database writes (all 3 layers)
- [ ] Load test (simulate 100 trades/day)
- [ ] Verify data retention policies
- [ ] Test disaster recovery (backup restore)
- [ ] Document runbooks

---

## 💰 **COST OPTIMIZATION TIPS**

1. **Use Gemini Free Tier** for paper trading (saves ₹1,500-2,000/month)
2. **Start with Railway** instead of GCP (saves ₹1,000/month initially)
3. **Compress TimescaleDB aggressively** (saves 90% storage)
4. **Archive to cold storage quickly** (S3 Glacier/GCS Coldline)
5. **Use Claude Haiku** for simple tasks instead of Sonnet (10x cheaper)
6. **Cache LLM responses** when possible (60 min for market intel)
7. **Batch operations** to reduce database connections

**Estimated Total Cost** (after free tiers):
- **Phase 1-2** (Railway + Gemini free): ₹1,500/month
- **Phase 3-4** (GCP + paid Gemini): ₹2,500/month
- **Plus LLM costs** (if exceeding free tier): ₹500-1,500/month

**Grand Total**: ₹2,000-4,000/month (well within budget!)

---

**Next Steps**: Ready to start implementation? I can:
1. Generate all the code files (LLM clients, database schemas, etc.)
2. Create deployment scripts (GCP or Railway)
3. Build the first agent (Market Intelligence)

Let me know which you'd like to tackle first!
