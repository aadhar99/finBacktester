"""
Database models for type safety and validation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from decimal import Decimal


@dataclass
class ActiveTrade:
    """Represents an active trading position."""
    trade_id: str
    symbol: str
    action: str  # BUY or SELL
    entry_time: datetime
    entry_price: Decimal
    quantity: int
    agent: str

    # Optional fields
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    current_price: Optional[Decimal] = None
    current_pnl: Optional[Decimal] = None
    current_pnl_pct: Optional[Decimal] = None
    confidence: Optional[Decimal] = None
    risk_score: Optional[int] = None
    last_updated: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'action': self.action,
            'entry_time': self.entry_time,
            'entry_price': float(self.entry_price),
            'quantity': self.quantity,
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': float(self.take_profit) if self.take_profit else None,
            'agent': self.agent,
            'confidence': float(self.confidence) if self.confidence else None,
            'risk_score': self.risk_score
        }


@dataclass
class PortfolioState:
    """Represents current portfolio state."""
    cash: Decimal
    total_value: Decimal
    invested_value: Decimal
    daily_pnl: Decimal
    daily_pnl_pct: Decimal
    total_exposure_pct: Decimal
    num_positions: int
    total_pnl: Decimal
    total_pnl_pct: Decimal
    max_drawdown_pct: Decimal
    last_updated: datetime


@dataclass
class DailySummary:
    """Represents end-of-day summary."""
    date: datetime
    starting_capital: Decimal
    ending_capital: Decimal
    pnl: Decimal
    pnl_pct: Decimal
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: Optional[Decimal] = None
    best_trade_pnl: Optional[Decimal] = None
    worst_trade_pnl: Optional[Decimal] = None
    sharpe_ratio: Optional[Decimal] = None
    max_drawdown_pct: Optional[Decimal] = None
    best_agent: Optional[str] = None
    worst_agent: Optional[str] = None


@dataclass
class TradeDecision:
    """Represents an AI trading decision (audit trail)."""
    decision_id: str
    timestamp: datetime
    symbol: str
    action: str
    agent: str
    ai_reasoning: str
    confidence: Decimal
    risk_score: int

    # Optional fields
    quantity: Optional[int] = None
    entry_price: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    market_regime: Optional[str] = None
    vix: Optional[Decimal] = None
    sector_sentiment: Optional[str] = None
    expected_return_pct: Optional[Decimal] = None
    expected_hold_hours: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    realized_pnl_pct: Optional[Decimal] = None
    actual_hold_hours: Optional[Decimal] = None
    outcome: Optional[str] = None
    required_human_approval: bool = True
    human_decision: Optional[str] = None
    human_decision_time: Optional[datetime] = None
    llm_provider: Optional[str] = None
    llm_cost: Optional[Decimal] = None
    decision_latency_ms: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            'decision_id': self.decision_id,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'action': self.action,
            'quantity': self.quantity,
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'agent': self.agent,
            'ai_reasoning': self.ai_reasoning,
            'confidence': float(self.confidence),
            'risk_score': self.risk_score,
            'market_regime': self.market_regime,
            'vix': float(self.vix) if self.vix else None,
            'sector_sentiment': self.sector_sentiment,
            'expected_return_pct': float(self.expected_return_pct) if self.expected_return_pct else None,
            'expected_hold_hours': float(self.expected_hold_hours) if self.expected_hold_hours else None,
            'required_human_approval': self.required_human_approval,
            'outcome': self.outcome or 'PENDING',
            'llm_provider': self.llm_provider,
            'llm_cost': float(self.llm_cost) if self.llm_cost else None,
            'decision_latency_ms': self.decision_latency_ms
        }
