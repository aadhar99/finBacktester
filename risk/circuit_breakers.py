"""
Circuit Breakers - Automatic Trading Halts

CRITICAL: Lesson from Flash Crash - need automatic trading halts to prevent
cascading losses.

Circuit breakers automatically trigger kill switch when:
1. Daily loss exceeds limit (-5%)
2. Too many consecutive losses (4 in a row)
3. VIX spikes above threshold (>35)
4. Order volume excessive (>10 orders/minute - runaway algo)
5. Max drawdown exceeded (-15%)
6. AI confidence drops below threshold (<50%)

Features:
- Real-time monitoring
- Automatic kill switch trigger
- Configurable thresholds
- Alert notifications
- Audit logging
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CircuitBreakerType(Enum):
    """Types of circuit breakers."""
    DAILY_LOSS = "daily_loss"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    VIX_SPIKE = "vix_spike"
    ORDER_VOLUME = "order_volume"
    MAX_DRAWDOWN = "max_drawdown"
    AI_CONFIDENCE = "ai_confidence"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a single circuit breaker."""
    breaker_type: CircuitBreakerType
    threshold: float
    description: str
    enabled: bool = True
    cooldown_minutes: int = 60  # How long before re-enabling after trip


@dataclass
class CircuitBreakerStatus:
    """Current status of a circuit breaker."""
    breaker_type: CircuitBreakerType
    enabled: bool
    tripped: bool
    trip_time: Optional[datetime] = None
    trip_reason: Optional[str] = None
    trip_count_today: int = 0


class CircuitBreakers:
    """
    Automatic trading halts - prevent catastrophic losses.

    Monitors trading activity and market conditions in real-time.
    Triggers kill switch when thresholds are breached.
    """

    # Default circuit breaker configurations
    DEFAULT_BREAKERS = {
        CircuitBreakerType.DAILY_LOSS: CircuitBreakerConfig(
            breaker_type=CircuitBreakerType.DAILY_LOSS,
            threshold=-5.0,  # -5% daily loss
            description="Daily loss limit (HARD STOP)",
            enabled=True,
            cooldown_minutes=1440  # 24 hours
        ),

        CircuitBreakerType.CONSECUTIVE_LOSSES: CircuitBreakerConfig(
            breaker_type=CircuitBreakerType.CONSECUTIVE_LOSSES,
            threshold=4.0,  # 4 consecutive losses
            description="Consecutive losing trades",
            enabled=True,
            cooldown_minutes=60
        ),

        CircuitBreakerType.VIX_SPIKE: CircuitBreakerConfig(
            breaker_type=CircuitBreakerType.VIX_SPIKE,
            threshold=35.0,  # VIX > 35 (extreme volatility)
            description="Market volatility spike",
            enabled=True,
            cooldown_minutes=30
        ),

        CircuitBreakerType.ORDER_VOLUME: CircuitBreakerConfig(
            breaker_type=CircuitBreakerType.ORDER_VOLUME,
            threshold=10.0,  # 10 orders per minute (runaway algo)
            description="Excessive order volume",
            enabled=True,
            cooldown_minutes=15
        ),

        CircuitBreakerType.MAX_DRAWDOWN: CircuitBreakerConfig(
            breaker_type=CircuitBreakerType.MAX_DRAWDOWN,
            threshold=-15.0,  # -15% from peak
            description="Maximum portfolio drawdown",
            enabled=True,
            cooldown_minutes=1440  # 24 hours
        ),

        CircuitBreakerType.AI_CONFIDENCE: CircuitBreakerConfig(
            breaker_type=CircuitBreakerType.AI_CONFIDENCE,
            threshold=0.50,  # AI confidence < 50% (model broken)
            description="AI confidence collapse",
            enabled=True,
            cooldown_minutes=120
        )
    }

    def __init__(self, kill_switch, config: Dict[CircuitBreakerType, CircuitBreakerConfig] = None):
        """
        Initialize circuit breakers.

        Args:
            kill_switch: KillSwitch instance to trigger
            config: Custom circuit breaker configurations (optional)
        """
        self.kill_switch = kill_switch

        # Use custom config or defaults
        self.config = config or self.DEFAULT_BREAKERS.copy()

        # Initialize status tracking
        self.status = {
            breaker_type: CircuitBreakerStatus(
                breaker_type=breaker_type,
                enabled=config.enabled,
                tripped=False
            )
            for breaker_type, config in self.config.items()
        }

        # Monitoring data
        self.daily_pnl = 0.0
        self.starting_capital = 100000.0  # Will be updated from portfolio
        self.portfolio_peak = 100000.0
        self.current_portfolio_value = 100000.0
        self.consecutive_losses = 0
        self.recent_orders: List[datetime] = []
        self.last_ai_confidence = 1.0
        self.current_vix = 15.0

        logger.info(f"✅ Circuit Breakers initialized ({len(self.config)} breakers)")

    def update_portfolio_metrics(
        self,
        current_value: float,
        daily_pnl: float = None,
        starting_capital: float = None
    ):
        """
        Update portfolio metrics for circuit breaker calculations.

        Args:
            current_value: Current portfolio value
            daily_pnl: Today's P&L (optional, will calculate if not provided)
            starting_capital: Starting capital for the day (optional)
        """
        if starting_capital:
            self.starting_capital = starting_capital

        self.current_portfolio_value = current_value

        # Update peak
        if current_value > self.portfolio_peak:
            self.portfolio_peak = current_value

        # Calculate daily P&L if not provided
        if daily_pnl is not None:
            self.daily_pnl = daily_pnl
        else:
            self.daily_pnl = current_value - self.starting_capital

    def update_trade_result(self, pnl: float):
        """
        Update with latest trade result.

        Args:
            pnl: Trade P&L (positive = win, negative = loss)
        """
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # Reset on win

    def update_order_activity(self):
        """Record new order placed (for order volume monitoring)."""
        self.recent_orders.append(datetime.now())

        # Clean old orders (keep last 5 minutes)
        cutoff = datetime.now() - timedelta(minutes=5)
        self.recent_orders = [t for t in self.recent_orders if t > cutoff]

    def update_ai_confidence(self, confidence: float):
        """
        Update AI confidence score.

        Args:
            confidence: AI confidence (0.0 to 1.0)
        """
        self.last_ai_confidence = confidence

    def update_market_vix(self, vix: float):
        """
        Update market volatility (VIX).

        Args:
            vix: VIX value
        """
        self.current_vix = vix

    async def check_all_breakers(self) -> List[Dict]:
        """
        Check all circuit breakers.

        Returns:
            List of triggered breakers (empty if none triggered)
        """
        triggered_breakers = []

        # Check each breaker
        for breaker_type, config in self.config.items():
            if not config.enabled:
                continue

            # Skip if breaker already tripped and in cooldown
            status = self.status[breaker_type]
            if status.tripped:
                if self._is_in_cooldown(status, config):
                    continue
                else:
                    # Cooldown expired, reset
                    status.tripped = False
                    status.trip_time = None

            # Check threshold
            triggered, value, message = self._check_breaker(breaker_type, config)

            if triggered:
                # Trigger this breaker
                await self._trigger_breaker(breaker_type, value, message)
                triggered_breakers.append({
                    'type': breaker_type.value,
                    'threshold': config.threshold,
                    'value': value,
                    'message': message
                })

        return triggered_breakers

    def _check_breaker(
        self,
        breaker_type: CircuitBreakerType,
        config: CircuitBreakerConfig
    ) -> tuple[bool, float, str]:
        """
        Check if a specific breaker should trip.

        Returns:
            (triggered, current_value, message)
        """
        if breaker_type == CircuitBreakerType.DAILY_LOSS:
            daily_loss_pct = (self.daily_pnl / self.starting_capital) * 100
            if daily_loss_pct <= config.threshold:
                return (
                    True,
                    daily_loss_pct,
                    f"Daily loss {daily_loss_pct:.2f}% exceeded {config.threshold}% limit"
                )

        elif breaker_type == CircuitBreakerType.CONSECUTIVE_LOSSES:
            if self.consecutive_losses >= config.threshold:
                return (
                    True,
                    self.consecutive_losses,
                    f"{self.consecutive_losses} consecutive losses (limit: {int(config.threshold)})"
                )

        elif breaker_type == CircuitBreakerType.VIX_SPIKE:
            if self.current_vix >= config.threshold:
                return (
                    True,
                    self.current_vix,
                    f"VIX spike to {self.current_vix:.1f} (limit: {config.threshold})"
                )

        elif breaker_type == CircuitBreakerType.ORDER_VOLUME:
            # Orders in last minute
            one_minute_ago = datetime.now() - timedelta(minutes=1)
            orders_last_minute = len([t for t in self.recent_orders if t > one_minute_ago])

            if orders_last_minute >= config.threshold:
                return (
                    True,
                    orders_last_minute,
                    f"Excessive order volume: {orders_last_minute} orders/minute (limit: {int(config.threshold)})"
                )

        elif breaker_type == CircuitBreakerType.MAX_DRAWDOWN:
            drawdown = ((self.current_portfolio_value - self.portfolio_peak) / self.portfolio_peak) * 100
            if drawdown <= config.threshold:
                return (
                    True,
                    drawdown,
                    f"Max drawdown {drawdown:.2f}% exceeded {config.threshold}% limit"
                )

        elif breaker_type == CircuitBreakerType.AI_CONFIDENCE:
            if self.last_ai_confidence <= config.threshold:
                return (
                    True,
                    self.last_ai_confidence,
                    f"AI confidence {self.last_ai_confidence:.2f} below {config.threshold} threshold"
                )

        return (False, 0.0, "")

    async def _trigger_breaker(self, breaker_type: CircuitBreakerType, value: float, message: str):
        """
        Trigger a circuit breaker.

        Args:
            breaker_type: Which breaker triggered
            value: Current value that exceeded threshold
            message: Description of trigger
        """
        logger.critical(f"⚡ CIRCUIT BREAKER TRIPPED: {breaker_type.value}")
        logger.critical(f"   {message}")

        # Update breaker status
        status = self.status[breaker_type]
        status.tripped = True
        status.trip_time = datetime.now()
        status.trip_reason = message
        status.trip_count_today += 1

        # Trigger kill switch
        if self.kill_switch:
            from execution.kill_switch import KillSwitchTrigger

            # Map circuit breaker type to kill switch trigger
            trigger_mapping = {
                CircuitBreakerType.DAILY_LOSS: KillSwitchTrigger.DAILY_LOSS,
                CircuitBreakerType.CONSECUTIVE_LOSSES: KillSwitchTrigger.CONSECUTIVE_LOSSES,
                CircuitBreakerType.VIX_SPIKE: KillSwitchTrigger.VIX_SPIKE,
                CircuitBreakerType.ORDER_VOLUME: KillSwitchTrigger.ORDER_VOLUME,
                CircuitBreakerType.MAX_DRAWDOWN: KillSwitchTrigger.MAX_DRAWDOWN,
                CircuitBreakerType.AI_CONFIDENCE: KillSwitchTrigger.AI_CONFIDENCE_DROP,
            }

            trigger = trigger_mapping.get(breaker_type, KillSwitchTrigger.SYSTEM_ERROR)

            await self.kill_switch.trigger(
                reason=trigger,
                details=message
            )
        else:
            logger.error("❌ No kill switch configured - cannot halt trading!")

    def _is_in_cooldown(self, status: CircuitBreakerStatus, config: CircuitBreakerConfig) -> bool:
        """Check if breaker is still in cooldown period."""
        if not status.trip_time:
            return False

        elapsed = (datetime.now() - status.trip_time).total_seconds() / 60
        return elapsed < config.cooldown_minutes

    def get_status(self) -> Dict:
        """Get current status of all circuit breakers."""
        return {
            'breakers': {
                breaker_type.value: {
                    'enabled': status.enabled,
                    'tripped': status.tripped,
                    'trip_time': status.trip_time.isoformat() if status.trip_time else None,
                    'trip_reason': status.trip_reason,
                    'trip_count_today': status.trip_count_today,
                    'threshold': self.config[breaker_type].threshold,
                    'in_cooldown': self._is_in_cooldown(status, self.config[breaker_type])
                }
                for breaker_type, status in self.status.items()
            },
            'current_metrics': {
                'daily_pnl': round(self.daily_pnl, 2),
                'daily_pnl_pct': round((self.daily_pnl / self.starting_capital) * 100, 2),
                'consecutive_losses': self.consecutive_losses,
                'current_vix': self.current_vix,
                'orders_last_minute': len([
                    t for t in self.recent_orders
                    if t > datetime.now() - timedelta(minutes=1)
                ]),
                'drawdown_pct': round(
                    ((self.current_portfolio_value - self.portfolio_peak) / self.portfolio_peak) * 100,
                    2
                ),
                'ai_confidence': round(self.last_ai_confidence, 2)
            }
        }

    def reset_daily_counters(self):
        """Reset daily counters (call at start of trading day)."""
        for status in self.status.values():
            status.trip_count_today = 0

        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.recent_orders = []

        logger.info("✅ Circuit breaker daily counters reset")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example usage of circuit breakers."""
    from execution.kill_switch import KillSwitch

    print("=" * 70)
    print("CIRCUIT BREAKER SYSTEM - Example Usage")
    print("=" * 70)

    # Initialize kill switch
    kill_switch = KillSwitch()

    # Initialize circuit breakers
    breakers = CircuitBreakers(kill_switch)

    # Simulate trading day
    print("\n📊 Starting trading day with ₹100,000 capital")
    breakers.update_portfolio_metrics(
        current_value=100000,
        starting_capital=100000
    )

    # Scenario 1: Normal trading (no triggers)
    print("\n✅ Scenario 1: Normal Trading")
    print("-" * 70)
    breakers.update_portfolio_metrics(current_value=100500)  # +0.5%
    triggered = await breakers.check_all_breakers()
    print(f"Triggered breakers: {len(triggered)}")

    # Scenario 2: Daily loss limit exceeded
    print("\n⚡ Scenario 2: Daily Loss Limit")
    print("-" * 70)
    breakers.update_portfolio_metrics(current_value=94500)  # -5.5% (exceeds -5% limit)
    triggered = await breakers.check_all_breakers()
    print(f"Triggered breakers: {len(triggered)}")
    if triggered:
        for breaker in triggered:
            print(f"  - {breaker['type']}: {breaker['message']}")

    # Get status
    print("\n📊 Circuit Breaker Status")
    print("-" * 70)
    import json
    status = breakers.get_status()
    print(json.dumps(status, indent=2))

    print("\n" + "=" * 70)
    print("✅ Circuit breaker system ready")
    print("=" * 70)


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
