"""
Kill Switch System - Emergency Trading Halt

CRITICAL: Lesson from Knight Capital - $440M lost in 45 minutes because
they couldn't stop the algorithm.

This kill switch MUST:
1. Respond in < 30 seconds
2. Cancel all pending orders
3. Close all positions
4. Disable all AI agents
5. Send emergency alerts
6. Be tested WEEKLY (every Sunday before market opens)

Features:
- Manual trigger (user initiated)
- Automatic triggers (circuit breakers)
- Health monitoring
- Test protocol
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
import json

logger = logging.getLogger(__name__)


class KillSwitchTrigger(Enum):
    """Reasons why kill switch can be activated."""
    MANUAL = "manual"  # User pressed emergency stop
    DAILY_LOSS = "daily_loss"  # Daily loss exceeded limit
    CONSECUTIVE_LOSSES = "consecutive_losses"  # Too many losses in a row
    VIX_SPIKE = "vix_spike"  # Market volatility spike
    ORDER_VOLUME = "order_volume"  # Runaway algorithm (too many orders)
    MAX_DRAWDOWN = "max_drawdown"  # Portfolio drawdown too large
    AI_CONFIDENCE_DROP = "ai_confidence_drop"  # AI model broken
    SYSTEM_ERROR = "system_error"  # Critical system failure
    REGULATORY = "regulatory"  # Regulatory issue detected
    TEST = "test"  # Weekly test (not real emergency)


class KillSwitchStatus(Enum):
    """Current status of kill switch."""
    ARMED = "armed"  # Ready to trigger
    TRIGGERED = "triggered"  # Emergency halt active
    TESTING = "testing"  # In test mode
    DISABLED = "disabled"  # Manually disabled (dangerous!)


class KillSwitch:
    """
    Emergency stop mechanism - HIGHEST PRIORITY SYSTEM

    Must be able to halt ALL trading in < 30 seconds
    """

    # Maximum response time (seconds)
    MAX_RESPONSE_TIME = 30

    # Test frequency (must test weekly)
    REQUIRED_TEST_INTERVAL = timedelta(days=7)

    def __init__(self, config: Dict = None):
        """
        Initialize kill switch.

        Args:
            config: Configuration dict with alert settings
        """
        self.status = KillSwitchStatus.ARMED
        self.last_test_date: Optional[datetime] = None
        self.last_trigger_date: Optional[datetime] = None
        self.test_history: List[Dict] = []
        self.trigger_history: List[Dict] = []

        # Configuration
        self.config = config or {}
        self.telegram_enabled = self.config.get('telegram_enabled', False)
        self.email_enabled = self.config.get('email_enabled', False)

        # Components (will be injected)
        self.order_manager = None
        self.position_manager = None
        self.agent_manager = None
        self.alert_system = None
        self.database = None

        logger.info("🚨 Kill Switch initialized and ARMED")

    def inject_dependencies(
        self,
        order_manager=None,
        position_manager=None,
        agent_manager=None,
        alert_system=None,
        database=None
    ):
        """
        Inject system dependencies.

        This allows kill switch to control all system components.
        """
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.agent_manager = agent_manager
        self.alert_system = alert_system
        self.database = database

        logger.info("✅ Kill Switch dependencies injected")

    async def trigger(
        self,
        reason: KillSwitchTrigger,
        details: str = "",
        force_close_positions: bool = True
    ) -> Dict[str, any]:
        """
        EMERGENCY HALT - Stop all trading immediately.

        Args:
            reason: Why kill switch was triggered
            details: Additional context
            force_close_positions: If True, close all positions (default)

        Returns:
            Dict with execution results and timing
        """
        start_time = time.time()

        logger.critical(
            f"🚨🚨🚨 KILL SWITCH ACTIVATED 🚨🚨🚨\n"
            f"Reason: {reason.value}\n"
            f"Details: {details}\n"
            f"Time: {datetime.now().isoformat()}"
        )

        # Update status
        old_status = self.status
        self.status = KillSwitchStatus.TRIGGERED if reason != KillSwitchTrigger.TEST else KillSwitchStatus.TESTING

        results = {
            'trigger_time': datetime.now().isoformat(),
            'reason': reason.value,
            'details': details,
            'previous_status': old_status.value,
            'steps_completed': [],
            'steps_failed': [],
            'timing': {}
        }

        try:
            # STEP 1: Cancel all pending orders
            step_start = time.time()
            logger.critical("📋 Step 1/5: Cancelling all pending orders...")

            if self.order_manager:
                try:
                    cancelled_orders = await self.order_manager.cancel_all_orders()
                    results['steps_completed'].append('cancel_orders')
                    results['cancelled_orders'] = cancelled_orders
                    logger.info(f"✅ Cancelled {len(cancelled_orders)} orders")
                except Exception as e:
                    results['steps_failed'].append(f'cancel_orders: {str(e)}')
                    logger.error(f"❌ Order cancellation failed: {e}")
            else:
                logger.warning("⚠️  No order manager - skipping order cancellation")

            results['timing']['cancel_orders'] = time.time() - step_start

            # STEP 2: Close all positions (if requested)
            if force_close_positions:
                step_start = time.time()
                logger.critical("💼 Step 2/5: Closing all positions...")

                if self.position_manager:
                    try:
                        closed_positions = await self.position_manager.close_all_positions()
                        results['steps_completed'].append('close_positions')
                        results['closed_positions'] = closed_positions
                        logger.info(f"✅ Closed {len(closed_positions)} positions")
                    except Exception as e:
                        results['steps_failed'].append(f'close_positions: {str(e)}')
                        logger.error(f"❌ Position closure failed: {e}")
                else:
                    logger.warning("⚠️  No position manager - skipping position closure")

                results['timing']['close_positions'] = time.time() - step_start
            else:
                logger.warning("⚠️  Position closure skipped (force_close_positions=False)")

            # STEP 3: Disable all AI agents
            step_start = time.time()
            logger.critical("🤖 Step 3/5: Disabling all AI agents...")

            if self.agent_manager:
                try:
                    disabled_agents = await self.agent_manager.disable_all_agents()
                    results['steps_completed'].append('disable_agents')
                    results['disabled_agents'] = disabled_agents
                    logger.info(f"✅ Disabled {len(disabled_agents)} agents")
                except Exception as e:
                    results['steps_failed'].append(f'disable_agents: {str(e)}')
                    logger.error(f"❌ Agent shutdown failed: {e}")
            else:
                logger.warning("⚠️  No agent manager - skipping agent shutdown")

            results['timing']['disable_agents'] = time.time() - step_start

            # STEP 4: Send emergency alerts
            step_start = time.time()
            logger.critical("📢 Step 4/5: Sending emergency alerts...")

            alert_message = (
                f"🚨 KILL SWITCH ACTIVATED\n\n"
                f"Reason: {reason.value}\n"
                f"Details: {details}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"Actions taken:\n"
                f"- Orders cancelled: {len(results.get('cancelled_orders', []))}\n"
                f"- Positions closed: {len(results.get('closed_positions', []))}\n"
                f"- Agents disabled: {len(results.get('disabled_agents', []))}\n\n"
                f"Trading system is now HALTED.\n"
                f"Manual intervention required to resume."
            )

            if self.alert_system:
                try:
                    await self.alert_system.send_emergency_alert(alert_message)
                    results['steps_completed'].append('send_alerts')
                    logger.info("✅ Emergency alerts sent")
                except Exception as e:
                    results['steps_failed'].append(f'send_alerts: {str(e)}')
                    logger.error(f"❌ Alert sending failed: {e}")
            else:
                # Fallback: at least log it
                logger.critical(alert_message)
                logger.warning("⚠️  No alert system - alerts not sent")

            results['timing']['send_alerts'] = time.time() - step_start

            # STEP 5: Log to audit trail
            step_start = time.time()
            logger.critical("📝 Step 5/5: Logging to audit trail...")

            if self.database:
                try:
                    await self.database.log_kill_switch_event({
                        'trigger_time': results['trigger_time'],
                        'reason': reason.value,
                        'details': details,
                        'results': json.dumps(results)
                    })
                    results['steps_completed'].append('log_audit')
                    logger.info("✅ Event logged to audit trail")
                except Exception as e:
                    results['steps_failed'].append(f'log_audit: {str(e)}')
                    logger.error(f"❌ Audit logging failed: {e}")
            else:
                logger.warning("⚠️  No database - audit trail not logged")

            results['timing']['log_audit'] = time.time() - step_start

        except Exception as e:
            logger.critical(f"❌ KILL SWITCH EXECUTION FAILED: {e}")
            results['critical_error'] = str(e)

        # Calculate total response time
        total_time = time.time() - start_time
        results['total_response_time'] = round(total_time, 2)

        # Check if we met the response time requirement
        if total_time > self.MAX_RESPONSE_TIME:
            logger.error(
                f"⚠️  KILL SWITCH TOO SLOW: {total_time:.2f}s > {self.MAX_RESPONSE_TIME}s target"
            )
            results['timing_warning'] = f"Response time exceeded target"
        else:
            logger.info(f"✅ Kill switch completed in {total_time:.2f}s (target: <{self.MAX_RESPONSE_TIME}s)")

        # Record trigger
        self.trigger_history.append(results)
        self.last_trigger_date = datetime.now()

        # Print summary
        logger.critical(
            f"\n{'='*70}\n"
            f"KILL SWITCH EXECUTION COMPLETE\n"
            f"{'='*70}\n"
            f"Total time: {total_time:.2f}s\n"
            f"Steps completed: {len(results['steps_completed'])}/5\n"
            f"Steps failed: {len(results['steps_failed'])}\n"
            f"Status: {self.status.value}\n"
            f"{'='*70}"
        )

        return results

    async def test_weekly(self) -> Dict[str, any]:
        """
        MANDATORY WEEKLY TEST

        Must be run every Sunday before market opens.
        Tests all kill switch functionality without real trades.

        Returns:
            Test results dict
        """
        logger.info("🧪 Starting WEEKLY KILL SWITCH TEST")
        logger.info("=" * 70)

        # Check if test is overdue
        if self.last_test_date:
            days_since_test = (datetime.now() - self.last_test_date).days
            if days_since_test > 7:
                logger.warning(f"⚠️  TEST OVERDUE by {days_since_test - 7} days!")

        # Run test (without closing real positions)
        test_result = await self.trigger(
            reason=KillSwitchTrigger.TEST,
            details="Weekly automated test",
            force_close_positions=False  # Don't close positions in test
        )

        # Update last test date
        self.last_test_date = datetime.now()
        self.test_history.append(test_result)

        # Re-arm kill switch after test
        self.status = KillSwitchStatus.ARMED

        # Evaluate test results
        test_passed = (
            len(test_result['steps_failed']) == 0 and
            test_result['total_response_time'] <= self.MAX_RESPONSE_TIME
        )

        if test_passed:
            logger.info("✅ WEEKLY TEST PASSED - Kill switch ready")
        else:
            logger.error("❌ WEEKLY TEST FAILED - Fix before trading!")

        logger.info("=" * 70)

        return {
            **test_result,
            'test_passed': test_passed,
            'next_test_due': (datetime.now() + self.REQUIRED_TEST_INTERVAL).isoformat()
        }

    def is_test_overdue(self) -> bool:
        """Check if weekly test is overdue."""
        if not self.last_test_date:
            return True

        days_since_test = (datetime.now() - self.last_test_date).days
        return days_since_test >= 7

    def get_status(self) -> Dict[str, any]:
        """Get current kill switch status."""
        return {
            'status': self.status.value,
            'last_test_date': self.last_test_date.isoformat() if self.last_test_date else None,
            'days_since_last_test': (
                (datetime.now() - self.last_test_date).days
                if self.last_test_date else None
            ),
            'test_overdue': self.is_test_overdue(),
            'last_trigger_date': self.last_trigger_date.isoformat() if self.last_trigger_date else None,
            'total_tests_run': len(self.test_history),
            'total_triggers': len(self.trigger_history),
            'ready_to_trade': (
                self.status == KillSwitchStatus.ARMED and
                not self.is_test_overdue()
            )
        }

    def disable(self, reason: str = ""):
        """
        Disable kill switch (DANGEROUS - only for emergencies).

        Args:
            reason: Why kill switch is being disabled
        """
        logger.warning(
            f"⚠️⚠️⚠️  KILL SWITCH DISABLED ⚠️⚠️⚠️ \n"
            f"Reason: {reason}\n"
            f"This is VERY DANGEROUS - re-enable ASAP!"
        )
        self.status = KillSwitchStatus.DISABLED

    def enable(self):
        """Re-enable kill switch."""
        self.status = KillSwitchStatus.ARMED
        logger.info("✅ Kill switch re-ARMED")


# ============================================================================
# MOCK MANAGERS FOR TESTING
# ============================================================================

class MockOrderManager:
    """Mock order manager for testing."""
    async def cancel_all_orders(self):
        await asyncio.sleep(0.1)  # Simulate API call
        return ['order_1', 'order_2', 'order_3']


class MockPositionManager:
    """Mock position manager for testing."""
    async def close_all_positions(self):
        await asyncio.sleep(0.5)  # Simulate closing positions
        return ['position_1', 'position_2']


class MockAgentManager:
    """Mock agent manager for testing."""
    async def disable_all_agents(self):
        await asyncio.sleep(0.05)  # Quick shutdown
        return ['market_intel', 'orchestrator', 'risk_sentinel']


class MockAlertSystem:
    """Mock alert system for testing."""
    async def send_emergency_alert(self, message):
        await asyncio.sleep(0.1)
        print(f"\n📢 ALERT SENT:\n{message}\n")


# ============================================================================
# TEST PROTOCOL
# ============================================================================

async def test_kill_switch():
    """Test kill switch functionality."""

    print("=" * 70)
    print("KILL SWITCH TEST PROTOCOL")
    print("=" * 70)

    # Initialize kill switch
    kill_switch = KillSwitch()

    # Inject mock dependencies
    kill_switch.inject_dependencies(
        order_manager=MockOrderManager(),
        position_manager=MockPositionManager(),
        agent_manager=MockAgentManager(),
        alert_system=MockAlertSystem()
    )

    # Test 1: Weekly test
    print("\n🧪 Test 1: Weekly Test Protocol")
    print("-" * 70)
    result = await kill_switch.test_weekly()

    print(f"\nTest Result: {'✅ PASSED' if result['test_passed'] else '❌ FAILED'}")
    print(f"Response Time: {result['total_response_time']}s (target: <30s)")
    print(f"Steps Completed: {len(result['steps_completed'])}/5")
    print(f"Steps Failed: {len(result['steps_failed'])}")

    # Test 2: Check status
    print("\n📊 Test 2: Status Check")
    print("-" * 70)
    status = kill_switch.get_status()
    print(json.dumps(status, indent=2))

    # Test 3: Manual trigger (simulated emergency)
    print("\n🚨 Test 3: Manual Trigger (Simulated Emergency)")
    print("-" * 70)
    result = await kill_switch.trigger(
        reason=KillSwitchTrigger.DAILY_LOSS,
        details="Daily loss -5.2% exceeded -5.0% limit",
        force_close_positions=True
    )

    print(f"\nEmergency Halt Complete!")
    print(f"Response Time: {result['total_response_time']}s")
    print(f"Orders Cancelled: {len(result.get('cancelled_orders', []))}")
    print(f"Positions Closed: {len(result.get('closed_positions', []))}")
    print(f"Agents Disabled: {len(result.get('disabled_agents', []))}")

    print("\n" + "=" * 70)
    print("✅ Kill switch test protocol complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_kill_switch())
