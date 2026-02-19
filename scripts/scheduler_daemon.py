#!/usr/bin/env python3
"""
Scheduler Daemon - Automated Daily Trading Pipeline

Runs scheduled jobs:
- Daily scan: 9:00 AM IST (scrape NSE, detect patterns, generate signals)
- Paper trading: 9:05 AM IST (execute trades, update positions)
- Position update: 3:00 PM IST (update with closing prices)

Usage:
    # Run in foreground (testing)
    python3 scripts/scheduler_daemon.py

    # Run in background (production)
    nohup python3 scripts/scheduler_daemon.py > logs/scheduler.log 2>&1 &

    # Or use screen/tmux
    screen -S trading-scheduler
    python3 scripts/scheduler_daemon.py
    # Ctrl+A, D to detach

Stop:
    pkill -f scheduler_daemon.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import logging
import subprocess
import signal
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)


class TradingScheduler:
    """Automated trading scheduler with APScheduler."""

    def __init__(self):
        """Initialize scheduler with job store and executors."""

        # Thread pool executor
        executors = {
            'default': ThreadPoolExecutor(max_workers=2)
        }

        # Job defaults
        job_defaults = {
            'coalesce': False,  # Run all missed jobs
            'max_instances': 1,  # Only one instance of each job at a time
            'misfire_grace_time': 300  # 5 minute grace period
        }

        # Use in-memory job store for cloud deployment (no persistence needed)
        self.scheduler = BlockingScheduler(
            executors=executors,
            job_defaults=job_defaults
        )

        # Add event listeners
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        logger.info("✅ Trading scheduler initialized")

    def _job_executed(self, event):
        """Callback when job completes successfully."""
        logger.info(f"  ✅ Job '{event.job_id}' executed successfully")

    def _job_error(self, event):
        """Callback when job fails."""
        logger.error(f"  ❌ Job '{event.job_id}' failed: {event.exception}")

    def _shutdown(self, signum, frame):
        """Graceful shutdown handler."""
        logger.info("⚠️  Shutdown signal received, stopping scheduler...")
        self.scheduler.shutdown()
        logger.info("✅ Scheduler stopped gracefully")
        sys.exit(0)

    def run_daily_scan(self):
        """Run daily NSE scan and pattern detection."""
        logger.info("=" * 70)
        logger.info("📊 Starting Daily Scan")
        logger.info("=" * 70)

        try:
            # Run daily scan script
            result = subprocess.run(
                ['python3', 'scripts/daily_scan.py'],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                logger.info("  ✅ Daily scan completed successfully")
                # Log last few lines of output
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-10:]:
                    logger.info(f"    {line}")
            else:
                logger.error(f"  ❌ Daily scan failed with code {result.returncode}")
                logger.error(f"    Error: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("  ❌ Daily scan timed out (>10 minutes)")
        except Exception as e:
            logger.error(f"  ❌ Daily scan error: {e}")

    def run_paper_trading(self):
        """Run paper trading update."""
        logger.info("=" * 70)
        logger.info("💼 Starting Paper Trading Update")
        logger.info("=" * 70)

        try:
            # Run paper trading script
            result = subprocess.run(
                ['python3', 'scripts/run_paper_trading.py'],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                logger.info("  ✅ Paper trading update completed")
                # Log last few lines of output
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-15:]:
                    logger.info(f"    {line}")
            else:
                logger.error(f"  ❌ Paper trading failed with code {result.returncode}")
                logger.error(f"    Error: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("  ❌ Paper trading timed out (>10 minutes)")
        except Exception as e:
            logger.error(f"  ❌ Paper trading error: {e}")

    def update_positions_eod(self):
        """Update positions with end-of-day prices."""
        logger.info("=" * 70)
        logger.info("📈 End-of-Day Position Update")
        logger.info("=" * 70)

        try:
            # Run paper trading with --show-only (just update, no new trades)
            result = subprocess.run(
                ['python3', 'scripts/run_paper_trading.py'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info("  ✅ EOD position update completed")
            else:
                logger.error(f"  ❌ EOD update failed with code {result.returncode}")

        except Exception as e:
            logger.error(f"  ❌ EOD update error: {e}")

    def schedule_jobs(self):
        """Schedule all trading jobs."""

        # Job 1: Daily Scan - 9:00 AM IST (every weekday)
        self.scheduler.add_job(
            self.run_daily_scan,
            'cron',
            day_of_week='mon-fri',
            hour=9,
            minute=0,
            id='daily_scan',
            replace_existing=True
        )

        # Job 2: Paper Trading - 9:05 AM IST (every weekday)
        self.scheduler.add_job(
            self.run_paper_trading,
            'cron',
            day_of_week='mon-fri',
            hour=9,
            minute=5,
            id='paper_trading',
            replace_existing=True
        )

        # Job 3: EOD Position Update - 3:35 PM IST (every weekday)
        self.scheduler.add_job(
            self.update_positions_eod,
            'cron',
            day_of_week='mon-fri',
            hour=15,
            minute=35,
            id='eod_update',
            replace_existing=True
        )

        logger.info("✅ Jobs scheduled:")
        logger.info("  📊 Daily Scan: Mon-Fri at 9:00 AM IST")
        logger.info("  💼 Paper Trading: Mon-Fri at 9:05 AM IST")
        logger.info("  📈 EOD Update: Mon-Fri at 3:35 PM IST")

    def start(self):
        """Start the scheduler."""
        self.schedule_jobs()

        logger.info("=" * 70)
        logger.info("🚀 Trading Scheduler Started")
        logger.info("=" * 70)

        # Print next run times
        self.scheduler.print_jobs()

        logger.info("")
        logger.info("⏰ Scheduler is running. Jobs will execute at their scheduled times.")
        logger.info("")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 70)

        # Start scheduler (blocking)
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("⚠️  Scheduler stopped by user")


def main():
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("Trading Scheduler Daemon v1.0")
    logger.info("=" * 70)

    # Check Python version
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")

    # Initialize and start scheduler
    scheduler = TradingScheduler()
    scheduler.start()


if __name__ == "__main__":
    main()
