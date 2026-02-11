#!/usr/bin/env python3
"""
Daily Smart Money Collector

Runs all scrapers for today's date and stores results in DB.
Designed for cron — logs results, exits 0 on success.

Cron example (run after market close at 6 PM IST, Mon-Fri):
    0 18 * * 1-5 cd /path/to/project && python3 scripts/daily_collector.py >> logs/collector.log 2>&1
"""

import logging
import sys
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._db_utils import SyncDB
from ai.agents.scrapers.nse_bulk_deals import BulkDealsScraper
from ai.agents.scrapers.nse_fii_dii import FIIDIIScraper
from ai.agents.scrapers.nse_corporate_actions import CorporateActionsScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def collect():
    """Run all scrapers for today and store results."""
    today = date.today()

    print(f"[{today.isoformat()}] Daily smart money collection starting...")

    # Connect to DB
    try:
        db = SyncDB()
    except Exception as e:
        print(f"  DB connection failed: {e}")
        sys.exit(1)

    scrapers = [
        ("BulkDeals", BulkDealsScraper()),
        ("FII_DII", FIIDIIScraper()),
        ("CorporateActions", CorporateActionsScraper()),
    ]

    results = []
    had_error = False

    for name, scraper in scrapers:
        try:
            inserted = scraper.run(today, today, db)
            results.append(f"{name}: {inserted} rows")
        except Exception as e:
            results.append(f"{name}: ERROR ({e})")
            logger.error(f"{name} failed: {e}")
            had_error = True

    db.close()

    summary = " | ".join(results)
    print(f"  {summary}")

    # Exit 0 even if some scrapers returned 0 rows (market may be closed).
    # Exit 1 only on DB failure (handled above) or if we want to flag errors.
    if had_error:
        print("  Some scrapers had errors (see above)")
    else:
        print("  Done.")


if __name__ == "__main__":
    collect()
