#!/usr/bin/env python3
"""
Backfill Smart Money Data

Fetches up to 6 months of historical smart money data and stores in DB.
Iterates week-by-week chunks (NSE limits date ranges), calling each
scraper's run(start, end, db) method.

Usage:
    python3 scripts/backfill_smart_money.py                          # default: 180 days, all scrapers
    python3 scripts/backfill_smart_money.py --days 7                 # last 7 days
    python3 scripts/backfill_smart_money.py --scraper bulk_deals     # one scraper only
    python3 scripts/backfill_smart_money.py --scraper fii_dii --days 30
"""

import argparse
import logging
import sys
from datetime import date, timedelta
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

SCRAPER_MAP = {
    "bulk_deals": BulkDealsScraper,
    "fii_dii": FIIDIIScraper,
    "corporate_actions": CorporateActionsScraper,
}

CHUNK_DAYS = 7  # process one week at a time


def backfill(days: int, scraper_names: list[str]):
    """Run the backfill for the given number of days and scrapers."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    print("=" * 70)
    print("SMART MONEY BACKFILL")
    print("=" * 70)
    print(f"  Range : {start_date} -> {end_date} ({days} days)")
    print(f"  Scrapers: {', '.join(scraper_names)}")
    print("=" * 70)

    # Connect to DB
    try:
        db = SyncDB()
    except Exception as e:
        print(f"\n  DB connection failed: {e}")
        sys.exit(1)

    # Instantiate requested scrapers
    scrapers = {}
    for name in scraper_names:
        cls = SCRAPER_MAP.get(name)
        if cls is None:
            print(f"  Unknown scraper: {name}")
            continue
        scrapers[name] = cls()

    # Process week-by-week chunks
    totals = {name: 0 for name in scrapers}
    chunk_start = start_date

    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end_date)
        print(f"\n  Chunk: {chunk_start} -> {chunk_end}")

        for name, scraper in scrapers.items():
            try:
                inserted = scraper.run(chunk_start, chunk_end, db)
                totals[name] += inserted
                print(f"    {name}: +{inserted} rows")
            except Exception as e:
                logger.error(f"    {name}: ERROR - {e}")

        chunk_start = chunk_end + timedelta(days=1)

    # Summary
    db.close()

    print("\n" + "=" * 70)
    print("BACKFILL SUMMARY")
    print("=" * 70)
    for name, count in totals.items():
        print(f"  {name:20s}: {count:>6} rows inserted")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Backfill smart money data into DB")
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days to backfill (default: 180)",
    )
    parser.add_argument(
        "--scraper",
        type=str,
        default="all",
        choices=list(SCRAPER_MAP.keys()) + ["all"],
        help="Which scraper to run (default: all)",
    )
    args = parser.parse_args()

    if args.scraper == "all":
        scraper_names = list(SCRAPER_MAP.keys())
    else:
        scraper_names = [args.scraper]

    backfill(args.days, scraper_names)


if __name__ == "__main__":
    main()
