"""
Bulk Deals Scraper — Fetches NSE bulk/block deal data.

Primary endpoint (snapshot, returns latest trading day):
    https://www.nseindia.com/api/snapshot-capital-market-largedeal

The old historical endpoint (/api/historical/bulk-deals) was retired by NSE.
The snapshot endpoint always returns the most recent trading day's data
regardless of date parameters, so historical backfill for bulk deals is
limited to the current day.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, List, Optional

from ai.agents.scrapers.base_scraper import BaseScraper, NSE_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class BulkDeal:
    """A single bulk/block deal record."""
    date: date
    symbol: str
    client_name: str
    deal_type: str  # BUY or SELL
    quantity: int
    price: float
    value: float  # quantity * price


class BulkDealsScraper(BaseScraper):
    """Scrapes NSE bulk/block deal data."""

    SNAPSHOT_URL = f"{NSE_BASE_URL}/api/snapshot-capital-market-largedeal"

    def __init__(self):
        super().__init__(name="BulkDeals")
        # Cache: the snapshot only returns the latest day, so avoid
        # re-fetching the same data on every iteration of run().
        self._snapshot_cache: Optional[dict] = None
        self._snapshot_date: Optional[str] = None

    def fetch(self, target_date: date) -> Optional[Any]:
        """Fetch bulk deals from NSE snapshot API.

        The snapshot endpoint returns the most recent trading day's data.
        If target_date matches the snapshot date we return the records;
        otherwise we return None (data not available for that date).
        """
        try:
            # Re-use cached snapshot within the same run()
            if self._snapshot_cache is None:
                resp = self._rate_limited_get(self.SNAPSHOT_URL)
                self._snapshot_cache = resp.json()
                self._snapshot_date = self._snapshot_cache.get("as_on_date", "")
                logger.info(f"  Snapshot as_on_date: {self._snapshot_date}")

            # Extract bulk deal records
            records = self._snapshot_cache.get("BULK_DEALS_DATA", [])
            if not records:
                return None

            # Filter to target_date — snapshot date is "DD-Mon-YYYY"
            try:
                snapshot_dt = self._parse_date(self._snapshot_date)
            except ValueError:
                # If we can't parse, return all records (best-effort)
                return records

            if target_date != snapshot_dt:
                logger.debug(
                    f"  ⏭️  Snapshot is for {snapshot_dt}, skipping {target_date}"
                )
                return None

            return records

        except Exception as e:
            logger.error(f"❌ BulkDeals fetch failed for {target_date}: {e}")
            return None

    def parse(self, raw_data: Any) -> List[BulkDeal]:
        """Parse raw JSON into BulkDeal dataclasses.

        Handles both legacy field names and current snapshot format:
            snapshot: buySell, qty, watp, clientName, date (DD-Mon-YYYY)
            legacy:   tradeType, quantity, avgPrice, clientName, dealDate
        """
        deals: List[BulkDeal] = []

        for record in raw_data:
            try:
                symbol = record.get("symbol", "").strip()
                client_name = record.get("clientName", record.get("client_name", "")).strip()

                # deal type: snapshot uses "buySell", legacy uses "tradeType"
                deal_type = record.get(
                    "buySell",
                    record.get("tradeType", record.get("deal_type", "")),
                ).strip().upper()

                # quantity: snapshot uses "qty" (string), legacy uses "quantity"
                raw_qty = record.get("qty", record.get("quantity", 0))
                quantity = int(str(raw_qty).replace(",", "").strip() or "0")

                # price: snapshot uses "watp" (weighted avg trade price), legacy uses "avgPrice"
                raw_price = record.get("watp", record.get("avgPrice", record.get("price", 0)))
                price = float(str(raw_price).replace(",", "").strip() or "0")

                # date
                deal_date_str = record.get("dealDate", record.get("date", ""))
                deal_date = self._parse_date(deal_date_str)

                # Validate
                if not symbol or quantity <= 0 or price <= 0:
                    continue

                if deal_type not in ("BUY", "SELL"):
                    continue

                value = round(quantity * price, 2)

                deals.append(BulkDeal(
                    date=deal_date,
                    symbol=symbol,
                    client_name=client_name,
                    deal_type=deal_type,
                    quantity=quantity,
                    price=price,
                    value=value,
                ))

            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"  Skipping record: {e}")
                continue

        return deals

    def store(self, records: List[BulkDeal], db_manager: Any) -> int:
        """
        Insert bulk deal records into the database.

        Uses ON CONFLICT DO NOTHING for deduplication.

        Args:
            records: List of BulkDeal dataclasses
            db_manager: Object with an execute() method (sync psycopg2 connection or similar)

        Returns:
            Number of rows inserted
        """
        if not records:
            return 0

        query = """
            INSERT INTO bulk_deals (date, symbol, client_name, deal_type, quantity, price, value)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """

        inserted = 0
        for deal in records:
            try:
                db_manager.execute(query, (
                    deal.date, deal.symbol, deal.client_name,
                    deal.deal_type, deal.quantity, deal.price, deal.value,
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"  Insert skipped: {e}")

        return inserted

    @staticmethod
    def _parse_date(date_str: str) -> date:
        """Parse date string from NSE (supports multiple formats)."""
        from datetime import datetime

        for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue

        raise ValueError(f"Cannot parse date: {date_str}")


# ============================================================================
# STANDALONE USAGE
# ============================================================================

def example_usage():
    """Example: fetch today's bulk deals (prints raw, does not store)."""
    import json
    from datetime import date as d

    print("=" * 70)
    print("BULK DEALS SCRAPER — Example Usage")
    print("=" * 70)

    scraper = BulkDealsScraper()
    today = d.today()

    print(f"\n📡 Fetching bulk deals for {today.isoformat()}...")
    raw = scraper.fetch(today)

    if raw is None:
        print("  No data available (market may be closed).")
        return

    deals = scraper.parse(raw)
    print(f"  Parsed {len(deals)} deals\n")

    for deal in deals[:5]:
        print(f"  {deal.symbol:12s} {deal.deal_type:4s} {deal.quantity:>12,} "
              f"@ ₹{deal.price:>10,.2f}  ({deal.client_name})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    example_usage()
