"""
FII/DII Flows Scraper — Fetches Foreign & Domestic Institutional Investor daily flows.

Source: https://www.nseindia.com/api/fiidiiTradeReact
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, List, Optional

from ai.agents.scrapers.base_scraper import BaseScraper, NSE_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class FIIDIIFlow:
    """A single FII/DII daily flow record."""
    date: date
    category: str  # FII or DII
    buy_value: float  # in crores
    sell_value: float  # in crores
    net_value: float  # buy_value - sell_value


class FIIDIIScraper(BaseScraper):
    """Scrapes FII/DII daily flow data from NSE."""

    FII_DII_URL = f"{NSE_BASE_URL}/api/fiidiiTradeReact"

    def __init__(self):
        super().__init__(name="FII_DII")

    def fetch(self, target_date: date) -> Optional[Any]:
        """Fetch FII/DII flow data for a given date."""
        try:
            resp = self._rate_limited_get(self.FII_DII_URL)
            data = resp.json()

            if not data:
                return None

            return data

        except Exception as e:
            logger.error(f"❌ FII/DII fetch failed for {target_date}: {e}")
            return None

    def parse(self, raw_data: Any) -> List[FIIDIIFlow]:
        """Parse raw JSON into FIIDIIFlow dataclasses."""
        flows: List[FIIDIIFlow] = []

        records = raw_data if isinstance(raw_data, list) else [raw_data]

        for record in records:
            try:
                category = record.get("category", "").strip().upper()

                # Normalize category names
                if "FII" in category or "FPI" in category:
                    category = "FII"
                elif "DII" in category:
                    category = "DII"
                else:
                    continue

                buy_value = self._parse_number(record.get("buyValue", record.get("buy_value", 0)))
                sell_value = self._parse_number(record.get("sellValue", record.get("sell_value", 0)))
                net_value = self._parse_number(record.get("netValue", record.get("net_value", 0)))

                # If net_value not provided, compute it
                if net_value == 0 and (buy_value != 0 or sell_value != 0):
                    net_value = round(buy_value - sell_value, 2)

                date_str = record.get("date", "")
                flow_date = self._parse_date(date_str) if date_str else date.today()

                flows.append(FIIDIIFlow(
                    date=flow_date,
                    category=category,
                    buy_value=buy_value,
                    sell_value=sell_value,
                    net_value=net_value,
                ))

            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"  Skipping record: {e}")
                continue

        return flows

    def store(self, records: List[FIIDIIFlow], db_manager: Any) -> int:
        """
        Insert FII/DII flow records into the database.

        Args:
            records: List of FIIDIIFlow dataclasses
            db_manager: Object with an execute() method

        Returns:
            Number of rows inserted
        """
        if not records:
            return 0

        query = """
            INSERT INTO fii_dii_flows (date, category, buy_value, sell_value, net_value)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """

        inserted = 0
        for flow in records:
            try:
                db_manager.execute(query, (
                    flow.date, flow.category,
                    flow.buy_value, flow.sell_value, flow.net_value,
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"  Insert skipped: {e}")

        return inserted

    @staticmethod
    def _parse_number(value: Any) -> float:
        """Parse a number that may contain commas or be a string."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.replace(",", "").strip() or "0")
        return 0.0

    @staticmethod
    def _parse_date(date_str: str) -> date:
        """Parse date string from NSE."""
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
    """Example: fetch today's FII/DII flows."""
    from datetime import date as d

    print("=" * 70)
    print("FII/DII FLOWS SCRAPER — Example Usage")
    print("=" * 70)

    scraper = FIIDIIScraper()
    today = d.today()

    print(f"\n📡 Fetching FII/DII flows for {today.isoformat()}...")
    raw = scraper.fetch(today)

    if raw is None:
        print("  No data available.")
        return

    flows = scraper.parse(raw)
    print(f"  Parsed {len(flows)} flow records\n")

    for flow in flows:
        sign = "+" if flow.net_value >= 0 else ""
        print(f"  {flow.category:3s}  Buy: ₹{flow.buy_value:>12,.2f} cr  "
              f"Sell: ₹{flow.sell_value:>12,.2f} cr  "
              f"Net: {sign}₹{flow.net_value:>12,.2f} cr")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    example_usage()
