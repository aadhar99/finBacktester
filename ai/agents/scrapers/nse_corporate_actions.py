"""
Corporate Actions Scraper — Fetches corporate announcements from NSE.

Source: https://www.nseindia.com/api/corporate-announcements
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, List, Optional

from ai.agents.scrapers.base_scraper import BaseScraper, NSE_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class CorporateAction:
    """A single corporate action announcement."""
    symbol: str
    announcement_date: date
    action_type: str
    description: str
    subject: str


class CorporateActionsScraper(BaseScraper):
    """Scrapes corporate action announcements from NSE."""

    CORP_ACTIONS_URL = f"{NSE_BASE_URL}/api/corporate-announcements"

    def __init__(self):
        super().__init__(name="CorporateActions")

    def fetch(self, target_date: date) -> Optional[Any]:
        """Fetch corporate announcements for a given date."""
        params = {
            "index": "equities",
            "from_date": target_date.strftime("%d-%m-%Y"),
            "to_date": target_date.strftime("%d-%m-%Y"),
        }

        try:
            resp = self._rate_limited_get(self.CORP_ACTIONS_URL, params=params)
            data = resp.json()

            if isinstance(data, dict):
                records = data.get("data", data.get("result", []))
            elif isinstance(data, list):
                records = data
            else:
                return None

            if not records:
                return None

            return records

        except Exception as e:
            logger.error(f"❌ CorporateActions fetch failed for {target_date}: {e}")
            return None

    def parse(self, raw_data: Any) -> List[CorporateAction]:
        """Parse raw JSON into CorporateAction dataclasses.

        NSE fields: symbol, desc, an_dt, attchmntText, sm_name, sm_isin, ...
        """
        actions: List[CorporateAction] = []

        for record in raw_data:
            try:
                symbol = record.get("symbol", record.get("sm_name", "")).strip()
                # "subject" is not in the NSE response; use attchmntText or desc
                subject = record.get("subject", record.get("attchmntText", record.get("sub", ""))).strip()
                description = record.get("desc", record.get("description", "")).strip()
                action_type = self._classify_action(subject, description)

                date_str = record.get("an_dt", record.get("date", record.get("announcementDate", "")))
                ann_date = self._parse_date(date_str) if date_str else date.today()

                if not symbol:
                    continue

                actions.append(CorporateAction(
                    symbol=symbol,
                    announcement_date=ann_date,
                    action_type=action_type,
                    description=description,
                    subject=subject,
                ))

            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"  Skipping record: {e}")
                continue

        return actions

    def store(self, records: List[CorporateAction], db_manager: Any) -> int:
        """
        Insert corporate action records into the database.

        Args:
            records: List of CorporateAction dataclasses
            db_manager: Object with an execute() method

        Returns:
            Number of rows inserted
        """
        if not records:
            return 0

        query = """
            INSERT INTO corporate_actions (symbol, announcement_date, action_type, description, subject)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """

        inserted = 0
        for action in records:
            try:
                db_manager.execute(query, (
                    action.symbol, action.announcement_date,
                    action.action_type, action.description, action.subject,
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"  Insert skipped: {e}")

        return inserted

    @staticmethod
    def _classify_action(subject: str, description: str) -> str:
        """Classify a corporate action based on subject/description keywords."""
        text = (subject + " " + description).lower()

        if "dividend" in text:
            return "DIVIDEND"
        elif "buyback" in text or "buy back" in text:
            return "BUYBACK"
        elif "board meeting" in text:
            return "BOARD_MEETING"
        elif "bonus" in text:
            return "BONUS"
        elif "split" in text:
            return "STOCK_SPLIT"
        elif "rights" in text:
            return "RIGHTS_ISSUE"
        elif "agm" in text or "annual general" in text:
            return "AGM"
        elif "merger" in text or "amalgamation" in text:
            return "MERGER"
        else:
            return "OTHER"

    @staticmethod
    def _parse_date(date_str: str) -> date:
        """Parse date string from NSE (supports datetime and date-only formats)."""
        from datetime import datetime

        # Datetime formats first (NSE an_dt: "06-Feb-2026 23:44:34")
        for fmt in (
            "%d-%b-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%d-%b-%Y",
            "%d-%m-%Y",
            "%Y-%m-%d",
            "%d/%m/%Y",
        ):
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue

        raise ValueError(f"Cannot parse date: {date_str}")


# ============================================================================
# STANDALONE USAGE
# ============================================================================

def example_usage():
    """Example: fetch today's corporate actions."""
    from datetime import date as d

    print("=" * 70)
    print("CORPORATE ACTIONS SCRAPER — Example Usage")
    print("=" * 70)

    scraper = CorporateActionsScraper()
    today = d.today()

    print(f"\n📡 Fetching corporate actions for {today.isoformat()}...")
    raw = scraper.fetch(today)

    if raw is None:
        print("  No data available.")
        return

    actions = scraper.parse(raw)
    print(f"  Parsed {len(actions)} actions\n")

    for a in actions[:10]:
        print(f"  {a.symbol:12s} [{a.action_type:15s}] {a.subject[:60]}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    example_usage()
