"""
Base Scraper — Abstract base class for all NSE data scrapers.

Provides:
- requests.Session with NSE-specific headers and cookie initialization
- Rate limiting (1 request per 2 seconds)
- Retry with exponential backoff
- fetch → parse → store pipeline via run()
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Any, List

import requests

logger = logging.getLogger(__name__)

NSE_BASE_URL = "https://www.nseindia.com"
NSE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Rate limiting: minimum seconds between requests
RATE_LIMIT_SECONDS = 2.0

# Retry config
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds — exponential backoff: 2s, 4s, 8s


class BaseScraper(ABC):
    """Abstract base class for NSE data scrapers."""

    def __init__(self, name: str):
        self.name = name
        self.session: requests.Session | None = None
        self._last_request_time: float = 0.0
        logger.info(f"✅ {self.name} scraper initialized")

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch(self, target_date: date) -> Any:
        """Fetch raw data for a given date from NSE. Returns raw JSON/HTML/bytes."""

    @abstractmethod
    def parse(self, raw_data: Any) -> List[Any]:
        """Parse raw data into a list of dataclass records."""

    @abstractmethod
    def store(self, records: List[Any], db_manager: Any) -> int:
        """Store parsed records in the database. Returns row count inserted."""

    # ------------------------------------------------------------------
    # Concrete: orchestrator
    # ------------------------------------------------------------------

    def run(self, start_date: date, end_date: date, db_manager: Any) -> int:
        """
        Run the full fetch → parse → store pipeline for a date range.

        Args:
            start_date: First date to fetch (inclusive)
            end_date: Last date to fetch (inclusive)
            db_manager: DatabaseManager instance (uses sync execute for store)

        Returns:
            Total rows inserted across all dates
        """
        total_inserted = 0
        current = start_date

        while current <= end_date:
            try:
                logger.info(f"📡 {self.name}: fetching {current.isoformat()}")
                raw = self.fetch(current)

                if raw is None:
                    logger.debug(f"  ⏭️  No data for {current.isoformat()}")
                    current += timedelta(days=1)
                    continue

                records = self.parse(raw)
                if not records:
                    logger.debug(f"  ⏭️  0 records parsed for {current.isoformat()}")
                    current += timedelta(days=1)
                    continue

                inserted = self.store(records, db_manager)
                total_inserted += inserted
                logger.info(f"  ✅ {inserted} rows inserted for {current.isoformat()}")

            except Exception as e:
                logger.error(f"  ❌ {self.name} failed for {current.isoformat()}: {e}")

            current += timedelta(days=1)

        logger.info(f"📊 {self.name}: total {total_inserted} rows inserted "
                     f"({start_date.isoformat()} → {end_date.isoformat()})")
        return total_inserted

    # ------------------------------------------------------------------
    # NSE session management
    # ------------------------------------------------------------------

    def _get_nse_session(self) -> requests.Session:
        """
        Get or create a requests.Session with NSE cookies.

        NSE requires a valid session cookie obtained by visiting the homepage first.
        """
        if self.session is not None:
            return self.session

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": NSE_USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{NSE_BASE_URL}/",
        })

        # Visit homepage to get session cookies
        try:
            logger.debug(f"🔑 {self.name}: initializing NSE session cookies")
            resp = self.session.get(NSE_BASE_URL, timeout=10)
            resp.raise_for_status()
            logger.debug(f"🔑 {self.name}: NSE session ready")
        except requests.RequestException as e:
            logger.warning(f"⚠️  {self.name}: cookie init failed: {e}")

        return self.session

    # ------------------------------------------------------------------
    # Rate-limited GET with retries
    # ------------------------------------------------------------------

    def _rate_limited_get(self, url: str, params: dict | None = None) -> requests.Response:
        """
        Perform a GET request with rate limiting and exponential backoff retries.

        Args:
            url: Target URL
            params: Optional query parameters

        Returns:
            requests.Response

        Raises:
            requests.RequestException after all retries exhausted
        """
        session = self._get_nse_session()

        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_SECONDS:
            sleep_for = RATE_LIMIT_SECONDS - elapsed
            logger.debug(f"⏳ Rate limit: sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)

        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._last_request_time = time.time()
                resp = session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                return resp

            except requests.RequestException as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"⚠️  {self.name}: attempt {attempt}/{MAX_RETRIES} failed "
                        f"({e}), retrying in {delay}s"
                    )
                    time.sleep(delay)

                    # Re-init session on connection errors
                    if isinstance(e, requests.ConnectionError):
                        self.session = None

        raise last_error  # type: ignore[misc]
