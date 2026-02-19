"""
Shared sync psycopg2 wrapper for scripts.

Provides a thin DB abstraction matching the execute() interface
that scrapers' store() methods expect.

Auto-detects and falls back to SQLite if PostgreSQL is not available.
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# Default from config/settings.py
DEFAULT_DB_URL = "postgresql://trading_user:trading_password_dev@localhost:5432/trading_system"


class SyncDB:
    """Thin database wrapper matching the execute() interface scrapers expect.

    Auto-detects database type:
    - PostgreSQL if available
    - SQLite as fallback
    """

    def __init__(self, url: str = None, use_sqlite: bool = None):
        """
        Initialize database connection.

        Args:
            url: Database URL (PostgreSQL format)
            use_sqlite: Force SQLite usage. If None, auto-detects.
        """
        self.url = url or DEFAULT_DB_URL
        self.use_sqlite = use_sqlite
        self.conn = None

        # Auto-detect if not specified
        if self.use_sqlite is None:
            self.use_sqlite = not self._postgres_available()

        if self.use_sqlite:
            self._connect_sqlite()
        else:
            self._connect_postgres()

    def _postgres_available(self) -> bool:
        """Check if PostgreSQL is available."""
        try:
            import psycopg2
            test_conn = psycopg2.connect(self.url)
            test_conn.close()
            return True
        except:
            return False

    def _connect_postgres(self):
        """Connect to PostgreSQL."""
        try:
            import psycopg2
            self.conn = psycopg2.connect(self.url)
            self.conn.autocommit = False
            self.db_type = 'postgresql'
            logger.info("✅ Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            logger.info("Falling back to SQLite...")
            self.use_sqlite = True
            self._connect_sqlite()

    def _connect_sqlite(self):
        """Connect to SQLite."""
        from utils.smart_money_sqlite import SmartMoneySQLite
        self.conn = SmartMoneySQLite()
        self.db_type = 'sqlite'
        logger.info("✅ Connected to SQLite database")

    def execute(self, query: str, params=None):
        """Execute a query and commit (matches scraper store() interface)."""
        if self.use_sqlite:
            # SQLite SmartMoneySQLite has its own execute method
            self.conn.execute(query, params or ())
        else:
            # PostgreSQL psycopg2
            cur = self.conn.cursor()
            cur.execute(query, params)
            self.conn.commit()
            cur.close()

    def fetchall(self, query: str, params=None):
        """Execute a query and return all rows."""
        cur = self.conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetchone(self, query: str, params=None):
        """Execute a query and return one row."""
        cur = self.conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        cur.close()
        return row

    def commit(self):
        """Explicit commit."""
        self.conn.commit()

    def close(self):
        """Close the connection."""
        if self.use_sqlite:
            self.conn.close()
            logger.info("SQLite connection closed")
        else:
            if self.conn and not self.conn.closed:
                self.conn.close()
                logger.info("PostgreSQL connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
