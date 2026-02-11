"""
Shared sync psycopg2 wrapper for scripts.

Provides a thin DB abstraction matching the execute() interface
that scrapers' store() methods expect.
"""

import logging
import psycopg2

logger = logging.getLogger(__name__)

# Default from config/settings.py
DEFAULT_DB_URL = "postgresql://trading_user:trading_password_dev@localhost:5432/trading_system"


class SyncDB:
    """Thin psycopg2 wrapper matching the execute() interface scrapers expect."""

    def __init__(self, url: str = None):
        self.url = url or DEFAULT_DB_URL
        self.conn = psycopg2.connect(self.url)
        self.conn.autocommit = False
        logger.info("Connected to database")

    def execute(self, query: str, params=None):
        """Execute a query and commit (matches scraper store() interface)."""
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
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
