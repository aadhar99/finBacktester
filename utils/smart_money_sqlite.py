"""
SQLite adapter for Smart Money data storage.

Lightweight alternative to PostgreSQL/TimescaleDB for development and testing.
Stores bulk deals, FII/DII flows, corporate actions, and promoter holdings.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, date

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/smart_money.db"


class SmartMoneySQLite:
    """SQLite-based storage for Smart Money tracking data."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                -- Bulk Deals (institutional trades >0.5% equity)
                CREATE TABLE IF NOT EXISTS bulk_deals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    client_name TEXT NOT NULL,
                    deal_type TEXT NOT NULL CHECK (deal_type IN ('BUY', 'SELL')),
                    quantity INTEGER NOT NULL CHECK (quantity > 0),
                    price REAL NOT NULL CHECK (price > 0),
                    value REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, symbol, client_name, deal_type, quantity)
                );

                CREATE INDEX IF NOT EXISTS idx_bulk_deals_symbol_date
                    ON bulk_deals(symbol, date DESC);
                CREATE INDEX IF NOT EXISTS idx_bulk_deals_client
                    ON bulk_deals(client_name, date DESC);
                CREATE INDEX IF NOT EXISTS idx_bulk_deals_date
                    ON bulk_deals(date DESC);

                -- FII/DII Flows (daily institutional flows)
                CREATE TABLE IF NOT EXISTS fii_dii_flows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    category TEXT NOT NULL CHECK (category IN ('FII', 'DII')),
                    buy_value REAL NOT NULL,
                    sell_value REAL NOT NULL,
                    net_value REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, category)
                );

                CREATE INDEX IF NOT EXISTS idx_fii_dii_date
                    ON fii_dii_flows(date DESC);
                CREATE INDEX IF NOT EXISTS idx_fii_dii_category
                    ON fii_dii_flows(category, date DESC);

                -- Corporate Actions (buybacks, dividends, board meetings)
                CREATE TABLE IF NOT EXISTS corporate_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    announcement_date TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    description TEXT,
                    board_meeting_date TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, announcement_date, action_type, description)
                );

                CREATE INDEX IF NOT EXISTS idx_corporate_actions_symbol_date
                    ON corporate_actions(symbol, announcement_date DESC);
                CREATE INDEX IF NOT EXISTS idx_corporate_actions_date
                    ON corporate_actions(announcement_date DESC);
                CREATE INDEX IF NOT EXISTS idx_corporate_actions_type
                    ON corporate_actions(action_type, announcement_date DESC);

                -- Promoter Holdings (quarterly stake changes)
                CREATE TABLE IF NOT EXISTS promoter_holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quarter TEXT NOT NULL,
                    filing_date TEXT NOT NULL,
                    promoter_pct REAL NOT NULL,
                    pledge_pct REAL,
                    change_from_prev_quarter REAL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, quarter)
                );

                CREATE INDEX IF NOT EXISTS idx_promoter_holdings_symbol
                    ON promoter_holdings(symbol, filing_date DESC);
            """)
            conn.commit()
            logger.info(f"✅ Smart Money SQLite initialized: {self.db_path}")
        finally:
            conn.close()

    def execute(self, query: str, params: tuple = ()):
        """
        Execute a query (for compatibility with PostgreSQL-style scrapers).

        This mimics the interface expected by scraper.store() methods.
        """
        conn = self._get_conn()
        try:
            # Convert PostgreSQL placeholders (%s) to SQLite placeholders (?)
            if "%s" in query:
                query = query.replace("%s", "?")

            # Handle ON CONFLICT DO NOTHING (PostgreSQL syntax) -> INSERT OR IGNORE (SQLite)
            if "ON CONFLICT DO NOTHING" in query:
                query = query.replace("ON CONFLICT DO NOTHING", "")
                query = query.replace("INSERT INTO", "INSERT OR IGNORE INTO")

            conn.execute(query, params)
            conn.commit()
        except sqlite3.IntegrityError:
            # Duplicate - ignore silently
            pass
        except Exception as e:
            logger.error(f"Execute error: {e}")
            raise
        finally:
            conn.close()

    def close(self):
        """Close database (no-op for SQLite, kept for compatibility)."""
        pass

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_bulk_deals(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get bulk deals with optional filters."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM bulk_deals WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date DESC LIMIT ?"
            params.append(limit)

            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()

    def get_fii_dii_flows(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None
    ) -> pd.DataFrame:
        """Get FII/DII flows with optional filters."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM fii_dii_flows WHERE 1=1"
            params = []

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            if category:
                query += " AND category = ?"
                params.append(category)

            query += " ORDER BY date DESC"

            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()

    def get_corporate_actions(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        action_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Get corporate actions with optional filters."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM corporate_actions WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if start_date:
                query += " AND announcement_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND announcement_date <= ?"
                params.append(end_date)
            if action_type:
                query += " AND action_type = ?"
                params.append(action_type)

            query += " ORDER BY announcement_date DESC"

            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        conn = self._get_conn()
        try:
            stats = {}

            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM bulk_deals")
            stats['bulk_deals'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM fii_dii_flows")
            stats['fii_dii_flows'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM corporate_actions")
            stats['corporate_actions'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM promoter_holdings")
            stats['promoter_holdings'] = cursor.fetchone()[0]

            # Date ranges
            cursor.execute("SELECT MIN(date), MAX(date) FROM bulk_deals")
            result = cursor.fetchone()
            if result[0]:
                stats['bulk_deals_date_range'] = f"{result[0]} to {result[1]}"

            return stats
        finally:
            conn.close()

    def get_accumulation_stocks(
        self,
        start_date: str,
        end_date: str,
        min_net_value: float = 1_00_00_000
    ) -> List[Dict[str, Any]]:
        """
        Get stocks being accumulated (net buying > threshold).

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_net_value: Minimum net buying value (default ₹1 crore)

        Returns:
            List of {symbol, net_value, buy_value, sell_value, num_deals}
        """
        conn = self._get_conn()
        try:
            query = """
                SELECT
                    symbol,
                    SUM(CASE WHEN deal_type = 'BUY' THEN value ELSE 0 END) as buy_value,
                    SUM(CASE WHEN deal_type = 'SELL' THEN value ELSE 0 END) as sell_value,
                    SUM(CASE WHEN deal_type = 'BUY' THEN value ELSE -value END) as net_value,
                    COUNT(*) as num_deals
                FROM bulk_deals
                WHERE date >= ? AND date <= ?
                GROUP BY symbol
                HAVING net_value >= ?
                ORDER BY net_value DESC
            """

            cursor = conn.cursor()
            cursor.execute(query, (start_date, end_date, min_net_value))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'symbol': row[0],
                    'buy_value': row[1],
                    'sell_value': row[2],
                    'net_value': row[3],
                    'num_deals': row[4]
                })

            return results
        finally:
            conn.close()

    def get_top_buyers(
        self,
        start_date: str,
        end_date: str,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top buyers by total value."""
        conn = self._get_conn()
        try:
            query = """
                SELECT
                    client_name,
                    SUM(value) as total_value,
                    COUNT(*) as num_deals
                FROM bulk_deals
                WHERE date >= ? AND date <= ? AND deal_type = 'BUY'
                GROUP BY client_name
                ORDER BY total_value DESC
                LIMIT ?
            """

            cursor = conn.cursor()
            cursor.execute(query, (start_date, end_date, top_n))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'client_name': row[0],
                    'total_value': row[1],
                    'num_deals': row[2]
                })

            return results
        finally:
            conn.close()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    db = SmartMoneySQLite()

    print("\n📊 Smart Money SQLite Stats:")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
