"""
SQLite audit store for intraday backtest results.

Stores backtest runs, trades, and candle-level evaluation audit trail.
Lightweight — no external dependencies beyond stdlib sqlite3.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/backtest_audit.db"


class SQLiteStore:
    """SQLite-based audit store for backtest trades and candle evaluations."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    params_json TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    total_pnl_points REAL,
                    total_pnl_rupees REAL,
                    total_trades INTEGER,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_time TEXT,
                    exit_price REAL,
                    direction TEXT NOT NULL,
                    pnl_points REAL,
                    pnl_rupees REAL,
                    exit_reason TEXT,
                    lot_size INTEGER,
                    brokerage REAL DEFAULT 0,
                    swing_high REAL,
                    candle_1_low REAL,
                    candle_1_high REAL,
                    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
                );

                CREATE TABLE IF NOT EXISTS candle_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    candle_number INTEGER NOT NULL,
                    time TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    prev_day_open REAL,
                    prev_day_close REAL,
                    prev_day_high REAL,
                    prev_day_low REAL,
                    condition_met INTEGER DEFAULT 0,
                    signal_generated INTEGER DEFAULT 0,
                    notes TEXT,
                    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);
                CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date);
                CREATE INDEX IF NOT EXISTS idx_audit_run ON candle_audit(run_id);
                CREATE INDEX IF NOT EXISTS idx_audit_date ON candle_audit(date);
            """)
            conn.commit()
        finally:
            conn.close()

    def create_run(
        self,
        strategy_name: str,
        params: Dict[str, Any],
        start_date: str,
        end_date: str
    ) -> int:
        """Create a new backtest run and return its ID."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO backtest_runs (strategy_name, params_json, start_date, end_date)
                   VALUES (?, ?, ?, ?)""",
                (strategy_name, json.dumps(params), start_date, end_date)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def update_run_metrics(
        self,
        run_id: int,
        total_pnl_points: float,
        total_pnl_rupees: float,
        total_trades: int,
        win_rate: float,
        sharpe_ratio: float,
        max_drawdown: float
    ):
        """Update a backtest run with final metrics."""
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE backtest_runs
                   SET total_pnl_points=?, total_pnl_rupees=?, total_trades=?,
                       win_rate=?, sharpe_ratio=?, max_drawdown=?
                   WHERE id=?""",
                (total_pnl_points, total_pnl_rupees, total_trades,
                 win_rate, sharpe_ratio, max_drawdown, run_id)
            )
            conn.commit()
        finally:
            conn.close()

    def insert_trade(
        self,
        run_id: int,
        date: str,
        entry_time: str,
        entry_price: float,
        exit_time: Optional[str],
        exit_price: Optional[float],
        direction: str,
        pnl_points: Optional[float],
        pnl_rupees: Optional[float],
        exit_reason: str,
        lot_size: int,
        brokerage: float = 0.0,
        swing_high: Optional[float] = None,
        candle_1_low: Optional[float] = None,
        candle_1_high: Optional[float] = None
    ):
        """Insert a trade record."""
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO trades
                   (run_id, date, entry_time, entry_price, exit_time, exit_price,
                    direction, pnl_points, pnl_rupees, exit_reason, lot_size, brokerage,
                    swing_high, candle_1_low, candle_1_high)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, date, entry_time, entry_price, exit_time, exit_price,
                 direction, pnl_points, pnl_rupees, exit_reason, lot_size, brokerage,
                 swing_high, candle_1_low, candle_1_high)
            )
            conn.commit()
        finally:
            conn.close()

    def insert_candle_audit(
        self,
        run_id: int,
        date: str,
        candle_number: int,
        time: str,
        ohlc: Dict[str, float],
        prev_day: Optional[Dict[str, float]] = None,
        condition_met: bool = False,
        signal_generated: bool = False,
        notes: str = ""
    ):
        """Insert a candle audit record."""
        prev = prev_day or {}
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO candle_audit
                   (run_id, date, candle_number, time, open, high, low, close,
                    prev_day_open, prev_day_close, prev_day_high, prev_day_low,
                    condition_met, signal_generated, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, date, candle_number, time,
                 ohlc.get('open'), ohlc.get('high'), ohlc.get('low'), ohlc.get('close'),
                 prev.get('open'), prev.get('close'), prev.get('high'), prev.get('low'),
                 int(condition_met), int(signal_generated), notes)
            )
            conn.commit()
        finally:
            conn.close()

    def insert_candle_audit_batch(self, records: List[tuple]):
        """Batch insert candle audit records for performance."""
        conn = self._get_conn()
        try:
            conn.executemany(
                """INSERT INTO candle_audit
                   (run_id, date, candle_number, time, open, high, low, close,
                    prev_day_open, prev_day_close, prev_day_high, prev_day_low,
                    condition_met, signal_generated, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                records
            )
            conn.commit()
        finally:
            conn.close()

    # ── Query methods ──

    def get_runs(self) -> pd.DataFrame:
        """Get all backtest runs."""
        conn = self._get_conn()
        try:
            df = pd.read_sql_query("SELECT * FROM backtest_runs ORDER BY created_at DESC", conn)
            return df
        finally:
            conn.close()

    def get_run(self, run_id: int) -> Optional[Dict]:
        """Get a single backtest run."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM backtest_runs WHERE id=?", (run_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_trades(self, run_id: int) -> pd.DataFrame:
        """Get all trades for a backtest run."""
        conn = self._get_conn()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE run_id=? ORDER BY date, entry_time",
                conn, params=(run_id,)
            )
            return df
        finally:
            conn.close()

    def get_candle_audit(self, run_id: int, date: Optional[str] = None) -> pd.DataFrame:
        """Get candle audit records, optionally filtered by date."""
        conn = self._get_conn()
        try:
            if date:
                df = pd.read_sql_query(
                    "SELECT * FROM candle_audit WHERE run_id=? AND date=? ORDER BY candle_number",
                    conn, params=(run_id, date)
                )
            else:
                df = pd.read_sql_query(
                    "SELECT * FROM candle_audit WHERE run_id=? ORDER BY date, candle_number",
                    conn, params=(run_id,)
                )
            return df
        finally:
            conn.close()

    def get_latest_run_id(self) -> Optional[int]:
        """Get the most recent backtest run ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT id FROM backtest_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return row['id'] if row else None
        finally:
            conn.close()

    def export_trades_csv(self, run_id: int, output_path: str):
        """Export trades for a run to CSV."""
        df = self.get_trades(run_id)
        if not df.empty:
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} trades to {output_path}")

    def export_audit_csv(self, run_id: int, output_path: str):
        """Export candle audit for a run to CSV."""
        df = self.get_candle_audit(run_id)
        if not df.empty:
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} audit records to {output_path}")
