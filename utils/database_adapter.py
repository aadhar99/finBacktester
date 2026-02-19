"""
Database Adapter - SQLite (local) or PostgreSQL (cloud)

Automatically detects environment and uses appropriate database:
- Local: SQLite (data/smart_money.db)
- Railway/Cloud: PostgreSQL (from DATABASE_URL env var)

Usage:
    from utils.database_adapter import get_database_url, DatabaseAdapter

    # Get connection string
    db_url = get_database_url()

    # Or use adapter
    db = DatabaseAdapter()
    db.execute_query("SELECT * FROM bulk_deals LIMIT 10")
"""

import os
import logging
from typing import Optional, Dict, List, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """
    Get database URL from environment or default to SQLite.

    Returns:
        Database connection string
    """
    # Check for Railway/Cloud database URL
    db_url = os.getenv('DATABASE_URL')

    if db_url:
        # PostgreSQL (Cloud)
        # Fix for SQLAlchemy 1.4+ (postgres:// -> postgresql://)
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)

        logger.info("✅ Using PostgreSQL (Cloud)")
        return db_url
    else:
        # SQLite (Local)
        logger.info("✅ Using SQLite (Local)")
        return 'sqlite:///data/smart_money.db'


def is_postgres() -> bool:
    """Check if using PostgreSQL."""
    return os.getenv('DATABASE_URL') is not None


class DatabaseAdapter:
    """
    Database adapter that works with both SQLite and PostgreSQL.

    Handles connection pooling, query execution, and migrations.
    """

    def __init__(self, echo: bool = False):
        """
        Initialize database adapter.

        Args:
            echo: Enable SQL query logging
        """
        self.db_url = get_database_url()
        self.is_postgres = is_postgres()

        # Create engine with appropriate settings
        if self.is_postgres:
            # PostgreSQL settings
            self.engine = create_engine(
                self.db_url,
                echo=echo,
                pool_pre_ping=True,  # Verify connections
                pool_recycle=3600,   # Recycle connections after 1 hour
                pool_size=5,
                max_overflow=10
            )
        else:
            # SQLite settings
            self.engine = create_engine(
                self.db_url,
                echo=echo,
                connect_args={'check_same_thread': False},
                poolclass=NullPool  # SQLite doesn't benefit from pooling
            )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        logger.info(f"✅ Database adapter initialized: {self.db_url.split('@')[0]}")

    def get_session(self) -> Session:
        """
        Get database session.

        Returns:
            SQLAlchemy session
        """
        return self.SessionLocal()

    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute raw SQL query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of result rows as dictionaries
        """
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def execute_update(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Execute INSERT/UPDATE/DELETE query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Number of affected rows
        """
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            session.commit()
            return result.rowcount

    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists.

        Args:
            table_name: Name of table to check

        Returns:
            True if table exists
        """
        if self.is_postgres:
            query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :table_name
                )
            """
        else:
            query = """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name = :table_name
            """

        result = self.execute_query(query, {'table_name': table_name})
        return bool(result and result[0])

    def migrate_sqlite_to_postgres(self, sqlite_path: str = 'data/smart_money.db'):
        """
        Migrate data from SQLite to PostgreSQL.

        Args:
            sqlite_path: Path to SQLite database file

        Raises:
            ValueError: If not using PostgreSQL
        """
        if not self.is_postgres:
            raise ValueError("Can only migrate TO PostgreSQL")

        import sqlite3
        import json

        logger.info(f"📦 Migrating from SQLite: {sqlite_path}")

        # Connect to SQLite
        sqlite_conn = sqlite3.connect(sqlite_path)
        sqlite_conn.row_factory = sqlite3.Row
        sqlite_cursor = sqlite_conn.cursor()

        # Get all tables
        sqlite_cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in sqlite_cursor.fetchall()]

        logger.info(f"  Found {len(tables)} tables: {', '.join(tables)}")

        # Migrate each table
        for table in tables:
            logger.info(f"  Migrating table: {table}")

            # Get data from SQLite
            sqlite_cursor.execute(f"SELECT * FROM {table}")
            rows = sqlite_cursor.fetchall()

            if not rows:
                logger.info(f"    ℹ️  Table {table} is empty, skipping")
                continue

            # Get column names
            columns = [desc[0] for desc in sqlite_cursor.description]

            # Insert into PostgreSQL
            with self.get_session() as session:
                for row in rows:
                    # Convert row to dict
                    row_dict = dict(zip(columns, row))

                    # Build INSERT query
                    cols = ', '.join(columns)
                    placeholders = ', '.join([f':{col}' for col in columns])
                    query = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"

                    try:
                        session.execute(text(query), row_dict)
                    except Exception as e:
                        logger.warning(f"    ⚠️  Failed to insert row: {e}")

                session.commit()

            logger.info(f"    ✅ Migrated {len(rows)} rows")

        sqlite_conn.close()
        logger.info("✅ Migration complete!")

    def close(self):
        """Close database connections."""
        self.engine.dispose()
        logger.info("✅ Database connections closed")


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test adapter
    db = DatabaseAdapter()

    # Test query
    try:
        result = db.execute_query("SELECT 1 as test")
        print(f"✅ Database connection successful: {result}")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

    # Show database type
    print(f"\nDatabase Type: {'PostgreSQL' if db.is_postgres else 'SQLite'}")
    print(f"Connection URL: {db.db_url.split('@')[0]}")

    db.close()
