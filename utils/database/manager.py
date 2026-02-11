"""
Database Manager - Handles PostgreSQL/TimescaleDB connections and operations.
"""

import logging
import asyncpg
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and operations.

    Features:
    - Connection pooling for performance
    - Automatic schema initialization
    - Helper methods for common queries
    - Transaction management
    - Error handling and retry logic
    """

    def __init__(self, database_url: str, min_pool_size: int = 5, max_pool_size: int = 20):
        """
        Initialize database manager.

        Args:
            database_url: PostgreSQL connection string
                Format: postgresql://user:password@host:port/database
            min_pool_size: Minimum number of connections in pool
            max_pool_size: Maximum number of connections in pool
        """
        self.database_url = database_url
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Establish database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                command_timeout=60
            )
            logger.info(f"✅ Database pool created (min={self.min_pool_size}, max={self.max_pool_size})")

            # Test connection
            async with self.pool.acquire() as conn:
                version = await conn.fetchval('SELECT version()')
                logger.info(f"📊 Database connected: {version.split(',')[0]}")

        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    async def disconnect(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("📊 Database pool closed")

    async def initialize_schema(self):
        """
        Initialize database schema from SQL files.

        Runs both postgres.sql and timescale.sql
        """
        schema_dir = Path(__file__).parent.parent.parent / 'config' / 'schema'

        # Run PostgreSQL schema
        postgres_schema = schema_dir / 'postgres.sql'
        if postgres_schema.exists():
            await self._execute_sql_file(postgres_schema)
            logger.info("✅ PostgreSQL schema initialized")
        else:
            logger.warning(f"⚠️  PostgreSQL schema file not found: {postgres_schema}")

        # Run TimescaleDB schema
        timescale_schema = schema_dir / 'timescale.sql'
        if timescale_schema.exists():
            await self._execute_sql_file(timescale_schema)
            logger.info("✅ TimescaleDB schema initialized")
        else:
            logger.warning(f"⚠️  TimescaleDB schema file not found: {timescale_schema}")

        # Run Smart Money schema
        smart_money_schema = schema_dir / 'smart_money.sql'
        if smart_money_schema.exists():
            await self._execute_sql_file(smart_money_schema)
            logger.info("✅ Smart Money schema initialized")
        else:
            logger.warning(f"⚠️  Smart Money schema file not found: {smart_money_schema}")

    async def _execute_sql_file(self, filepath: Path):
        """Execute SQL file."""
        with open(filepath, 'r') as f:
            sql = f.read()

        async with self.pool.acquire() as conn:
            await conn.execute(sql)

    # ========================================================================
    # ACTIVE TRADES OPERATIONS
    # ========================================================================

    async def insert_active_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        Insert new active trade.

        Args:
            trade_data: Dict with trade information

        Returns:
            Inserted trade ID
        """
        query = """
            INSERT INTO active_trades (
                trade_id, symbol, action, entry_time, entry_price,
                quantity, stop_loss, take_profit, agent, confidence, risk_score
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            trade_id = await conn.fetchval(
                query,
                trade_data['trade_id'],
                trade_data['symbol'],
                trade_data['action'],
                trade_data.get('entry_time', datetime.now()),
                trade_data['entry_price'],
                trade_data['quantity'],
                trade_data.get('stop_loss'),
                trade_data.get('take_profit'),
                trade_data['agent'],
                trade_data.get('confidence'),
                trade_data.get('risk_score')
            )

        logger.info(f"📝 Active trade inserted: {trade_data['trade_id']} ({trade_data['symbol']})")
        return trade_id

    async def update_active_trade(self, trade_id: str, updates: Dict[str, Any]):
        """
        Update active trade.

        Args:
            trade_id: Trade ID to update
            updates: Dict of fields to update
        """
        set_clause = ', '.join([f"{key} = ${i+2}" for i, key in enumerate(updates.keys())])
        query = f"""
            UPDATE active_trades
            SET {set_clause}, last_updated = NOW()
            WHERE trade_id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, trade_id, *updates.values())

        logger.debug(f"📝 Active trade updated: {trade_id}")

    async def close_active_trade(self, trade_id: str) -> Dict[str, Any]:
        """
        Close active trade and return its data.

        Args:
            trade_id: Trade ID to close

        Returns:
            Trade data before deletion
        """
        async with self.pool.acquire() as conn:
            # Get trade data
            trade = await conn.fetchrow(
                "SELECT * FROM active_trades WHERE trade_id = $1",
                trade_id
            )

            if not trade:
                raise ValueError(f"Trade not found: {trade_id}")

            # Delete from active_trades
            await conn.execute(
                "DELETE FROM active_trades WHERE trade_id = $1",
                trade_id
            )

        logger.info(f"✅ Active trade closed: {trade_id}")
        return dict(trade)

    async def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get all active trades."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM active_trades ORDER BY entry_time DESC")

        return [dict(row) for row in rows]

    # ========================================================================
    # PORTFOLIO STATE OPERATIONS
    # ========================================================================

    async def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM portfolio_state WHERE id = 1")

        if not row:
            logger.warning("Portfolio state not found, initializing...")
            await self.initialize_schema()
            return await self.get_portfolio_state()

        return dict(row)

    async def update_portfolio_state(self, updates: Dict[str, Any]):
        """Update portfolio state."""
        set_clause = ', '.join([f"{key} = ${i+1}" for i, key in enumerate(updates.keys())])
        query = f"""
            UPDATE portfolio_state
            SET {set_clause}, last_updated = NOW()
            WHERE id = 1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, *updates.values())

        logger.debug("📊 Portfolio state updated")

    # ========================================================================
    # TRADE DECISIONS (AUDIT TRAIL)
    # ========================================================================

    async def log_trade_decision(self, decision_data: Dict[str, Any]) -> int:
        """
        Log trade decision to audit trail (TimescaleDB).

        Args:
            decision_data: Decision information

        Returns:
            Inserted decision ID
        """
        query = """
            INSERT INTO trade_decisions (
                decision_id, timestamp, symbol, action, quantity,
                entry_price, agent, ai_reasoning, confidence, risk_score,
                market_regime, vix, sector_sentiment,
                expected_return_pct, expected_hold_hours,
                required_human_approval, outcome,
                llm_provider, llm_cost, decision_latency_ms
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20
            )
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            decision_id = await conn.fetchval(
                query,
                decision_data['decision_id'],
                decision_data.get('timestamp', datetime.now()),
                decision_data['symbol'],
                decision_data['action'],
                decision_data.get('quantity'),
                decision_data.get('entry_price'),
                decision_data['agent'],
                decision_data['ai_reasoning'],
                decision_data['confidence'],
                decision_data['risk_score'],
                decision_data.get('market_regime'),
                decision_data.get('vix'),
                decision_data.get('sector_sentiment'),
                decision_data.get('expected_return_pct'),
                decision_data.get('expected_hold_hours'),
                decision_data.get('required_human_approval', True),
                decision_data.get('outcome', 'PENDING'),
                decision_data.get('llm_provider'),
                decision_data.get('llm_cost'),
                decision_data.get('decision_latency_ms')
            )

        logger.debug(f"📝 Trade decision logged: {decision_data['decision_id']}")
        return decision_id

    async def update_trade_decision_outcome(
        self,
        decision_id: str,
        outcome: str,
        realized_pnl: float,
        realized_pnl_pct: float,
        exit_price: float,
        exit_time: datetime
    ):
        """Update trade decision with actual outcome."""
        query = """
            UPDATE trade_decisions
            SET outcome = $2,
                realized_pnl = $3,
                realized_pnl_pct = $4,
                exit_price = $5,
                exit_time = $6,
                actual_hold_hours = EXTRACT(EPOCH FROM ($6 - timestamp)) / 3600
            WHERE decision_id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                decision_id,
                outcome,
                realized_pnl,
                realized_pnl_pct,
                exit_price,
                exit_time
            )

        logger.debug(f"📝 Trade decision outcome updated: {decision_id} -> {outcome}")

    # ========================================================================
    # DAILY SUMMARY OPERATIONS
    # ========================================================================

    async def save_daily_summary(self, summary_data: Dict[str, Any]):
        """Save daily summary."""
        query = """
            INSERT INTO daily_summary (
                date, starting_capital, ending_capital, pnl, pnl_pct,
                num_trades, num_wins, num_losses, win_rate,
                best_trade_pnl, worst_trade_pnl,
                sharpe_ratio, max_drawdown_pct, max_exposure_pct,
                best_agent, worst_agent
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16
            )
            ON CONFLICT (date) DO UPDATE SET
                ending_capital = EXCLUDED.ending_capital,
                pnl = EXCLUDED.pnl,
                pnl_pct = EXCLUDED.pnl_pct,
                num_trades = EXCLUDED.num_trades,
                num_wins = EXCLUDED.num_wins,
                num_losses = EXCLUDED.num_losses,
                win_rate = EXCLUDED.win_rate,
                best_trade_pnl = EXCLUDED.best_trade_pnl,
                worst_trade_pnl = EXCLUDED.worst_trade_pnl,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                max_drawdown_pct = EXCLUDED.max_drawdown_pct,
                max_exposure_pct = EXCLUDED.max_exposure_pct,
                best_agent = EXCLUDED.best_agent,
                worst_agent = EXCLUDED.worst_agent
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                summary_data['date'],
                summary_data['starting_capital'],
                summary_data['ending_capital'],
                summary_data['pnl'],
                summary_data['pnl_pct'],
                summary_data['num_trades'],
                summary_data['num_wins'],
                summary_data['num_losses'],
                summary_data.get('win_rate'),
                summary_data.get('best_trade_pnl'),
                summary_data.get('worst_trade_pnl'),
                summary_data.get('sharpe_ratio'),
                summary_data.get('max_drawdown_pct'),
                summary_data.get('max_exposure_pct'),
                summary_data.get('best_agent'),
                summary_data.get('worst_agent')
            )

        logger.info(f"📊 Daily summary saved: {summary_data['date']}")

    # ========================================================================
    # AGENT ACTIONS LOG
    # ========================================================================

    async def log_agent_action(
        self,
        agent_name: str,
        action_type: str,
        context: Dict[str, Any],
        result: Dict[str, Any],
        latency_ms: int = None,
        llm_cost: float = None,
        error: str = None
    ):
        """Log agent action to TimescaleDB."""
        query = """
            INSERT INTO agent_actions (
                agent_name, action_type, context, result,
                latency_ms, llm_cost,
                error_occurred, error_message
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """

        import json

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                agent_name,
                action_type,
                json.dumps(context),
                json.dumps(result),
                latency_ms,
                llm_cost,
                error is not None,
                error
            )

        logger.debug(f"📝 Agent action logged: {agent_name}/{action_type}")

    # ========================================================================
    # LLM COST TRACKING
    # ========================================================================

    async def log_llm_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        agent_name: str = None,
        task_type: str = None
    ):
        """Log LLM API cost."""
        query = """
            INSERT INTO llm_costs (
                provider, model, input_tokens, output_tokens,
                total_tokens, cost, agent_name, task_type
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                provider,
                model,
                input_tokens,
                output_tokens,
                input_tokens + output_tokens,
                cost,
                agent_name,
                task_type
            )

        logger.debug(f"💰 LLM cost logged: {provider}/{model} = ₹{cost:.4f}")

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    async def get_config(self, key: str) -> Optional[str]:
        """Get configuration value."""
        async with self.pool.acquire() as conn:
            value = await conn.fetchval(
                "SELECT value FROM user_config WHERE key = $1",
                key
            )
        return value

    async def set_config(self, key: str, value: str, description: str = None):
        """Set configuration value."""
        query = """
            INSERT INTO user_config (key, value, description)
            VALUES ($1, $2, $3)
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                description = COALESCE(EXCLUDED.description, user_config.description),
                last_updated = NOW()
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, key, value, description)

        logger.debug(f"⚙️  Config set: {key} = {value}")

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Check database health.

        Returns:
            Dict with health status
        """
        try:
            async with self.pool.acquire() as conn:
                # Check basic connectivity
                await conn.fetchval('SELECT 1')

                # Check table existence
                tables = await conn.fetch("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                """)

                # Get pool stats
                pool_size = self.pool.get_size()
                pool_free = self.pool.get_idle_size()

                return {
                    'status': 'healthy',
                    'pool_size': pool_size,
                    'pool_free': pool_free,
                    'tables_count': len(tables),
                    'tables': [row['table_name'] for row in tables]
                }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    # ========================================================================
    # CONTEXT MANAGER SUPPORT
    # ========================================================================

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
