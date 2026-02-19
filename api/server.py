"""
REST API Server for Cloud Deployment

Provides endpoints for dashboard to fetch data:
- /health - Health check
- /api/portfolio - Current portfolio state
- /api/stats - Database statistics
- /api/deals - Recent bulk deals
- /api/signals - Recent trading signals

Usage:
    python3 api/server.py
    # Or with gunicorn:
    gunicorn api.server:app --bind 0.0.0.0:8000
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.database_adapter import DatabaseAdapter

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Database will be initialized on-demand
_db = None

def get_db():
    """Get database instance (lazy initialization)."""
    global _db
    if _db is None:
        _db = DatabaseAdapter()
    return _db


@app.route('/', methods=['GET'])
def index():
    """API root endpoint with documentation."""
    return jsonify({
        'service': 'Smart Money Trading API',
        'version': '1.0.0',
        'status': 'online',
        'endpoints': {
            '/health': 'Health check and database status',
            '/api/portfolio': 'Current portfolio state',
            '/api/stats': 'Database statistics',
            '/api/deals': 'Recent bulk deals',
            '/api/signals': 'Recent trading signals'
        },
        'documentation': 'https://github.com/aadhar99/finBacktester'
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for cloud providers."""
    try:
        # Test database connection
        result = get_db().execute_query("SELECT 1 as test")
        db_status = 'healthy' if result else 'unhealthy'
    except:
        db_status = 'unhealthy'

    return jsonify({
        'status': 'healthy',
        'service': 'smart-money-trading',
        'database': db_status,
        'version': '1.0.0'
    }), 200


@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Get current paper trading portfolio state."""
    try:
        portfolio_file = 'paper_trading/portfolio_state.json'

        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                data = json.load(f)
            return jsonify(data), 200
        else:
            # Return empty portfolio
            return jsonify({
                'initial_capital': 1000000,
                'current_value': 1000000,
                'cash': 1000000,
                'total_return_pct': 0,
                'total_pnl': 0,
                'total_trades': 0,
                'open_positions': 0,
                'closed_positions': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate_pct': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown_pct': 0,
                'positions': [],
                'closed_trades': []
            }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics."""
    try:
        # Check if table exists
        if not get_db().table_exists('bulk_deals'):
            return jsonify({
                'total_bulk_deals': 0,
                'unique_symbols': 0,
                'date_range_start': None,
                'date_range_end': None,
                'total_buy_value': 0,
                'total_sell_value': 0
            }), 200

        # Total bulk deals
        total_deals = get_db().execute_query("SELECT COUNT(*) as count FROM bulk_deals")
        total_count = total_deals[0]['count'] if total_deals else 0

        # Unique symbols
        unique_symbols = get_db().execute_query("SELECT COUNT(DISTINCT symbol) as count FROM bulk_deals")
        symbol_count = unique_symbols[0]['count'] if unique_symbols else 0

        # Date range
        date_range = get_db().execute_query("""
            SELECT MIN(deal_date) as start_date, MAX(deal_date) as end_date
            FROM bulk_deals
        """)
        start_date = date_range[0]['start_date'] if date_range else None
        end_date = date_range[0]['end_date'] if date_range else None

        # Buy/sell values
        values = get_db().execute_query("""
            SELECT
                SUM(CASE WHEN deal_type = 'buy' THEN quantity * price ELSE 0 END) as buy_value,
                SUM(CASE WHEN deal_type = 'sell' THEN quantity * price ELSE 0 END) as sell_value
            FROM bulk_deals
        """)
        buy_value = float(values[0]['buy_value'] or 0) if values else 0
        sell_value = float(values[0]['sell_value'] or 0) if values else 0

        return jsonify({
            'total_bulk_deals': total_count,
            'unique_symbols': symbol_count,
            'date_range_start': start_date,
            'date_range_end': end_date,
            'total_buy_value': buy_value,
            'total_sell_value': sell_value
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/deals', methods=['GET'])
def get_deals():
    """Get recent bulk deals."""
    try:
        limit = request.args.get('limit', 100, type=int)

        if not get_db().table_exists('bulk_deals'):
            return jsonify([]), 200

        deals = get_db().execute_query(f"""
            SELECT * FROM bulk_deals
            ORDER BY deal_date DESC, id DESC
            LIMIT {limit}
        """)

        return jsonify(deals), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/signals', methods=['GET'])
def get_signals():
    """Get recent trading signals."""
    try:
        # Return from latest daily report
        reports_dir = Path('paper_trading/reports')

        if not reports_dir.exists():
            return jsonify([]), 200

        # Get latest report
        report_files = sorted(reports_dir.glob('daily_*.json'), reverse=True)

        if not report_files:
            return jsonify([]), 200

        with open(report_files[0], 'r') as f:
            report_data = json.load(f)

        # Extract signals from positions
        signals = []
        for pos in report_data.get('positions', []):
            signals.append({
                'symbol': pos['symbol'],
                'signal_type': pos['signal_type'],
                'entry_price': pos['entry_price'],
                'confidence': pos['confidence'],
                'pattern': pos['pattern_type'],
                'date': pos['entry_date']
            })

        return jsonify(signals), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    print(f"🚀 Starting API server on port {port}")
    print(f"   Health check: http://localhost:{port}/health")
    print(f"   API docs: See CLOUD_DEPLOYMENT_GUIDE.md")

    app.run(host='0.0.0.0', port=port, debug=debug)
