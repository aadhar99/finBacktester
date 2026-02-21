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

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import json
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.database_adapter import DatabaseAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Get current paper trading portfolio state with real-time prices."""
    try:
        from datetime import datetime, date
        import yfinance as yf

        portfolio_file = 'paper_trading/portfolio_state.json'

        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                data = json.load(f)

            positions = data.get('positions', [])

            # Fetch real-time prices for all positions (individual fetch for reliability)
            total_position_value = 0
            total_unrealized_pnl = 0

            for position in positions:
                symbol_yf = position['symbol'] + '.NS'

                try:
                    # Fetch price individually for each position
                    logger.info(f"Fetching price for {symbol_yf}")
                    ticker = yf.Ticker(symbol_yf)
                    hist = ticker.history(period='5d')  # Get 5 days to handle weekends
                    logger.info(f"{symbol_yf}: Got {len(hist)} data points")

                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        position['current_price'] = round(current_price, 2)
                        logger.info(f"{symbol_yf}: Current price = ₹{current_price:.2f}")

                        # Recalculate P&L
                        entry_price = position['entry_price']
                        quantity = position['quantity']

                        if position['signal_type'] == 'BUY':
                            unrealized_pnl = (current_price - entry_price) * quantity
                        else:  # SELL/SHORT
                            unrealized_pnl = (entry_price - current_price) * quantity

                        unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

                        position['unrealized_pnl'] = round(unrealized_pnl, 2)
                        position['unrealized_pnl_pct'] = round(unrealized_pnl_pct, 2)

                        # Track position value and total P&L
                        position_value = current_price * quantity
                        total_position_value += position_value
                        total_unrealized_pnl += unrealized_pnl
                    else:
                        # If no hist data, use entry price as fallback
                        logger.warning(f"{symbol_yf}: No history data available, using entry price")
                        position_value = position['entry_price'] * position['quantity']
                        total_position_value += position_value

                except Exception as e:
                    # If price fetch fails, use entry price as fallback
                    logger.error(f"{symbol_yf}: Error fetching price - {str(e)}")
                    position_value = position['entry_price'] * position['quantity']
                    total_position_value += position_value

                # Recalculate days_held
                if position.get('entry_date'):
                    entry_date = datetime.strptime(position['entry_date'], '%Y-%m-%d').date()
                    position['days_held'] = (date.today() - entry_date).days

            # Update portfolio-level metrics
            cash = data.get('cash', 0)
            initial_capital = data.get('initial_capital', 1000000)

            data['current_value'] = round(cash + total_position_value, 2)
            data['total_pnl'] = round(total_unrealized_pnl, 2)
            data['total_return_pct'] = round(((data['current_value'] - initial_capital) / initial_capital) * 100, 2)

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


@app.route('/api/debug/yfinance', methods=['GET'])
def debug_yfinance():
    """Debug endpoint to test yfinance data fetching."""
    try:
        import yfinance as yf
        from datetime import datetime

        test_symbol = request.args.get('symbol', 'APEX')
        symbol_yf = test_symbol + '.NS'

        result = {
            'symbol': symbol_yf,
            'timestamp': datetime.now().isoformat(),
            'yfinance_version': yf.__version__ if hasattr(yf, '__version__') else 'unknown'
        }

        try:
            ticker = yf.Ticker(symbol_yf)
            hist = ticker.history(period='5d')

            if not hist.empty:
                result['success'] = True
                result['data_points'] = len(hist)
                result['latest_date'] = str(hist.index[-1])
                result['latest_close'] = float(hist['Close'].iloc[-1])
                result['data_sample'] = {
                    'dates': [str(d) for d in hist.index[-3:]],
                    'closes': [float(c) for c in hist['Close'].iloc[-3:]]
                }
            else:
                result['success'] = False
                result['error'] = 'No data returned from yfinance'

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            result['error_type'] = type(e).__name__

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500


@app.route('/dashboard')
def dashboard():
    """Serve the dashboard HTML page."""
    dashboard_dir = Path(__file__).resolve().parent.parent / 'dashboard'
    return send_from_directory(dashboard_dir, 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    print(f"🚀 Starting API server on port {port}")
    print(f"   Health check: http://localhost:{port}/health")
    print(f"   Dashboard: http://localhost:{port}/dashboard")
    print(f"   API docs: See CLOUD_DEPLOYMENT_GUIDE.md")

    app.run(host='0.0.0.0', port=port, debug=debug)
