"""
Backtest Anti-Pump & Dump Filter

Validates the new confidence scoring system by:
1. Replaying historical signals with old vs new system
2. Calculating performance metrics for both
3. Optimizing volume spike thresholds
4. Generating comparison reports

Usage:
    python3 scripts/backtest_pump_filter.py
    python3 scripts/backtest_pump_filter.py --optimize
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta
import logging
from typing import Dict, List, Tuple
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PumpFilterBacktester:
    """Backtest the anti-pump & dump filter on historical data."""

    def __init__(self, volume_thresholds: Dict[str, float] = None):
        """
        Initialize backtester.

        Args:
            volume_thresholds: Dict with 'severe', 'high', 'moderate' spike ratios
                              Default: {'severe': 3.0, 'high': 2.0, 'moderate': 1.5}
        """
        self.volume_thresholds = volume_thresholds or {
            'severe': 3.0,    # -30 penalty
            'high': 2.0,      # -20 penalty
            'moderate': 1.5   # -10 penalty
        }

        self.penalties = {
            'severe': 30,
            'high': 20,
            'moderate': 10
        }

    def get_volume_spike_ratio(self, symbol: str, entry_date: str) -> float:
        """
        Calculate volume spike ratio for a symbol on entry date.

        Args:
            symbol: Stock symbol (without .NS)
            entry_date: Entry date in YYYY-MM-DD format

        Returns:
            Volume spike ratio (entry_volume / avg_volume)
        """
        try:
            symbol_yf = f"{symbol}.NS"
            ticker = yf.Ticker(symbol_yf)

            # Get 30 days of history before entry
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            start_date = (entry_dt - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = (entry_dt + timedelta(days=1)).strftime('%Y-%m-%d')

            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty or len(hist) < 5:
                logger.warning(f"  {symbol}: Insufficient history")
                return 1.0

            # Entry day volume
            entry_data = hist[hist.index.date == entry_dt.date()]
            if entry_data.empty:
                logger.warning(f"  {symbol}: No data for {entry_date}")
                return 1.0

            entry_volume = float(entry_data['Volume'].iloc[0])

            # Average volume before entry (exclude entry day)
            pre_entry = hist[hist.index.date < entry_dt.date()]
            if len(pre_entry) < 5:
                logger.warning(f"  {symbol}: Not enough pre-entry data")
                return 1.0

            avg_volume = pre_entry['Volume'].tail(10).mean()

            if avg_volume == 0:
                return 1.0

            spike_ratio = entry_volume / avg_volume

            return spike_ratio

        except Exception as e:
            logger.error(f"  {symbol}: Error calculating volume - {e}")
            return 1.0

    def apply_pump_filter(self, confidence: float, volume_spike_ratio: float) -> Tuple[float, int, str]:
        """
        Apply pump & dump filter to confidence score.

        Args:
            confidence: Original confidence (0-100)
            volume_spike_ratio: Volume spike ratio

        Returns:
            Tuple of (final_confidence, penalty, risk_level)
        """
        penalty = 0
        risk_level = "LOW"

        if volume_spike_ratio >= self.volume_thresholds['severe']:
            penalty = self.penalties['severe']
            risk_level = "SEVERE"
        elif volume_spike_ratio >= self.volume_thresholds['high']:
            penalty = self.penalties['high']
            risk_level = "HIGH"
        elif volume_spike_ratio >= self.volume_thresholds['moderate']:
            penalty = self.penalties['moderate']
            risk_level = "MODERATE"

        final_confidence = max(confidence - penalty, 0)

        return final_confidence, penalty, risk_level

    def get_actual_performance(self, symbol: str, entry_date: str, entry_price: float,
                               hold_days: int = 5) -> Dict:
        """
        Get actual performance of a stock after entry.

        Args:
            symbol: Stock symbol
            entry_date: Entry date
            entry_price: Entry price
            hold_days: Number of days to hold (default 5)

        Returns:
            Dict with performance metrics
        """
        try:
            symbol_yf = f"{symbol}.NS"
            ticker = yf.Ticker(symbol_yf)

            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            end_date = (entry_dt + timedelta(days=hold_days + 5)).strftime('%Y-%m-%d')  # +5 for weekends

            hist = ticker.history(start=entry_date, end=end_date)

            if hist.empty or len(hist) < 2:
                return {'success': False, 'reason': 'No data'}

            # Get exit price (after hold_days trading days)
            trading_days = hist[hist.index.date > entry_dt.date()]
            if len(trading_days) < hold_days:
                exit_idx = -1  # Use last available
            else:
                exit_idx = hold_days - 1

            exit_price = float(trading_days['Close'].iloc[exit_idx])
            exit_date = trading_days.index[exit_idx].strftime('%Y-%m-%d')

            # Calculate returns
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100

            # Check if hit stop loss (5%) or take profit (15%)
            max_price = float(trading_days['High'].max())
            min_price = float(trading_days['Low'].min())

            max_gain_pct = ((max_price - entry_price) / entry_price) * 100
            max_loss_pct = ((min_price - entry_price) / entry_price) * 100

            hit_stop_loss = min_price <= entry_price * 0.95
            hit_take_profit = max_price >= entry_price * 1.15

            return {
                'success': True,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_date': exit_date,
                'pnl_pct': pnl_pct,
                'max_gain_pct': max_gain_pct,
                'max_loss_pct': max_loss_pct,
                'hit_stop_loss': hit_stop_loss,
                'hit_take_profit': hit_take_profit,
                'is_winner': pnl_pct > 0
            }

        except Exception as e:
            logger.error(f"  {symbol}: Error calculating performance - {e}")
            return {'success': False, 'reason': str(e)}

    def backtest_historical_signals(self, signals_file: str) -> Dict:
        """
        Backtest historical signals with old vs new system.

        Args:
            signals_file: Path to JSON file with historical signals

        Returns:
            Dict with comparison results
        """
        logger.info("=" * 80)
        logger.info("BACKTESTING ANTI-PUMP & DUMP FILTER")
        logger.info("=" * 80)

        # Load historical signals
        with open(signals_file, 'r') as f:
            data = json.load(f)

        positions = data.get('positions', [])

        if not positions:
            logger.warning("No positions found in signals file")
            return {}

        logger.info(f"\nAnalyzing {len(positions)} historical signals...")
        logger.info(f"Entry date: {positions[0].get('entry_date', 'N/A')}")

        old_system_trades = []
        new_system_trades = []
        filtered_out = []

        for pos in positions:
            symbol = pos['symbol']
            entry_date = pos['entry_date']
            entry_price = pos['entry_price']
            original_confidence = pos.get('confidence', 95.0)

            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing: {symbol}")
            logger.info(f"  Entry Date: {entry_date}")
            logger.info(f"  Entry Price: ₹{entry_price:.2f}")
            logger.info(f"  Original Confidence: {original_confidence:.1f}%")

            # Calculate volume spike
            volume_spike = self.get_volume_spike_ratio(symbol, entry_date)
            logger.info(f"  Volume Spike: {volume_spike:.1f}x average")

            # Apply filter
            final_confidence, penalty, risk_level = self.apply_pump_filter(
                original_confidence, volume_spike
            )
            logger.info(f"  Risk Level: {risk_level}")
            logger.info(f"  Penalty: -{penalty}")
            logger.info(f"  Final Confidence: {final_confidence:.1f}%")

            # Get actual performance
            performance = self.get_actual_performance(symbol, entry_date, entry_price, hold_days=5)

            if performance['success']:
                logger.info(f"  Actual Performance: {performance['pnl_pct']:+.2f}%")
                logger.info(f"  Exit Price: ₹{performance['exit_price']:.2f}")

                # Old system: Would have traded
                old_system_trades.append({
                    'symbol': symbol,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'confidence': original_confidence,
                    'volume_spike': volume_spike,
                    'performance': performance,
                    'risk_level': risk_level
                })

                # New system: Trade only if confidence >= 70
                if final_confidence >= 70:
                    new_system_trades.append({
                        'symbol': symbol,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'confidence': final_confidence,
                        'volume_spike': volume_spike,
                        'performance': performance,
                        'risk_level': risk_level
                    })
                    logger.info(f"  ✅ NEW SYSTEM: WOULD TRADE")
                else:
                    filtered_out.append({
                        'symbol': symbol,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'original_confidence': original_confidence,
                        'final_confidence': final_confidence,
                        'volume_spike': volume_spike,
                        'performance': performance,
                        'risk_level': risk_level,
                        'reason': f'{risk_level} pump risk ({volume_spike:.1f}x volume)'
                    })
                    logger.info(f"  ❌ NEW SYSTEM: FILTERED OUT ({risk_level} risk)")
            else:
                logger.warning(f"  ⚠️  Could not calculate performance: {performance.get('reason', 'Unknown')}")

        # Calculate metrics
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)

        old_metrics = self._calculate_metrics(old_system_trades, "OLD SYSTEM")
        new_metrics = self._calculate_metrics(new_system_trades, "NEW SYSTEM")

        # Show filtered trades
        logger.info(f"\n🚫 FILTERED OUT BY NEW SYSTEM ({len(filtered_out)} trades):")
        logger.info("-" * 80)
        for trade in filtered_out:
            perf = trade['performance']
            logger.info(f"{trade['symbol']}: {perf['pnl_pct']:+.2f}% | {trade['risk_level']} risk | "
                       f"{trade['volume_spike']:.1f}x volume | {trade['reason']}")

        # Summary comparison
        self._print_comparison(old_metrics, new_metrics, filtered_out)

        # Save report
        report = {
            'backtest_date': datetime.now().isoformat(),
            'thresholds': self.volume_thresholds,
            'penalties': self.penalties,
            'old_system': old_metrics,
            'new_system': new_metrics,
            'filtered_out': filtered_out,
            'old_system_trades': old_system_trades,
            'new_system_trades': new_system_trades
        }

        report_file = f'paper_trading/reports/backtest_pump_filter_{date.today().strftime("%Y-%m-%d")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"\n✅ Backtest report saved to {report_file}")

        return report

    def _calculate_metrics(self, trades: List[Dict], system_name: str) -> Dict:
        """Calculate performance metrics for a set of trades."""
        if not trades:
            return {
                'system': system_name,
                'total_trades': 0,
                'winners': 0,
                'losers': 0,
                'win_rate_pct': 0,
                'avg_return_pct': 0,
                'total_return_pct': 0,
                'avg_winner_pct': 0,
                'avg_loser_pct': 0,
                'profit_factor': 0
            }

        winners = [t for t in trades if t['performance']['is_winner']]
        losers = [t for t in trades if not t['performance']['is_winner']]

        total_return = sum(t['performance']['pnl_pct'] for t in trades)
        avg_return = total_return / len(trades)

        avg_winner = sum(t['performance']['pnl_pct'] for t in winners) / len(winners) if winners else 0
        avg_loser = sum(t['performance']['pnl_pct'] for t in losers) / len(losers) if losers else 0

        total_gains = sum(t['performance']['pnl_pct'] for t in winners)
        total_losses = abs(sum(t['performance']['pnl_pct'] for t in losers))
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')

        return {
            'system': system_name,
            'total_trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate_pct': (len(winners) / len(trades)) * 100,
            'avg_return_pct': avg_return,
            'total_return_pct': total_return,
            'avg_winner_pct': avg_winner,
            'avg_loser_pct': avg_loser,
            'profit_factor': profit_factor
        }

    def _print_comparison(self, old_metrics: Dict, new_metrics: Dict, filtered: List[Dict]):
        """Print comparison table."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 PERFORMANCE COMPARISON")
        logger.info("=" * 80)

        logger.info(f"\n{'Metric':<25} {'Old System':<20} {'New System':<20} {'Change'}")
        logger.info("-" * 80)

        metrics_to_compare = [
            ('Total Trades', 'total_trades', ''),
            ('Winners', 'winners', ''),
            ('Losers', 'losers', ''),
            ('Win Rate', 'win_rate_pct', '%'),
            ('Avg Return', 'avg_return_pct', '%'),
            ('Total Return', 'total_return_pct', '%'),
            ('Avg Winner', 'avg_winner_pct', '%'),
            ('Avg Loser', 'avg_loser_pct', '%'),
            ('Profit Factor', 'profit_factor', 'x')
        ]

        for label, key, suffix in metrics_to_compare:
            old_val = old_metrics.get(key, 0)
            new_val = new_metrics.get(key, 0)

            if suffix == '%':
                old_str = f"{old_val:.2f}%"
                new_str = f"{new_val:.2f}%"
                change = new_val - old_val
                change_str = f"{change:+.2f}%"
            elif suffix == 'x':
                old_str = f"{old_val:.2f}x"
                new_str = f"{new_val:.2f}x"
                change = new_val - old_val
                change_str = f"{change:+.2f}x"
            else:
                old_str = str(int(old_val))
                new_str = str(int(new_val))
                change = new_val - old_val
                change_str = f"{int(change):+d}"

            logger.info(f"{label:<25} {old_str:<20} {new_str:<20} {change_str}")

        # Highlight key improvements
        logger.info("\n" + "=" * 80)
        logger.info("🎯 KEY IMPROVEMENTS")
        logger.info("=" * 80)

        win_rate_improvement = new_metrics['win_rate_pct'] - old_metrics['win_rate_pct']
        return_improvement = new_metrics['avg_return_pct'] - old_metrics['avg_return_pct']

        logger.info(f"✅ Win Rate: {old_metrics['win_rate_pct']:.1f}% → {new_metrics['win_rate_pct']:.1f}% "
                   f"({win_rate_improvement:+.1f}%)")
        logger.info(f"✅ Avg Return: {old_metrics['avg_return_pct']:.2f}% → {new_metrics['avg_return_pct']:.2f}% "
                   f"({return_improvement:+.2f}%)")

        if filtered:
            avoided_losses = [f for f in filtered if not f['performance']['is_winner']]
            avoided_gains = [f for f in filtered if f['performance']['is_winner']]

            logger.info(f"\n🛡️  RISK AVOIDANCE:")
            logger.info(f"   Avoided {len(avoided_losses)} losing trades")
            if avoided_losses:
                avg_avoided_loss = sum(f['performance']['pnl_pct'] for f in avoided_losses) / len(avoided_losses)
                logger.info(f"   Average avoided loss: {avg_avoided_loss:.2f}%")

            if avoided_gains:
                logger.info(f"   ⚠️  Missed {len(avoided_gains)} winning trades")
                avg_missed_gain = sum(f['performance']['pnl_pct'] for f in avoided_gains) / len(avoided_gains)
                logger.info(f"   Average missed gain: {avg_missed_gain:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Backtest anti-pump & dump filter')
    parser.add_argument('--signals-file', default='paper_trading/portfolio_state.json',
                       help='Path to signals file (default: current portfolio)')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize volume spike thresholds')

    args = parser.parse_args()

    if args.optimize:
        logger.info("🔧 Threshold optimization not yet implemented")
        logger.info("   Using default thresholds: 1.5x, 2.0x, 3.0x")

    backtester = PumpFilterBacktester()
    report = backtester.backtest_historical_signals(args.signals_file)

    logger.info("\n" + "=" * 80)
    logger.info("✅ BACKTEST COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
