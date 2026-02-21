"""
Generate Detailed Position Analysis Report

Creates comprehensive reports with:
- Volume analysis (accumulation vs distribution)
- Technical indicators
- Post-entry performance tracking
- Risk metrics
- Pattern validation

Usage:
    python3 scripts/generate_detailed_report.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DetailedPositionAnalyzer:
    """Analyze positions with detailed metrics and risk factors."""

    def __init__(self, portfolio_file='paper_trading/portfolio_state.json'):
        self.portfolio_file = portfolio_file

    def load_portfolio(self):
        """Load current portfolio."""
        with open(self.portfolio_file, 'r') as f:
            return json.load(f)

    def analyze_position(self, position):
        """Analyze a single position with detailed metrics."""
        symbol = position['symbol']
        entry_date = position['entry_date']
        entry_price = position['entry_price']

        logger.info(f"Analyzing {symbol}...")

        symbol_yf = f"{symbol}.NS"
        ticker = yf.Ticker(symbol_yf)

        # Get historical data (7 days before entry to now)
        entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
        start_date = (entry_dt - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = date.today().strftime('%Y-%m-%d')

        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            return None

        # Current price
        current_price = float(hist['Close'].iloc[-1])

        # Volume analysis
        entry_idx = hist.index[hist.index.date == entry_dt.date()]
        if len(entry_idx) == 0:
            return None

        entry_volume = float(hist.loc[entry_idx[0], 'Volume'])

        # Pre-entry average volume (5 days before)
        pre_entry = hist[hist.index.date < entry_dt.date()]
        avg_volume_before = pre_entry['Volume'].tail(5).mean()

        # Post-entry average volume
        post_entry = hist[hist.index.date > entry_dt.date()]
        avg_volume_after = post_entry['Volume'].mean() if len(post_entry) > 0 else entry_volume

        # Volume spike ratio
        volume_spike_ratio = entry_volume / avg_volume_before if avg_volume_before > 0 else 1.0

        # Post-entry volume change
        volume_change_pct = ((avg_volume_after - avg_volume_before) / avg_volume_before * 100) if avg_volume_before > 0 else 0

        # Price action analysis
        price_change_pct = ((current_price - entry_price) / entry_price) * 100

        # Volatility
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = float(daily_returns.std() * 100)

        # Technical indicators
        hist['SMA_5'] = hist['Close'].rolling(5).mean()
        hist['SMA_10'] = hist['Close'].rolling(10).mean()

        sma_5 = float(hist['SMA_5'].iloc[-1]) if not hist['SMA_5'].isna().iloc[-1] else current_price
        sma_10 = float(hist['SMA_10'].iloc[-1]) if not hist['SMA_10'].isna().iloc[-1] else current_price

        trend = "Uptrend" if current_price > sma_5 else "Downtrend"

        # Risk assessment
        risk_score = 0
        risk_factors = []

        # Risk Factor 1: Volume spike (>2.5x = pump risk)
        if volume_spike_ratio > 2.5:
            risk_score += 30
            risk_factors.append(f"HIGH volume spike ({volume_spike_ratio:.1f}x avg) - Pump & dump risk")
        elif volume_spike_ratio > 1.5:
            risk_score += 15
            risk_factors.append(f"Elevated volume spike ({volume_spike_ratio:.1f}x avg)")

        # Risk Factor 2: Post-entry volume increase (distribution)
        if volume_change_pct > 50:
            risk_score += 25
            risk_factors.append(f"High post-entry volume (+{volume_change_pct:.1f}%) - Distribution pattern")
        elif volume_change_pct > 25:
            risk_score += 10
            risk_factors.append(f"Elevated post-entry volume (+{volume_change_pct:.1f}%)")

        # Risk Factor 3: High volatility
        if volatility > 10:
            risk_score += 20
            risk_factors.append(f"High volatility ({volatility:.1f}%) - Unstable price")
        elif volatility > 7:
            risk_score += 10
            risk_factors.append(f"Elevated volatility ({volatility:.1f}%)")

        # Risk Factor 4: Downtrend
        if trend == "Downtrend":
            risk_score += 15
            risk_factors.append("Price below 5-day SMA - Downtrend")

        # Risk Factor 5: Immediate decline
        first_close_after = post_entry['Close'].iloc[0] if len(post_entry) > 0 else current_price
        immediate_change = ((first_close_after - entry_price) / entry_price) * 100
        if immediate_change < -3:
            risk_score += 10
            risk_factors.append(f"Sharp decline after entry ({immediate_change:.1f}%)")

        # Pattern classification
        if risk_score > 50:
            pattern_quality = "🔴 HIGH RISK - Distribution/Pump"
        elif risk_score > 30:
            pattern_quality = "🟡 MODERATE RISK - Mixed signals"
        else:
            pattern_quality = "🟢 LOW RISK - Genuine accumulation"

        # Get company info
        info = ticker.info

        return {
            'symbol': symbol,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'days_held': position.get('days_held', 0),
            'unrealized_pnl': position.get('unrealized_pnl', 0),

            # Volume metrics
            'entry_volume': entry_volume,
            'avg_volume_before': avg_volume_before,
            'avg_volume_after': avg_volume_after,
            'volume_spike_ratio': volume_spike_ratio,
            'volume_change_pct': volume_change_pct,

            # Technical metrics
            'volatility': volatility,
            'sma_5': sma_5,
            'sma_10': sma_10,
            'trend': trend,

            # Risk assessment
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'pattern_quality': pattern_quality,

            # Company info
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap_cr': info.get('marketCap', 0) / 10000000 if info.get('marketCap') else 0,

            # Original pattern
            'original_pattern': position.get('pattern_type', 'N/A'),
            'original_confidence': position.get('confidence', 0)
        }

    def generate_report(self):
        """Generate detailed report for all positions."""
        logger.info("=" * 80)
        logger.info("GENERATING DETAILED POSITION REPORT")
        logger.info("=" * 80)

        portfolio = self.load_portfolio()
        positions = portfolio.get('positions', [])

        if not positions:
            logger.info("No positions to analyze")
            return

        analyzed_positions = []

        for pos in positions:
            analysis = self.analyze_position(pos)
            if analysis:
                analyzed_positions.append(analysis)

        # Sort by risk score (highest first)
        analyzed_positions.sort(key=lambda x: x['risk_score'], reverse=True)

        # Generate report
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_positions': len(analyzed_positions),
            'portfolio_summary': {
                'total_value': portfolio['current_value'],
                'total_pnl': portfolio['total_pnl'],
                'total_return_pct': portfolio['total_return_pct']
            },
            'positions': analyzed_positions
        }

        # Save JSON report
        report_file = f'paper_trading/reports/detailed/detailed_report_{date.today().strftime("%Y-%m-%d")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"✅ Detailed report saved to {report_file}")

        # Print summary
        self.print_report_summary(analyzed_positions)

        return report

    def print_report_summary(self, positions):
        """Print human-readable report summary."""
        print("\n" + "=" * 80)
        print("📊 DETAILED POSITION ANALYSIS REPORT")
        print("=" * 80)

        for pos in positions:
            print(f"\n{pos['pattern_quality']} {pos['symbol']}")
            print("-" * 80)
            print(f"Performance: {pos['price_change_pct']:+.2f}% | P&L: ₹{pos['unrealized_pnl']:,.2f}")
            print(f"Entry: ₹{pos['entry_price']:.2f} → Current: ₹{pos['current_price']:.2f} | Days: {pos['days_held']}")

            print(f"\nVolume Analysis:")
            print(f"  Entry Volume: {pos['entry_volume']:,.0f} ({pos['volume_spike_ratio']:.1f}x avg)")
            print(f"  Avg Before: {pos['avg_volume_before']:,.0f} | Avg After: {pos['avg_volume_after']:,.0f}")
            print(f"  Post-Entry Change: {pos['volume_change_pct']:+.1f}%")

            print(f"\nTechnical Indicators:")
            print(f"  Trend: {pos['trend']} | Volatility: {pos['volatility']:.2f}%")
            print(f"  5-Day SMA: ₹{pos['sma_5']:.2f} | 10-Day SMA: ₹{pos['sma_10']:.2f}")

            print(f"\nRisk Assessment (Score: {pos['risk_score']}/100):")
            for factor in pos['risk_factors']:
                print(f"  • {factor}")
            if not pos['risk_factors']:
                print("  ✅ No significant risk factors detected")

            print(f"\nCompany: {pos['sector']} - {pos['industry']}")
            print(f"Market Cap: ₹{pos['market_cap_cr']:.2f} Cr")
            print(f"Original Pattern: {pos['original_pattern']} (Confidence: {pos['original_confidence']:.1f}%)")

        print("\n" + "=" * 80)


if __name__ == '__main__':
    analyzer = DetailedPositionAnalyzer()
    analyzer.generate_report()
