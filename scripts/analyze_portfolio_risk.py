"""
Portfolio Risk Analysis Script

Analyzes current portfolio using Advanced Risk Manager:
- Calculates comprehensive risk metrics
- Generates risk report
- Provides rebalancing recommendations

Usage:
    python3 scripts/analyze_portfolio_risk.py
    python3 scripts/analyze_portfolio_risk.py --portfolio paper_trading/portfolio_state.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
import argparse
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta

from risk.advanced_risk_manager import AdvancedRiskManager, RiskLimits, PortfolioRiskMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_portfolio(portfolio_file: str) -> dict:
    """Load portfolio from JSON file."""
    with open(portfolio_file, 'r') as f:
        return json.load(f)


def fetch_sector_info(positions: list) -> dict:
    """Fetch sector information for all positions."""
    logger.info("Fetching sector information...")

    sector_map = {}
    for pos in positions:
        symbol = pos['symbol']
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info
            sector = info.get('sector', 'Unknown')
            sector_map[symbol] = sector
            logger.info(f"  {symbol}: {sector}")
        except Exception as e:
            logger.warning(f"  {symbol}: Could not fetch sector - {e}")
            sector_map[symbol] = 'Unknown'

    return sector_map


def calculate_historical_returns(portfolio_value: float, days: int = 30) -> pd.Series:
    """
    Calculate historical daily returns.

    For now, returns synthetic data. In production, would fetch actual
    portfolio value history from database.
    """
    logger.info(f"Calculating historical returns ({days} days)...")

    # Synthetic returns for demonstration
    # In production, fetch actual portfolio values from database
    np = __import__('numpy')
    dates = pd.date_range(end=date.today(), periods=days)

    # Simulate daily returns (mean=0.1%, std=2%)
    returns = np.random.normal(0.001, 0.02, days)

    return pd.Series(returns, index=dates)


def analyze_portfolio_risk(portfolio_file: str = 'paper_trading/portfolio_state.json'):
    """Main analysis function."""
    logger.info("=" * 80)
    logger.info("📊 PORTFOLIO RISK ANALYSIS")
    logger.info("=" * 80)

    # Load portfolio
    portfolio = load_portfolio(portfolio_file)
    positions = portfolio.get('positions', [])

    logger.info(f"\n📁 Portfolio loaded:")
    logger.info(f"   File: {portfolio_file}")
    logger.info(f"   Total Value: ₹{portfolio['current_value']:,.2f}")
    logger.info(f"   Cash: ₹{portfolio['cash']:,.2f}")
    logger.info(f"   Positions: {len(positions)}")

    # Fetch sector information
    sector_map = fetch_sector_info(positions)

    # Add sector info to positions
    for pos in positions:
        pos['sector'] = sector_map.get(pos['symbol'], 'Unknown')

    # Initialize risk manager
    risk_manager = AdvancedRiskManager()

    # Calculate historical returns
    historical_returns = calculate_historical_returns(portfolio['current_value'])

    # Calculate risk metrics
    logger.info("\n" + "=" * 80)
    logger.info("CALCULATING RISK METRICS")
    logger.info("=" * 80)

    metrics = risk_manager.calculate_portfolio_risk_metrics(
        positions=positions,
        portfolio_value=portfolio['current_value'],
        cash=portfolio['cash'],
        historical_returns=historical_returns
    )

    # Generate and print report
    logger.info("\n")
    report = risk_manager.generate_risk_report(metrics)
    print(report)

    # Check if rebalancing is needed
    should_rebalance, reasons = risk_manager.should_rebalance(metrics)

    if should_rebalance:
        logger.info("\n" + "=" * 80)
        logger.info("🔄 REBALANCING RECOMMENDATIONS")
        logger.info("=" * 80)
        logger.info("\nRebalancing is recommended:")
        for reason in reasons:
            logger.info(f"  • {reason}")

        logger.info("\nSuggested actions:")
        if metrics.total_exposure_pct > risk_manager.limits.max_total_exposure_pct:
            excess = metrics.total_exposure_pct - risk_manager.limits.max_total_exposure_pct
            logger.info(f"  1. Reduce exposure by {excess:.1f}% (trim largest positions)")

        if metrics.cash_reserve_pct < risk_manager.limits.min_cash_reserve_pct:
            needed = risk_manager.limits.min_cash_reserve_pct - metrics.cash_reserve_pct
            logger.info(f"  2. Increase cash reserve by {needed:.1f}% (sell losing positions)")

        for sector, exposure in metrics.sector_exposure.items():
            if exposure > risk_manager.limits.max_sector_exposure_pct:
                excess = exposure - risk_manager.limits.max_sector_exposure_pct
                logger.info(f"  3. Reduce {sector} sector exposure by {excess:.1f}%")
    else:
        logger.info("\n✅ Portfolio is well-balanced - no rebalancing needed")

    # Test position sizing for new signals
    logger.info("\n" + "=" * 80)
    logger.info("🎯 POSITION SIZING SIMULATION")
    logger.info("=" * 80)
    logger.info("\nSimulating position sizing for new signals:")

    test_signals = [
        {'symbol': 'TEST1', 'confidence': 95, 'price': 500, 'sector': 'Technology'},
        {'symbol': 'TEST2', 'confidence': 85, 'price': 300, 'sector': 'Industrials'},
        {'symbol': 'TEST3', 'confidence': 75, 'price': 200, 'sector': 'Consumer Defensive'},
    ]

    # Convert positions to dict format
    current_positions_dict = {pos['symbol']: pos for pos in positions}

    for signal in test_signals:
        logger.info(f"\n{signal['symbol']} ({signal['sector']}):")
        logger.info(f"  Confidence: {signal['confidence']}%")
        logger.info(f"  Price: ₹{signal['price']:.2f}")

        size, details = risk_manager.calculate_risk_adjusted_size(
            signal_confidence=signal['confidence'],
            symbol=signal['symbol'],
            price=signal['price'],
            portfolio_value=portfolio['current_value'],
            current_positions=current_positions_dict,
            sector=signal['sector']
        )

        logger.info(f"  Recommended Size: {size} shares")
        logger.info(f"  Capital Allocation: ₹{details['capital_allocated']:,.2f} ({details['final_position_pct']:.1f}%)")

        if details['risk_factors']:
            logger.info(f"  Risk Factors:")
            for factor in details['risk_factors']:
                logger.info(f"    • {factor}")

    # Save risk report
    report_data = {
        'analysis_date': datetime.now().isoformat(),
        'portfolio_file': portfolio_file,
        'metrics': {
            'total_value': metrics.total_value,
            'exposure_pct': metrics.total_exposure_pct,
            'cash_reserve_pct': metrics.cash_reserve_pct,
            'position_count': metrics.position_count,
            'volatility': metrics.portfolio_volatility,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown_pct': metrics.max_drawdown_pct,
            'value_at_risk_95': metrics.value_at_risk_95,
            'largest_position_pct': metrics.largest_position_pct,
            'sector_exposure': metrics.sector_exposure,
            'risk_score': metrics.risk_score,
            'risk_alerts': metrics.risk_alerts
        },
        'rebalancing': {
            'needed': should_rebalance,
            'reasons': reasons
        }
    }

    report_file = f'paper_trading/reports/risk_analysis_{date.today().strftime("%Y-%m-%d")}.json'
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"\n✅ Risk analysis report saved to {report_file}")
    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze portfolio risk')
    parser.add_argument('--portfolio', default='paper_trading/portfolio_state.json',
                       help='Path to portfolio JSON file')

    args = parser.parse_args()

    analyze_portfolio_risk(args.portfolio)
