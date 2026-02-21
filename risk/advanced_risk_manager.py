"""
Advanced Risk Management System

Implements:
1. Dynamic position sizing based on risk scores
2. Portfolio-level risk metrics (VaR, Sharpe, etc.)
3. Sector exposure limits
4. Risk-adjusted rebalancing
5. Real-time risk monitoring

Usage:
    from risk.advanced_risk_manager import AdvancedRiskManager

    risk_mgr = AdvancedRiskManager(portfolio)
    position_size = risk_mgr.calculate_risk_adjusted_size(signal, current_portfolio)
    risk_report = risk_mgr.generate_risk_report()
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_pct: float = 15.0          # Max % of portfolio per position
    max_sector_exposure_pct: float = 40.0   # Max % of portfolio per sector
    max_total_exposure_pct: float = 70.0    # Max % invested (keep 30% cash)
    max_correlation: float = 0.7            # Max correlation between positions
    min_confidence: float = 70.0            # Min confidence to trade
    max_daily_loss_pct: float = 5.0         # Max portfolio loss in a day
    max_position_count: int = 10            # Max number of positions
    min_cash_reserve_pct: float = 20.0     # Min cash reserve


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics."""
    total_value: float
    total_exposure_pct: float
    cash_reserve_pct: float
    position_count: int

    # Risk metrics
    portfolio_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    value_at_risk_95: float  # 95% VaR

    # Concentration metrics
    largest_position_pct: float
    sector_exposure: Dict[str, float]
    correlation_matrix: Optional[pd.DataFrame] = None

    # Risk alerts
    risk_alerts: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0-100 (higher = more risky)


class AdvancedRiskManager:
    """
    Advanced risk management system.

    Features:
    - Dynamic position sizing based on confidence and risk
    - Portfolio-level risk metrics (VaR, Sharpe, etc.)
    - Sector exposure limits
    - Correlation-based diversification
    - Real-time risk monitoring
    """

    def __init__(self, limits: RiskLimits = None):
        """
        Initialize risk manager.

        Args:
            limits: Risk limits configuration (uses defaults if None)
        """
        self.limits = limits or RiskLimits()
        logger.info("✅ AdvancedRiskManager initialized")
        logger.info(f"   Max Position: {self.limits.max_position_pct}%")
        logger.info(f"   Max Sector: {self.limits.max_sector_exposure_pct}%")
        logger.info(f"   Min Confidence: {self.limits.min_confidence}%")

    def calculate_risk_adjusted_size(
        self,
        signal_confidence: float,
        symbol: str,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, Dict],
        sector: Optional[str] = None
    ) -> Tuple[int, Dict]:
        """
        Calculate risk-adjusted position size.

        Args:
            signal_confidence: Signal confidence (0-100)
            symbol: Stock symbol
            price: Current price
            portfolio_value: Total portfolio value
            current_positions: Dict of current positions
            sector: Stock sector (optional)

        Returns:
            Tuple of (position_size, sizing_details)
        """
        sizing_details = {
            'base_confidence': signal_confidence,
            'adjustments': [],
            'final_size': 0,
            'capital_allocated': 0,
            'risk_factors': []
        }

        # 1. Base sizing from confidence (50-100% confidence → 5-15% position)
        confidence_factor = (signal_confidence - 50) / 50  # 0 to 1
        base_position_pct = 5.0 + (confidence_factor * 10.0)  # 5% to 15%

        sizing_details['base_position_pct'] = base_position_pct
        sizing_details['adjustments'].append(f"Base from confidence: {base_position_pct:.1f}%")

        # 2. Risk score adjustment
        risk_score = self._calculate_risk_score(symbol, current_positions)
        if risk_score > 70:
            risk_adjustment = -3.0  # Reduce by 3% for high risk
            base_position_pct += risk_adjustment
            sizing_details['adjustments'].append(f"High portfolio risk: {risk_adjustment:.1f}%")
            sizing_details['risk_factors'].append("High portfolio risk")

        # 3. Correlation adjustment (reduce size if correlated with existing)
        if current_positions:
            correlation_adjustment = self._correlation_adjustment(
                symbol, current_positions, portfolio_value
            )
            base_position_pct += correlation_adjustment
            if correlation_adjustment != 0:
                sizing_details['adjustments'].append(
                    f"Correlation adjustment: {correlation_adjustment:.1f}%"
                )

        # 4. Sector exposure check
        if sector:
            sector_exposure = self._get_sector_exposure(current_positions, sector)
            if sector_exposure > self.limits.max_sector_exposure_pct - base_position_pct:
                max_allowed = self.limits.max_sector_exposure_pct - sector_exposure
                if max_allowed < base_position_pct:
                    sizing_details['adjustments'].append(
                        f"Sector limit: reduced to {max_allowed:.1f}%"
                    )
                    sizing_details['risk_factors'].append(
                        f"Sector exposure at {sector_exposure:.1f}%"
                    )
                    base_position_pct = max(max_allowed, 0)

        # 5. Apply position limits
        base_position_pct = min(base_position_pct, self.limits.max_position_pct)
        base_position_pct = max(base_position_pct, 0)

        # 6. Check total exposure
        current_exposure = self._get_total_exposure(current_positions, portfolio_value)
        available_exposure = self.limits.max_total_exposure_pct - current_exposure

        if base_position_pct > available_exposure:
            sizing_details['adjustments'].append(
                f"Exposure limit: reduced to {available_exposure:.1f}%"
            )
            sizing_details['risk_factors'].append(
                f"Portfolio exposure at {current_exposure:.1f}%"
            )
            base_position_pct = max(available_exposure, 0)

        # 7. Calculate final position size
        capital_allocated = (portfolio_value * base_position_pct) / 100
        position_size = int(capital_allocated / price)

        sizing_details['final_position_pct'] = base_position_pct
        sizing_details['final_size'] = position_size
        sizing_details['capital_allocated'] = capital_allocated

        logger.info(f"  Position sizing for {symbol}:")
        logger.info(f"    Confidence: {signal_confidence:.1f}%")
        logger.info(f"    Base allocation: {base_position_pct:.1f}%")
        logger.info(f"    Position size: {position_size} shares")
        logger.info(f"    Capital: ₹{capital_allocated:,.2f}")

        return position_size, sizing_details

    def calculate_portfolio_risk_metrics(
        self,
        positions: List[Dict],
        portfolio_value: float,
        cash: float,
        historical_returns: Optional[pd.Series] = None
    ) -> PortfolioRiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.

        Args:
            positions: List of current positions
            portfolio_value: Total portfolio value
            cash: Cash available
            historical_returns: Historical daily returns (optional)

        Returns:
            PortfolioRiskMetrics with all risk indicators
        """
        logger.info("📊 Calculating portfolio risk metrics...")

        # Basic metrics
        total_exposure = sum(
            pos.get('quantity', 0) * pos.get('current_price', pos.get('entry_price', 0))
            for pos in positions
        )
        exposure_pct = (total_exposure / portfolio_value * 100) if portfolio_value > 0 else 0
        cash_reserve_pct = (cash / portfolio_value * 100) if portfolio_value > 0 else 100

        # Position concentration
        position_values = [
            pos.get('quantity', 0) * pos.get('current_price', pos.get('entry_price', 0))
            for pos in positions
        ]
        largest_position_pct = (max(position_values) / portfolio_value * 100) if position_values else 0

        # Sector exposure
        sector_exposure = self._calculate_sector_exposure(positions, portfolio_value)

        # Calculate volatility and risk metrics
        if historical_returns is not None and len(historical_returns) > 5:
            volatility = float(historical_returns.std() * np.sqrt(252) * 100)  # Annualized
            mean_return = float(historical_returns.mean() * 252)

            # Sharpe ratio (assuming 5% risk-free rate)
            risk_free_rate = 0.05
            sharpe = (mean_return - risk_free_rate) / (volatility / 100) if volatility > 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = historical_returns[historical_returns < 0]
            downside_std = float(downside_returns.std() * np.sqrt(252))
            sortino = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0

            # Value at Risk (95% confidence)
            var_95 = float(np.percentile(historical_returns, 5) * portfolio_value)

            # Max drawdown
            cumulative = (1 + historical_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = float(drawdown.min() * 100)
        else:
            # No historical data - use conservative estimates
            volatility = 20.0  # Assume 20% annual volatility
            sharpe = 0.0
            sortino = 0.0
            var_95 = -portfolio_value * 0.05  # -5%
            max_drawdown = 0.0

        # Risk alerts
        alerts = []
        if exposure_pct > self.limits.max_total_exposure_pct:
            alerts.append(f"⚠️ Total exposure ({exposure_pct:.1f}%) exceeds limit ({self.limits.max_total_exposure_pct}%)")

        if cash_reserve_pct < self.limits.min_cash_reserve_pct:
            alerts.append(f"⚠️ Cash reserve ({cash_reserve_pct:.1f}%) below minimum ({self.limits.min_cash_reserve_pct}%)")

        if largest_position_pct > self.limits.max_position_pct:
            alerts.append(f"⚠️ Largest position ({largest_position_pct:.1f}%) exceeds limit ({self.limits.max_position_pct}%)")

        for sector, exposure in sector_exposure.items():
            if exposure > self.limits.max_sector_exposure_pct:
                alerts.append(f"⚠️ {sector} sector ({exposure:.1f}%) exceeds limit ({self.limits.max_sector_exposure_pct}%)")

        if len(positions) > self.limits.max_position_count:
            alerts.append(f"⚠️ Position count ({len(positions)}) exceeds limit ({self.limits.max_position_count})")

        # Calculate overall risk score (0-100)
        risk_score = self._calculate_portfolio_risk_score(
            exposure_pct, cash_reserve_pct, volatility, len(positions), sector_exposure
        )

        metrics = PortfolioRiskMetrics(
            total_value=portfolio_value,
            total_exposure_pct=exposure_pct,
            cash_reserve_pct=cash_reserve_pct,
            position_count=len(positions),
            portfolio_volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_drawdown,
            value_at_risk_95=var_95,
            largest_position_pct=largest_position_pct,
            sector_exposure=sector_exposure,
            risk_alerts=alerts,
            risk_score=risk_score
        )

        logger.info("✅ Risk metrics calculated")
        logger.info(f"   Risk Score: {risk_score:.1f}/100")
        logger.info(f"   Exposure: {exposure_pct:.1f}%")
        logger.info(f"   Cash Reserve: {cash_reserve_pct:.1f}%")
        logger.info(f"   Alerts: {len(alerts)}")

        return metrics

    def should_rebalance(self, metrics: PortfolioRiskMetrics) -> Tuple[bool, List[str]]:
        """
        Determine if portfolio should be rebalanced.

        Args:
            metrics: Current portfolio risk metrics

        Returns:
            Tuple of (should_rebalance, reasons)
        """
        reasons = []

        # Check if any limits are breached
        if metrics.total_exposure_pct > self.limits.max_total_exposure_pct:
            reasons.append(f"Total exposure ({metrics.total_exposure_pct:.1f}%) exceeds limit")

        if metrics.cash_reserve_pct < self.limits.min_cash_reserve_pct:
            reasons.append(f"Cash reserve ({metrics.cash_reserve_pct:.1f}%) below minimum")

        if metrics.largest_position_pct > self.limits.max_position_pct * 1.2:  # 20% tolerance
            reasons.append(f"Position concentration too high ({metrics.largest_position_pct:.1f}%)")

        for sector, exposure in metrics.sector_exposure.items():
            if exposure > self.limits.max_sector_exposure_pct * 1.1:  # 10% tolerance
                reasons.append(f"{sector} sector exposure ({exposure:.1f}%) too high")

        if metrics.risk_score > 80:
            reasons.append(f"Overall risk score ({metrics.risk_score:.1f}) too high")

        should_rebalance = len(reasons) > 0

        if should_rebalance:
            logger.warning("⚠️ Portfolio rebalancing recommended:")
            for reason in reasons:
                logger.warning(f"   - {reason}")

        return should_rebalance, reasons

    def generate_risk_report(self, metrics: PortfolioRiskMetrics) -> str:
        """Generate human-readable risk report."""
        report = []
        report.append("=" * 80)
        report.append("📊 PORTFOLIO RISK REPORT")
        report.append("=" * 80)

        # Overall risk assessment
        if metrics.risk_score < 30:
            risk_level = "🟢 LOW RISK"
        elif metrics.risk_score < 60:
            risk_level = "🟡 MODERATE RISK"
        else:
            risk_level = "🔴 HIGH RISK"

        report.append(f"\nOverall Risk Level: {risk_level} (Score: {metrics.risk_score:.1f}/100)")

        # Portfolio overview
        report.append(f"\n📈 Portfolio Overview:")
        report.append(f"   Total Value: ₹{metrics.total_value:,.2f}")
        report.append(f"   Total Exposure: {metrics.total_exposure_pct:.1f}%")
        report.append(f"   Cash Reserve: {metrics.cash_reserve_pct:.1f}%")
        report.append(f"   Position Count: {metrics.position_count}")

        # Risk metrics
        report.append(f"\n📊 Risk Metrics:")
        report.append(f"   Volatility: {metrics.portfolio_volatility:.2f}% (annualized)")
        report.append(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        report.append(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
        report.append(f"   Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        report.append(f"   Value at Risk (95%): ₹{metrics.value_at_risk_95:,.2f}")

        # Concentration
        report.append(f"\n🎯 Concentration:")
        report.append(f"   Largest Position: {metrics.largest_position_pct:.1f}%")

        if metrics.sector_exposure:
            report.append(f"\n🏢 Sector Exposure:")
            for sector, exposure in sorted(metrics.sector_exposure.items(), key=lambda x: x[1], reverse=True):
                status = "⚠️" if exposure > self.limits.max_sector_exposure_pct else "✅"
                report.append(f"   {status} {sector}: {exposure:.1f}%")

        # Alerts
        if metrics.risk_alerts:
            report.append(f"\n⚠️ Risk Alerts ({len(metrics.risk_alerts)}):")
            for alert in metrics.risk_alerts:
                report.append(f"   {alert}")
        else:
            report.append(f"\n✅ No risk alerts - portfolio within limits")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    # Helper methods

    def _calculate_risk_score(self, symbol: str, current_positions: Dict) -> float:
        """Calculate overall portfolio risk score."""
        if not current_positions:
            return 0.0

        # Simple risk score based on position count and concentration
        position_count = len(current_positions)
        risk_score = min(position_count * 5, 50)  # More positions = more risk (up to 50)

        return risk_score

    def _correlation_adjustment(
        self, symbol: str, current_positions: Dict, portfolio_value: float
    ) -> float:
        """Adjust position size based on correlation with existing positions."""
        # Simplified - would need actual correlation calculation
        # For now, just reduce if we have many positions in same sector
        return 0.0

    def _get_sector_exposure(self, current_positions: Dict, sector: str) -> float:
        """Get current exposure to a specific sector."""
        # Simplified - would need actual sector classification
        return 0.0

    def _get_total_exposure(self, current_positions: Dict, portfolio_value: float) -> float:
        """Get total portfolio exposure percentage."""
        if not current_positions or portfolio_value == 0:
            return 0.0

        total_exposure = sum(
            pos.get('quantity', 0) * pos.get('current_price', pos.get('entry_price', 0))
            for pos in current_positions.values()
        )

        return (total_exposure / portfolio_value) * 100

    def _calculate_sector_exposure(self, positions: List[Dict], portfolio_value: float) -> Dict[str, float]:
        """Calculate exposure by sector."""
        sector_values = {}

        for pos in positions:
            sector = pos.get('sector', 'Unknown')
            value = pos.get('quantity', 0) * pos.get('current_price', pos.get('entry_price', 0))
            sector_values[sector] = sector_values.get(sector, 0) + value

        # Convert to percentages
        sector_exposure = {
            sector: (value / portfolio_value * 100) if portfolio_value > 0 else 0
            for sector, value in sector_values.items()
        }

        return sector_exposure

    def _calculate_portfolio_risk_score(
        self,
        exposure_pct: float,
        cash_reserve_pct: float,
        volatility: float,
        position_count: int,
        sector_exposure: Dict[str, float]
    ) -> float:
        """
        Calculate overall portfolio risk score (0-100).

        Higher score = higher risk
        """
        score = 0.0

        # Exposure risk (0-25 points)
        if exposure_pct > self.limits.max_total_exposure_pct:
            score += 25
        else:
            score += (exposure_pct / self.limits.max_total_exposure_pct) * 15

        # Cash reserve risk (0-15 points)
        if cash_reserve_pct < self.limits.min_cash_reserve_pct:
            score += 15
        else:
            score += max(0, 15 - (cash_reserve_pct / self.limits.min_cash_reserve_pct) * 15)

        # Volatility risk (0-25 points)
        # 0-10% vol = 0 pts, 10-30% = linear, >30% = 25 pts
        if volatility > 30:
            score += 25
        else:
            score += max(0, (volatility - 10) / 20 * 25)

        # Concentration risk (0-20 points)
        max_sector = max(sector_exposure.values()) if sector_exposure else 0
        if max_sector > self.limits.max_sector_exposure_pct:
            score += 20
        else:
            score += (max_sector / self.limits.max_sector_exposure_pct) * 20

        # Position count risk (0-15 points)
        if position_count > self.limits.max_position_count:
            score += 15
        elif position_count == 0:
            score += 5  # Some risk in having no positions
        else:
            score += (position_count / self.limits.max_position_count) * 10

        return min(score, 100)
