"""
Strategy Complexity Enforcer

CRITICAL: Lesson from research - "Simpler, clear-cut strategies tend to work better
in the long run. It's more about testing and refining than making everything overly complicated."

This module enforces complexity limits on trading strategies to prevent overfitting.

Rules:
1. Maximum 5 indicators per strategy
2. Maximum 3 entry conditions
3. Maximum 2 exit conditions
4. Maximum 10 tunable parameters
5. Maximum 100 lines of code
6. Must be explainable in 2 sentences or less

Purpose: Prevent AI from creating overly complex, overfitted strategies
"""

import re
import ast
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ComplexityLimits:
    """
    Configuration for complexity limits.

    Two-tier system:
    - SOFT limits: Warning threshold (flag but allow)
    - HARD limits: Absolute maximum (reject strategy)
    """
    # SOFT LIMITS (warnings only)
    soft_max_indicators: int = 5
    soft_max_entry_conditions: int = 3
    soft_max_exit_conditions: int = 2
    soft_max_parameters: int = 10
    soft_max_lines_of_code: int = 100

    # HARD LIMITS (reject strategy)
    max_indicators: int = 10  # More lenient - up to 10 indicators
    max_entry_conditions: int = 8  # Allow more complex entry logic
    max_exit_conditions: int = 5  # Exit can be complex
    max_parameters: int = 20  # More room for parameter tuning
    max_lines_of_code: int = 200  # Reasonable implementation size
    max_description_words: int = 100


@dataclass
class ComplexityAnalysis:
    """Results of complexity analysis."""
    num_indicators: int
    num_entry_conditions: int
    num_exit_conditions: int
    num_parameters: int
    lines_of_code: int
    description_words: int
    violations: List[str]
    warnings: List[str]
    passes: bool
    score: float  # 0-100, lower is simpler


class ComplexityEnforcer:
    """
    Enforces complexity limits on trading strategies.

    Prevents overfitted "kitchen sink" strategies with too many indicators.
    """

    # Common technical indicators (for detection)
    KNOWN_INDICATORS = [
        'sma', 'ema', 'rsi', 'macd', 'bollinger', 'bb', 'atr', 'adx',
        'stochastic', 'cci', 'williams', 'obv', 'vwap', 'pivot',
        'fibonacci', 'ichimoku', 'keltner', 'dmi', 'mfi', 'roc',
        'trix', 'ultimate_oscillator', 'aroon', 'supertrend'
    ]

    def __init__(self, limits: ComplexityLimits = None):
        """
        Initialize complexity enforcer.

        Args:
            limits: Custom complexity limits (optional)
        """
        self.limits = limits or ComplexityLimits()
        logger.info(
            f"✅ Complexity Enforcer initialized\n"
            f"   Soft limits: {self.limits.soft_max_indicators} indicators, "
            f"{self.limits.soft_max_entry_conditions} entry conditions, "
            f"{self.limits.soft_max_parameters} params\n"
            f"   Hard limits: {self.limits.max_indicators} indicators, "
            f"{self.limits.max_entry_conditions} entry conditions, "
            f"{self.limits.max_parameters} params"
        )

    def analyze_strategy_code(self, code: str, description: str = "") -> ComplexityAnalysis:
        """
        Analyze strategy code for complexity.

        Args:
            code: Strategy Python code
            description: Strategy description (plain English)

        Returns:
            ComplexityAnalysis with results
        """
        violations = []
        warnings = []

        # 1. Count indicators used
        num_indicators = self._count_indicators(code)
        if num_indicators > self.limits.max_indicators:
            violations.append(
                f"Too many indicators: {num_indicators} > {self.limits.max_indicators} HARD limit"
            )
        elif num_indicators > self.limits.soft_max_indicators:
            warnings.append(
                f"High indicator count: {num_indicators} (soft limit: {self.limits.soft_max_indicators}). "
                f"Consider simplifying to avoid overfitting."
            )

        # 2. Count entry/exit conditions
        num_entry, num_exit = self._count_conditions(code)
        if num_entry > self.limits.max_entry_conditions:
            violations.append(
                f"Too many entry conditions: {num_entry} > {self.limits.max_entry_conditions} HARD limit"
            )
        elif num_entry > self.limits.soft_max_entry_conditions:
            warnings.append(
                f"Complex entry logic: {num_entry} conditions (soft limit: {self.limits.soft_max_entry_conditions})"
            )

        if num_exit > self.limits.max_exit_conditions:
            violations.append(
                f"Too many exit conditions: {num_exit} > {self.limits.max_exit_conditions} HARD limit"
            )
        elif num_exit > self.limits.soft_max_exit_conditions:
            warnings.append(
                f"Complex exit logic: {num_exit} conditions (soft limit: {self.limits.soft_max_exit_conditions})"
            )

        # 3. Count tunable parameters
        num_parameters = self._count_parameters(code)
        if num_parameters > self.limits.max_parameters:
            violations.append(
                f"Too many parameters: {num_parameters} > {self.limits.max_parameters} HARD limit"
            )
        elif num_parameters > self.limits.soft_max_parameters:
            warnings.append(
                f"Many parameters: {num_parameters} (soft limit: {self.limits.soft_max_parameters}). "
                f"Ensure walk-forward validation to prevent overfitting."
            )

        # 4. Count lines of code
        lines_of_code = self._count_lines_of_code(code)
        if lines_of_code > self.limits.max_lines_of_code:
            violations.append(
                f"Too many lines: {lines_of_code} > {self.limits.max_lines_of_code} HARD limit"
            )
        elif lines_of_code > self.limits.soft_max_lines_of_code:
            warnings.append(
                f"Large implementation: {lines_of_code} lines (soft limit: {self.limits.soft_max_lines_of_code})"
            )

        # 5. Check description length
        description_words = len(description.split()) if description else 0
        if description and description_words > self.limits.max_description_words:
            warnings.append(
                f"Description too long: {description_words} words (aim for 2 sentences or less)"
            )

        # Calculate complexity score (0-100, lower is better)
        score = self._calculate_complexity_score(
            num_indicators,
            num_entry,
            num_exit,
            num_parameters,
            lines_of_code
        )

        # Determine pass/fail
        passes = len(violations) == 0

        return ComplexityAnalysis(
            num_indicators=num_indicators,
            num_entry_conditions=num_entry,
            num_exit_conditions=num_exit,
            num_parameters=num_parameters,
            lines_of_code=lines_of_code,
            description_words=description_words,
            violations=violations,
            warnings=warnings,
            passes=passes,
            score=score
        )

    def _count_indicators(self, code: str) -> int:
        """Count technical indicators used in code."""
        code_lower = code.lower()
        count = 0

        for indicator in self.KNOWN_INDICATORS:
            # Look for indicator name in code
            if re.search(rf'\b{indicator}\b', code_lower):
                count += 1

        # Also look for ta-lib style calls: ta.SMA, ta.RSI, etc.
        talib_calls = re.findall(r'ta\.[A-Z]+', code)
        count += len(set(talib_calls))  # Unique indicators

        return min(count, 20)  # Cap at 20 for sanity

    def _count_conditions(self, code: str) -> Tuple[int, int]:
        """
        Count entry and exit conditions.

        Returns:
            (num_entry_conditions, num_exit_conditions)
        """
        # Look for entry logic
        entry_conditions = 0
        exit_conditions = 0

        # Pattern: if ... and ... and ... (count 'and' operators)
        entry_block = self._extract_block(code, 'entry', 'should_buy', 'buy_signal')
        if entry_block:
            entry_conditions = entry_block.count(' and ') + 1  # +1 for first condition

        exit_block = self._extract_block(code, 'exit', 'should_sell', 'sell_signal')
        if exit_block:
            exit_conditions = exit_block.count(' and ') + 1

        # Cap at reasonable number
        return (min(entry_conditions, 10), min(exit_conditions, 10))

    def _count_parameters(self, code: str) -> int:
        """Count tunable parameters (class attributes, config values)."""
        count = 0

        # Look for class attributes (self.param_name = value)
        params = re.findall(r'self\.([\w_]+)\s*=\s*[\d.]+', code)
        count += len(set(params))

        # Look for config dict keys
        config_params = re.findall(r'[\'"](\w+)[\'"]\s*:\s*[\d.]+', code)
        count += len(set(config_params))

        return min(count, 20)

    def _count_lines_of_code(self, code: str) -> int:
        """Count non-blank, non-comment lines of code."""
        lines = code.split('\n')
        count = 0

        for line in lines:
            stripped = line.strip()
            # Skip blank lines and comments
            if stripped and not stripped.startswith('#'):
                count += 1

        return count

    def _extract_block(self, code: str, *keywords) -> str:
        """Extract code block containing any of the keywords."""
        for keyword in keywords:
            # Find function/method containing keyword
            pattern = rf'def\s+\w*{keyword}\w*.*?(?=\n\s*def|\Z)'
            match = re.search(pattern, code, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0)
        return ""

    def _calculate_complexity_score(
        self,
        num_indicators: int,
        num_entry: int,
        num_exit: int,
        num_parameters: int,
        lines_of_code: int
    ) -> float:
        """
        Calculate overall complexity score (0-100).

        Lower is simpler/better.
        """
        # Weighted scoring
        score = (
            (num_indicators / self.limits.max_indicators) * 30 +
            (num_entry / self.limits.max_entry_conditions) * 20 +
            (num_exit / self.limits.max_exit_conditions) * 15 +
            (num_parameters / self.limits.max_parameters) * 20 +
            (lines_of_code / self.limits.max_lines_of_code) * 15
        )

        return min(round(score, 1), 100.0)

    def validate_strategy(
        self,
        code: str,
        description: str = "",
        strategy_name: str = "Unknown"
    ) -> Tuple[bool, ComplexityAnalysis]:
        """
        Validate strategy complexity.

        Args:
            code: Strategy code
            description: Strategy description
            strategy_name: Strategy name (for logging)

        Returns:
            (passes, analysis_result)
        """
        logger.info(f"🔍 Analyzing complexity: {strategy_name}")

        analysis = self.analyze_strategy_code(code, description)

        if analysis.passes:
            logger.info(
                f"✅ {strategy_name} passes complexity check "
                f"(score: {analysis.score}/100)"
            )
        else:
            logger.warning(
                f"❌ {strategy_name} FAILED complexity check "
                f"(score: {analysis.score}/100)"
            )
            for violation in analysis.violations:
                logger.warning(f"   - {violation}")

        if analysis.warnings:
            for warning in analysis.warnings:
                logger.info(f"   ⚠️  {warning}")

        return (analysis.passes, analysis)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example usage of complexity enforcer."""

    print("=" * 70)
    print("COMPLEXITY ENFORCER - Example Usage")
    print("=" * 70)

    enforcer = ComplexityEnforcer()

    # Example 1: Simple strategy (GOOD)
    simple_strategy = """
class MomentumStrategy:
    def __init__(self):
        self.rsi_period = 14
        self.rsi_oversold = 30

    def should_buy(self, data):
        rsi = calculate_rsi(data, self.rsi_period)
        return rsi < self.rsi_oversold
"""

    simple_description = "Buy when RSI drops below 30 (oversold)"

    print("\n📊 Example 1: Simple Momentum Strategy")
    print("-" * 70)
    passes, analysis = enforcer.validate_strategy(
        simple_strategy,
        simple_description,
        "SimpleMomentum"
    )
    print(f"Indicators: {analysis.num_indicators}/{enforcer.limits.max_indicators}")
    print(f"Entry conditions: {analysis.num_entry_conditions}/{enforcer.limits.max_entry_conditions}")
    print(f"Parameters: {analysis.num_parameters}/{enforcer.limits.max_parameters}")
    print(f"Lines of code: {analysis.lines_of_code}/{enforcer.limits.max_lines_of_code}")
    print(f"Complexity score: {analysis.score}/100")
    print(f"Result: {'✅ PASS' if passes else '❌ FAIL'}")

    # Example 2: Complex strategy (BAD)
    complex_strategy = """
class KitchenSinkStrategy:
    def __init__(self):
        self.sma_short = 10
        self.sma_medium = 20
        self.sma_long = 50
        self.ema_fast = 12
        self.ema_slow = 26
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.bb_period = 20
        self.atr_period = 14
        self.adx_period = 14

    def should_buy(self, data):
        sma10 = calculate_sma(data, self.sma_short)
        sma20 = calculate_sma(data, self.sma_medium)
        sma50 = calculate_sma(data, self.sma_long)
        ema12 = calculate_ema(data, self.ema_fast)
        rsi = calculate_rsi(data, self.rsi_period)
        macd = calculate_macd(data, self.macd_fast, self.macd_slow)
        bb_upper, bb_lower = calculate_bollinger(data, self.bb_period)
        atr = calculate_atr(data, self.atr_period)
        adx = calculate_adx(data, self.adx_period)

        return (
            sma10 > sma20 and
            sma20 > sma50 and
            ema12 > sma20 and
            rsi < 70 and
            rsi > 30 and
            macd > 0 and
            data['close'] < bb_upper and
            atr > 0.5
        )
"""

    complex_description = "Multi-indicator strategy combining SMA crossovers, EMA, RSI, MACD, Bollinger Bands, ATR, and ADX with multiple entry conditions"

    print("\n📊 Example 2: Kitchen Sink Strategy (Overfitted)")
    print("-" * 70)
    passes, analysis = enforcer.validate_strategy(
        complex_strategy,
        complex_description,
        "KitchenSink"
    )
    print(f"Indicators: {analysis.num_indicators}/{enforcer.limits.max_indicators}")
    print(f"Entry conditions: {analysis.num_entry_conditions}/{enforcer.limits.max_entry_conditions}")
    print(f"Parameters: {analysis.num_parameters}/{enforcer.limits.max_parameters}")
    print(f"Lines of code: {analysis.lines_of_code}/{enforcer.limits.max_lines_of_code}")
    print(f"Complexity score: {analysis.score}/100")
    print(f"Result: {'✅ PASS' if passes else '❌ FAIL'}")

    if analysis.violations:
        print("\nViolations:")
        for violation in analysis.violations:
            print(f"  ❌ {violation}")

    print("\n" + "=" * 70)
    print("✅ Complexity enforcer ready to prevent overfitting")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
