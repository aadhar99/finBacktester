"""
LLM Pattern Enhancer - AI-Powered Smart Money Analysis

Uses LLM (Gemini free tier) to add context and narrative to detected patterns:
- Why this pattern is significant
- What it tells us about institutional sentiment
- Risk factors to consider
- Actionable recommendations

Enhances rule-based pattern detection with human-like reasoning.
"""

import logging
import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from ai.agents.pattern_detector import Pattern

logger = logging.getLogger(__name__)


@dataclass
class LLMInsight:
    """AI-generated insights for a detected pattern."""
    reasoning: str
    significance: str
    risk_factors: List[str]
    recommendation: str  # BUY, HOLD, AVOID
    target_confidence: float  # 0-100
    action_items: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_json(cls, json_str: str) -> 'LLMInsight':
        """Parse LLM JSON response into LLMInsight."""
        try:
            data = json.loads(json_str)
            return cls(
                reasoning=data.get('reasoning', ''),
                significance=data.get('significance', ''),
                risk_factors=data.get('risk_factors', []),
                recommendation=data.get('recommendation', 'HOLD'),
                target_confidence=float(data.get('target_confidence', 0)),
                action_items=data.get('action_items', [])
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}")
            logger.error(f"Raw response: {json_str}")
            # Return default insight
            return cls(
                reasoning="Unable to parse LLM response",
                significance="",
                risk_factors=["LLM parsing error"],
                recommendation="HOLD",
                target_confidence=0,
                action_items=[]
            )


class LLMPatternEnhancer:
    """
    Uses LLM to enhance pattern detection with narrative analysis.

    Converts raw pattern data into actionable insights with:
    - Human-readable explanations
    - Market context
    - Risk assessment
    - Trading recommendations
    """

    def __init__(self, llm_manager):
        """
        Initialize LLM Pattern Enhancer.

        Args:
            llm_manager: LLMManager instance (configured with Gemini/Claude/etc)
        """
        self.llm = llm_manager
        logger.info("✅ LLMPatternEnhancer initialized")

    async def enhance_pattern(
        self,
        symbol: str,
        pattern: Pattern,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> LLMInsight:
        """
        Use LLM to analyze pattern and provide narrative insights.

        Args:
            symbol: Stock symbol (e.g., 'BIOPOL')
            pattern: Detected pattern from PatternDetector
            additional_context: Optional extra context (sector, market cap, etc.)

        Returns:
            LLMInsight with reasoning, risks, recommendations
        """
        prompt = self._build_prompt(symbol, pattern, additional_context)

        try:
            response = await self.llm.complete(
                prompt,
                task_type='analysis',
                max_tokens=2000  # Increased for full JSON response
            )

            # Extract and parse JSON from LLM response
            insight = self._extract_json_insight(response.text)

            logger.info(f"  ✅ LLM enhanced {symbol} {pattern.type}: {insight.recommendation}")

            return insight

        except Exception as e:
            logger.error(f"LLM enhancement failed for {symbol}: {e}")
            # Return default insight on error
            return LLMInsight(
                reasoning=f"LLM error: {str(e)}",
                significance="",
                risk_factors=["LLM enhancement unavailable"],
                recommendation="HOLD",
                target_confidence=pattern.confidence,
                action_items=[]
            )

    def _extract_json_insight(self, text: str) -> LLMInsight:
        """
        Extract JSON from LLM response with robust parsing.

        Handles:
        - Markdown code blocks
        - Partial/truncated JSON
        - Extra text before/after JSON
        """
        original_text = text
        text = text.strip()

        # Remove markdown code blocks if present
        if '```json' in text:
            json_start = text.find('```json') + 7
            json_end = text.find('```', json_start)
            if json_end == -1:
                # No closing ```, take rest of text
                text = text[json_start:].strip()
            else:
                text = text[json_start:json_end].strip()
        elif '```' in text:
            json_start = text.find('```') + 3
            json_end = text.find('```', json_start)
            if json_end == -1:
                text = text[json_start:].strip()
            else:
                text = text[json_start:json_end].strip()

        # Find JSON object boundaries
        if '{' in text:
            json_start = text.find('{')
            text = text[json_start:]

            # Try to find matching closing brace
            # Count braces to find the end
            brace_count = 0
            json_end = -1
            in_string = False
            escape_next = False

            for i, char in enumerate(text):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break

            if json_end > 0:
                text = text[:json_end]

        # Try to parse JSON
        try:
            data = json.loads(text)
            return LLMInsight(
                reasoning=data.get('reasoning', ''),
                significance=data.get('significance', ''),
                risk_factors=data.get('risk_factors', []),
                recommendation=data.get('recommendation', 'HOLD'),
                target_confidence=float(data.get('target_confidence', 0)),
                action_items=data.get('action_items', [])
            )
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")

            # Try to extract partial data manually
            return self._parse_partial_json(original_text)

    def _parse_partial_json(self, text: str) -> LLMInsight:
        """
        Fallback parser for incomplete/malformed JSON.

        Extracts fields using regex when JSON parsing fails.
        """
        import re

        def extract_field(pattern: str, default: str = '') -> str:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip().strip('"').strip(',')
            return default

        def extract_array(pattern: str) -> List[str]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                array_text = match.group(1)
                # Extract quoted strings
                items = re.findall(r'"([^"]*)"', array_text)
                return items
            return []

        reasoning = extract_field(r'"reasoning"\s*:\s*"([^"]*(?:"[^"]*)*)"')
        significance = extract_field(r'"significance"\s*:\s*"([^"]*)"')
        recommendation = extract_field(r'"recommendation"\s*:\s*"([^"]*)"', 'HOLD')

        # Extract confidence
        conf_match = re.search(r'"target_confidence"\s*:\s*(\d+)', text)
        target_confidence = float(conf_match.group(1)) if conf_match else 0

        # Extract arrays
        risk_factors = extract_array(r'"risk_factors"\s*:\s*\[(.*?)\]')
        action_items = extract_array(r'"action_items"\s*:\s*\[(.*?)\]')

        logger.info(f"  ℹ️  Extracted partial JSON: {len(reasoning)} chars reasoning, {len(risk_factors)} risks")

        return LLMInsight(
            reasoning=reasoning or "Unable to parse complete response",
            significance=significance,
            risk_factors=risk_factors if risk_factors else ["LLM response incomplete"],
            recommendation=recommendation,
            target_confidence=target_confidence,
            action_items=action_items
        )

    def _build_prompt(
        self,
        symbol: str,
        pattern: Pattern,
        additional_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build LLM prompt for pattern analysis."""

        # Format evidence for readability
        evidence_str = json.dumps(pattern.evidence, indent=2, default=str)

        # Build context section
        context_str = ""
        if additional_context:
            context_str = f"\nAdditional Context:\n{json.dumps(additional_context, indent=2)}\n"

        prompt = f"""You are an expert Indian stock market analyst specializing in smart money tracking.

Analyze this institutional trading pattern and provide actionable insights.

Stock: {symbol}
Pattern Type: {pattern.type}
Signal: {pattern.signal}
Confidence: {pattern.confidence:.0f}%

Pattern Evidence:
{evidence_str}
{context_str}

Your task:
1. Explain WHY this pattern is significant for retail traders
2. What does this tell us about institutional sentiment?
3. What are the key risk factors to consider?
4. Should a retail trader BUY, HOLD, or AVOID this stock?
5. What are specific action items?

IMPORTANT GUIDELINES:
- Be concise and actionable (2-3 sentences for reasoning)
- Focus on institutional behavior insights (what smart money is doing)
- Consider Indian market context (NSE regulations, FII behavior, etc.)
- Risk factors should be specific, not generic
- Action items should be concrete steps

Output ONLY valid JSON in this exact format:
{{
  "reasoning": "Brief explanation of pattern significance (2-3 sentences)",
  "significance": "One sentence on why this matters",
  "risk_factors": ["Specific risk 1", "Specific risk 2", "Specific risk 3"],
  "recommendation": "BUY" or "HOLD" or "AVOID",
  "target_confidence": 0-100,
  "action_items": ["Specific action 1", "Specific action 2"]
}}

Respond ONLY with the JSON, no other text."""

        return prompt

    async def enhance_multiple_patterns(
        self,
        symbol: str,
        patterns: List[Pattern],
        additional_context: Optional[Dict[str, Any]] = None
    ) -> List[LLMInsight]:
        """
        Enhance multiple patterns for same stock.

        Args:
            symbol: Stock symbol
            patterns: List of detected patterns
            additional_context: Optional context

        Returns:
            List of LLMInsight objects (one per pattern)
        """
        tasks = [
            self.enhance_pattern(symbol, pattern, additional_context)
            for pattern in patterns
        ]

        return await asyncio.gather(*tasks)


# ============================================================================
# Testing with Real BIOPOL Pattern
# ============================================================================

async def test_with_real_data():
    """Test LLM enhancer with real BIOPOL accumulation pattern."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from utils.llm import LLMManager, LLMConfig, LLMProvider
    import os

    print("=" * 70)
    print("LLM PATTERN ENHANCER - Testing with Real BIOPOL Data")
    print("=" * 70)

    # Check for Gemini API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n❌ GEMINI_API_KEY not found in .env")
        print("   Please add your Gemini API key to continue")
        return

    # Initialize LLM (Gemini free tier)
    llm_config = LLMConfig(
        provider=LLMProvider.GEMINI,
        model='flash',
        api_key=api_key
    )
    llm = LLMManager([llm_config])

    # Initialize enhancer
    enhancer = LLMPatternEnhancer(llm)

    # Create real BIOPOL pattern from our detection
    biopol_pattern = Pattern(
        type="SUSTAINED_ACCUMULATION",
        signal="BULLISH",
        confidence=95.0,
        evidence={
            'buy_deals': 5,
            'sell_deals': 2,
            'buy_value': 152799420.0,
            'sell_value': 14252400.0,
            'net_value': 138547020.0,
            'buy_sell_ratio': 10.72,
            'unique_buyers': 5,
            'top_buyers': [
                {'name': 'CRAFT EMERGING MARKET FUND PCC- ELITE CAPITAL FUND', 'value': 41292000.0},
                {'name': 'CRAFT EMERGING MARKET FUND PCC- CITADEL CAPITAL FUND', 'value': 41292000.0},
                {'name': 'BEACON STONE CAPITAL VCC - BEACON STONE I', 'value': 39960000.0}
            ],
            'date_range': '2026-02-13 to 2026-02-13'
        }
    )

    print(f"\n📊 Pattern Detected:")
    print(f"   Symbol: BIOPOL")
    print(f"   Pattern: {biopol_pattern.type}")
    print(f"   Signal: {biopol_pattern.signal}")
    print(f"   Confidence: {biopol_pattern.confidence:.0f}%")
    print(f"   Net Buying: ₹{biopol_pattern.evidence['net_value']:,.0f}")
    print(f"   Buy/Sell Ratio: {biopol_pattern.evidence['buy_sell_ratio']:.2f}:1")

    print("\n🤖 Asking LLM for insights...")

    # Get LLM enhancement
    insight = await enhancer.enhance_pattern('BIOPOL', biopol_pattern)

    # Display results
    print("\n" + "=" * 70)
    print("LLM-ENHANCED ANALYSIS")
    print("=" * 70)

    print(f"\n💡 Reasoning:")
    print(f"   {insight.reasoning}")

    print(f"\n⭐ Significance:")
    print(f"   {insight.significance}")

    print(f"\n⚠️  Risk Factors:")
    for i, risk in enumerate(insight.risk_factors, 1):
        print(f"   {i}. {risk}")

    print(f"\n📈 Recommendation: {insight.recommendation}")
    print(f"   LLM Confidence: {insight.target_confidence:.0f}%")

    print(f"\n✅ Action Items:")
    for i, action in enumerate(insight.action_items, 1):
        print(f"   {i}. {action}")

    print("\n" + "=" * 70)

    # Show LLM stats
    stats = llm.get_stats()
    print(f"\n💰 LLM Usage:")
    print(f"   Cost: ₹{stats['total_cost']:.4f}")
    if 'total_tokens' in stats:
        print(f"   Tokens: {stats['total_tokens']}")
    print(f"   Provider: Gemini (free tier)")

    print("\n✅ Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    asyncio.run(test_with_real_data())
