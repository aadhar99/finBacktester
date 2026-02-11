"""
Anthropic Claude integration.
"""

import time
import logging
from typing import Dict
from .base import BaseLLMClient, LLMResponse, LLMProvider, LLMError, LLMAuthenticationError, LLMTimeoutError

logger = logging.getLogger(__name__)


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude integration."""

    def __init__(self, config):
        super().__init__(config)

        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        self.client = anthropic.Anthropic(api_key=config.api_key)

        # Model selection
        self.model_map = {
            'sonnet': 'claude-sonnet-4-5-20250929',  # Latest Sonnet
            'haiku': 'claude-haiku-4-20250311',      # Fast & cheap
            'opus': 'claude-opus-4-20250514'         # Most capable
        }
        self.model = self.model_map.get(config.model, config.model)

        logger.info(f"✅ Claude client initialized with model: {self.model}")

    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using Claude."""
        start_time = time.time()

        try:
            # Prepare request parameters
            params = {
                'model': self.model,
                'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
                'temperature': kwargs.get('temperature', self.config.temperature),
                'messages': [{"role": "user", "content": prompt}]
            }

            # Add system prompt if provided
            if 'system' in kwargs:
                params['system'] = kwargs['system']

            # Make API call
            response = self.client.messages.create(**params)

            latency_ms = int((time.time() - start_time) * 1000)

            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self.calculate_cost(input_tokens, output_tokens)

            # Update stats
            self.total_cost += cost
            self.total_calls += 1

            # Extract text from response
            text = response.content[0].text

            logger.debug(f"Claude API call: {latency_ms}ms, {input_tokens + output_tokens} tokens, ₹{cost:.4f}")

            return LLMResponse(
                text=text,
                model=self.model,
                tokens_used=input_tokens + output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                provider=LLMProvider.CLAUDE,
                metadata={
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'stop_reason': response.stop_reason
                }
            )

        except self.anthropic.AuthenticationError as e:
            self.failed_calls += 1
            logger.error(f"Claude authentication failed: {e}")
            raise LLMAuthenticationError(f"Claude API authentication failed: {e}")

        except self.anthropic.RateLimitError as e:
            self.failed_calls += 1
            logger.warning(f"Claude rate limit exceeded: {e}")
            raise LLMError(f"Claude rate limit exceeded: {e}")

        except self.anthropic.APITimeoutError as e:
            self.failed_calls += 1
            logger.error(f"Claude API timeout: {e}")
            raise LLMTimeoutError(f"Claude API timeout: {e}")

        except Exception as e:
            self.failed_calls += 1
            logger.error(f"Claude API error: {e}")
            raise LLMError(f"Claude API error: {e}")

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Claude.

        Claude uses a similar tokenizer to GPT-4.
        Rough estimate: 1 token ≈ 4 characters.
        """
        return len(text) // 4

    def get_cost_per_1k_tokens(self) -> Dict[str, float]:
        """
        Pricing in INR (1 USD = ~83 INR as of Feb 2026).

        Returns:
            Dict with 'input' and 'output' costs per 1K tokens
        """
        # Pricing as of Feb 2026
        if 'haiku' in self.model.lower():
            # Haiku: $0.25/$1.25 per million tokens
            return {
                'input': 0.025,   # ₹2.08 per million / 1000 = ₹0.002 per 1K
                'output': 0.125   # ₹10.38 per million / 1000 = ₹0.010 per 1K
            }
        elif 'opus' in self.model.lower():
            # Opus: $15/$75 per million tokens
            return {
                'input': 1.50,    # ₹1,245 per million / 1000
                'output': 7.50    # ₹6,225 per million / 1000
            }
        else:  # Sonnet (default)
            # Sonnet: $3/$15 per million tokens
            return {
                'input': 0.30,    # ₹249 per million / 1000
                'output': 1.50    # ₹1,245 per million / 1000
            }
