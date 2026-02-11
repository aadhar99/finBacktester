"""
OpenAI GPT-4 integration.
"""

import time
import logging
from typing import Dict
from .base import BaseLLMClient, LLMResponse, LLMProvider, LLMError, LLMAuthenticationError, LLMTimeoutError

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT-4 integration."""

    def __init__(self, config):
        super().__init__(config)

        try:
            from openai import AsyncOpenAI
            self.AsyncOpenAI = AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self.client = AsyncOpenAI(api_key=config.api_key)

        # Model selection
        self.model_map = {
            'gpt4': 'gpt-4-turbo-2024-04-09',
            'gpt4o': 'gpt-4o',
            'gpt4o-mini': 'gpt-4o-mini'  # Cheaper option
        }
        self.model = self.model_map.get(config.model, config.model)

        logger.info(f"✅ OpenAI client initialized with model: {self.model}")

    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using GPT-4."""
        start_time = time.time()

        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]

            # Add system message if provided
            if 'system' in kwargs:
                messages.insert(0, {"role": "system", "content": kwargs['system']})

            # Prepare request parameters
            params = {
                'model': self.model,
                'messages': messages,
                'temperature': kwargs.get('temperature', self.config.temperature),
                'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            }

            # Enable JSON mode if requested
            if kwargs.get('json_mode', False):
                params['response_format'] = {"type": "json_object"}

            # Make API call
            response = await self.client.chat.completions.create(**params)

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract usage stats
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            # Calculate cost
            cost = self.calculate_cost(input_tokens, output_tokens)

            # Update stats
            self.total_cost += cost
            self.total_calls += 1

            # Extract text
            text = response.choices[0].message.content

            logger.debug(f"OpenAI API call: {latency_ms}ms, {input_tokens + output_tokens} tokens, ₹{cost:.4f}")

            return LLMResponse(
                text=text,
                model=self.model,
                tokens_used=input_tokens + output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                provider=LLMProvider.GPT4,
                metadata={
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'finish_reason': response.choices[0].finish_reason
                }
            )

        except Exception as e:
            self.failed_calls += 1

            error_msg = str(e).lower()

            if 'api key' in error_msg or 'authentication' in error_msg or 'unauthorized' in error_msg:
                logger.error(f"OpenAI authentication failed: {e}")
                raise LLMAuthenticationError(f"OpenAI API authentication failed: {e}")

            elif 'rate limit' in error_msg or 'quota' in error_msg:
                logger.warning(f"OpenAI rate limit exceeded: {e}")
                raise LLMError(f"OpenAI rate limit exceeded: {e}")

            elif 'timeout' in error_msg:
                logger.error(f"OpenAI API timeout: {e}")
                raise LLMTimeoutError(f"OpenAI API timeout: {e}")

            else:
                logger.error(f"OpenAI API error: {e}")
                raise LLMError(f"OpenAI API error: {e}")

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for GPT-4.

        Uses tiktoken library if available, otherwise estimates.
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except ImportError:
            logger.debug("tiktoken not installed, using estimation")
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
        except Exception as e:
            logger.debug(f"Token counting failed, using estimation: {e}")
            return len(text) // 4

    def get_cost_per_1k_tokens(self) -> Dict[str, float]:
        """
        Pricing in INR (1 USD = ~83 INR as of Feb 2026).

        Returns:
            Dict with 'input' and 'output' costs per 1K tokens
        """
        if 'mini' in self.model.lower():
            # GPT-4o Mini: $0.15/$0.60 per million tokens
            return {
                'input': 0.012,   # ₹12.45 per million / 1000
                'output': 0.050   # ₹49.80 per million / 1000
            }
        elif 'gpt-4o' in self.model.lower():
            # GPT-4o: $2.50/$10 per million tokens
            return {
                'input': 0.21,    # ₹207.50 per million / 1000
                'output': 0.83    # ₹830 per million / 1000
            }
        else:  # GPT-4 Turbo
            # GPT-4 Turbo: $10/$30 per million tokens
            return {
                'input': 0.83,    # ₹830 per million / 1000
                'output': 2.49    # ₹2,490 per million / 1000
            }
