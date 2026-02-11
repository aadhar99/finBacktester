"""
Google Gemini integration.
"""

import time
import logging
from typing import Dict
from .base import BaseLLMClient, LLMResponse, LLMProvider, LLMError, LLMAuthenticationError, LLMTimeoutError

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """Google Gemini integration."""

    def __init__(self, config):
        super().__init__(config)

        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. Install with: pip install google-generativeai"
            )

        # Configure API key
        genai.configure(api_key=config.api_key)

        # Model selection - using latest stable versions
        self.model_map = {
            'flash': 'gemini-flash-latest',               # Fastest, free tier (alias to latest)
            'pro': 'gemini-pro-latest',                   # Most capable (alias to latest)
            'flash-thinking': 'gemini-flash-latest'       # Use flash for now
        }
        self.model_name = self.model_map.get(config.model, config.model)

        # Initialize model
        self.model = genai.GenerativeModel(self.model_name)

        logger.info(f"✅ Gemini client initialized with model: {self.model_name}")

    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using Gemini."""
        start_time = time.time()

        try:
            # Prepare generation config
            generation_config = {
                'temperature': kwargs.get('temperature', self.config.temperature),
                'max_output_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            }

            # Add candidate count if specified
            if 'n' in kwargs:
                generation_config['candidate_count'] = kwargs['n']

            # Make API call (synchronous - Gemini doesn't have native async yet)
            # We'll use asyncio to run it in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Count tokens (Gemini provides token counting)
            input_tokens = self.count_tokens(prompt)
            output_tokens = self.count_tokens(response.text)

            # Calculate cost
            cost = self.calculate_cost(input_tokens, output_tokens)

            # Update stats
            self.total_cost += cost
            self.total_calls += 1

            logger.debug(f"Gemini API call: {latency_ms}ms, {input_tokens + output_tokens} tokens, ₹{cost:.4f}")

            return LLMResponse(
                text=response.text,
                model=self.model_name,
                tokens_used=input_tokens + output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                provider=LLMProvider.GEMINI,
                metadata={
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'finish_reason': response.candidates[0].finish_reason if response.candidates else None
                }
            )

        except Exception as e:
            self.failed_calls += 1

            error_msg = str(e).lower()

            if 'api key' in error_msg or 'authentication' in error_msg:
                logger.error(f"Gemini authentication failed: {e}")
                raise LLMAuthenticationError(f"Gemini API authentication failed: {e}")

            elif 'quota' in error_msg or 'rate limit' in error_msg:
                logger.warning(f"Gemini rate limit exceeded: {e}")
                raise LLMError(f"Gemini rate limit exceeded: {e}")

            elif 'timeout' in error_msg:
                logger.error(f"Gemini API timeout: {e}")
                raise LLMTimeoutError(f"Gemini API timeout: {e}")

            else:
                logger.error(f"Gemini API error: {e}")
                raise LLMError(f"Gemini API error: {e}")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Gemini's tokenizer.

        Falls back to estimation if tokenizer unavailable.
        """
        try:
            return self.model.count_tokens(text).total_tokens
        except Exception as e:
            logger.debug(f"Token counting failed, using estimation: {e}")
            # Fallback: rough estimate (similar to GPT-4)
            return len(text) // 4

    def get_cost_per_1k_tokens(self) -> Dict[str, float]:
        """
        Pricing in INR (1 USD = ~83 INR as of Feb 2026).

        Note: Gemini Flash has generous free tier (1M tokens/day).

        Returns:
            Dict with 'input' and 'output' costs per 1K tokens
        """
        if 'flash' in self.model_name.lower():
            # Gemini Flash: FREE tier up to 1M tokens/day
            # Paid tier (if exceeding free): $0.075/$0.30 per million tokens
            # For now, assume free tier (return 0 cost)
            return {
                'input': 0.0,
                'output': 0.0
            }
            # Uncomment below if tracking paid usage:
            # return {
            #     'input': 0.006,   # ₹6.23 per million / 1000
            #     'output': 0.024   # ₹24.90 per million / 1000
            # }

        elif 'pro' in self.model_name.lower():
            # Gemini Pro: $1.25/$5 per million tokens
            return {
                'input': 0.10,    # ₹103.75 per million / 1000
                'output': 0.42    # ₹415 per million / 1000
            }

        else:
            # Default to Flash pricing
            return {
                'input': 0.0,
                'output': 0.0
            }
