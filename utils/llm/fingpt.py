"""
FinGPT integration (finance-specific LLM).

FinGPT is an open-source LLM trained on financial data.
This client supports both Hugging Face Inference API and self-hosted deployments.
"""

import time
import logging
import aiohttp
from typing import Dict
from .base import BaseLLMClient, LLMResponse, LLMProvider, LLMError, LLMAuthenticationError, LLMTimeoutError

logger = logging.getLogger(__name__)


class FinGPTClient(BaseLLMClient):
    """
    FinGPT integration.

    Supports:
    1. Hugging Face Inference API (easiest, low cost)
    2. Self-hosted endpoint (cheapest for high volume)
    """

    def __init__(self, config):
        super().__init__(config)

        # Default to Hugging Face Inference API
        if config.base_url is None:
            self.api_url = "https://api-inference.huggingface.co/models/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
        else:
            self.api_url = config.base_url

        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"✅ FinGPT client initialized with endpoint: {self.api_url}")

    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using FinGPT."""
        start_time = time.time()

        try:
            # Prepare payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                    "temperature": kwargs.get('temperature', self.config.temperature),
                    "return_full_text": False
                }
            }

            # Make API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:

                    if response.status == 401:
                        raise LLMAuthenticationError("FinGPT API authentication failed")

                    if response.status == 429:
                        raise LLMError("FinGPT rate limit exceeded")

                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"FinGPT API error: {response.status} - {error_text}")

                    result = await response.json()

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract text from response
            # Hugging Face API returns array of dicts
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                text = result.get('generated_text', result.get('text', ''))
            else:
                text = str(result)

            # Token counting (not provided by HF API, estimate)
            input_tokens = self.count_tokens(prompt)
            output_tokens = self.count_tokens(text)

            # Calculate cost (fixed rate for HF API)
            cost = self.calculate_cost(input_tokens, output_tokens)

            # Update stats
            self.total_cost += cost
            self.total_calls += 1

            logger.debug(f"FinGPT API call: {latency_ms}ms, ~{input_tokens + output_tokens} tokens, ₹{cost:.4f}")

            return LLMResponse(
                text=text,
                model='fingpt-forecaster',
                tokens_used=input_tokens + output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                provider=LLMProvider.FINGPT,
                metadata={
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'endpoint': 'huggingface' if 'huggingface' in self.api_url else 'custom'
                }
            )

        except aiohttp.ClientError as e:
            self.failed_calls += 1
            logger.error(f"FinGPT network error: {e}")
            raise LLMError(f"FinGPT network error: {e}")

        except asyncio.TimeoutError:
            self.failed_calls += 1
            logger.error("FinGPT request timeout")
            raise LLMTimeoutError("FinGPT request timeout")

        except (LLMAuthenticationError, LLMError, LLMTimeoutError):
            # Re-raise our custom errors
            raise

        except Exception as e:
            self.failed_calls += 1
            logger.error(f"FinGPT error: {e}")
            raise LLMError(f"FinGPT error: {e}")

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for FinGPT.

        FinGPT uses Llama2 tokenizer (similar to GPT).
        """
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def get_cost_per_1k_tokens(self) -> Dict[str, float]:
        """
        Pricing in INR.

        Hugging Face Inference API:
        - Free tier: Limited requests
        - Paid: ~$0.06 per 1K input tokens, $0.12 per 1K output tokens

        For simplicity, using flat rate since HF charges per request.

        Returns:
            Dict with 'input' and 'output' costs per 1K tokens
        """
        if 'huggingface' in self.api_url:
            # HF Inference API: ~₹0.02 per request (amortized)
            # Approximate as token cost
            return {
                'input': 0.005,   # ~₹0.41 per 1K tokens
                'output': 0.010   # ~₹0.83 per 1K tokens
            }
        else:
            # Self-hosted: Cost depends on infrastructure
            # Assume free (user pays for compute separately)
            return {
                'input': 0.0,
                'output': 0.0
            }


# Import asyncio for timeout handling
import asyncio
