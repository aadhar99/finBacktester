"""
LLM Manager - Unified interface for multiple LLM providers with automatic fallback.
"""

import logging
from typing import List, Optional
from .base import LLMProvider, LLMConfig, LLMResponse, BaseLLMClient, LLMError
from .claude import ClaudeClient
from .gemini import GeminiClient
from .openai import OpenAIClient
from .fingpt import FinGPTClient

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Unified LLM manager with automatic fallback.

    Features:
    - Plug-and-play provider switching
    - Automatic fallback on errors
    - Cost tracking across all providers
    - Smart routing based on task type
    - Performance monitoring

    Example:
        configs = [
            LLMConfig(provider=LLMProvider.CLAUDE, model='sonnet', api_key='...'),
            LLMConfig(provider=LLMProvider.GEMINI, model='flash', api_key='...'),
        ]

        llm = LLMManager(configs)
        response = await llm.complete("Analyze this trade...")
    """

    def __init__(self, configs: List[LLMConfig]):
        """
        Initialize with multiple providers (in priority order).

        Args:
            configs: List of LLMConfig objects in priority order
                    First config = primary provider
                    Rest = fallback providers
        """
        self.providers: List[BaseLLMClient] = []

        for config in configs:
            provider = self._create_provider(config)
            if provider:
                self.providers.append(provider)

        if not self.providers:
            raise ValueError("At least one LLM provider must be configured")

        self.primary = self.providers[0]
        self.fallbacks = self.providers[1:] if len(self.providers) > 1 else []

        logger.info(f"✅ LLM Manager initialized with {len(self.providers)} provider(s)")
        logger.info(f"   Primary: {self.primary.config.provider.value} ({self.primary.config.model})")
        if self.fallbacks:
            fallback_info = [f"{p.config.provider.value} ({p.config.model})" for p in self.fallbacks]
            logger.info(f"   Fallbacks: {', '.join(fallback_info)}")

    def _create_provider(self, config: LLMConfig) -> Optional[BaseLLMClient]:
        """Create provider client based on config."""
        try:
            if config.provider == LLMProvider.CLAUDE:
                return ClaudeClient(config)
            elif config.provider == LLMProvider.GEMINI:
                return GeminiClient(config)
            elif config.provider == LLMProvider.GPT4:
                return OpenAIClient(config)
            elif config.provider == LLMProvider.FINGPT:
                return FinGPTClient(config)
            else:
                logger.warning(f"Unknown provider: {config.provider}")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize {config.provider.value}: {e}")
            return None

    async def complete(
        self,
        prompt: str,
        task_type: str = 'general',
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion with automatic fallback.

        Args:
            prompt: Input prompt
            task_type: Type of task (used for smart routing)
                - 'general': Use primary provider
                - 'sentiment': Use FinGPT if available, else primary
                - 'fast': Use cheapest/fastest provider (Gemini Flash)
                - 'critical': Use most capable provider (Claude Opus/GPT-4)
            **kwargs: Additional parameters passed to provider

        Returns:
            LLMResponse with result

        Raises:
            LLMError: If all providers fail
        """
        # Smart routing based on task type
        provider = self._select_provider(task_type)

        # Try selected provider
        try:
            response = await provider.complete(prompt, **kwargs)
            logger.info(
                f"✅ {provider.config.provider.value}: {response.latency_ms}ms, "
                f"{response.tokens_used} tokens, ₹{response.cost:.4f}"
            )
            return response

        except Exception as e:
            logger.warning(
                f"⚠️ {provider.config.provider.value} failed: {e}"
            )

            # Try fallbacks
            for fallback in self.fallbacks:
                # Skip if this fallback was the one that just failed
                if fallback == provider:
                    continue

                try:
                    logger.info(f"🔄 Trying fallback: {fallback.config.provider.value}")
                    response = await fallback.complete(prompt, **kwargs)
                    logger.info(
                        f"✅ {fallback.config.provider.value} (fallback): {response.latency_ms}ms, "
                        f"{response.tokens_used} tokens, ₹{response.cost:.4f}"
                    )
                    return response

                except Exception as fallback_error:
                    logger.warning(
                        f"⚠️ {fallback.config.provider.value} (fallback) failed: {fallback_error}"
                    )
                    continue

            # All providers failed
            error_msg = f"All LLM providers failed. Primary error: {e}"
            logger.error(error_msg)
            raise LLMError(error_msg)

    def _select_provider(self, task_type: str) -> BaseLLMClient:
        """
        Select optimal provider based on task type.

        Args:
            task_type: Type of task

        Returns:
            Selected provider
        """
        if task_type == 'sentiment':
            # Use FinGPT for sentiment if available
            fingpt = next(
                (p for p in self.providers if p.config.provider == LLMProvider.FINGPT),
                None
            )
            if fingpt:
                logger.debug("Routing sentiment task to FinGPT")
                return fingpt

        elif task_type == 'fast':
            # Use Gemini Flash for speed and cost
            gemini = next(
                (p for p in self.providers if p.config.provider == LLMProvider.GEMINI),
                None
            )
            if gemini:
                logger.debug("Routing fast task to Gemini")
                return gemini

        elif task_type == 'critical':
            # Use most capable: Claude Opus > GPT-4 > Claude Sonnet
            opus = next(
                (p for p in self.providers if 'opus' in p.config.model.lower()),
                None
            )
            if opus:
                logger.debug("Routing critical task to Claude Opus")
                return opus

            gpt4 = next(
                (p for p in self.providers if p.config.provider == LLMProvider.GPT4 and 'mini' not in p.config.model.lower()),
                None
            )
            if gpt4:
                logger.debug("Routing critical task to GPT-4")
                return gpt4

        # Default: use primary
        logger.debug(f"Using primary provider for {task_type} task")
        return self.primary

    def get_total_cost(self) -> float:
        """Get total cost across all providers in INR."""
        return sum(p.total_cost for p in self.providers)

    def get_stats(self) -> dict:
        """
        Get comprehensive usage statistics.

        Returns:
            Dict with overall stats and per-provider breakdown
        """
        total_calls = sum(p.total_calls for p in self.providers)
        total_failed = sum(p.failed_calls for p in self.providers)

        return {
            'total_cost': self.get_total_cost(),
            'total_calls': total_calls,
            'total_failed': total_failed,
            'success_rate': (total_calls - total_failed) / max(total_calls, 1),
            'by_provider': {
                p.config.provider.value: p.get_stats()
                for p in self.providers
            }
        }

    def reset_stats(self):
        """Reset all usage statistics."""
        for provider in self.providers:
            provider.total_cost = 0.0
            provider.total_calls = 0
            provider.failed_calls = 0
        logger.info("📊 LLM stats reset")

    async def test_providers(self) -> dict:
        """
        Test all configured providers.

        Returns:
            Dict with test results for each provider
        """
        test_prompt = "Respond with 'OK' if you receive this."

        results = {}

        for provider in self.providers:
            provider_name = f"{provider.config.provider.value} ({provider.config.model})"
            try:
                response = await provider.complete(test_prompt, max_tokens=10)
                results[provider_name] = {
                    'status': 'success',
                    'latency_ms': response.latency_ms,
                    'cost': response.cost
                }
                logger.info(f"✅ {provider_name} test passed")
            except Exception as e:
                results[provider_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                logger.error(f"❌ {provider_name} test failed: {e}")

        return results
