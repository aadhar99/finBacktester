"""
Base classes and interfaces for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    CLAUDE = "claude"
    GEMINI = "gemini"
    GPT4 = "gpt4"
    FINGPT = "fingpt"
    CUSTOM = "custom"


@dataclass
class LLMResponse:
    """Standardized response from any LLM."""
    text: str
    model: str
    tokens_used: int
    cost: float  # In INR
    latency_ms: int
    provider: LLMProvider
    metadata: Optional[Dict] = None


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: LLMProvider
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30  # seconds
    rate_limit: Optional[int] = None  # calls per minute
    base_url: Optional[str] = None  # For custom endpoints


class BaseLLMClient(ABC):
    """Base class for all LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.total_cost = 0.0
        self.total_calls = 0
        self.failed_calls = 0

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate completion from prompt.

        Args:
            prompt: Input text prompt
            **kwargs: Additional provider-specific parameters
                - temperature: Override default temperature
                - max_tokens: Override default max tokens
                - json_mode: Request JSON response (if supported)

        Returns:
            LLMResponse with generated text and metadata
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Approximate token count
        """
        pass

    @abstractmethod
    def get_cost_per_1k_tokens(self) -> Dict[str, float]:
        """
        Get pricing (input and output per 1k tokens in INR).

        Returns:
            Dict with 'input' and 'output' keys
        """
        pass

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost in INR.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in INR
        """
        pricing = self.get_cost_per_1k_tokens()
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        return input_cost + output_cost

    def get_stats(self) -> Dict:
        """Get usage statistics for this provider."""
        return {
            'provider': self.config.provider.value,
            'model': self.config.model,
            'total_calls': self.total_calls,
            'failed_calls': self.failed_calls,
            'success_rate': (self.total_calls - self.failed_calls) / max(self.total_calls, 1),
            'total_cost': self.total_cost
        }


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication fails."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when request times out."""
    pass
