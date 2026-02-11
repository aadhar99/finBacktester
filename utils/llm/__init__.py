"""
LLM Integration Module

Provides unified interface for multiple LLM providers:
- Claude (Anthropic)
- Gemini (Google)
- GPT-4 (OpenAI)
- FinGPT (Finance-specific)

Usage:
    from utils.llm import LLMManager, LLMConfig, LLMProvider

    configs = [
        LLMConfig(provider=LLMProvider.CLAUDE, model='sonnet', api_key='...'),
        LLMConfig(provider=LLMProvider.GEMINI, model='flash', api_key='...')
    ]

    llm = LLMManager(configs)
    response = await llm.complete("Analyze this trade...")
"""

from .base import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    BaseLLMClient
)
from .manager import LLMManager

__all__ = [
    'LLMProvider',
    'LLMConfig',
    'LLMResponse',
    'BaseLLMClient',
    'LLMManager'
]
