"""
Thread Manager Factory - Creates appropriate thread manager based on model provider.
Allows seamless switching between OpenAI and Gemini without breaking existing code.
"""

import os
import logging
from typing import Union

from app.core.openai_thread_manager import OpenAIThreadManager
from app.core.gemini_thread_manager import GeminiThreadManager

logger = logging.getLogger(__name__)


class ThreadManagerFactory:
    """Factory to create appropriate thread manager based on model provider."""

    @staticmethod
    def create_thread_manager(model_name: str = None) -> Union[OpenAIThreadManager, GeminiThreadManager]:
        """
        Create appropriate thread manager based on model.

        Args:
            model_name: The LLM model name (e.g., "gpt-4o", "gemini/gemini-2.0-flash")

        Returns:
            Appropriate thread manager instance
        """
        # If no model specified, use environment or default
        if not model_name:
            from app.core.config import DEFAULT_MODEL
            model_name = DEFAULT_MODEL

        # Check for manual override first
        from app.core.config import THREAD_MANAGER_PROVIDER
        if THREAD_MANAGER_PROVIDER != "auto":
            if THREAD_MANAGER_PROVIDER == "gemini":
                logger.info(f"Using GeminiThreadManager (manual override) for model: {model_name}")
                return GeminiThreadManager()
            elif THREAD_MANAGER_PROVIDER == "openai":
                logger.info(f"Using OpenAIThreadManager (manual override) for model: {model_name}")
                return OpenAIThreadManager()

        # Auto-determine provider from model name
        if model_name.startswith(("gpt-", "gpt4", "o1-")):
            provider = "openai"
        elif model_name.startswith(("gemini/", "gemini-")):
            provider = "gemini"
        elif "claude" in model_name.lower():
            provider = "openai"  # Claude models share the OpenAI-style thread manager integration
        else:
            # Default to OpenAI for unknown providers
            provider = "openai"
            logger.warning(f"Unknown model provider for {model_name}, defaulting to OpenAI thread manager")

        # Create appropriate manager
        if provider == "gemini":
            logger.info(f"Creating GeminiThreadManager (auto-detected) for model: {model_name}")
            return GeminiThreadManager()
        else:
            logger.info(f"Creating OpenAIThreadManager (auto-detected) for model: {model_name}")
            return OpenAIThreadManager()


# Convenience function for backward compatibility
def get_thread_manager(model_name: str = None) -> Union[OpenAIThreadManager, GeminiThreadManager]:
    """Get appropriate thread manager instance."""
    return ThreadManagerFactory.create_thread_manager(model_name)


# Global instance for current session (backwards compatibility)
# This will be initialized when first imported
_default_thread_manager = None


def get_default_thread_manager() -> Union[OpenAIThreadManager, GeminiThreadManager]:
    """Get default thread manager instance (singleton pattern)."""
    global _default_thread_manager

    if _default_thread_manager is None:
        _default_thread_manager = ThreadManagerFactory.create_thread_manager()
        logger.info(f"Initialized default thread manager: {type(_default_thread_manager).__name__}")

    return _default_thread_manager


def set_thread_manager_for_model(model_name: str) -> Union[OpenAIThreadManager, GeminiThreadManager]:
    """Set and return thread manager appropriate for specified model."""
    global _default_thread_manager

    _default_thread_manager = ThreadManagerFactory.create_thread_manager(model_name)
    logger.info(f"Switched to thread manager: {type(_default_thread_manager).__name__} for model: {model_name}")

    return _default_thread_manager
