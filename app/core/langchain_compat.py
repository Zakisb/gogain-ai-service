"""Compatibility helpers for LangChain LLM classes.

Ensures ChatOpenAI is fully initialised with required forward references so that
Pydantic schema rebuilding does not fail at runtime.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _ensure_chat_openai_ready() -> None:
    """Patch ChatOpenAI module globals and rebuild Pydantic schema if needed."""
    try:
        import langchain_openai.chat_models.base as base_module
        from langchain_core.caches import BaseCache  # type: ignore
        from langchain_core.callbacks import Callbacks  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.debug("ChatOpenAI compatibility import skipped: %s", exc)
        return

    try:
        if getattr(base_module, "BaseCache", None) is None:
            setattr(base_module, "BaseCache", BaseCache)
        if getattr(base_module, "Callbacks", None) is None:
            setattr(base_module, "Callbacks", Callbacks)
        ChatOpenAI.model_rebuild(raise_errors=False)
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("ChatOpenAI model_rebuild skipped: %s", exc)


_ensure_chat_openai_ready()

__all__ = ["ChatOpenAI"]
