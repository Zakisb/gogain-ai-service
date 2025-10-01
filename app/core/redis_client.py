"""Async Redis client factory.

Provides a single shared asyncio Redis client configured via environment vars.

Env vars supported:
- `REDIS_URL` (e.g., redis://localhost:6379/0) takes precedence
- or `REDIS_HOST` (default: localhost), `REDIS_PORT` (default: 6379),
  `REDIS_DB` (default: 0), `REDIS_PASSWORD` (optional)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

try:
    import redis.asyncio as redis  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    redis = None  # Will raise at runtime if used without dependency installed


def _build_redis_url() -> str:
    url = os.getenv("REDIS_URL")
    if url:
        return url

    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    password = os.getenv("REDIS_PASSWORD")

    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


@lru_cache(maxsize=1)
def get_redis() -> "redis.Redis":
    """Return a shared asyncio Redis client instance.

    Raises RuntimeError if `redis` package is not installed.
    """
    if redis is None:
        raise RuntimeError(
            "Redis dependency not installed. Add 'redis' to requirements/minreqs."
        )
    url = _build_redis_url()
    return redis.from_url(url, encoding="utf-8", decode_responses=True)

