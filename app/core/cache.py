from __future__ import annotations

"""High-level cache helpers backed by Redis with TTLs.

Caches:
- Conversation history per user: list of JSON entries with role/content
- Pain assessment sessions per user: active flag + messages list

All functions are async and resilient: on Redis errors they log and return safe defaults.
"""

import os
import json
import logging
from typing import Any, Dict, List

from .redis_client import get_redis

logger = logging.getLogger(__name__)


# --- Configuration (with sane defaults) ---
CONVERSATION_TTL_SECONDS = int(os.getenv("CONVERSATION_TTL_SECONDS", "7200"))  # 2h
CONVERSATION_MAX_LENGTH = int(os.getenv("CONVERSATION_MAX_LENGTH", "20"))

PAIN_SESSION_TTL_SECONDS = int(os.getenv("PAIN_SESSION_TTL_SECONDS", "10800"))  # 3h


def _conv_key(user_id: str) -> str:
    return f"conv:{user_id}"


def _pain_msgs_key(user_id: str) -> str:
    return f"pain:msgs:{user_id}"


def _pain_active_key(user_id: str) -> str:
    return f"pain:active:{user_id}"

# --- Conversation history ---
async def get_conversation_history(user_id: str, limit: int = 3) -> List[Dict[str, str]]:
    try:
        r = get_redis()
        key = _conv_key(user_id)
        if limit and limit > 0:
            raw = await r.lrange(key, -limit, -1)
        else:
            raw = await r.lrange(key, 0, -1)
        out: List[Dict[str, str]] = []
        for item in raw:
            try:
                obj = json.loads(item)
                if isinstance(obj, dict) and "role" in obj and "content" in obj:
                    out.append({"role": obj["role"], "content": obj["content"]})
            except Exception:
                continue
        return out
    except Exception as e:
        logger.warning(f"Conversation cache read failed for {user_id}: {e}")
        return []


async def add_conversation_message(user_id: str, message: str, role: str) -> None:
    try:
        r = get_redis()
        key = _conv_key(user_id)
        payload = json.dumps({"role": role, "content": message}, ensure_ascii=False)
        await r.rpush(key, payload)
        # Keep only the last N messages and refresh TTL
        await r.ltrim(key, -CONVERSATION_MAX_LENGTH, -1)
        await r.expire(key, CONVERSATION_TTL_SECONDS)
    except Exception as e:
        logger.warning(f"Conversation cache write failed for {user_id}: {e}")


# --- Pain agent sessions ---
async def start_pain_agent_session(user_id: str) -> None:
    try:
        r = get_redis()
        # Reset messages and mark active
        await r.delete(_pain_msgs_key(user_id))
        await r.set(_pain_active_key(user_id), "1", ex=PAIN_SESSION_TTL_SECONDS)
    except Exception as e:
        logger.warning(f"Pain session start failed for {user_id}: {e}")


async def add_to_pain_agent_session(user_id: str, message: str, role: str) -> None:
    try:
        r = get_redis()
        active_key = _pain_active_key(user_id)
        if (await r.get(active_key)) != "1":
            await start_pain_agent_session(user_id)

        payload = json.dumps({"role": role, "content": message}, ensure_ascii=False)
        msgs_key = _pain_msgs_key(user_id)
        await r.rpush(msgs_key, payload)
        # Refresh TTLs on both keys to keep the session alive while active
        await r.expire(active_key, PAIN_SESSION_TTL_SECONDS)
        await r.expire(msgs_key, PAIN_SESSION_TTL_SECONDS)
    except Exception as e:
        logger.warning(f"Pain session append failed for {user_id}: {e}")


async def get_pain_agent_session_history(user_id: str) -> List[Dict[str, str]]:
    try:
        r = get_redis()
        if (await r.get(_pain_active_key(user_id))) != "1":
            return []
        raw = await r.lrange(_pain_msgs_key(user_id), 0, -1)
        out: List[Dict[str, str]] = []
        for item in raw:
            try:
                obj = json.loads(item)
                if isinstance(obj, dict) and "role" in obj and "content" in obj:
                    out.append({"role": obj["role"], "content": obj["content"]})
            except Exception:
                continue
        return out
    except Exception as e:
        logger.warning(f"Pain session read failed for {user_id}: {e}")
        return []


async def end_pain_agent_session(user_id: str) -> None:
    try:
        r = get_redis()
        await r.set(_pain_active_key(user_id), "0", ex=PAIN_SESSION_TTL_SECONDS)
        await r.expire(_pain_msgs_key(user_id), PAIN_SESSION_TTL_SECONDS)
    except Exception as e:
        logger.warning(f"Pain session end failed for {user_id}: {e}")


async def is_pain_agent_session_active(user_id: str) -> bool:
    try:
        r = get_redis()
        return (await r.get(_pain_active_key(user_id))) == "1"
    except Exception:
        return False
