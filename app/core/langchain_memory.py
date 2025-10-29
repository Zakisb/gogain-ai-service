"""LangChain-native message history management using Redis.

This module provides conversation history management using LangChain's
RedisChatMessageHistory for persistent, server-side conversation tracking.
"""

from typing import Optional
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from app.core.redis_client import get_redis_url
import logging

logger = logging.getLogger(__name__)


def get_conversation_history(
    user_id: str,
    session_type: str = "general",
    ttl: Optional[int] = 3600 * 24 * 7,  # 7 days default
) -> RedisChatMessageHistory:
    """Get LangChain RedisChatMessageHistory for a user's conversation.

    Args:
        user_id: User identifier
        session_type: Type of session ("general", "pain", "training")
        ttl: Time-to-live in seconds (None = no expiration)

    Returns:
        RedisChatMessageHistory instance connected to Redis

    Example:
        >>> history = get_conversation_history("user_123")
        >>> history.add_user_message("Hello!")
        >>> history.add_ai_message("Hi there!")
        >>> messages = history.messages  # List[BaseMessage]
    """
    session_id = f"chat:{user_id}:{session_type}"

    try:
        redis_url = get_redis_url()

        history = RedisChatMessageHistory(
            session_id=session_id,
            url=redis_url,
            ttl=ttl,
        )

        logger.debug(f"Created RedisChatMessageHistory for session_id={session_id}")
        return history

    except Exception as e:
        logger.error(f"Failed to create RedisChatMessageHistory: {e}")
        raise


def clear_conversation_history(user_id: str, session_type: str = "general") -> None:
    """Clear all messages from a user's conversation history.

    Args:
        user_id: User identifier
        session_type: Type of session to clear
    """
    history = get_conversation_history(user_id, session_type)
    history.clear()
    logger.info(f"Cleared conversation history for user_id={user_id}, session_type={session_type}")
