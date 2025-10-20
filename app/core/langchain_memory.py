"""LangChain-native message history management using Redis.

This module provides conversation history management using LangChain's
RedisChatMessageHistory for persistent, server-side conversation tracking.
"""

from typing import List, Optional
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


def add_system_context_messages(
    history: RedisChatMessageHistory,
    user_profile: Optional[str] = None,
    user_memories: Optional[str] = None,
    training_snapshot: Optional[str] = None,
) -> None:
    """Add system context messages to history (user data, memories, etc.).

    These are typically added at the start of a conversation or when context changes.

    Args:
        history: RedisChatMessageHistory instance
        user_profile: Formatted user profile text
        user_memories: Formatted user memories text
        training_snapshot: Formatted training program text
    """
    # Note: System messages in LangChain are typically added via prompt template,
    # not persisted in history. This function is for reference only.
    #
    # For persistent system context, use the prompt template's {context} placeholder
    # as currently implemented.
    pass


def get_messages_for_prompt(
    user_id: str,
    session_type: str = "general",
    limit: Optional[int] = None,
) -> List[BaseMessage]:
    """Get conversation messages formatted for LangChain prompt.

    Args:
        user_id: User identifier
        session_type: Type of session
        limit: Maximum number of messages to retrieve (None = all)

    Returns:
        List of LangChain BaseMessage instances (most recent first if limited)
    """
    history = get_conversation_history(user_id, session_type)
    messages = history.messages

    if limit and len(messages) > limit:
        # Keep most recent messages
        messages = messages[-limit:]

    logger.debug(f"Retrieved {len(messages)} messages for user_id={user_id}")
    return messages


def clear_conversation_history(user_id: str, session_type: str = "general") -> None:
    """Clear all messages from a user's conversation history.

    Args:
        user_id: User identifier
        session_type: Type of session to clear
    """
    history = get_conversation_history(user_id, session_type)
    history.clear()
    logger.info(f"Cleared conversation history for user_id={user_id}, session_type={session_type}")
