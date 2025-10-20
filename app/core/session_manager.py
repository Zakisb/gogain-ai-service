"""Session state management for multi-agent conversations.

This module provides session state tracking to manage which specialist agent
is currently active for each user, enabling proper context isolation and
handoff between the general agent and specialist agents (pain, training).

Architecture:
- General agent: Always sees FULL conversation history
- Specialist agents (pain, training): See only their own isolated history
- Active agent tracked in Redis: "general" | "pain" | "training"
- Automatic handoff when specialist returns done=True
- Manual handoff support for user-initiated switches
"""

from __future__ import annotations

import logging
from typing import Optional, Literal
from datetime import datetime

from app.core.redis_client import get_redis
from app.core.langchain_memory import (
    get_conversation_history,
    clear_conversation_history,
)

logger = logging.getLogger(__name__)

# Type alias for agent types
AgentType = Literal["general", "pain", "training"]

# Redis key pattern for active agent tracking
def _active_agent_key(user_id: str) -> str:
    return f"session:active_agent:{user_id}"

def _session_metadata_key(user_id: str) -> str:
    return f"session:metadata:{user_id}"

# Session TTL (keep session state for 3 hours of inactivity)
SESSION_TTL_SECONDS = 3600 * 3


class SessionManager:
    """Manages multi-agent session state for users."""

    @staticmethod
    async def get_active_agent(user_id: str) -> AgentType:
        """Get the currently active agent for a user.

        Args:
            user_id: User identifier

        Returns:
            Active agent type ("general", "pain", or "training")
            Defaults to "general" if no active session
        """
        try:
            r = get_redis()
            active = await r.get(_active_agent_key(user_id))

            if active in ("pain", "training", "general"):
                logger.debug(f"🎯 Active agent for user {user_id}: {active}")
                return active  # type: ignore

            logger.debug(f"🎯 No active agent for user {user_id}, defaulting to general")
            return "general"
        except Exception as e:
            logger.warning(f"Failed to get active agent for {user_id}: {e}")
            return "general"

    @staticmethod
    async def set_active_agent(
        user_id: str,
        agent_type: AgentType,
        metadata: Optional[dict] = None,
    ) -> None:
        """Set the active agent for a user.

        Args:
            user_id: User identifier
            agent_type: Agent to activate ("general", "pain", or "training")
            metadata: Optional metadata to store with session (e.g., body_region for pain)
        """
        try:
            r = get_redis()

            # Store active agent
            await r.set(
                _active_agent_key(user_id),
                agent_type,
                ex=SESSION_TTL_SECONDS
            )

            # Store metadata if provided
            if metadata:
                import json
                await r.set(
                    _session_metadata_key(user_id),
                    json.dumps(metadata),
                    ex=SESSION_TTL_SECONDS
                )

            logger.info(f"🎯 Set active agent for user {user_id}: {agent_type}")
        except Exception as e:
            logger.error(f"Failed to set active agent for {user_id}: {e}")

    @staticmethod
    async def get_session_metadata(user_id: str) -> dict:
        """Get session metadata for a user.

        Args:
            user_id: User identifier

        Returns:
            Session metadata dict (empty if none exists)
        """
        try:
            r = get_redis()
            import json
            raw = await r.get(_session_metadata_key(user_id))
            if raw:
                return json.loads(raw)
            return {}
        except Exception as e:
            logger.warning(f"Failed to get session metadata for {user_id}: {e}")
            return {}

    @staticmethod
    async def start_specialist_session(
        user_id: str,
        agent_type: Literal["pain", "training"],
        metadata: Optional[dict] = None,
    ) -> None:
        """Start a new specialist agent session.

        This:
        1. Sets the specialist as active agent
        2. Clears any existing specialist conversation history
        3. Stores session metadata

        Args:
            user_id: User identifier
            agent_type: Specialist agent type ("pain" or "training")
            metadata: Optional metadata (e.g., body_region for pain)
        """
        try:
            # Clear specialist history for fresh start
            clear_conversation_history(user_id, session_type=agent_type)

            # Set as active agent
            await SessionManager.set_active_agent(user_id, agent_type, metadata)

            logger.info(f"🚀 Started {agent_type} session for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to start {agent_type} session for {user_id}: {e}")

    @staticmethod
    async def end_specialist_session(
        user_id: str,
        agent_type: Literal["pain", "training"],
        clear_history: bool = True,
    ) -> None:
        """End a specialist agent session and return to general agent.

        This:
        1. Switches active agent back to "general"
        2. Optionally clears specialist conversation history
        3. Preserves general conversation history (which contains everything)

        Args:
            user_id: User identifier
            agent_type: Specialist agent that's ending ("pain" or "training")
            clear_history: Whether to clear specialist history (default: True)
        """
        try:
            # Clear specialist history if requested
            if clear_history:
                clear_conversation_history(user_id, session_type=agent_type)
                logger.info(f"🧹 Cleared {agent_type} history for user {user_id}")

            # Switch back to general agent
            await SessionManager.set_active_agent(user_id, "general")

            logger.info(f"🏁 Ended {agent_type} session for user {user_id}, returned to general")
        except Exception as e:
            logger.error(f"Failed to end {agent_type} session for {user_id}: {e}")

    @staticmethod
    async def store_message(
        user_id: str,
        message: str,
        role: Literal["user", "assistant", "system"],
        agent_context: Optional[AgentType] = None,
    ) -> None:
        """Store a message in the appropriate conversation histories.

        Messages are stored differently based on the active agent:
        - General agent: Messages stored ONLY in general history
        - Specialist agents: Messages stored in BOTH specialist and general history

        Args:
            user_id: User identifier
            message: Message content
            role: Message role ("user", "assistant", or "system")
            agent_context: Which agent context to use (None = use active agent)
        """
        try:
            # Determine which agent context to use
            if agent_context is None:
                agent_context = await SessionManager.get_active_agent(user_id)

            # Always store in general history (full conversation)
            general_history = get_conversation_history(user_id, session_type="general")

            if role == "user":
                general_history.add_user_message(message)
            elif role == "assistant":
                general_history.add_ai_message(message)
            # System messages not stored in LangChain history (added via prompt template)

            # If specialist is active, also store in specialist history
            if agent_context in ("pain", "training"):
                specialist_history = get_conversation_history(user_id, session_type=agent_context)

                if role == "user":
                    specialist_history.add_user_message(message)
                elif role == "assistant":
                    specialist_history.add_ai_message(message)

                logger.debug(f"📝 Stored {role} message in both general and {agent_context} history")
            else:
                logger.debug(f"📝 Stored {role} message in general history only")

        except Exception as e:
            logger.error(f"Failed to store message for {user_id}: {e}")

    @staticmethod
    async def should_switch_agent(
        user_id: str,
        user_message: str,
    ) -> Optional[AgentType]:
        """Detect if user wants to manually switch agents.

        Checks for explicit user commands to switch contexts:
        - "stop pain assessment", "cancel pain" → switch to general
        - "stop training", "cancel program" → switch to general

        Args:
            user_id: User identifier
            user_message: User's message

        Returns:
            Agent to switch to, or None if no switch detected
        """
        message_lower = user_message.lower()

        # Check for pain session cancellation
        if any(phrase in message_lower for phrase in [
            "stop pain", "cancel pain", "arrêter douleur", "annuler douleur",
            "stop assessment", "cancel assessment"
        ]):
            logger.info(f"🛑 User {user_id} manually ending pain session")
            return "general"

        # Check for training session cancellation
        if any(phrase in message_lower for phrase in [
            "stop training", "cancel training", "arrêter entraînement",
            "annuler entraînement", "stop program", "cancel program"
        ]):
            logger.info(f"🛑 User {user_id} manually ending training session")
            return "general"

        return None

    @staticmethod
    async def get_conversation_for_agent(
        user_id: str,
        agent_type: AgentType,
        limit: Optional[int] = None,
    ) -> list:
        """Get conversation history appropriate for a specific agent.

        - General agent: Gets FULL conversation history (no filtering)
        - Specialist agents: Get ONLY their isolated history

        Args:
            user_id: User identifier
            agent_type: Which agent's perspective to get
            limit: Maximum number of messages (None = all, only applies to general)

        Returns:
            List of LangChain BaseMessage instances
        """
        try:
            history = get_conversation_history(user_id, session_type=agent_type)
            messages = history.messages

            # Only apply limit to general agent (specialists see all their messages)
            if agent_type == "general" and limit and len(messages) > limit:
                messages = messages[-limit:]

            logger.debug(f"📖 Retrieved {len(messages)} messages for {agent_type} agent")
            return messages

        except Exception as e:
            logger.error(f"Failed to get conversation for {agent_type} agent: {e}")
            return []
