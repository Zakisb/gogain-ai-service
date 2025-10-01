"""
Gemini Thread-based session management for persistent conversations.
Implements the same interface as OpenAIThreadManager but uses local storage
since Gemini doesn't have a native threads API.
"""

import json
import logging
import os
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class GeminiThreadManager:
    """Manages Gemini thread-based conversations for users using local storage."""

    def __init__(self, storage_dir: str = "app/thread_sessions/gemini"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized GeminiThreadManager with storage: {self.storage_dir}")

    def _get_thread_mapping_file(self, user_id: str) -> Path:
        """Get the thread mapping file path for a user."""
        return self.storage_dir / f"{user_id}_thread.json"

    def _get_conversation_file(self, thread_id: str) -> Path:
        """Get the conversation history file for a thread."""
        return self.storage_dir / f"conversation_{thread_id}.json"

    async def get_or_create_thread(self, user_id: str, force_new: bool = False) -> str:
        """Get existing thread ID or create a new one for user.

        Args:
            user_id: User identifier
            force_new: If True, create a new thread even if one exists
        """
        thread_file = self._get_thread_mapping_file(user_id)

        # If force_new is True, skip existing thread lookup and create new one
        if force_new:
            logger.info(f"Force creating new thread for user {user_id} (cleanup requested)")

        # Try to load existing thread
        elif thread_file.exists():
            try:
                with open(thread_file, "r") as f:
                    thread_data = json.load(f)
                    thread_id = thread_data.get("thread_id")

                if thread_id:
                    # Verify thread conversation file exists
                    conversation_file = self._get_conversation_file(thread_id)
                    if conversation_file.exists():
                        logger.info(f"Using existing thread {thread_id} for user {user_id}")
                        return thread_id
                    else:
                        logger.warning(f"Thread {thread_id} conversation file missing, creating new one")

            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Failed to load thread mapping for user {user_id}: {e}")

        # Create new thread
        try:
            thread_id = f"gemini_thread_{uuid.uuid4().hex[:16]}"
            created_at = int(datetime.now().timestamp())

            # Save thread mapping
            thread_data = {
                "user_id": user_id,
                "thread_id": thread_id,
                "created_at": created_at,
                "provider": "gemini"
            }

            with open(thread_file, "w") as f:
                json.dump(thread_data, f, indent=2)

            # Initialize empty conversation file
            conversation_data = {
                "thread_id": thread_id,
                "user_id": user_id,
                "created_at": created_at,
                "messages": []
            }

            conversation_file = self._get_conversation_file(thread_id)
            with open(conversation_file, "w") as f:
                json.dump(conversation_data, f, indent=2)

            logger.info(f"Created new thread {thread_id} for user {user_id}")
            return thread_id

        except Exception as e:
            logger.error(f"Failed to create thread for user {user_id}: {e}")
            raise

    async def clear_thread(self, user_id: str) -> str:
        """Create a fresh thread for user, effectively 'clearing' conversation history.

        Args:
            user_id: User identifier

        Returns:
            New thread ID
        """
        return await self.get_or_create_thread(user_id, force_new=True)

    async def add_message(self, user_id: str, content: str, role: str = "user") -> str:
        """Add a message to user's thread and return message ID."""
        thread_id = await self.get_or_create_thread(user_id)

        try:
            conversation_file = self._get_conversation_file(thread_id)

            # Load existing conversation
            if conversation_file.exists():
                with open(conversation_file, "r") as f:
                    conversation_data = json.load(f)
            else:
                conversation_data = {
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "created_at": int(datetime.now().timestamp()),
                    "messages": []
                }

            # Create message
            message_id = f"msg_{uuid.uuid4().hex[:12]}"
            message = {
                "id": message_id,
                "role": role,
                "content": content,
                "created_at": int(datetime.now().timestamp())
            }

            # Add message to conversation
            conversation_data["messages"].append(message)

            # Keep only last 50 messages to prevent file bloat
            if len(conversation_data["messages"]) > 50:
                conversation_data["messages"] = conversation_data["messages"][-50:]

            # Save updated conversation
            with open(conversation_file, "w") as f:
                json.dump(conversation_data, f, indent=2)

            logger.info(f"Added {role} message to thread {thread_id} for user {user_id}")
            return message_id

        except Exception as e:
            logger.error(f"Failed to add message to thread for user {user_id}: {e}")
            raise

    async def get_conversation_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history from user's thread."""
        try:
            thread_id = await self.get_or_create_thread(user_id)
            conversation_file = self._get_conversation_file(thread_id)

            if not conversation_file.exists():
                logger.info(f"No conversation history found for user {user_id}")
                return []

            with open(conversation_file, "r") as f:
                conversation_data = json.load(f)

            messages = conversation_data.get("messages", [])

            # Apply limit
            if limit and len(messages) > limit:
                messages = messages[-limit:]

            # Convert to OpenAI-compatible format
            history = []
            for message in messages:
                history.append({
                    "role": message["role"],
                    "content": message["content"],
                    "created_at": message["created_at"],
                    "message_id": message["id"]
                })

            logger.info(f"Retrieved {len(history)} messages from thread for user {user_id}")
            return history

        except Exception as e:
            logger.error(f"Failed to get conversation history for user {user_id}: {e}")
            return []

    async def create_run_with_assistant(self, user_id: str, assistant_id: str) -> str:
        """Create a run with a specific assistant (Gemini doesn't use this concept)."""
        # Gemini doesn't have assistants/runs, so we'll just return a dummy run ID
        thread_id = await self.get_or_create_thread(user_id)
        run_id = f"gemini_run_{uuid.uuid4().hex[:12]}"

        logger.info(f"Created dummy run {run_id} for thread {thread_id} (Gemini doesn't use assistants)")
        return run_id

    async def wait_for_run_completion(self, user_id: str, run_id: str, timeout: int = 30) -> Dict[str, Any]:
        """Wait for run to complete (Gemini doesn't use this concept)."""
        # Gemini doesn't have runs, so we'll immediately return completed status
        logger.info(f"Dummy run {run_id} marked as completed (Gemini doesn't use runs)")
        return {
            "status": "completed",
            "run_id": run_id,
            "completed_at": int(datetime.now().timestamp())
        }

    async def get_latest_assistant_message(self, user_id: str) -> Optional[str]:
        """Get the latest assistant message from the thread."""
        try:
            history = await self.get_conversation_history(user_id, limit=10)

            # Find the most recent assistant message
            for message in reversed(history):
                if message["role"] == "assistant":
                    return message["content"]

            return None

        except Exception as e:
            logger.error(f"Failed to get latest assistant message for user {user_id}: {e}")
            return None

    async def clear_thread(self, user_id: str) -> None:
        """Delete thread and mapping for user."""
        thread_file = self._get_thread_mapping_file(user_id)

        try:
            if thread_file.exists():
                with open(thread_file, "r") as f:
                    thread_data = json.load(f)
                    thread_id = thread_data.get("thread_id")

                if thread_id:
                    # Delete the conversation file
                    conversation_file = self._get_conversation_file(thread_id)
                    if conversation_file.exists():
                        conversation_file.unlink()
                        logger.info(f"Deleted conversation file for thread {thread_id}")

                # Remove local mapping
                thread_file.unlink()
                logger.info(f"Cleared thread mapping for user {user_id}")

        except Exception as e:
            logger.error(f"Failed to clear thread for user {user_id}: {e}")

    async def get_thread_info(self, user_id: str) -> Dict[str, Any]:
        """Get thread information for debugging."""
        try:
            thread_id = await self.get_or_create_thread(user_id)
            thread_file = self._get_thread_mapping_file(user_id)

            if thread_file.exists():
                with open(thread_file, "r") as f:
                    thread_data = json.load(f)

                return {
                    "thread_id": thread_id,
                    "created_at": thread_data.get("created_at"),
                    "user_id": user_id,
                    "provider": "gemini"
                }

            return {"error": "Thread mapping not found"}

        except Exception as e:
            logger.error(f"Failed to get thread info for user {user_id}: {e}")
            return {"error": str(e)}