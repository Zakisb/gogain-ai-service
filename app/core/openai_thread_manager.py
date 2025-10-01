"""
OpenAI Thread-based session management for persistent conversations.
Uses OpenAI's Threads API for conversation history instead of manual message management.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from pathlib import Path

logger = logging.getLogger(__name__)

class OpenAIThreadManager:
    """Manages OpenAI thread-based conversations for users."""
    
    def __init__(self, storage_dir: str = "app/thread_sessions"):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_thread_mapping_file(self, user_id: str) -> Path:
        """Get the thread mapping file path for a user."""
        return self.storage_dir / f"{user_id}_thread.json"
    
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
                    # Verify thread still exists
                    try:
                        await self.client.beta.threads.retrieve(thread_id)
                        logger.info(f"Using existing thread {thread_id} for user {user_id}")
                        return thread_id
                    except Exception:
                        logger.warning(f"Thread {thread_id} no longer exists, creating new one")
                        
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Failed to load thread mapping for user {user_id}: {e}")
        
        # Create new thread
        try:
            thread = await self.client.beta.threads.create()
            thread_id = thread.id
            
            # Save thread mapping
            thread_data = {
                "user_id": user_id,
                "thread_id": thread_id,
                "created_at": thread.created_at
            }
            
            with open(thread_file, "w") as f:
                json.dump(thread_data, f, indent=2)
                
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
            message = await self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=content
            )
            
            logger.info(f"Added {role} message to thread {thread_id} for user {user_id}")
            return message.id
            
        except Exception as e:
            logger.error(f"Failed to add message to thread for user {user_id}: {e}")
            raise
    
    async def get_conversation_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history from user's thread."""
        try:
            thread_id = await self.get_or_create_thread(user_id)
            
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread_id,
                limit=limit,
                order="asc"
            )
            
            history = []
            for message in messages.data:
                content = ""
                if message.content and len(message.content) > 0:
                    # Handle different content types
                    if hasattr(message.content[0], 'text'):
                        content = message.content[0].text.value
                    else:
                        content = str(message.content[0])
                
                history.append({
                    "role": message.role,
                    "content": content,
                    "created_at": message.created_at,
                    "message_id": message.id
                })
            
            logger.info(f"Retrieved {len(history)} messages from thread for user {user_id}")
            return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history for user {user_id}: {e}")
            return []
    
    async def create_run_with_assistant(self, user_id: str, assistant_id: str) -> str:
        """Create a run with a specific assistant."""
        thread_id = await self.get_or_create_thread(user_id)
        
        try:
            run = await self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id
            )
            
            logger.info(f"Created run {run.id} with assistant {assistant_id} for user {user_id}")
            return run.id
            
        except Exception as e:
            logger.error(f"Failed to create run for user {user_id}: {e}")
            raise
    
    async def wait_for_run_completion(self, user_id: str, run_id: str, timeout: int = 30) -> Dict[str, Any]:
        """Wait for run to complete and return final status."""
        thread_id = await self.get_or_create_thread(user_id)
        
        try:
            import asyncio
            for _ in range(timeout):
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run_id
                )
                
                if run.status in ["completed", "failed", "cancelled", "expired"]:
                    logger.info(f"Run {run_id} finished with status: {run.status}")
                    return {
                        "status": run.status,
                        "run_id": run_id,
                        "completed_at": run.completed_at
                    }
                
                await asyncio.sleep(1)
            
            logger.warning(f"Run {run_id} timed out after {timeout}s")
            return {"status": "timeout", "run_id": run_id}
            
        except Exception as e:
            logger.error(f"Failed to wait for run completion: {e}")
            return {"status": "error", "run_id": run_id, "error": str(e)}
    
    async def get_latest_assistant_message(self, user_id: str) -> Optional[str]:
        """Get the latest assistant message from the thread."""
        try:
            thread_id = await self.get_or_create_thread(user_id)
            
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread_id,
                limit=1,
                order="desc"
            )
            
            if messages.data and messages.data[0].role == "assistant":
                message = messages.data[0]
                if message.content and len(message.content) > 0:
                    if hasattr(message.content[0], 'text'):
                        return message.content[0].text.value
                    else:
                        return str(message.content[0])
            
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
                    # Delete the OpenAI thread
                    try:
                        await self.client.beta.threads.delete(thread_id)
                        logger.info(f"Deleted OpenAI thread {thread_id}")
                    except Exception as e:
                        logger.warning(f"Failed to delete OpenAI thread {thread_id}: {e}")
                
                # Remove local mapping
                thread_file.unlink()
                logger.info(f"Cleared thread mapping for user {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to clear thread for user {user_id}: {e}")
    
    async def get_thread_info(self, user_id: str) -> Dict[str, Any]:
        """Get thread information for debugging."""
        try:
            thread_id = await self.get_or_create_thread(user_id)
            thread = await self.client.beta.threads.retrieve(thread_id)
            
            return {
                "thread_id": thread_id,
                "created_at": thread.created_at,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get thread info for user {user_id}: {e}")
            return {"error": str(e)}