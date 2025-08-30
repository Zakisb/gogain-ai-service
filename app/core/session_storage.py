"""
File-based session storage for AgentWorkflow context and conversation memory.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.workflow import Context, JsonSerializer, JsonPickleSerializer

logger = logging.getLogger(__name__)

class SessionStorage:
    """File-based storage for user sessions (context + memory)."""
    
    def __init__(self, storage_dir: str = "app/chat_sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_context_file(self, user_id: str) -> Path:
        """Get the context file path for a user."""
        return self.storage_dir / f"{user_id}_context.json"
    
    def _get_memory_file(self, user_id: str) -> Path:
        """Get the memory file path for a user."""
        return self.storage_dir / f"{user_id}_memory.json"
    
    async def load_context(self, user_id: str) -> Dict[str, Any]:
        """Load workflow context for a user with proper serializer handling."""
        context_file = self._get_context_file(user_id)
        
        if context_file.exists():
            try:
                with open(context_file, "r") as f:
                    context_data = json.load(f)
                logger.info(f"Loaded context for user {user_id}")
                return context_data
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Failed to load context for user {user_id}: {e}")
                return {}
        else:
            logger.info(f"No existing context found for user {user_id}")
            return {}
    
    def restore_context_from_dict(self, workflow, context_data: Dict[str, Any]) -> Context:
        """Restore Context from dictionary using appropriate serializer."""
        if not context_data:
            return Context(workflow)
            
        # ------------------------------------------------------------------
        # Sanitize context payload
        # ------------------------------------------------------------------
        # Serialized Context objects produced by LlamaIndex may include the
        # in-memory chat buffer under the key `globals -> memory`.  This causes
        # deserialisation failures on subsequent loads because the pickled
        # payload tries to instantiate the abstract `BaseMemory` class instead
        # of a concrete implementation.  To avoid this we simply strip the
        # memory entry – we persist conversation memory separately in
        # `save_memory()` / `load_memory()`, so removing it here is safe and
        # desirable.


        # Try JsonPickleSerializer first (for contexts with complex objects)
        try:
            serializer = JsonPickleSerializer()
            ctx = Context.from_dict(workflow, context_data, serializer=serializer)
            logger.info("Restored context using JsonPickleSerializer")
            return ctx
        except Exception as e:
            logger.warning(f"Failed to restore context with JsonPickleSerializer: {e}")
            
        # Fallback to JsonSerializer
        try:
            serializer = JsonSerializer()
            ctx = Context.from_dict(workflow, context_data, serializer=serializer)
            logger.info("Restored context using JsonSerializer fallback")
            return ctx
        except Exception as e:
            logger.warning(f"Failed to restore context with JsonSerializer: {e}")
            
        # Final fallback to fresh context
        logger.info("Creating fresh context due to restoration failures")
        return Context(workflow)
    
    async def save_context(self, user_id: str, context: Context) -> None:
        """Save workflow context for a user using proper LlamaIndex serializers."""
        context_file = self._get_context_file(user_id)
        
        try:
            # Use JsonPickleSerializer to handle complex objects like memory
            serializer = JsonPickleSerializer()

            # Convert context to dictionary **before** touching the memory key
            context_data = context.to_dict(serializer=serializer)

            # ------------------------------------------------------------------
            # Remove volatile / non-serialisable fields
            # ------------------------------------------------------------------
            # Memory is persisted independently, therefore omit it from the
            # workflow context snapshot to avoid deserialisation errors later.
            # if "globals" in context_data and "memory" in context_data["globals"]:
            #     context_data["globals"].pop("memory", None)

            with open(context_file, "w") as f:
                json.dump(context_data, f, indent=2)
            
            logger.info(f"Saved context for user {user_id} using JsonPickleSerializer")
        except Exception as e:
            logger.error(f"Failed to save context with JsonPickleSerializer for user {user_id}: {e}")
            # Fallback to JsonSerializer
            try:
                logger.info(f"Trying fallback JsonSerializer for user {user_id}")
                serializer = JsonSerializer()
                context_data = context.to_dict(serializer=serializer)
                

                with open(context_file, "w") as f:
                    json.dump(context_data, f, indent=2)
                logger.info(f"Saved context for user {user_id} using JsonSerializer fallback")
            except Exception as fallback_error:
                logger.error(f"Both serializers failed for user {user_id}: {fallback_error}")
                # Final fallback to simple context
                try:
                    simple_context = {
                        "user_id": user_id,
                        "saved": True,
                        "global_context": {}
                    }
                    with open(context_file, "w") as f:
                        json.dump(simple_context, f, indent=2)
                    logger.info(f"Saved minimal fallback context for user {user_id}")
                except Exception as final_error:
                    logger.error(f"Complete context save failure for user {user_id}: {final_error}")
    
    async def load_memory(self, user_id: str) -> ChatMemoryBuffer:
        """Load conversation memory for a user."""
        memory_file = self._get_memory_file(user_id)
        
        if memory_file.exists():
            try:
                with open(memory_file, "r") as f:
                    chat_history_data = json.load(f)
                
                # Recreate ChatMessage objects
                messages = [
                    ChatMessage(role=MessageRole(msg["role"]), content=msg["content"])
                    for msg in chat_history_data
                ]
                
                # Create buffer and add messages
                memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
                if messages:
                    memory.put_messages(messages)
                
                logger.info(f"Loaded {len(messages)} messages from memory for user {user_id}")
                return memory
                
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.error(f"Failed to load memory for user {user_id}: {e}")
                return ChatMemoryBuffer.from_defaults(token_limit=40000)
        else:
            logger.info(f"No existing memory found for user {user_id}")
            return ChatMemoryBuffer.from_defaults(token_limit=40000)
    
    async def save_memory(self, user_id: str, memory: ChatMemoryBuffer) -> None:
        """Save conversation memory for a user."""
        memory_file = self._get_memory_file(user_id)
        
        try:
            messages = memory.get_all()
            # Convert ChatMessage objects to JSON serializable dicts
            chat_history_data = [
                {"role": msg.role.value, "content": msg.content} for msg in messages
            ]
            
            with open(memory_file, "w") as f:
                json.dump(chat_history_data, f, indent=2)
            
            logger.info(f"Saved {len(messages)} messages to memory for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to save memory for user {user_id}: {e}")
    
    async def load_chat_history(self, user_id: str) -> List[ChatMessage]:
        """Load chat history as a list of ChatMessage objects for a user."""
        memory_file = self._get_memory_file(user_id)
        
        if memory_file.exists():
            try:
                with open(memory_file, "r") as f:
                    chat_history_data = json.load(f)
                
                # Recreate ChatMessage objects
                messages = [
                    ChatMessage(role=MessageRole(msg["role"]), content=msg["content"])
                    for msg in chat_history_data
                ]
                
                logger.info(f"Loaded {len(messages)} messages from chat history for user {user_id}")
                return messages
                
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.error(f"Failed to load chat history for user {user_id}: {e}")
                return []
        else:
            logger.info(f"No existing chat history found for user {user_id}")
            return []
    
    async def clear_session(self, user_id: str) -> None:
        """Clear both context and memory for a user."""
        context_file = self._get_context_file(user_id)
        memory_file = self._get_memory_file(user_id)
        
        try:
            if context_file.exists():
                context_file.unlink()
            if memory_file.exists():
                memory_file.unlink()
            logger.info(f"Cleared session for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to clear session for user {user_id}: {e}")