"""Memory retrieval tool for QA agent to access user information from Mem0.

This tool allows the QA agent to search and retrieve user memories such as:
- Dietary preferences (vegetarian, vegan, allergies)
- Gym access and equipment availability
- Training schedule and constraints
- Health conditions and medications
- Personal goals and preferences
"""

from llama_index.core.tools import FunctionTool
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def search_user_memories(query: str, user_id: str, limit: int = 5) -> str:
    """Search user's stored memories/information using semantic search.

    Use this tool when the user asks about their personal information, such as:
    - "Am I vegetarian?" → query: "dietary preferences vegetarian vegan"
    - "Do I have gym access?" → query: "gym access equipment availability"
    - "Am I allergic to lactose?" → query: "allergies intolerances lactose dairy"
    - "What's my training schedule?" → query: "training schedule workout frequency"
    - "What are my fitness goals?" → query: "fitness goals objectives targets"

    Args:
        query: Search query to find relevant memories (use keywords related to the question)
        user_id: User identifier (automatically provided by the system)
        limit: Maximum number of memories to return (default: 5)

    Returns:
        String containing found memories, or a message if no memories found
    """
    try:
        from app.core.memory import get_mem0_client

        mem0_client = get_mem0_client()
        if not mem0_client:
            return "Memory system is not available. Please provide the information directly."

        # Search memories
        result = mem0_client.search_memories(query=query, user_id=user_id, limit=limit)

        if not result.get("success"):
            error = result.get("error", "Unknown error")
            logger.error(f"Memory search failed for user {user_id}: {error}")
            return f"Could not search memories: {error}"

        memories = result.get("memories", [])

        if not memories:
            return "No information found in your profile. Could you please provide this information?"

        # Format memories for the agent
        formatted_memories = []
        for idx, memory in enumerate(memories, 1):
            if isinstance(memory, dict):
                # Extract memory text
                memory_text = memory.get("memory", memory.get("text", ""))
                if memory_text:
                    formatted_memories.append(f"{idx}. {memory_text}")

        if not formatted_memories:
            return "No information found in your profile. Could you please provide this information?"

        # Return formatted results
        result_text = "Found the following information in your profile:\n\n" + "\n".join(formatted_memories)
        logger.info(f"🧠 Retrieved {len(formatted_memories)} memories for user {user_id} with query: {query}")
        return result_text

    except Exception as e:
        logger.error(f"Error searching user memories: {e}", exc_info=True)
        return "Could not access memory system. Please provide the information directly."


def get_memory_search_tool() -> FunctionTool:
    """Factory function to create the memory search tool for QA agent."""
    return FunctionTool.from_defaults(
        fn=search_user_memories,
        name="search_user_memories",
        description=(
            "Search user's stored personal information (diet, allergies, gym access, schedule, goals). "
            "Use when user asks about their own information like 'Am I vegetarian?', 'Do I have gym access?', "
            "'Am I allergic to X?'. Provide a descriptive search query with keywords."
        )
    )
