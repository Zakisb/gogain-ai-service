"""Utility helpers for LangChain-based agents.

All tools have been fully migrated to LangChain StructuredTool.
No more LlamaIndex tool dependencies in the agent layer.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def convert_dict_messages_to_langchain(
    messages: Iterable[Dict[str, Any]]
) -> List[BaseMessage]:
    """Convert simple {role, content} dict messages into LangChain messages."""

    converted: List[BaseMessage] = []

    for entry in messages:
        if not isinstance(entry, dict):
            raise ValueError(f"Expected dict message, got: {type(entry)!r}")

        role = entry.get("role")
        content = entry.get("content", "")

        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        elif role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "tool":
            # Support both "name" and "tool_name" keys for compatibility
            tool_name = entry.get("name") or entry.get("tool_name") or "tool"
            tool_call_id = (
                entry.get("tool_call_id")
                or entry.get("id")
                or f"{tool_name}-call"
            )
            converted.append(
                ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)
            )
        else:
            # Default to human message to avoid silently dropping content
            converted.append(HumanMessage(content=content))

    return converted
