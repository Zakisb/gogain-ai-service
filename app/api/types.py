"""
Type definitions for chat endpoint streaming events.
These define the exact structure of Server-Sent Events sent to the frontend.
"""

from typing import Literal, Union, List
from pydantic import BaseModel


# Individual Event Types
class StartEvent(BaseModel):
    type: Literal["start"]
    message: str


class StatusEvent(BaseModel):
    type: Literal["status"]
    message: str


class AgentStartEvent(BaseModel):
    type: Literal["agent_start"]
    agent: str


class TokenEvent(BaseModel):
    type: Literal["token"]
    content: str


class ToolCallsEvent(BaseModel):
    type: Literal["tool_calls"]
    tools: List[str]


class ToolStartEvent(BaseModel):
    type: Literal["tool_start"]
    tool: str


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"]
    tool: str


class CompleteEvent(BaseModel):
    type: Literal["complete"]
    final_response: str


class DoneEvent(BaseModel):
    type: Literal["done"]


class ErrorEvent(BaseModel):
    type: Literal["error"]
    message: str


# Union type for all possible streaming events
StreamingEvent = Union[
    StartEvent,
    StatusEvent,
    AgentStartEvent,
    TokenEvent,
    ToolCallsEvent,
    ToolStartEvent,
    ToolResultEvent,
    CompleteEvent,
    DoneEvent,
    ErrorEvent,
]


# Helper function to create typed events
def create_start_event(message: str) -> StartEvent:
    return StartEvent(type="start", message=message)


def create_status_event(message: str) -> StatusEvent:
    return StatusEvent(type="status", message=message)


def create_agent_start_event(agent: str) -> AgentStartEvent:
    return AgentStartEvent(type="agent_start", agent=agent)


def create_token_event(content: str) -> TokenEvent:
    return TokenEvent(type="token", content=content)


def create_tool_calls_event(tools: List[str]) -> ToolCallsEvent:
    return ToolCallsEvent(type="tool_calls", tools=tools)


def create_tool_start_event(tool: str) -> ToolStartEvent:
    return ToolStartEvent(type="tool_start", tool=tool)


def create_tool_result_event(tool: str) -> ToolResultEvent:
    return ToolResultEvent(type="tool_result", tool=tool)


def create_complete_event(final_response: str) -> CompleteEvent:
    return CompleteEvent(type="complete", final_response=final_response)


def create_done_event() -> DoneEvent:
    return DoneEvent(type="done")


def create_error_event(message: str) -> ErrorEvent:
    return ErrorEvent(type="error", message=message)