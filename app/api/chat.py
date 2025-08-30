"""Basic chat endpoint using AgentWorkflow with CentralRouterAgent and PainAgent."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import json
import asyncio
from pathlib import Path

from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.memory.mem0 import Mem0Memory
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput, 
    AgentStream,
    ToolCall, 
    ToolCallResult,
    FunctionAgent
)

from llama_index.core.workflow import Context, JsonSerializer
import os
from llama_index.core.agent.workflow import AgentWorkflow

from app.agents.central import build_workflow
from app.agents.pain import build as build_pain_agent
from app.core.session_storage import SessionStorage
from app.api.types import (
    StreamingEvent,
    create_start_event,
    create_status_event,
    create_agent_start_event,
    create_token_event,
    create_tool_calls_event,
    create_tool_start_event,
    create_tool_result_event,
    create_complete_event,
    create_done_event,
    create_error_event,
)
from llama_index.core.memory import Memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: str = "test_user_01"  # Default for development

# Create API Router and session storage
router = APIRouter(prefix="/chat")
session_storage = SessionStorage()


def create_mem0_memory(user_id: str, conversation_id: str) -> Mem0Memory:
    """Create Mem0Memory instance for the given user."""
    try:
        # Check if MEM0_API_KEY is set (for Mem0 platform)
        mem0_api_key = os.getenv("MEM0_API_KEY")
        if mem0_api_key:
            # Using Mem0 platform
            logger.info(f"Initializing Mem0Memory with platform API for user {user_id}")
            return Mem0Memory.from_client(
                context={"user_id": user_id, "conversation_id": conversation_id},
                api_key=mem0_api_key,   
                search_msg_limit=10  # Number of historical messages for retrieval
            )
        else:
            # Using local Mem0 (fallback)
            logger.info(f"Initializing Mem0Memory with local config for user {user_id}")
            return Mem0Memory.from_client(
                context={"user_id": user_id, "conversation_id": conversation_id},
                search_msg_limit=10
            )
    except Exception as e:
        logger.warning(f"Failed to create Mem0Memory for user {user_id}: {e}")
        logger.info("Falling back to local memory only")
        return None


def emit_event(event: StreamingEvent) -> str:
    """Helper function to emit a typed streaming event as Server-Sent Event format."""
    return f"data: {event.model_dump_json()}\n\n"

@router.post("")

async def handle_chat(request: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint with session persistence for context and memory.
    Always returns Server-Sent Events for real-time response streaming.
    
    Returns:
        StreamingResponse: Server-Sent Events containing typed StreamingEvent objects
        
    Event Types:
        - StartEvent: Initialization
        - StatusEvent: Processing status updates  
        - AgentStartEvent: Agent transitions
        - TokenEvent: Real-time text streaming
        - ToolCallsEvent: Tool usage notifications
        - ToolStartEvent: Individual tool start
        - ToolResultEvent: Individual tool completion
        - CompleteEvent: Final response with complete text
        - DoneEvent: Stream completion
        - ErrorEvent: Error information
    """
    
    async def stream_generator():
        try:
            logger.info(f"Starting streaming chat for user {request.user_id}: {request.message}")
            print(f"Request: {request.user_id}")
            conversation_id = request.user_id
            
            # Initialize LLM - Testing OpenAI for finer streaming granularity
            llm = OpenAI(model="gpt-4o-mini", temperature=0.7)
            Settings.llm = llm
            
            # Load existing session data  
            existing_context_data = await session_storage.load_context(request.user_id)
            local_memory = await session_storage.load_memory(request.user_id)
            chat_history = await session_storage.load_chat_history(request.user_id)
            
            # Initialize Mem0 memory for enhanced personalization
            mem0_memory = create_mem0_memory(request.user_id, conversation_id)
            
            # Use Mem0 if available, otherwise fall back to local memory
            # memory = mem0_memory if mem0_memory else local_memory
            
            # yield emit_event(create_status_event(f"Memory system: {'Mem0' if mem0_memory else 'Local'}"))
            
            
            # Build agents
            pain_agent = build_pain_agent(llm)
            specialist_agents = [pain_agent]
            
            # Build workflow with loaded or 
            # default state
            initial_state = existing_context_data.get("global_context", {})
            workflow = build_workflow(
                llm,
                specialist_agents,
                initial_state={}
            )
            
            # Create or restore context using proper serializers
            if existing_context_data:
                try:
                    ctx = session_storage.restore_context_from_dict(workflow, existing_context_data)
                    logger.info("Restored context from previous session")
                    yield emit_event(create_status_event("Context restored"))
                except Exception as e:
                    logger.warning(f"Failed to restore context, using fresh: {e}")
                    ctx = Context(workflow)
                    yield emit_event(create_status_event("Using fresh context"))
            else:
                ctx = Context(workflow)
                yield emit_event(create_status_event("Using fresh context"))
            
            
            # Run workflow with streaming
            handler = workflow.run(user_msg=request.message, ctx=ctx)
            
            current_agent = None
            final_response_content = "No response generated."
            
            # Stream events as they come
            async for event in handler.stream_events():
                # logger.info(f"🔄 Received event: {type(event).__name__}")
                # logger.debug(f"📋 Full event data: {event}")
                
                # # AgentInput: the full message object that begins the agent's execution
                # if isinstance(event, AgentInput):
                #     logger.info()
                #     if hasattr(event, 'current_agent_name'):
                #         agent_name = event.current_agent_name
                #         logger.info(f"🤖 Agent name: {agent_name}")
                #         if agent_name != current_agent:
                #             current_agent = agent_name
                #             logger.info(f"🔄 Switching to agent: {current_agent}")
                #             yield emit_event(create_agent_start_event(current_agent))
                
                # AgentStream: real-time streaming output from the agent
                if isinstance(event, AgentStream):
                    if hasattr(event, 'delta') and event.delta:
                        logger.info(f"📤 Streaming token: '{event.delta}'")
                        # Stream individual tokens as they come
                        yield emit_event(create_token_event(event.delta))
                    else:
                        logger.warning("⚠️ AgentStream event has no delta attribute or delta is empty")
                
                # AgentOutput: the complete response from the agent
                elif isinstance(event, AgentOutput):
                    if event.response and hasattr(event.response, 'response'):
                        response_text = str(event.response.response).strip()
                        logger.info(f"📄 Full response: '{response_text}'")
                        if response_text:
                            final_response_content = response_text
                    elif event.response and hasattr(event.response, 'content'):
                        response_text = str(event.response.content).strip()
                        logger.info(f"📄 Response content: '{response_text}'")
                        if response_text:
                            final_response_content = response_text
                    
                    # Check for tool calls
                    if event.response and hasattr(event.response, 'tool_calls') and event.response.tool_calls:
                        tool_names = [call.tool_name for call in event.response.tool_calls]
                        logger.info(f"🛠️ Tool calls detected: {tool_names}")
                        yield emit_event(create_tool_calls_event(tool_names))
                
                # # ToolCall: which tools were called and with what arguments
                # elif isinstance(event, ToolCall):
                #     logger.info(f"🔨 ToolCall: {event.tool_name} with args: {getattr(event, 'tool_kwargs', 'N/A')}")
                #     yield emit_event(create_tool_start_event(event.tool_name))
                
                # # ToolCallResult: the result of a tool call
                # elif isinstance(event, ToolCallResult):
                #     logger.info(f"✅ ToolCallResult: {event.tool_name} -> {getattr(event, 'tool_output', 'N/A')}")
                #     yield emit_event(create_tool_result_event(event.tool_name))
                
                # Unknown event type
                # else:
                #     logger.warning(f"❓ Unknown event type: {type(event).__name__}")
            
            
            # Save context
            await session_storage.save_context(request.user_id, ctx)
            
            # # Always save local chat history + handle Mem0
            # if mem0_memory:
            #     # Mem0 automatically handles persistence, but we still save locally
            #     logger.info("Using Mem0 - memory automatically persisted")
                
            #     # Add current exchange to local memory for local history tracking
            #     from llama_index.core.llms import ChatMessage, MessageRole
            #     local_memory.put(ChatMessage(role=MessageRole.USER, content=request.message))
            #     if final_response_content and final_response_content != "No response generated.":
            #         local_memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=final_response_content))
                
            #     # Save updated local memory 
            #     await session_storage.save_memory(request.user_id, local_memory)
            #     logger.info("Saved local chat history alongside Mem0")
            # else:
            #     # Save local memory (it already contains the conversation)
            #     await session_storage.save_memory(request.user_id, local_memory)
            #     logger.info("Saved local memory to file storage")
            
            # Send completion event
            yield emit_event(create_complete_event(final_response_content))
            yield emit_event(create_done_event())
            
            
        except Exception as e:
            logger.error(f"Error in streaming chat endpoint: {e}", exc_info=True)
            yield emit_event(create_error_event(str(e)))
            yield emit_event(create_done_event())
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

## create simple endpoint to test the workflow without streaming


@router.post("/test")
async def test_workflow(request: ChatRequest) -> bool:
    """
    Test the workflow without streaming.
    """
    llm = OpenAI(model="gpt-4o-mini", temperature=0.7)
    Settings.llm = llm
    mem0_memory = create_mem0_memory(request.user_id, request.user_id)
    memory = Memory.from_defaults(session_id=f"session_{request.user_id}", token_limit=400000)

   
    review_agent = FunctionAgent(
        name="ReviewAgent",
        description="Reviews a report and gives feedback.",
        system_prompt="You are a reviewer…",  # etc.
        llm=llm,
        tools=[],
        can_handoff_to=["WriteAgent"],
    )

    workflow = AgentWorkflow(
        agents=[review_agent],
        root_agent=review_agent.name,
    )

    # ------------------------------------------------------------
    # Very simple file-based context persistence for demonstration
    # ------------------------------------------------------------
    # Use JsonPickleSerializer so complex objects (e.g. Mem0Memory) are serialised safely
    # from llama_index.core.workflow import JsonPickleSerializer
    # serializer = JsonPickleSerializer()
    # context_file = Path("test_workflow_context.json")

    # if context_file.exists():
    #     # Load previously saved context
    #     with context_file.open("r") as f:
    #         context_data = json.load(f)
    #     try:
    #         ctx = Context.from_dict(workflow, context_data, serializer=serializer)
    #         logger.info("Loaded context from file for /test workflow")
    #     except Exception as e:
    #         logger.warning(f"Failed to restore context, starting fresh: {e}")
    #         ctx = Context(workflow)
    # else:
    #     # No saved context yet
    #     ctx = Context(workflow)
    #     logger.info("No saved context found – starting fresh")
    ctx = Context(workflow)
    # Run workflow
    response = await workflow.run(user_msg=request.message, ctx=ctx, memory=memory)

    # Save updated context back to file so it can be reused next time
    # try:
    #     # Convert context to dict first, then drop volatile memory entry
    #     context_payload = ctx.to_dict(serializer=serializer)

    #     with context_file.open("w") as f:
    #         json.dump(context_payload, f, indent=2)
    #     logger.info("Context saved to file for /test workflow")
    # except Exception as e:
    #     logger.error(f"Failed to save context: {e}")

    
    print(response)
    
    return True