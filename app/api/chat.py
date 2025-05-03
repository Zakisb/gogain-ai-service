# GOGAIN-AI-SERVICE/app/api/chat.py
from fastapi import APIRouter
from pydantic import BaseModel
import logging
import os

from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage

from app.agents import build_all
from app.agents.central import build_workflow
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.workflow import Context
from llama_index.core.workflow import JsonPickleSerializer, JsonSerializer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Request/Response models (simplified without user/session IDs for now)
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Create an API Router instance
router = APIRouter(prefix="/chat")                  # no tags

# === Agent Definitions ===
# NOTE: llm instance will be passed dynamically based on request

# === Chat Endpoint using AgentWorkflow (aka AgentRunner) ===


@router.post("", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Handles chat interactions using a multi-agent workflow (stateless),
    streaming and logging events.
    """

   
    # --- Initialize LLM for this request ---
    # Using the global default or the one specified in the request
  
    # Ensure your GOOGLE_API_KEY is set in the environment before starting the app
    # e.g. export GOOGLE_API_KEY="your_real_key" (do NOT hard-code secrets)

    # Gemini 2.0 Flash is only available under the "models/" prefix
    llm = Gemini(model="models/gemini-2.0-flash")
    Settings.llm = llm

    # Build specialist agents (tools)
    specialist_agents = build_all(llm)

    # Build a multi-agent workflow with shared state
    workflow = build_workflow(
        llm,
        specialist_agents,
        initial_state={
            "pain_summary": None,
            "training_plan": None,
            "diet_plan": None,
        },
    )
    ctx = Context(workflow)
    memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
    # memory.put(ChatMessage(role="user", content="Hello, world!"))
    # memory.put(ChatMessage(role="assistant", content="Hello, world to you too!"))
    # 

    # logger.info("Running AgentWorkflow with streaming...")
    # # Get the async task handler
    handler = await workflow.run(request.message,  ctx=ctx, memory=memory)

    print(f"Handler: {handler}")
    final_response_content = "No response generated."

    response2 = await workflow.run(user_msg="What's my name?", ctx=ctx, memory=memory)
    print(response2)
    ctx_dict = ctx.to_dict(serializer=JsonSerializer())
    print(f"Context: {ctx_dict}")
    chat_history = memory.get()
    print(f"Chat History: {chat_history}")


    # Stream events asynchronously
    # async for event in handler.astream_events():
    #     # Check for agent transitions using WorkflowStepEvent context
    #     if isinstance(event, WorkflowStepEvent) and event.step_context:
    #         agent_name = event.step_context.current_agent_name
    #         if agent_name != current_agent:
    #             current_agent = agent_name
    #             logger.info(f"\n{'='*30} Agent Start: {current_agent} {'='*30}\n")
    #     # Log Agent Output
    #     elif isinstance(event, AgentOutput):
    #         if event.response and event.response.response: # Check response exists
    #             response_text = event.response.response.strip()
    #             if response_text:
    #                 logger.info(f"📤 Agent Output ({current_agent}): {response_text}")
    #                 # Capture the latest non-empty response content as the final one
    #                 final_response_content = response_text
    #         # Log planned tool calls (if any)
    #         if event.response and event.response.tool_calls:
    #             tool_names = [call.tool_name for call in event.response.tool_calls]
    #             logger.info(f"🛠️ Planning to use tools: {tool_names}")
    #     # Log Tool Calls
    #     elif isinstance(event, ToolCall):
    #         logger.info(f"🔨 Calling Tool: {event.tool_name}")
    #         logger.info(f"  With arguments: {event.tool_kwargs}")
    #     # Log Tool Results
    #     elif isinstance(event, ToolCallResult):
    #         logger.info(f"🔧 Tool Result ({event.tool_name}):")
    #         # Be cautious logging potentially large/sensitive tool outputs fully
    #         # logger.info(f"  Output: {event.tool_output}")
    #         logger.info(f"  Output: [Result received for {event.tool_name}]")


    # logger.info("Workflow finished.")
    # logger.info(f"Final response captured: {final_response_content}")

    return ChatResponse(response=final_response_content)
