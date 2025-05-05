# GOGAIN-AI-SERVICE/app/api/chat.py
from fastapi import APIRouter, HTTPException

from pydantic import BaseModel
import logging
import os
import json
from pathlib import Path
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage, MessageRole

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





# MEMORY_DIR = Path("app/chat_memory")
# MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# # --- Configuration for Mock User Data --- Added
# MOCK_DATA_DIR = Path("app/mock_data")
# MOCK_DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure 

# Create an API Router instance
router = APIRouter(prefix="/chat")                  # no tags

# === Agent Definitions ===
# NOTE: llm instance will be passed dynamically based on request

# === Chat Endpoint using AgentWorkflow (aka AgentRunner) ===


# def load_memory_from_file(user_id: str) -> ChatMemoryBuffer:
#     """Loads chat history for a user from a JSON file."""
#     memory_file = MEMORY_DIR / f"{user_id}_memory.json"
#     if memory_file.exists():
#         try:
#             with open(memory_file, "r") as f:
#                 chat_history_data = json.load(f)
#                 # Recreate ChatMessage objects
#                 messages = [
#                     ChatMessage(role=MessageRole(msg["role"]), content=msg["content"])
#                     for msg in chat_history_data
#                 ]
#                 logger.info(f"Loaded {len(messages)} messages from memory for user {user_id}")
#                 # Consider token limit if loading large histories
#                 return ChatMemoryBuffer.from_defaults(messages=messages, token_limit=40000)
#         except (json.JSONDecodeError, KeyError, Exception) as e:
#             logger.error(f"Failed to load or parse memory file {memory_file}: {e}. Starting fresh.")
#             # If loading fails, start with empty memory
#             return ChatMemoryBuffer.from_defaults(token_limit=40000)
#     else:
#         logger.info(f"No memory file found for user {user_id}. Starting fresh.")
#         return ChatMemoryBuffer.from_defaults(token_limit=40000)

# def save_memory_to_file(user_id: str, memory: ChatMemoryBuffer):
#     """Saves chat history for a user to a JSON file."""
#     memory_file = MEMORY_DIR / f"{user_id}_memory.json"
#     messages = memory.get_all()
#     # Convert ChatMessage objects to JSON serializable dicts
#     chat_history_data = [
#         {"role": msg.role.value, "content": msg.content} for msg in messages
#     ]
#     try:
#         with open(memory_file, "w") as f:
#             json.dump(chat_history_data, f, indent=2)
#         logger.info(f"Saved {len(messages)} messages to memory for user {user_id}")
#     except Exception as e:
#         logger.error(f"Failed to save memory file {memory_file}: {e}")


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





# @router.post("", response_model=ChatResponse)
# async def handle_chat(request: ChatRequest):
#     """
#     Handles chat interactions using a multi-agent workflow,
#     using simple file-based persistence for chat memory based on user_id.
#     Loads mock user data if available.
#     """
#     logger.info(f"Received chat request for user_id: {request.user_id}")

#     # --- Load Mock User Data --- Modified section
#     user_mock_data = {} # Default to empty dict
#     mock_file_path = MOCK_DATA_DIR / f"{request.user_id}.json"
#     if mock_file_path.exists():
#         try:
#             with open(mock_file_path, 'r') as f:
#                 user_mock_data = json.load(f)
#             logger.info(f"Loaded mock data for user {request.user_id} from {mock_file_path}")
#         except (json.JSONDecodeError, Exception) as e:
#             logger.error(f"Failed to load or parse mock data file {mock_file_path}: {e}. Proceeding without mock data.")
#             user_mock_data = {} # Reset on error
#     else:
#         logger.info(f"No mock data file found for user {request.user_id} at {mock_file_path}. Proceeding without mock data.")


#     # --- Initialize LLM (as before) ---
#     # Consider initializing LLM outside the request handler if it's always the same
#     llm = Gemini(model="models/gemini-2.0-flash")
#     Settings.llm = llm

#     # --- Build Agents (as before) ---
#     specialist_agents = build_all(llm)
#     # --- Prepare Initial Workflow State (Include mock data if loaded) ---
#     initial_state = {
#         "user_profile": user_mock_data,
#         "pain_summary": None,
#         "training_plan": None,
#         "diet_plan": None,
#     }



#     # --- Build Workflow (as before) ---
#     workflow = build_workflow(
#         llm,
#         specialist_agents,
#         initial_state=initial_state,
#     )

#     # --- Manage Memory (Using File Persistence) ---
#     ctx = Context(workflow)
#     memory = load_memory_from_file(request.user_id)


#     # --- Run Workflow ---
#     logger.info("Running AgentWorkflow...")
#     final_response_content = "Error: Workflow did not produce a response."

#     try:
#         output: AgentOutput = await workflow.run(
#             input=request.message,
#             ctx=ctx,
#             memory=memory
#         )

#         if isinstance(output, AgentOutput):
#             if output.response:
#                 final_response_content = str(output.response).strip()
#                 print(f"[DEBUG] Extracted response from AgentOutput: {final_response_content}")
#             else:
#                 final_response_content = "[AgentOutput received, but .response attribute was empty]"
#                 logger.warning(f"AgentOutput received, but .response was empty. Output object: {output}")
#         else:
#             logger.warning(f"Workflow output was not an AgentOutput object. Type: {type(output)}, Value: {output}")
#             final_response_content = f"[Unexpected workflow output type: {type(output)}]"

#         logger.info(f"Workflow finished. Captured response attempt: {final_response_content}")

#         save_memory_to_file(request.user_id, memory)

#     except Exception as e:
#         logger.error(f"Error running workflow or handling memory for user {request.user_id}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal server error during chat processing.")

#     if not final_response_content:
#         final_response_content = "[No response generated]"


