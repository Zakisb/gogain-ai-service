# GOGAIN-AI-SERVICE/app/api/chat.py
from fastapi import APIRouter, Depends # Import APIRouter and Depends
from pydantic import BaseModel

from llama_index.core.settings import Settings # Use global settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.litellm import LiteLLM # Import LiteLLM integration

# Define Request/Response models specific to this router (or reuse from a central models file)
class ChatRequest(BaseModel):
    message: str
    model: str | None = None # Optional: allow client to specify model

class ChatResponse(BaseModel):
    response: str

# Create an API Router instance
router = APIRouter(
    prefix="/chat", # All routes in this file will start with /chat
    tags=["Chat"]   # Tag for API documentation
)

# --- Helper function for LiteLLM (optional, but shows flexibility) ---
# You could put this in a shared 'dependencies' or 'utils' file later
def get_llm_instance(model_name: str | None = None) -> LiteLLM:
    """Gets a LiteLLM instance, potentially overriding the default model."""
    # This function isn't strictly needed if you only use the global Settings.llm
    # But useful if you want endpoint-specific models or configurations.
    # For now, we'll rely on the global Settings.llm configured in main.py
    # and potentially override the model name if provided in the request.
    # However, directly modifying Settings.llm per request isn't thread-safe.
    # A better approach for per-request models is to instantiate LiteLLM here.
    active_model = model_name or Settings.llm.model # Use request model or default
    # Note: LiteLLM needs API keys set as environment variables.
    # It automatically picks the correct one based on the model name.
    return LiteLLM(model=active_model)


# --- Chat Endpoint ---
@router.post("", response_model=ChatResponse) # Path is now relative to the router prefix ("/chat")
async def handle_chat(request: ChatRequest):
    """
    Main endpoint to handle chat interactions with the AI coach using LiteLLM.
    Receives a user message and returns the AI's response.
    Optionally allows specifying the LLM model to use.
    """
    # Instantiate LiteLLM for this request to allow model selection
    # Note: This assumes API keys are set globally via environment variables
    llm_instance = LiteLLM(model=request.model or Settings.llm.model) # Use requested or default model

    user_message = ChatMessage(role=MessageRole.USER, content=request.message)

    # Use the specific LiteLLM instance for this request
    # Using achat for async operation
    llm_response = await llm_instance.achat([user_message])

    ai_response_content = llm_response.message.content

    return ChatResponse(response=ai_response_content)
