# GOGAIN-AI-SERVICE/app/core/config.py
import os
from dotenv import load_dotenv

# Load .env file located in the project root
# Adjust the path if your .env is elsewhere
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Retrieve API Keys ---
# LlamaIndex OpenAI wrapper will use the relevant key based on the model requested
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Add other keys as needed (COHERE_API_KEY, etc.)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Audio transcription ---
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "whisper-1")
TRANSCRIPTION_UPLOAD_DIR = os.getenv("TRANSCRIPTION_UPLOAD_DIR", "uploads")

# --- Application Settings ---
APP_TITLE = "GoGain AI Coach Service"
APP_DESCRIPTION = "API endpoints for the AI Virtual Coach powered by LlamaIndex & OpenAI"
APP_VERSION = "0.2.0" # Incremented version

# --- Vector store / Qdrant settings ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# --- LlamaIndex / OpenAI Settings ---
# Default model to use if not specified elsewhere
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", os.getenv("DEFAULT_ROUTER_MODEL", "gpt-4o"))
PAIN_AGENT_MODEL = os.getenv("PAIN_AGENT_MODEL", "gpt-4o")

# --- Thread Manager Settings ---
# Override thread manager provider (optional)
# If not set, will auto-detect based on DEFAULT_MODEL
# Options: "openai", "gemini", "auto"
THREAD_MANAGER_PROVIDER = os.getenv("THREAD_MANAGER_PROVIDER", "auto")

# You could add other settings here like embedding models, database URIs, etc.
# EMBEDDING_MODEL = "text-embedding-ada-002"

# --- Mem0 Configuration ---
# Enable/disable message-based memory retrieval on every user message
ENABLE_MEM0_MESSAGE_RETRIEVAL = os.getenv("ENABLE_MEM0_MESSAGE_RETRIEVAL", "true").lower() == "true"

# --- Basic validation ---
if not OPENAI_API_KEY and not ANTHROPIC_API_KEY: # Check if at least one key is present
     print("Warning: No primary API keys (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY) found in environment variables.")
     # Consider raising an exception in production if a key is required:
     # raise ValueError("Missing required API keys in environment variables.")

if DEFAULT_MODEL.startswith("gemini/") and not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in environment/.env")
