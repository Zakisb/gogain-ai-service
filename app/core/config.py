# GOGAIN-AI-SERVICE/app/core/config.py
import os
from dotenv import load_dotenv

# Load .env file located in the project root
# Adjust the path if your .env is elsewhere
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Retrieve API Keys ---
# LiteLLM will use the relevant key based on the model requested
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Add other keys as needed (COHERE_API_KEY, etc.)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Application Settings ---
APP_TITLE = "GoGain AI Coach Service"
APP_DESCRIPTION = "API endpoints for the AI Virtual Coach powered by LlamaIndex & LiteLLM"
APP_VERSION = "0.2.0" # Incremented version

# --- LlamaIndex / LiteLLM Settings ---
# Default model to use if not specified elsewhere
# LiteLLM uses model names like "gpt-3.5-turbo", "claude-3-sonnet-20240229", etc.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini/gemini-pro")

# You could add other settings here like embedding models, database URIs, etc.
# EMBEDDING_MODEL = "text-embedding-ada-002"

# --- Basic validation ---
if not OPENAI_API_KEY and not ANTHROPIC_API_KEY: # Check if at least one key is present
     print("Warning: No primary API keys (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY) found in environment variables.")
     # Consider raising an exception in production if a key is required:
     # raise ValueError("Missing required API keys in environment variables.")

if DEFAULT_MODEL.startswith("gemini/") and not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in environment/.env")
