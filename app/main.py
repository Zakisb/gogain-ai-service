from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables before importing other modules
load_dotenv()

from app.core import config
from app.api.chat import (
    router as chat_router,
    legacy_router as vc_legacy_router,
    get_or_build_workflow,
)
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from app.core.redis_client import get_redis   # Redis client for caches


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Redis connection (optional but recommended)
    try:
        redis = get_redis()
        await redis.ping()
        app.state.redis = redis
        print("✅ Connected to Redis for caching")
    except Exception as e:
        print(f"⚠️ Redis connection failed or not configured: {e}")

    # Startup: Warmup - pre-build agents to eliminate first-request delay
    print("🔥 Warming up agents...")
    from app.core.config import DEFAULT_MODEL

    try:
        warmup_llm = OpenAI(model=DEFAULT_MODEL, temperature=0.7)
        get_or_build_workflow(warmup_llm)
        print("✅ Agents warmed up successfully")
    except Exception as e:
        print(f"⚠️ Agent warmup failed (will build on first request): {e}")

    # Startup: Pre-initialize assessment tools for better performance
    try:
        print("🔥 Pre-warming assessment tools...")
        from app.knowledge.enhanced_assessment_tools import initialize_global_assessment_tools
        initialize_global_assessment_tools()
        print("✅ Assessment tools pre-warmed successfully")
    except Exception as e:
        print(f"⚠️ Assessment tool warmup failed (will initialize on first request): {e}")

    yield
    # Shutdown: Clean up resources if needed
    pass


# Create an instance of the FastAPI class with lifespan
app = FastAPI(lifespan=lifespan)


# Include both the namespaced and legacy-compatible chat routes
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
# adding virtual coach router
app.include_router(vc_legacy_router, tags=["Virtual Coach"])

# Configure global LlamaIndex default LLM once
Settings.llm = OpenAI(model=config.DEFAULT_MODEL)

# Define a path operation decorator for the root endpoint ("/")
# This tells FastAPI that the function below handles GET requests to "/"
@app.get("/")
async def read_root():
    """
    Root endpoint that returns a simple greeting.
    """
    return {"message": "Hello World"}

# You can add more endpoints later
# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: str | None = None):
#     return {"item_id": item_id, "q": q}
