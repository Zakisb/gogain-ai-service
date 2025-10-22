from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables before importing other modules
load_dotenv()

from app.api.chat import (
    router as chat_router,
    legacy_router as vc_legacy_router,
    warmup_agents,
)
from app.core.redis_client import get_redis   # Redis client for caches
from app.core.prisma_client import ensure_client_connected
from app.core.database import execute_query, close_pool
from app.core.memory import get_mem0_client


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

    # Startup: Warmup - pre-build LangChain agents
    print("🔥 Warming up LangChain agents...")

    try:
        await warmup_agents()
        print("✅ Agent manager initialised successfully")
    except Exception as e:
        print(f"⚠️ Agent warmup failed (will build on first request): {e}")

    # Startup: Warm Mem0 client to avoid first-request penalty
    try:
        if get_mem0_client():
            print("✅ Mem0 client warmed successfully")
        else:
            print("ℹ️ Mem0 client not configured (MEM0_API_KEY missing)")
    except Exception as e:
        print(f"⚠️ Mem0 warmup failed (will initialize lazily): {e}")

    # Startup: Ensure Prisma client is ready and database reachable
    try:
        ensure_client_connected()
        status = execute_query("SELECT NOW() AS ts", fetch_one=True)
        ts = status.get("ts") if status else "unknown"
        print(f"✅ Prisma database connection established (server time: {ts})")
    except Exception as e:
        print(f"❌ Prisma database connection failed: {e}")

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
    try:
        close_pool()
        print("✅ Prisma client shutdown")
    except Exception:
        pass


# Create an instance of the FastAPI class with lifespan
app = FastAPI(lifespan=lifespan)


# Include both the namespaced and legacy-compatible chat routes
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
# adding virtual coach router
app.include_router(vc_legacy_router, tags=["Virtual Coach"])


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
