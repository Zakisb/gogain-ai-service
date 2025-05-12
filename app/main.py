from fastapi import FastAPI
from app.core import config
from app.api import chat
from llama_index.core.settings import Settings
from llama_index.llms.litellm import LiteLLM
from qdrant_client import QdrantClient
from app.core.qdrant import get_qdrant_client   # <- import

# Create an instance of the FastAPI class
app = FastAPI()
client = get_qdrant_client()


# Include the chat router
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])

# Configure global LlamaIndex default LLM once
Settings.llm = LiteLLM(model=config.DEFAULT_MODEL)

# Define a path operation decorator for the root endpoint ("/")
# This tells FastAPI that the function below handles GET requests to "/"
@app.get("/")
async def read_root():
    """
    Root endpoint that returns a simple greeting.
    """
    return {"message": "Hello World"}


# Create & store on startup so routers can access via request.app.state
@app.on_event("startup")
async def _init_vector_store() -> None:
    app.state.qdrant = get_qdrant_client()      # one global instance

# You can add more endpoints later
# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: str | None = None):
#     return {"item_id": item_id, "q": q}
