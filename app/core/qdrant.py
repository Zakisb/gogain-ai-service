# app/core/qdrant.py
from functools import lru_cache
from qdrant_client import QdrantClient

# You can also read these from environment variables:
_QDRANT_URL  = "http://localhost:6333"
_QDRANT_PORT = 6333   # optional if already in the URL

@lru_cache            # builds once, then re-uses the same instance
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=_QDRANT_URL, port=_QDRANT_PORT)