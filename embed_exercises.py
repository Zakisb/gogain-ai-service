"""embed_exercises.py
Standalone script to:
1. Fetch/parse exercises (DB or cached JSON)
2. Embed them with Google Gemini embeddings through Llama-Index
3. Upsert into a local Qdrant collection called "exercises"

Run examples:
    # Fresh fetch from DB, then embed & ingest
    python embed_exercises.py --source db

    # Re-use previously exported JSON for faster iteration
    python embed_exercises.py --source json --path formatted_exercises_sqlalchemy.json

Env vars required
-----------------
GEMINI_API_KEY  – for Google Generative AI embeddings (preferred)
OPENAI_API_KEY  – fallback if GEMINI_API_KEY not provided
QDRANT_HOST     – default "localhost"
QDRANT_PORT     – default "6333"

Prerequisites (add to requirements.txt):
    google-generativeai
    voyageai
    litellm
    llama-index-embeddings-google
    llama-index-embeddings-voyageai
    llama-index-vector-stores
    qdrant-client
"""
# Standard library imports
import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Sequence

# Third-party imports (new Google GenAI SDK)
from google import genai                             # modern SDK entry-point
from google.genai.types import EmbedContentConfig

# LlamaIndex imports
from llama_index.core import (Document, VectorStoreIndex, StorageContext)
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client import QdrantClient, models as qmodels

# Local imports

## Gemini custom embedding model
# --------------- Local imports ----------------- #
try:
    from fetch_exercises import fetch_and_format_exercises_sqlalchemy  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import fetch_and_format_exercises_sqlalchemy. "
        "Ensure fetch_exercises.py is in PYTHONPATH."
    ) from exc

# ---------------------------------------------------------------------------
# Debug imports – used only for troubleshooting version mismatches
# ---------------------------------------------------------------------------
import voyageai  # type: ignore  # noqa: E402
import llama_index  # type: ignore  # noqa: E402
import inspect  # noqa: E402
import sys, importlib.util
try:
    from importlib.metadata import version as pkg_version  # Python 3.8+
except ImportError:  # pragma: no cover
    # For older Python versions, fall back to importlib-metadata backport
    from importlib_metadata import version as pkg_version  # type: ignore

# ------------------------------------------------ #

def load_documents_from_json(path: str) -> List[Dict]:
    """Load pre-formatted exercise dicts from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of documents")
    return data



def convert_to_documents(items: List[Dict]) -> List[Document]:
    """Convert our plain-dict format into Llama-Index Document objects."""
    docs: List[Document] = []
    for itm in items:
        docs.append(
            Document(id_=itm["doc_id"], text=itm["text"], metadata=itm["metadata"])
        )
    return docs


def ensure_qdrant_collection(client: QdrantClient, collection: str, dim: int):
    """Create or make sure the Qdrant collection exists with the right dim."""
    try:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )
    except Exception as exc:
        print(f"[warn] Could not recreate collection {collection}: {exc}. Assuming it exists.")


async def main():
    # Create a tool that lets users provide options when running this script from the command line
    # For example, they can run: python embed_exercises.py --source json --path my_file.json
    # The description explains what this script does to anyone running --help
    parser = argparse.ArgumentParser(description="Embed and ingest exercises into Qdrant")

    # Add argument for choosing data source - either database or JSON file
    # This lets users choose where to get the exercise data from:
    # Example: python embed_exercises.py --source db
    #   - If --source=db: Fetches fresh data from database
    #   - If --source=json: Loads from a saved JSON file (faster for testing)
    parser.add_argument(
        "--source",
        choices=["db", "json"],
        default="db",
        help="Where to load exercise documents from. 'db' runs SQLAlchemy fetch, 'json' loads a file.",
    )

    # If using JSON source (--source json), this specifies which file to load
    # Example: python embed_exercises.py --source json --path my_exercises.json
    # If not specified, defaults to "formatted_exercises_sqlalchemy.json"
    parser.add_argument(
        "--path",
        default="formatted_exercises_sqlalchemy.json",
        help="Path to JSON file when --source json is used.",
    )

    # Lets you specify which Qdrant collection to save the embeddings to
    # Example: python embed_exercises.py --collection my_exercises
    # Think of collections like different tables in a database
    # If not specified, defaults to "exercises"
    parser.add_argument(
        "--collection",
        default="exercises",
        help="Qdrant collection name to upsert into.",
    )

    # Takes all the command line arguments above and puts them in an 'args' object
    # So we can later use args.source, args.path, args.collection to get the values
    args = parser.parse_args()

    # --------------------------------------------------------------------------------
    # Step 1: Load items
    if args.source == "db":
        print("[load] Fetching exercises from PostgreSQL via SQLAlchemy …")
        items = fetch_and_format_exercises_sqlalchemy()
    else:
        print(f"[load] Loading exercises from JSON {args.path} …")
        items = load_documents_from_json(args.path)

    if not items:
        raise SystemExit("No exercises found. Aborting.")
    print(f"[load] Loaded {len(items)} items")

    # --------------------------------------------------------------------------------
    # Step 2: Convert to Llama-Index Documents
    docs = convert_to_documents(items)

    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    if voyage_api_key:  # --- use VoyageAI out-of-the-box wrapper --------------------
        embed_model = VoyageEmbedding(
            voyage_api_key=voyage_api_key,
            model_name="voyage-3-large",  # 1024-dim generalist model
            input_type="document",  # we are indexing corpus docs
        )
        vector_dim = 1_024
        print("[embed] Using VoyageAI model 'voyage-3' (dim 1024)")

    else:
        raise SystemExit(
            "Neither VOYAGE_API_KEY nor GEMINI_API_KEY set. Please export one env var."
        )

    # # --------------------------------------------------------------------------------
    # Step 4: import qdrant client instance from app.core.qdrant
    client = qdrant_client.QdrantClient(
        host="localhost",
        port=6333
    )
    vector_store = QdrantVectorStore(client=client, collection_name=args.collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    test_docs = docs[:5]  
    print(f"[info] Testing with {len(test_docs)} documents")
    # # # --------------------------------------------------------------------------------
    # # # Step 5: Build / upsert index
    print("[embed] Generating embeddings and upserting into Qdrant …")
    # VectorStoreIndex.from_documents(
    #     test_docs,
    #     storage_context=storage_context,
    #     embed_model=embed_model,
    #     show_progress=True,
    #     batch_size=128,
    #     store_nodes_override=False,  # don't duplicate nodes if same id
    # )

    index = VectorStoreIndex.from_vector_store(
       vector_store=vector_store,
       embed_model=embed_model
    )
    
    query_engine = index.as_query_engine()
    response = query_engine.query("i need an exercise for my neck problem")

    
    print(response)


    print("[embed] Done!\n")

   
    


if __name__ == "__main__":
    asyncio.run(main()) 