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
    litellm
    llama-index-embeddings-google
    llama-index-vector-stores
    qdrant-client
"""
# Standard library imports
import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Sequence

# Third-party imports
import google.generativeai as genai
from google.genai.types import EmbedContentConfig
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models as qmodels

# Local imports
from app.core.qdrant import get_qdrant_client   # ← reuse the cached client

## Gemini custom embedding model

class GeminiEmbedding(BaseEmbedding):
    """
    LlamaIndex‑compatible wrapper around the Gemini API embedding model
    (`gemini-embedding-exp-03-07`).  Works with an API‑key only – no Vertex /
    service‑account creds required.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-embedding-exp-03-07",
        task_type: str = "RETRIEVAL_DOCUMENT",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # configure the SDK to talk to the Gemini endpoint (NOT Vertex)
        genai.configure(api_key=api_key)

        self._client = genai.Client()
        self._model_name = model_name
        self._cfg = EmbedContentConfig(task_type=task_type)

    # ---------- internal helper -------------------------------------------------
    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Gemini `embed_content` takes a single string, so loop (simple & reliable).
        If you hit throughput limits, add your own asyncio/gather wrapper.
        """
        out: List[List[float]] = []
        for t in texts:
            res = self._client.models.embed_content(
                model=self._model_name,
                contents=t,
                config=self._cfg,
            )
            out.append(res.embeddings[0].values)
        return out

    # ---------- required sync overrides ----------------------------------------
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_batch([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    # ---------- optional async stubs (run sync work in threadpool) --------------
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
# --------------- Local imports ----------------- #
try:
    from fetch_exercises import fetch_and_format_exercises_sqlalchemy  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import fetch_and_format_exercises_sqlalchemy. "
        "Ensure fetch_exercises.py is in PYTHONPATH."
    ) from exc


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

    print(docs[0])

    # --------------------------------------------------------------------------------
    # Step 3: Prepare embedding model
    
    # ①  Make sure your key is set (Gemini API key, not Vertex) // get it from env variables
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    print(gemini_api_key)
    if not gemini_api_key:
        raise SystemExit("GEMINI_API_KEY not set. Provide at least one.")
    #    └─ LlamaIndex will look here, but we’ll also pass it explicitly


    # ③  One wrapper object that LlamaIndex understands
    # ③  One wrapper object that LlamaIndex understands
    embed_model = GeminiEmbedding(api_key=gemini_api_key)
    print(embed_model)
    # vector_dim = embed_model._get_query_embedding("test").size  # quick hack to infer dim

    # # --------------------------------------------------------------------------------
    # Step 4: import qdrant client instance from app.core.qdrant
    q_client = get_qdrant_client()
    vector_dim = 3_072    # gemini-embedding-exp-03-07 outputs 3 072-dim vectors
    ensure_qdrant_collection(q_client, args.collection, vector_dim)
    vector_store = QdrantVectorStore(client=q_client, collection_name=args.collection)


    # # # --------------------------------------------------------------------------------
    # # # Step 5: Build / upsert index
    print("[embed] Generating embeddings and upserting into Qdrant …")
    # VectorStoreIndex.from_documents(
    #     docs,
    #     vector_store=vector_store,
    #     embed_model=embed_model,
    #     show_progress=True,
    #     batch_size=128,
    #     store_nodes_override=False,  # don't duplicate nodes if same id
    # )
    print("[embed] Done!\n")

    # # --------------------------------------------------------------------------------
    # # Step 6: Sanity-check retrieval
    # print("[test] Example retrieval → 'abdominaux sans matériel'")
    # retriever = vector_store.as_retriever(similarity_top_k=3, embed_model=embed_model)
    # results = retriever.retrieve("abdominaux sans matériel")
    # for res in results:
    #     meta = res.metadata
    #     print(f" > ID={meta['exercise_id']} (score {res.score:.3f}) | Title snippet: {res.get_content()[:50]}…")


if __name__ == "__main__":
    asyncio.run(main()) 