#!/usr/bin/env python3
"""
Simple CLI to query fitness_questions and fitness_guidelines collections in Qdrant.

Usage examples:
  python scripts/test_fitness_rag_query.py --query "perdre du poids" --top-k 5
  python scripts/test_fitness_rag_query.py --query "warm-up guidelines" --collections fitness_guidelines

Env (.env):
  QDRANT_URL, QDRANT_API_KEY (optional), OPENAI_API_KEY
  EMBEDDING_MODEL (defaults to text-embedding-3-small)
"""

import argparse
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query Qdrant fitness RAG collections")
    p.add_argument("--query", required=True, help="Natural language query text")
    p.add_argument("--top-k", type=int, default=5, help="Top K results per collection")
    p.add_argument(
        "--collections",
        nargs="+",
        default=["fitness_questions", "fitness_guidelines"],
        help="Collections to search (space-separated)")
    p.add_argument("--no-questions-filter", action="store_true", help="Do not filter fitness_questions to question chunks only")
    return p.parse_args()


def get_embedder() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment")
    return OpenAI(api_key=api_key)


def embed(oai: OpenAI, text: str, model: str) -> List[float]:
    text = (text or "").strip()
    if not text:
        raise SystemExit("Empty query provided")
    resp = oai.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding


def get_qdrant() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if not url:
        raise SystemExit("Missing QDRANT_URL in environment")
    api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def search_collection(client: QdrantClient, collection: str, vector: List[float], top_k: int, filter_questions_only: bool):
    query_filter = None
    if filter_questions_only and collection == "fitness_questions":
        # Only return actual question chunks
        query_filter = qm.Filter(must=[qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value="question"))])

    return client.search(
        collection_name=collection,
        query_vector=qm.NamedVector(name="text-dense", vector=vector),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=query_filter,
    )


def main():
    load_dotenv()
    args = get_args()

    oai = get_embedder()
    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    vec = embed(oai, args.query, embed_model)

    qdr = get_qdrant()

    print(f"Query: {args.query}")
    print(f"Model: {embed_model}")
    print("")

    for col in args.collections:
        try:
            results = search_collection(
                qdr, col, vec, args.top_k, filter_questions_only=not args.no_questions_filter
            )
        except Exception as e:
            print(f"Collection {col}: ERROR - {e}")
            continue

        print(f"=== {col} (top {args.top_k}) ===")
        if not results:
            print("  No results")
            print("")
            continue

        for i, sp in enumerate(results, start=1):
            pl = sp.payload or {}
            text = (pl.get("text") or "").replace("\n", "\\n")
            preview = text[:160] + ("…" if len(text) > 160 else "")
            print(
                f"  {i}. id={sp.id} score={sp.score:.4f} type={pl.get('chunk_type')} "
                f"slug={pl.get('section_slug')} file={pl.get('file_name')}"
            )
            qp = pl.get("question_plain")
            if qp:
                print(f"     Q: {qp}")
            print(f"     Text: {preview}")

        print("")


if __name__ == "__main__":
    main()

