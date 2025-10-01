#!/usr/bin/env python3
"""
Chunk + embed all pain documents into two Qdrant collections across all regions.

Collections:
  - questions_all_regions: atomic items (question, red_flag, follow_up)
  - context_all_regions:   explanations/causes (150-word windows, 25-word overlap)

Embeddings: Voyage "voyage-3-large" via LlamaIndex wrapper.
Runs inside the ai_service container; uses `QDRANT_URL` from env/config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import QDRANT_URL
from app.core.embeddings import get_embedding_model, EmbeddingConfig
from chunk_embed_two_pass import TwoPassChunker, Atom, AtomType


# -----------------------------
# Config / inputs
# -----------------------------

PAIN_DOCS = [
    "documents/lower_back_pain_final_report.md",
    "documents/knees_pain_final_report.md",
    "documents/shoulder_pain_final_report.md",
    "documents/neck_pain_final_report.md",
    "documents/foot_ankle_final_pain_report.md",
    "documents/hips_final_pain_report.md",
    "documents/Pain report.md",
]

BODY_REGION_MAP = {
    "lower_back_pain_final_report.md": "back",
    "knees_pain_final_report.md": "knee",
    "shoulder_pain_final_report.md": "shoulder",
    "neck_pain_final_report.md": "neck",
    "foot_ankle_final_pain_report.md": "foot_ankle",
    "hips_final_pain_report.md": "hip",
    "Pain report.md": "general",
}

COL_QUESTIONS = "questions_all_regions"
COL_CONTEXT = "context_all_regions"


# -----------------------------
# Qdrant helpers
# -----------------------------

def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    try:
        info = client.get_collection(collection_name=name)
        cfg = getattr(info, "config", None)
        if cfg and getattr(cfg, "params", None) and getattr(cfg.params, "vectors", None):
            size = getattr(cfg.params.vectors, "size", None)
            if size and size != vector_size:
                client.delete_collection(collection_name=name)
                client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
        return
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def to_payload(a: Atom) -> Dict:
    """Map Atom to the required payload contract.

    Fields:
      - id (implicit as point id)
      - type, body_region, phase, priority, tags, signals
      - related_ids (from Atom.related_chunks)
      - text
      - section_path, block_index, segment_char_start, segment_char_end
    """
    return {
        "text": a.text,
        "type": a.type.value,
        "body_region": a.body_region,
        "phase": a.phase.value,
        "priority": a.priority.value,
        "tags": a.tags,
        "signals": a.signals,
        "related_ids": a.related_chunks or [],
        "section_path": a.section_path,
        "block_index": a.block_index,
        "segment_char_start": a.segment_char_start,
        "segment_char_end": a.segment_char_end,
    }


# -----------------------------
# Main pipeline
# -----------------------------

def chunk_doc(path: Path, body_region: str) -> List[Atom]:
    text = path.read_text(encoding="utf-8")
    chunker = TwoPassChunker(max_words=150, overlap_words=25)
    return chunker.process_document(text, body_region)


def partition_atoms(atoms: List[Atom]) -> Tuple[List[Atom], List[Atom]]:
    """Split into (questions/red_flags/followups) and (explanations/causes)."""
    a_side: List[Atom] = []
    b_side: List[Atom] = []
    for a in atoms:
        if a.type in (AtomType.QUESTION, AtomType.RED_FLAG, AtomType.FOLLOW_UP):
            a_side.append(a)
        else:
            b_side.append(a)
    return a_side, b_side


def upsert(client: QdrantClient, collection: str, embed_model, atoms: List[Atom]) -> int:
    if not atoms:
        return 0
    # Prepare point dicts to maintain compatibility with older client/server
    pts = []
    for a in atoms:
        vec = embed_model.get_text_embedding(a.text)
        pts.append({"id": str(a.id), "vector": vec, "payload": to_payload(a)})
    try:
        client.openapi_client.points_api.upsert_points(
            collection_name=collection,
            point_insert_operations={"points": pts},
            wait=True,
        )
    except Exception:
        client.openapi_client.points_api.upsert_points(
            collection_name=collection,
            upsert_request={"points": pts},
            wait=True,
        )
    return len(pts)


def ingest_all(selected_docs: List[str] | None = None, dry_run: bool = False) -> Dict:
    load_dotenv()
    client = QdrantClient(url=QDRANT_URL)
    embed_model = get_embedding_model()
    vec_dim = EmbeddingConfig.DEFAULT_VECTOR_DIM

    ensure_collection(client, COL_QUESTIONS, vec_dim)
    ensure_collection(client, COL_CONTEXT, vec_dim)

    docs = selected_docs or PAIN_DOCS
    processed = 0
    upserts = {COL_QUESTIONS: 0, COL_CONTEXT: 0}

    for p in docs:
        path = Path(p)
        if not path.exists():
            print(f"⚠️  Missing: {p}")
            continue
        body_region = BODY_REGION_MAP.get(path.name, "general")
        print(f"📄 Processing {path.name} → {body_region}")
        atoms = chunk_doc(path, body_region)
        a_side, b_side = partition_atoms(atoms)

        if dry_run:
            print(f"  • atoms: {len(atoms)} | A-side: {len(a_side)} | B-side: {len(b_side)}")
        else:
            upserts[COL_QUESTIONS] += upsert(client, COL_QUESTIONS, embed_model, a_side)
            upserts[COL_CONTEXT] += upsert(client, COL_CONTEXT, embed_model, b_side)
        processed += 1

    # Summary
    out = {"processed_docs": processed, "upserts": upserts}
    try:
        qi = client.get_collection(COL_QUESTIONS)
        ci = client.get_collection(COL_CONTEXT)
        out["collection_counts"] = {
            COL_QUESTIONS: getattr(qi, "points_count", None),
            COL_CONTEXT: getattr(ci, "points_count", None),
        }
    except Exception:
        pass
    return out


def main():
    parser = argparse.ArgumentParser(description="Ingest pain docs into all-regions Qdrant collections")
    parser.add_argument("--docs", nargs="*", help="Optional specific doc paths to ingest")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, do not upsert")
    args = parser.parse_args()

    summary = ingest_all(selected_docs=args.docs or None, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

