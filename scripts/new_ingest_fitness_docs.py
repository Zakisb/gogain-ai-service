#!/usr/bin/env python3
"""
Ingest fitness docs into Qdrant with lossless chunking (named-vector API, v2).

Project structure assumed:
  project_root/
  ├── documents/
  │   ├── fitness_training_guidelines_report.md
  │   └── fitness_goals_questions.md
  └── scripts/
      └── ingest_fitness_v2.py

Improvements:
- Auto-resolves paths (../documents/)
- Logs resolved paths and counts before ingestion
- Creates new collection `fitness_questions_v2` to preserve old data
- Purges only entries from the same file
- Keeps deterministic IDs and payload indexes
"""

import os
import re
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ---------- Helpers ----------

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s.strip().lower())
    return s

def count_words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))

@dataclass
class Chunk:
    text: str
    start: int
    end: int
    order: int
    meta: Dict[str, Any]

# ---------- Chunkers ----------

def chunk_questions_atomic(text: str, file_name: str) -> List[Chunk]:
    """One question per chunk. Keeps exact text and section_path."""
    lines = text.splitlines(keepends=True)
    chunks: List[Chunk] = []
    order = 0
    pos = 0
    headings: List[str] = []
    heading_pat = re.compile(r"^(#{1,6})\s+(.*)$")
    list_pat = re.compile(r"^([ \t]*)([-*+]\s+|\d+\.\s+)(.*)$")

    for line in lines:
        cur_pos = pos
        line_len = len(line)

        # Headings
        m_h = heading_pat.match(line)
        if m_h:
            level = len(m_h.group(1))
            title = m_h.group(2).strip()
            if level <= len(headings):
                headings = headings[:level-1]
            while len(headings) < level-1:
                headings.append("")
            if level == 1:
                headings = [title]
            else:
                if len(headings) == level-1:
                    headings.append(title)
                else:
                    headings[level-1] = title
            chunks.append(Chunk(
                text=line,
                start=cur_pos,
                end=cur_pos + line_len,
                order=order,
                meta={
                    "chunk_type": "heading",
                    "heading_level": level,
                    "section_path": headings.copy(),
                    "section_slug": "/".join(slugify(h) for h in headings),
                    "file_name": file_name,
                }
            ))
            order += 1
            pos += line_len
            continue

        # Bullet or numbered line
        m_list = list_pat.match(line)
        if m_list:
            content = m_list.group(3)
            is_question = content.strip().endswith("?")
            ck_type = "question" if is_question else "list_item"
            payload = {
                "chunk_type": ck_type,
                "section_path": headings.copy(),
                "section_slug": "/".join(slugify(h) for h in headings),
                "file_name": file_name,
                "is_question": is_question,
                "goal_section": headings[1] if len(headings) > 1 else None,
                "subsection": headings[2] if len(headings) > 2 else None,
            }
            if is_question:
                payload["question_plain"] = content.strip()
            chunks.append(Chunk(
                text=line,
                start=cur_pos,
                end=cur_pos + line_len,
                order=order,
                meta=payload,
            ))
            order += 1
            pos += line_len
            continue

        # Standalone question
        if line.strip().endswith("?"):
            chunks.append(Chunk(
                text=line,
                start=cur_pos,
                end=cur_pos + line_len,
                order=order,
                meta={
                    "chunk_type": "question",
                    "is_question": True,
                    "section_path": headings.copy(),
                    "section_slug": "/".join(slugify(h) for h in headings),
                    "file_name": file_name,
                    "goal_section": headings[1] if len(headings) > 1 else None,
                    "subsection": headings[2] if len(headings) > 2 else None,
                    "question_plain": line.strip(),
                }
            ))
            order += 1
            pos += line_len
            continue

        # Everything else
        chunks.append(Chunk(
            text=line,
            start=cur_pos,
            end=cur_pos + line_len,
            order=order,
            meta={
                "chunk_type": "other",
                "section_path": headings.copy(),
                "section_slug": "/".join(slugify(h) for h in headings),
                "file_name": file_name,
            }
        ))
        order += 1
        pos += line_len

    return chunks

# ---------- Embeddings ----------

def embed_texts(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def clean_for_embedding(chunk: Chunk) -> str:
    t = chunk.meta.get("question_plain")
    if t:
        return re.sub(r"\s+", " ", t).strip()
    return re.sub(r"\s+", " ", chunk.text).strip()

# ---------- Qdrant setup ----------

def ensure_collection(client: QdrantClient, name: str, vector_size: int):
    vectors_config = {"text-dense": qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE)}
    try:
        info = client.get_collection(name)
        existing = info.vectors
        existing_size = None
        if isinstance(existing, dict):
            vp = existing.get("text-dense")
            if vp and hasattr(vp, "size"):
                existing_size = vp.size
        elif hasattr(existing, "size"):
            existing_size = existing.size
        if existing_size is None or existing_size != vector_size:
            client.recreate_collection(collection_name=name, vectors_config=vectors_config)
    except Exception:
        client.recreate_collection(collection_name=name, vectors_config=vectors_config)

def create_payload_indexes(client: QdrantClient, collection: str):
    for field, ftype in [
        ("doc_id", qm.PayloadSchemaType.KEYWORD),
        ("section_slug", qm.PayloadSchemaType.KEYWORD),
        ("chunk_type", qm.PayloadSchemaType.KEYWORD),
        ("file_name", qm.PayloadSchemaType.KEYWORD),
        ("order", qm.PayloadSchemaType.INTEGER),
    ]:
        try:
            client.create_payload_index(collection_name=collection, field_name=field, field_schema=ftype)
        except Exception:
            pass

def deterministic_point_id(doc_id: str, start: int, end: int) -> str:
    name = f"{doc_id}:{start}:{end}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))

def upsert_chunks(client: QdrantClient, collection: str, doc_id: str, doc_sha1: str,
                  chunks: List[Chunk], embedder: OpenAI, model: str):
    embed_indices, embed_texts_list = [], []
    for idx, c in enumerate(chunks):
        if c.meta.get("chunk_type") in {"blank", "heading"}:
            continue
        cleaned = clean_for_embedding(c)
        if not cleaned:
            continue
        embed_indices.append(idx)
        embed_texts_list.append(cleaned)

    if not embed_texts_list:
        ensure_collection(client, collection, 1536)
        create_payload_indexes(client, collection)
        return

    vectors = embed_texts(embedder, embed_texts_list, model)
    dim = len(vectors[0])
    ensure_collection(client, collection, dim)
    create_payload_indexes(client, collection)

    file_name = chunks[0].meta.get("file_name", "")
    try:
        client.delete(
            collection_name=collection,
            points_selector=qm.Filter(
                must=[qm.FieldCondition(key="file_name", match=qm.MatchValue(value=file_name))]
            )
        )
        print(f"[Cleanup] Removed old entries for {file_name}")
    except Exception:
        pass

    points = []
    for idx, vec in zip(embed_indices, vectors):
        c = chunks[idx]
        payload = {
            **c.meta,
            "doc_id": doc_id,
            "doc_sha1": doc_sha1,
            "order": c.order,
            "char_start": c.start,
            "char_end": c.end,
            "text": c.text,
            "char_count": len(c.text),
            "word_count": count_words(c.text),
        }
        pid = deterministic_point_id(doc_id, c.start, c.end)
        points.append(qm.PointStruct(id=pid, vector={"text-dense": vec}, payload=payload))

    client.upsert(collection_name=collection, points=points)

# ---------- Main ----------

def main():
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not qdrant_url or not openai_api_key:
        raise SystemExit("❌ Missing QDRANT_URL or OPENAI_API_KEY in .env")

    # Path resolution (documents/ is sibling to scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    documents_dir = project_root / "documents"

    guidelines_path = Path(os.getenv("GUIDELINES_PATH", documents_dir / "fitness_training_guidelines_report.md"))
    questions_path = Path(os.getenv("QUESTIONS_PATH", documents_dir / "fitness_goals_questions.md"))

    collection_guidelines = os.getenv("COLLECTION_GUIDELINES", "fitness_guidelines")
    collection_questions = os.getenv("COLLECTION_QUESTIONS", "fitness_questions_v2")
    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    print(f"[Paths] GUIDELINES_PATH -> {guidelines_path}")
    print(f"[Paths] QUESTIONS_PATH  -> {questions_path}")
    print(f"[Paths] COLLECTION      -> {collection_questions}")

    qdr = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    oai = OpenAI(api_key=openai_api_key)

    # ---- Ingest questions (atomic chunks) ----
    if questions_path.exists():
        with questions_path.open("rb") as f:
            b = f.read()
        text = b.decode("utf-8")
        doc_sha = sha1_bytes(b)
        doc_id = f"questions::{questions_path.name}::{doc_sha[:8]}"
        chunks = chunk_questions_atomic(text, questions_path.name)

        sec_counts = Counter(c.meta.get("section_slug", "<none>") for c in chunks if c.meta.get("chunk_type") == "question")
        total_q = sum(sec_counts.values())
        print("[Questions] per-section counts from file:", dict(sec_counts))
        print(f"[Questions] total questions in file: {total_q}")

        upsert_chunks(qdr, collection_questions, doc_id, doc_sha, chunks, oai, embed_model)
        print(f"✅ Ingestion complete into '{collection_questions}' ({total_q} question chunks).")
    else:
        print(f"❌ Questions file not found: {questions_path}")

if __name__ == "__main__":
    main()

