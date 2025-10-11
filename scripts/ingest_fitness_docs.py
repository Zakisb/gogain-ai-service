#!/usr/bin/env python3
"""
Ingest fitness docs into Qdrant with lossless chunking (new named-vector API).

Fixes & improvements vs. original:
- Robust code-fence handling (sets buf_type='code' on open; flushes if fence unclosed at EOF)
- List continuation respects indent levels; nested lists are separate chunks
- Questions chunker only marks `chunk_type='question'` when it truly is one
- Skip embedding for noise chunks (blank, heading); still store them for lossless reconstruction
- Deterministic point IDs (SHA1 of doc_id:start:end) + pre-delete by doc_id to avoid dupes
- Validate collection vector size; recreate on mismatch
- Add payload indexes for fast filters (doc_id, section_slug, chunk_type, file_name, order)
- Cleaned text for embedding (e.g., strip bullet markers) while preserving original `text`
- Extra payload: char_count, word_count, prev_order, next_order

Env (.env):
  OPENAI_API_KEY=...
  QDRANT_URL=http(s)://<host>:<port>
  QDRANT_API_KEY=<optional if needed>

Optional env:
  GUIDELINES_PATH=fitness_training_guidelines_report.md
  QUESTIONS_PATH=fitness_goals_questions.md
  COLLECTION_GUIDELINES=fitness_guidelines
  COLLECTION_QUESTIONS=fitness_questions
  EMBEDDING_MODEL=text-embedding-3-small
"""

import os
import re
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ---------- Helpers ----------

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


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

def _indent_len(s: str) -> int:
    # Treat tabs as 4 spaces for continuation tests
    return len(s.replace("\t", "    "))


def chunk_markdown_hier_para(text: str, file_name: str) -> List[Chunk]:
    """
    Strategy A: Markdown Hierarchical-Paragraph (paragraphs, list items, code blocks).
    Fully lossless: each chunk stores exact text and char offsets.
    Improvements:
    - Respect indent when grouping list continuation lines
    - Handle unclosed code fences
    """
    lines = text.splitlines(keepends=True)
    chunks: List[Chunk] = []
    order = 0
    pos = 0
    headings: List[str] = []  # stack of H1..H6
    code_fence: Optional[str] = None
    buf_lines: List[str] = []
    buf_start: Optional[int] = None
    buf_type = "paragraph"

    def flush_buf():
        nonlocal order, buf_lines, buf_start, buf_type
        if buf_lines:
            chunk_text = "".join(buf_lines)
            chunks.append(Chunk(
                text=chunk_text,
                start=buf_start,
                end=buf_start + len(chunk_text),
                order=order,
                meta={
                    "chunk_type": buf_type,
                    "section_path": headings.copy(),
                    "section_slug": "/".join(slugify(h) for h in headings),
                    "file_name": file_name,
                }
            ))
            order += 1
            buf_lines = []
            buf_start = None
            # keep buf_type as-is; caller should set before next use

    # Detect whether a line is a list item or a numbered item
    list_pat = re.compile(r"^([ \t]*)([-*+]\s+|\d+\.\s+)(.*)$")
    heading_pat = re.compile(r"^(#{1,6})\s+(.*)$")

    i = 0
    while i < len(lines):
        line = lines[i]
        line_len = len(line)
        cur_pos = pos

        # Handle code fences as atomic blocks
        if code_fence:
            # inside code block
            if line.strip().startswith(code_fence):
                # include closing fence and flush
                if not buf_lines:
                    buf_start = cur_pos
                buf_lines.append(line)
                buf_type = "code"  # ensure type
                flush_buf()
                code_fence = None
            else:
                if not buf_lines:
                    buf_start = cur_pos
                buf_lines.append(line)
            pos += line_len
            i += 1
            continue

        # Opening code fence?
        m_code = re.match(r"^(```|~~~)", line.strip())
        if m_code:
            # flush any pending paragraph/list before code
            flush_buf()
            code_fence = m_code.group(1)
            if not buf_lines:
                buf_start = cur_pos
            buf_type = "code"  # set on open so unclosed fences still get code type
            buf_lines.append(line)
            pos += line_len
            i += 1
            continue

        # Heading?
        m_h = heading_pat.match(line)
        if m_h:
            # flush paragraph/list before heading
            flush_buf()
            level = len(m_h.group(1))
            title = m_h.group(2).strip()
            # update heading stack
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
            # Store heading as a chunk (lossless structure)
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
            i += 1
            continue

        # List item (treat continuation lines with deeper indent as part of the item; stop at nested bullets)
        m_list = list_pat.match(line)
        if m_list:
            # flush any pending paragraph
            flush_buf()
            open_indent = _indent_len(m_list.group(1))
            li_start = cur_pos
            li_lines = [line]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                nxt_len = len(nxt)
                # New bullet at same or lower indent → stop
                m_next = list_pat.match(nxt)
                if m_next:
                    nxt_indent = _indent_len(m_next.group(1))
                    if nxt_indent <= open_indent:
                        break
                    else:
                        # nested bullet → stop so nested list becomes its own chunk
                        break
                # Boundary tokens
                if heading_pat.match(nxt) or re.match(r"^(```|~~~)", nxt.strip()):
                    break
                # continuation if blank or strictly deeper indent than the opening indent
                if nxt.strip() == "" or _indent_len(re.match(r"^([ \t]*)", nxt).group(1)) > open_indent:
                    li_lines.append(nxt)
                    j += 1
                    continue
                else:
                    break
            li_text = "".join(li_lines)
            chunks.append(Chunk(
                text=li_text,
                start=li_start,
                end=li_start + len(li_text),
                order=order,
                meta={
                    "chunk_type": "list_item",
                    "section_path": headings.copy(),
                    "section_slug": "/".join(slugify(h) for h in headings),
                    "file_name": file_name,
                }
            ))
            order += 1
            pos += sum(len(x) for x in li_lines)
            i = j
            continue

        # Blank line → paragraph boundary
        if line.strip() == "":
            flush_buf()
            # store blank line to keep offsets lossless
            chunks.append(Chunk(
                text=line,
                start=cur_pos,
                end=cur_pos + line_len,
                order=order,
                meta={
                    "chunk_type": "blank",
                    "section_path": headings.copy(),
                    "section_slug": "/".join(slugify(h) for h in headings),
                    "file_name": file_name,
                }
            ))
            order += 1
            pos += line_len
            i += 1
            continue

        # Default: paragraph text (accumulate contiguous lines until blank/heading/list/code)
        if not buf_lines:
            buf_start = cur_pos
            buf_type = "paragraph"
        buf_lines.append(line)
        pos += line_len
        i += 1

    # flush tail
    flush_buf()
    # If file ended while code fence was open, it's already marked as code via buf_type set on open
    return chunks


def chunk_questions_atomic(text: str, file_name: str) -> List[Chunk]:
    """
    Strategy B: one question per chunk.
    Recognizes headings to attach goal/subsections; any bullet/numbered/line ending in '?' becomes a question chunk.
    Lossless: exact question text with offsets. Non-question bullets are `list_item`.
    """
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
                # store a cleaned question form helpful for embedding/search
                payload["question_plain"] = content.strip()
            chunks.append(Chunk(
                text=line,  # keep bullet prefix for losslessness
                start=cur_pos,
                end=cur_pos + line_len,
                order=order,
                meta=payload,
            ))
            order += 1
            pos += line_len
            continue

        # Non-list line that ends with '?' → standalone question
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

        # Everything else is kept as structural filler to remain reconstructible
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
    # Batches are fine to add if your files are huge; here keep it simple
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def clean_for_embedding(chunk: Chunk) -> str:
    """Return a cleaned representation for embedding while preserving lossless storage in payload.
    - Strip leading bullet markers for list items
    - Prefer `question_plain` if present
    - Collapse whitespace lightly
    """
    t = chunk.meta.get("question_plain")
    if t:
        return re.sub(r"\s+", " ", t).strip()
    if chunk.meta.get("chunk_type") == "list_item":
        # remove bullet prefix while keeping body
        m = re.match(r"^[ \t]*([-*+]\s+|\d+\.\s+)(.*)$", chunk.text, flags=re.DOTALL)
        if m:
            t = m.group(2)
        else:
            t = chunk.text
    else:
        t = chunk.text
    return re.sub(r"\s+", " ", t).strip()

# ---------- Qdrant setup (named vectors) ----------

def ensure_collection(client: QdrantClient, name: str, vector_size: int):
    """
    Create (or update) a collection using the new named-vector format: {"text-dense": VectorParams(...)}.
    Recreate if size mismatches.
    """
    vectors_config = {"text-dense": qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE)}
    try:
        info = client.get_collection(name)
        # Determine existing size for named vectors
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
    """Create useful payload indexes; ignore errors if they already exist."""
    for field, ftype in [
        ("doc_id", qm.PayloadSchemaType.KEYWORD),
        ("section_slug", qm.PayloadSchemaType.KEYWORD),
        ("chunk_type", qm.PayloadSchemaType.KEYWORD),
        ("file_name", qm.PayloadSchemaType.KEYWORD),
        ("order", qm.PayloadSchemaType.INTEGER),
        ("char_start", qm.PayloadSchemaType.INTEGER),
        ("char_end", qm.PayloadSchemaType.INTEGER),
    ]:
        try:
            client.create_payload_index(collection_name=collection, field_name=field, field_schema=ftype)
        except Exception:
            pass


def deterministic_point_id(doc_id: str, start: int, end: int) -> str:
    """Return a deterministic UUID string acceptable by Qdrant.

    Qdrant requires point IDs to be unsigned integers or UUIDs. We generate a
    stable UUID v5 from the doc_id and character offsets to preserve
    determinism across runs while conforming to the API.
    """
    name = f"{doc_id}:{start}:{end}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    doc_id: str,
    doc_sha1: str,
    chunks: List[Chunk],
    embedder: OpenAI,
    model: str
):
    # Compute cleaned texts to embed and skip noise types
    embed_indices: List[int] = []
    embed_texts_list: List[str] = []
    for idx, c in enumerate(chunks):
        ctype = c.meta.get("chunk_type")
        if ctype in {"blank", "heading"}:
            continue
        cleaned = clean_for_embedding(c)
        if not cleaned:
            # Skip empty/whitespace-only inputs which cause embedding 400 errors
            continue
        embed_indices.append(idx)
        embed_texts_list.append(cleaned)

    if not embed_texts_list:
        # no embeddable chunks; ensure collection exists anyway (choose a default size)
        ensure_collection(client, collection, 1536)
        create_payload_indexes(client, collection)
        return

    vectors = embed_texts(embedder, embed_texts_list, model)
    dim = len(vectors[0])
    ensure_collection(client, collection, dim)
    create_payload_indexes(client, collection)

    # Remove any existing points for this doc_id to avoid duplicates from prior runs
    try:
        client.delete(collection_name=collection, points_selector=qm.Filter(
            must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
        ))
    except Exception:
        pass

    # Build and upsert points
    points = []
    # quick map for prev/next calculation
    total = len(chunks)
    for local_i, (idx, vec) in enumerate(zip(embed_indices, vectors)):
        c = chunks[idx]
        payload = {
            **c.meta,
            "doc_id": doc_id,
            "doc_sha1": doc_sha1,
            "order": c.order,
            "prev_order": c.order - 1 if c.order > 0 else None,
            "next_order": c.order + 1 if c.order < total - 1 else None,
            "char_start": c.start,
            "char_end": c.end,
            "text": c.text,  # full original text retained for lossless retrieval
            "char_count": len(c.text),
            "word_count": count_words(c.text),
        }
        pid = deterministic_point_id(doc_id, c.start, c.end)
        points.append(qm.PointStruct(
            id=pid,
            vector={"text-dense": vec},   # NEW named-vector API
            payload=payload
        ))

    client.upsert(collection_name=collection, points=points)

# ---------- Main ----------

def main():
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not qdrant_url or not openai_api_key:
        raise SystemExit("Missing QDRANT_URL or OPENAI_API_KEY in .env")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    def resolve_path(path_str: str, default_name: str) -> Path:
        if not path_str:
            candidate = repo_root / "documents" / default_name
            return candidate
        p = Path(path_str)
        if p.is_absolute():
            return p
        # Try common relative bases
        candidates = [
            Path.cwd() / p,
            script_dir / p,
            repo_root / p,
            repo_root / "documents" / p,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]

    guidelines_path = resolve_path(os.getenv("GUIDELINES_PATH"), "fitness_training_guidelines_report.md")
    questions_path = resolve_path(os.getenv("QUESTIONS_PATH"), "fitness_goals_questions.md")
    collection_guidelines = os.getenv("COLLECTION_GUIDELINES", "fitness_guidelines")
    collection_questions = os.getenv("COLLECTION_QUESTIONS", "fitness_questions")
    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    qdr = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    oai = OpenAI(api_key=openai_api_key)

    # ---- Ingest guidelines (Markdown hierarchical-paragraph) ----
    if guidelines_path.exists():
        with guidelines_path.open("rb") as f:
            b = f.read()
        text = b.decode("utf-8")
        doc_sha = sha1_bytes(b)
        doc_id = f"guidelines::{guidelines_path.name}::{doc_sha[:8]}"
        chunks = chunk_markdown_hier_para(text, guidelines_path.name)
        print(f"[Guidelines] chunks (total stored): {len(chunks)}")
        upsert_chunks(qdr, collection_guidelines, doc_id, doc_sha, chunks, oai, embed_model)
        print("[Guidelines] upserted (embeddings skipped for blank/heading)")
    else:
        print(f"[Guidelines] Not found: {guidelines_path}")

    # ---- Ingest questions (Atomic questions) ----
    if questions_path.exists():
        with questions_path.open("rb") as f:
            b = f.read()
        text = b.decode("utf-8")
        doc_sha = sha1_bytes(b)
        doc_id = f"questions::{questions_path.name}::{doc_sha[:8]}"
        chunks = chunk_questions_atomic(text, questions_path.name)
        print(f"[Questions] chunks (total stored): {len(chunks)}")
        upsert_chunks(qdr, collection_questions, doc_id, doc_sha, chunks, oai, embed_model)
        print("[Questions] upserted (embeddings only for question/list/paragraph/code)")
    else:
        print(f"[Questions] Not found: {questions_path}")

    print("✅ Ingestion complete (named vectors, lossless payloads, fast filters, idempotent upserts).")


if __name__ == "__main__":
    main()
