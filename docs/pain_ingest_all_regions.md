Pain Docs: All-Regions Ingestion

Overview
- Creates two Qdrant collections shared across all regions:
  - `questions_all_regions`: atomic items (question, red_flag, follow_up)
  - `context_all_regions`: explanations/causes (150-word windows, 25-word overlap)
- Uses Voyage `voyage-3-large` embeddings (1024-dim).

Prereqs
- Env vars: `VOYAGE_API_KEY`, `QDRANT_URL` (Docker default: `http://qdrant:6333`).
- Docs live under `documents/`.

Run
- Inside the ai_service container:
  - `python -m app.knowledge.pain_ingest_all_regions` (ingest all docs)
  - `python -m app.knowledge.pain_ingest_all_regions --dry-run` (no upsert)
  - `python -m app.knowledge.pain_ingest_all_regions --docs documents/lower_back_pain_final_report.md --dry-run`

Notes
- Payload fields: `type, body_region, phase, priority, tags, signals, related_ids, text`.
- Retrieval engine: `app/knowledge/pain_retrieval_engine.py` uses embedding-only classification (no keyword matching on user queries), dense retrieval, and optional Voyage rerank.
