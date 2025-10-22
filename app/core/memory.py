"""Mem0 client that enriches conversations with GPT-4o mini and stores them in Qdrant."""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from mem0 import Memory
from openai import OpenAI
from qdrant_client import QdrantClient


logger = logging.getLogger(__name__)


class Mem0Client:
    """High-level helper that enriches chat messages and persists them via Mem0/Qdrant."""

    def __init__(self) -> None:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for Mem0 enrichment")

        self.llm_model = os.getenv("MEM0_ENRICH_MODEL", "gpt-4o-mini")
        try:
            self.openai_client: Optional[OpenAI] = OpenAI(api_key=openai_key)
        except Exception as exc:  # pragma: no cover - defensive initialisation
            logger.warning("⚠️ Failed to initialise OpenAI client for Mem0 enrichment: %s", exc)
            self.openai_client = None

        self.fact_confidence_threshold = float(os.getenv("MEM0_FACT_CONFIDENCE_MIN", "0.7"))

        qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        collection_name = os.getenv("MEM0_QDRANT_COLLECTION", "mem0")

        # Quick connectivity test to surface configuration issues early
        try:
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=5.0)
            qdrant_client.get_collections()
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to Qdrant at {qdrant_url}: {exc}") from exc

        config: Dict[str, Any] = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": collection_name,
                    "url": qdrant_url,
                },
            }
        }
        if qdrant_api_key:
            config["vector_store"]["config"]["api_key"] = qdrant_api_key

        self.client = Memory.from_config(config)

    # ------------------------------------------------------------------
    # Public API (compatible with previous Mem0Client usage)
    # ------------------------------------------------------------------
    def add_memory(
        self,
        message: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Enrich a chat message and persist it for the given user."""

        try:
            result = self.ingest_chat_message(user_id=user_id, message_text=message, metadata=metadata)
            return {"success": True, "result": result}
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("❌ [Mem0] add_memory failed: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    def get_memories(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        try:
            raw = self.client.get_all(user_id=user_id, limit=limit)
            items = raw.get("results", []) if isinstance(raw, dict) else raw
            normalized = [self._format_memory_item(self._normalize_mem0_item(item)) for item in items or []]
            return {"success": True, "memories": normalized}
        except Exception as exc:
            logger.error("❌ [Mem0] get_memories failed: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    def search_memories(self, query: str, user_id: str, limit: int = 5) -> Dict[str, Any]:
        try:
            raw = self.client.search(query=query, user_id=user_id, limit=limit)
            items = raw.get("results", []) if isinstance(raw, dict) else raw
            normalized = [self._format_memory_item(self._normalize_mem0_item(item)) for item in items or []]
            return {"success": True, "memories": normalized}
        except Exception as exc:
            logger.error("❌ [Mem0] search_memories failed: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    def get_recent_memories(
        self,
        user_id: str,
        limit: int = 10,
        within_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            base = self.get_memories(user_id=user_id, limit=1000)
            if not base.get("success"):
                return base

            items = base.get("memories", [])
            now_ts = _now_unix()

            def parse_timestamp(memory: Dict[str, Any]) -> float:
                metadata = (memory or {}).get("metadata", {}) if isinstance(memory, dict) else {}
                ts = metadata.get("timestamp_unix")
                if isinstance(ts, (int, float)):
                    return float(ts)
                iso = metadata.get("timestamp_iso")
                return _iso_to_unix(iso) if isinstance(iso, str) else 0.0

            filtered: List[Tuple[float, Dict[str, Any]]] = []
            for memory in items:
                ts = parse_timestamp(memory)
                if within_seconds is not None and ts > 0:
                    if (now_ts - ts) <= within_seconds:
                        filtered.append((ts, memory))
                else:
                    filtered.append((ts, memory))

            filtered.sort(key=lambda pair: pair[0], reverse=True)
            return {"success": True, "memories": [m for _, m in filtered[:limit]]}
        except Exception as exc:
            logger.error("❌ [Mem0] get_recent_memories failed: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    def get_memories_in_range(
        self,
        user_id: str,
        start_iso: str,
        end_iso: str,
        limit: int = 100,
    ) -> Dict[str, Any]:
        try:
            start_ts = _iso_to_unix(start_iso)
            end_ts = _iso_to_unix(end_iso)
            base = self.get_memories(user_id=user_id, limit=1000)
            if not base.get("success"):
                return base

            selected: List[Dict[str, Any]] = []
            for memory in base.get("memories", []):
                metadata = memory.get("metadata", {}) if isinstance(memory, dict) else {}
                ts = metadata.get("timestamp_unix")
                if isinstance(ts, (int, float)) and start_ts <= float(ts) <= end_ts:
                    selected.append(memory)

            selected.sort(
                key=lambda item: (item.get("metadata", {}).get("timestamp_unix", 0) if isinstance(item, dict) else 0),
                reverse=True,
            )
            return {"success": True, "memories": selected[:limit]}
        except Exception as exc:
            logger.error("❌ [Mem0] get_memories_in_range failed: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    def search_memories_with_recency(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        recency_weight: float = 0.3,
    ) -> Dict[str, Any]:
        try:
            base = self.search_memories(query=query, user_id=user_id, limit=50)
            if not base.get("success"):
                return base

            items = base.get("memories", [])
            now_ts = _now_unix()
            scored: List[Tuple[float, Dict[str, Any]]] = []

            for memory in items:
                if not isinstance(memory, dict):
                    scored.append((0.0, memory))
                    continue

                score = float(memory.get("score", 0.0))
                metadata = memory.get("metadata", {})
                ts = metadata.get("timestamp_unix")
                if isinstance(ts, (int, float)):
                    age = max(0.0, now_ts - float(ts))
                    tau = 7 * 24 * 3600  # ~7 days
                    decay = math.exp(-(age / tau))
                else:
                    decay = 0.0

                combined = (1 - recency_weight) * score + recency_weight * decay
                scored.append((combined, memory))

            scored.sort(key=lambda pair: pair[0], reverse=True)
            return {"success": True, "memories": [m for _, m in scored[:limit]]}
        except Exception as exc:
            logger.error("❌ [Mem0] search_memories_with_recency failed: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    def update_memory(self, memory_id: str, data: Any, user_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            payload = data if isinstance(data, dict) else {"data": data}
            if user_id:
                payload.setdefault("user_id", user_id)
            result = self.client.update(memory_id=memory_id, data=payload)
            return {"success": True, "result": result}
        except Exception as exc:
            logger.error("❌ [Mem0] update_memory failed: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            result = self.client.delete(memory_id=memory_id)
            return {"success": True, "result": result}
        except Exception as exc:
            logger.error("❌ [Mem0] delete_memory failed: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    def delete_all_memories(self, user_id: str) -> Dict[str, Any]:
        try:
            base = self.get_memories(user_id=user_id, limit=1000)
            if not base.get("success"):
                return base

            memories = base.get("memories", [])
            if not memories:
                return {"success": True, "message": "No memories found to delete", "deleted_count": 0}

            deleted = 0
            for memory in memories:
                memory_id = memory.get("id") if isinstance(memory, dict) else None
                if not memory_id:
                    continue
                if self.delete_memory(memory_id).get("success"):
                    deleted += 1

            return {"success": True, "message": f"Successfully deleted {deleted} memories", "deleted_count": deleted}
        except Exception as exc:
            logger.error("❌ [Mem0] delete_all_memories failed: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Enrichment pipeline
    # ------------------------------------------------------------------
    def ingest_chat_message(
        self,
        user_id: str,
        message_text: str,
        message_id: Optional[str] = None,
        lang_hint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        message_id = message_id or str(uuid.uuid4())
        ts_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        extraction = self.call_llm(message_text)
        if lang_hint:
            extraction["lang"] = lang_hint

        base_metadata = {
            "lang": extraction.get("lang", "auto"),
            "source": "chat_message",
            "timestamp_iso": ts_iso,
            "timestamp_unix": _safe_unix_time(),
            "summary_short": extraction.get("summary_short") or message_text[:200],
            "summary_long": extraction.get("summary_long") or extraction.get("summary_short") or message_text,
            "facts": extraction.get("facts", []),
            "message_id": message_id,
        }
        if metadata:
            base_metadata.update(metadata)

        # Merge any known facts into existing records before storing the raw message
        for fact in list(base_metadata.get("facts", [])):
            attribute = fact.get("attribute")
            if not attribute:
                continue
            try:
                existing = self.find_existing_fact(user_id, attribute)
            except Exception as exc:  # pragma: no cover - defensive search
                logger.debug("ℹ️ [Mem0] fact search failed for attribute '%s': %s", attribute, exc)
                continue

            if not existing or not isinstance(existing, dict) or not existing.get("id"):
                continue

            merged_metadata = self._merge_fact_metadata(existing.get("metadata", {}), fact, ts_iso)
            merged_metadata["summary_short"] = base_metadata["summary_short"]
            merged_metadata["summary_long"] = base_metadata["summary_long"]
            merged_metadata["timestamp_iso"] = ts_iso
            merged_metadata["timestamp_unix"] = base_metadata["timestamp_unix"]

            try:
                self.client.update(
                    memory_id=existing["id"],
                    data={
                        "content": existing.get("content") or message_text,
                        "metadata": merged_metadata,
                    },
                )
                logger.debug("🔁 [Mem0] Merged fact '%s' into memory %s", attribute, existing["id"])
            except Exception as exc:
                logger.warning(
                    "⚠️ [Mem0] Failed to merge fact '%s' into memory %s: %s",
                    attribute,
                    existing.get("id"),
                    exc,
                )

        result = self.client.add(
            messages=message_text,
            user_id=user_id,
            metadata=base_metadata,
            infer=False,
        )
        logger.debug("🧠 [Mem0] Created new memory entry for user=%s", user_id)
        return result

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def call_llm(self, message: str) -> Dict[str, Any]:
        cleaned = (message or "").strip()
        if not cleaned:
            return {"summary_short": "", "summary_long": "", "facts": [], "lang": "auto"}

        if not self.openai_client:
            return {
                "summary_short": cleaned[:200],
                "summary_long": cleaned,
                "facts": [],
                "lang": "auto",
            }

        system_prompt = (
            "Tu es un module d'extraction de mémoire clinique pour un coach santé. "
            "Analyse le message utilisateur et produis un JSON STRICT avec ce schéma :\n"
            "{\n"
            "  'summary_short': string // 1-2 phrases maximum\n"
            "  'summary_long': string\n"
            "  'facts': [\n"
            "     {\n"
            "       'attribute': string\n"
            "       'value': string\n"
            "       'subject': 'user'\n"
            "       'category': string\n"
            "       'polarity': 'asserted' | 'revoked'\n"
            "       'confidence': number\n"
            "       'evidence_span': string\n"
            "     }\n"
            "   ]\n"
            "}\n"
            "summary_short <= 2 phrases. Réponds uniquement avec du JSON valide."
        )

        try:
            chat = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": cleaned},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            output_text = (chat.choices[0].message.content or "").strip()
            parsed = self._coerce_json(output_text)
        except Exception as exc:  # pragma: no cover - API failures
            logger.warning("⚠️ [Mem0] LLM extraction failed, using fallback: %s", exc)
            parsed = None

        if not isinstance(parsed, dict):
            return {
                "summary_short": cleaned[:200],
                "summary_long": cleaned,
                "facts": [],
                "lang": "auto",
            }

        facts = parsed.get("facts")
        normalized_facts: List[Dict[str, Any]] = []
        if isinstance(facts, list):
            for fact in facts:
                if not isinstance(fact, dict):
                    continue
                normalized_facts.append(
                    {
                        "attribute": fact.get("attribute"),
                        "value": fact.get("value"),
                        "subject": fact.get("subject", "user"),
                        "category": fact.get("category"),
                        "polarity": fact.get("polarity", "asserted"),
                        "confidence": _safe_float(fact.get("confidence", 0.7)),
                        "evidence_span": fact.get("evidence_span"),
                    }
                )

        return {
            "summary_short": parsed.get("summary_short", cleaned[:200]),
            "summary_long": parsed.get("summary_long", parsed.get("summary_short", cleaned)),
            "facts": normalized_facts,
            "lang": parsed.get("lang", "auto"),
        }

    def find_existing_fact(self, user_id: str, attribute: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self.client.search(query=attribute, user_id=user_id, limit=20)
            items = raw.get("results", []) if isinstance(raw, dict) else raw
        except Exception as exc:
            logger.debug("ℹ️ [Mem0] find_existing_fact search failed: %s", exc)
            return None

        for item in items or []:
            normalized = self._normalize_mem0_item(item)
            facts = normalized.get("metadata", {}).get("facts") or []
            for fact in facts:
                if (
                    isinstance(fact, dict)
                    and fact.get("attribute") == attribute
                    and fact.get("subject", "user") == "user"
                ):
                    return normalized
        return None

    def _merge_fact_metadata(
        self,
        existing_metadata: Dict[str, Any],
        new_fact: Dict[str, Any],
        timestamp_iso: str,
    ) -> Dict[str, Any]:
        metadata = dict(existing_metadata or {})
        facts = list(metadata.get("facts") or [])
        merged = False

        for fact in facts:
            if (
                isinstance(fact, dict)
                and fact.get("attribute") == new_fact.get("attribute")
                and fact.get("subject", "user") == "user"
            ):
                fact["last_seen"] = timestamp_iso
                polarity = new_fact.get("polarity")
                if polarity == "revoked":
                    fact["polarity"] = "revoked"
                elif polarity == "asserted":
                    fact["polarity"] = "asserted"
                    confidence = _safe_float(new_fact.get("confidence", 0.7))
                    fact["confidence"] = min(0.99, max(fact.get("confidence", 0.5), confidence))
                if new_fact.get("evidence_span"):
                    fact["evidence_span"] = new_fact["evidence_span"]
                if new_fact.get("value"):
                    fact["value"] = new_fact["value"]
                if new_fact.get("category"):
                    fact["category"] = new_fact["category"]
                merged = True
                break

        if not merged:
            new_copy = dict(new_fact)
            new_copy.setdefault("subject", "user")
            new_copy.setdefault("confidence", 0.7)
            new_copy.setdefault("polarity", "asserted")
            new_copy["last_seen"] = timestamp_iso
            facts.append(new_copy)

        metadata["facts"] = facts
        return metadata

    def _format_memory_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(item.get("metadata") or {})
        summary_short = (metadata.get("summary_short") or "").strip()
        summary_long = metadata.get("summary_long")
        raw_facts = metadata.get("facts") or []

        filtered_facts: List[Dict[str, Any]] = []
        for fact in raw_facts:
            if not isinstance(fact, dict):
                continue
            if str(fact.get("subject", "user")) != "user":
                continue
            polarity = str(fact.get("polarity", "asserted")).lower()
            if polarity != "asserted":
                continue
            confidence = _safe_float(fact.get("confidence", 0.0))
            if confidence < self.fact_confidence_threshold:
                continue
            filtered_facts.append(dict(fact, confidence=confidence))

        display_lines: List[str] = []
        if summary_short:
            display_lines.append(summary_short)

        if filtered_facts:
            display_lines.append("Faits confirmés :")
            for fact in filtered_facts:
                attribute = fact.get("attribute")
                value = fact.get("value")
                description = fact.get("description") or fact.get("summary")
                if attribute and value:
                    line = f"- {attribute}: {value}"
                elif attribute:
                    line = f"- {attribute}"
                elif value:
                    line = f"- {value}"
                elif description:
                    line = f"- {description}"
                else:
                    line = "- (fact)"
                display_lines.append(line)

        if not display_lines:
            fallback = item.get("content") or summary_long or metadata.get("raw")
            if fallback:
                display_lines.append(str(fallback))

        formatted: Dict[str, Any] = {
            "id": item.get("id"),
            "score": item.get("score"),
            "content": item.get("content"),
            "metadata": metadata,
            "summary_short": summary_short,
            "summary_long": summary_long,
            "facts": filtered_facts,
            "memory": "\n".join(display_lines).strip(),
        }

        for key, value in item.items():
            if key not in formatted:
                formatted[key] = value

        return formatted

    def _normalize_mem0_item(self, item: Any) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {"memory": str(item)}

        normalized = dict(item)
        if "payload" in normalized and isinstance(normalized["payload"], dict):
            payload = normalized["payload"]
            normalized.update(payload)
            normalized.pop("payload", None)

        if "content" not in normalized and "memory" in normalized:
            normalized["content"] = normalized["memory"]

        metadata = normalized.get("metadata") or normalized.get("meta")
        if metadata and not isinstance(metadata, dict):
            try:
                normalized["metadata"] = json.loads(metadata)
            except Exception:
                normalized["metadata"] = {}
        elif metadata is None:
            normalized["metadata"] = {}

        return normalized

    def _coerce_json(self, text: str) -> Optional[Dict[str, Any]]:
        candidate = (text or "").strip()
        if not candidate:
            return None

        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if candidate.lower().startswith("json"):
                candidate = candidate[4:]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = candidate[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
            return None


_mem0_client: Optional[Mem0Client] = None


def get_mem0_client() -> Optional[Mem0Client]:
    global _mem0_client
    if _mem0_client is None:
        if not os.getenv("MEM0_API_KEY"):
            return None
        try:
            _mem0_client = Mem0Client()
        except Exception as exc:
            logger.warning("⚠️ Failed to initialise Mem0 client: %s", exc)
            return None
    return _mem0_client


mem0_client: Optional[Mem0Client] = None


def _now_unix() -> float:
    return datetime.now(timezone.utc).timestamp()


def _iso_to_unix(iso_ts: str) -> float:
    try:
        if iso_ts.endswith("Z"):
            iso_ts = iso_ts.replace("Z", "+00:00")
        return datetime.fromisoformat(iso_ts).timestamp()
    except Exception:
        return 0.0


def _safe_unix_time() -> float:
    try:
        return _now_unix()
    except Exception:
        return time.time()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default
