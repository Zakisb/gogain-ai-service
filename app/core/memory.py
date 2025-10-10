import os
import time
from typing import List, Optional, Dict, Any, Tuple
from mem0 import Memory
from qdrant_client import QdrantClient
from datetime import datetime, timezone
import math

class Mem0Client:
    def __init__(self):
        # Ensure OpenAI API key is available for Mem0's embedding model
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for Mem0")

        # Get Qdrant server URL from environment or default
        qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")

        # Test Qdrant connection at startup to catch issues early
        try:
            print(f"🔌 Testing Qdrant connection at {qdrant_url} ...")
            qdrant_client = QdrantClient(url=qdrant_url, timeout=5.0)
            qdrant_client.get_collections()
            print(f"✅ Qdrant connection successful.")
        except Exception as e:
            print(f"❌ Qdrant connection test failed: {e}")
            raise RuntimeError(f"Failed to connect to Qdrant at {qdrant_url}: {e}")

        # Configure Mem0 to use Qdrant server
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0",
                    "url": qdrant_url
                    # timeout parameter not supported by Mem0's Qdrant config validation
                }
            }
        }

        self.client = Memory.from_config(config)

    def _build_temporal_metadata(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        hour = now.hour
        # Coarse time-of-day bucket in French
        if 5 <= hour < 12:
            tod = "matin"
        elif 12 <= hour < 17:
            tod = "après-midi"
        elif 17 <= hour < 22:
            tod = "soir"
        else:
            tod = "nuit"

        french_weekdays = [
            "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"
        ]
        meta = {
            "timestamp_iso": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "timestamp_unix": _safe_unix_time(),
            "day_of_week": french_weekdays[now.weekday()],
            "time_of_day": tod,
        }
        return meta

    def add_memory(self, message: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            # Auto-enrich metadata with temporal information (non-breaking)
            base_meta = metadata.copy() if isinstance(metadata, dict) else {}
            base_meta.update(self._build_temporal_metadata())
            result = self.client.add(messages=message, user_id=user_id, metadata=base_meta)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_memories(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        try:
            result = self.client.get_all(user_id=user_id, limit=limit)
            memories = result.get('results', []) if isinstance(result, dict) else result
            return {"success": True, "memories": memories}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_memories(self, query: str, user_id: str, limit: int = 5) -> Dict[str, Any]:
        try:
            result = self.client.search(query=query, user_id=user_id, limit=limit)
            memories = result.get('results', []) if isinstance(result, dict) else result
            return {"success": True, "memories": memories}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # --- Temporal-aware helpers (non-breaking, opt-in) ---
    def get_recent_memories(self, user_id: str, limit: int = 10, within_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Return memories sorted by most recent timestamp in metadata.

        within_seconds: if provided, filter to memories within this age from now.
        """
        try:
            base = self.get_memories(user_id=user_id, limit=1000)
            if not base.get("success"):
                return base
            items = base.get("memories", [])
            now = _now_unix()

            def parse_ts(m: Dict[str, Any]) -> float:
                meta = (m or {}).get("metadata", {}) if isinstance(m, dict) else {}
                ts = meta.get("timestamp_unix")
                if isinstance(ts, (int, float)):
                    return float(ts)
                # Fallback: absent timestamp -> 0
                return 0.0

            filtered = []
            for m in items:
                ts = parse_ts(m)
                if within_seconds is not None and ts > 0:
                    if (now - ts) <= within_seconds:
                        filtered.append((ts, m))
                else:
                    filtered.append((ts, m))

            filtered.sort(key=lambda x: x[0], reverse=True)
            top = [m for _, m in filtered[:limit]]
            return {"success": True, "memories": top}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_memories_in_range(self, user_id: str, start_iso: str, end_iso: str, limit: int = 100) -> Dict[str, Any]:
        """Return memories with timestamp_iso between start_iso and end_iso (inclusive)."""
        try:
            start = _iso_to_unix(start_iso)
            end = _iso_to_unix(end_iso)
            base = self.get_memories(user_id=user_id, limit=1000)
            if not base.get("success"):
                return base
            items = base.get("memories", [])
            selected = []
            for m in items:
                meta = (m or {}).get("metadata", {}) if isinstance(m, dict) else {}
                ts = meta.get("timestamp_unix")
                if isinstance(ts, (int, float)) and start <= float(ts) <= end:
                    selected.append(m)
            # Keep chronological order (most recent first)
            selected.sort(key=lambda mm: (mm.get("metadata", {}).get("timestamp_unix", 0) if isinstance(mm, dict) else 0), reverse=True)
            return {"success": True, "memories": selected[:limit]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_memories_with_recency(self, query: str, user_id: str, limit: int = 5, recency_weight: float = 0.3) -> Dict[str, Any]:
        """Search with a simple recency boost to ranking. Returns top results.

        recency_weight in [0,1]: 0 = no recency boost, 1 = heavy recency.
        """
        try:
            base = self.search_memories(query=query, user_id=user_id, limit=50)
            if not base.get("success"):
                return base
            items = base.get("memories", [])
            now = _now_unix()

            scored: List[Tuple[float, Any]] = []
            for m in items:
                if isinstance(m, dict):
                    score = float(m.get("score", 0.0))
                    meta = m.get("metadata", {})
                    ts = meta.get("timestamp_unix")
                    if isinstance(ts, (int, float)):
                        age = max(0.0, now - float(ts))
                        # Half-life ~ 7 days decay as an example; avoid hardcoding keyword logic
                        # decay = exp(-age / tau); choose tau = 7 days in seconds
                        tau = 7 * 24 * 3600
                        decay = math.exp(-(age / tau))
                    else:
                        decay = 0.0
                    combined = (1 - recency_weight) * score + recency_weight * decay
                    scored.append((combined, m))
                else:
                    scored.append((0.0, m))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = [m for _, m in scored[:limit]]
            return {"success": True, "memories": top}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_memory(self, memory_id: str, data: str, user_id: str = None) -> Dict[str, Any]:
        try:
            result = self.client.update(memory_id=memory_id, data=data)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_memory(self, memory_id: str, user_id: str = None) -> Dict[str, Any]:
        try:
            result = self.client.delete(memory_id=memory_id)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_all_memories(self, user_id: str) -> Dict[str, Any]:
        try:
            memories_result = self.get_memories(user_id, limit=1000)
            if not memories_result["success"]:
                return memories_result

            memories = memories_result["memories"]
            if not memories:
                return {"success": True, "message": "No memories found to delete", "deleted_count": 0}

            deleted_count = 0
            errors = []

            for memory in memories:
                memory_id = memory.get("id")
                if memory_id:
                    delete_result = self.delete_memory(memory_id)
                    if delete_result["success"]:
                        deleted_count += 1
                    else:
                        errors.append(f"Failed to delete memory {memory_id}: {delete_result['error']}")

            if errors:
                return {
                    "success": False,
                    "error": f"Deleted {deleted_count} memories but had {len(errors)} errors: {'; '.join(errors)}"
                }
            else:
                return {"success": True, "message": f"Successfully deleted {deleted_count} memories", "deleted_count": deleted_count}

        except Exception as e:
            return {"success": False, "error": str(e)}


# Lazy-loaded global client to prevent early locks
_mem0_client = None

def get_mem0_client():
    global _mem0_client
    if _mem0_client is None:
        # Only initialize if MEM0_API_KEY is set
        mem0_key = os.getenv("MEM0_API_KEY")
        if not mem0_key:
            return None
        _mem0_client = Mem0Client()
    return _mem0_client

# Export lazy-loaded client (will be None if no MEM0_API_KEY)
# Skip initialization at module load to prevent blocking startup
mem0_client = None
print("💡 Mem0 client will be initialized on first use to prevent blocking startup")


# ---- Internal time helpers ----
def _now_unix() -> float:
    return datetime.now(timezone.utc).timestamp()


def _iso_to_unix(iso_ts: str) -> float:
    try:
        # Accept 'Z' suffix
        if iso_ts.endswith('Z'):
            iso_ts = iso_ts.replace('Z', '+00:00')
        return datetime.fromisoformat(iso_ts).timestamp()
    except Exception:
        return 0.0


def _safe_unix_time() -> float:
    try:
        return _now_unix()
    except Exception:
        return 0.0
