"""Enhanced AI-powered assessment tools with memory integration and clinical reasoning."""

from typing import List, Dict, Any, Optional
import logging
import json
import re
import os
import atexit
import concurrent.futures
import threading
import contextvars
from enum import Enum
from datetime import datetime
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from qdrant_client import QdrantClient
from app.core.config import QDRANT_URL
from app.core.embeddings import configure_global_embedding
from app.core.redis_client import get_sync_redis
# TEMPORARILY DISABLED: from app.core.memory import get_mem0_client
# Removed import of deleted tools.py file

logger = logging.getLogger(__name__)

# Context variable for user_id (supports async/await - safe for concurrent requests)
_user_id_context: contextvars.ContextVar[str] = contextvars.ContextVar('user_id', default='unknown_user')


def set_user_context(user_id: str):
    """Set user_id in context variable (called by API layer).

    This allows tools to access user_id programmatically without the AI knowing about it.
    Uses contextvars for async safety - each request gets isolated context.
    """
    _user_id_context.set(user_id)


def get_user_context() -> str:
    """Return the current user_id from contextvars.

    Used by tools that should not trust the LLM to pass user_id.
    """
    try:
        return _user_id_context.get()
    except Exception:
        return "unknown_user"


# ============================================================================
# MEM0 CONTEXT MANAGEMENT (Redis-backed, async retrieval)
# ============================================================================

def _mem0_context_key(user_id: str, body_region: str) -> str:
    """Build Redis key for storing Mem0 memory context."""
    safe_user = _sanitize_for_key(user_id)
    safe_region = _sanitize_for_key(body_region)
    return f"pain:mem0:{safe_user}:{safe_region}"


def store_mem0_context(user_id: str, body_region: str, context: Dict[str, Any], ttl: int = 1800) -> None:
    """Store Mem0 memory context in Redis for pain agent consumption.

    Args:
        user_id: User identifier
        body_region: Body region being assessed
        context: Memory context dict with categorized memories
        ttl: Time-to-live in seconds (default 30 minutes)
    """
    try:
        redis_client = get_sync_redis()
        key = _mem0_context_key(user_id, body_region)

        redis_client.setex(
            key,
            ttl,
            json.dumps(context, ensure_ascii=False)
        )

        logger.info(f"✅ [Mem0] Stored memory context for user={user_id} region={body_region}")

    except Exception as exc:
        logger.error(f"❌ [Mem0] Failed to store context for user={user_id}: {exc}")


def get_mem0_context(user_id: str, body_region: str) -> Optional[Dict[str, Any]]:
    """Retrieve Mem0 memory context from Redis.

    Returns None if not available yet (async retrieval still in progress).
    """
    try:
        redis_client = get_sync_redis()
        key = _mem0_context_key(user_id, body_region)

        data = redis_client.get(key)
        if not data:
            logger.debug(f"🔍 [Mem0] No context found for user={user_id} region={body_region}")
            return None

        context = json.loads(data)
        logger.info(f"✅ [Mem0] Retrieved memory context for user={user_id} region={body_region}")
        return context

    except Exception as exc:
        logger.error(f"❌ [Mem0] Failed to retrieve context for user={user_id}: {exc}")
        return None

# Global instances for startup optimization
_COMPILED_PATTERNS = None

# Simple in-memory cache for fetched assessment questions across turns
# Keyed by (context_key, body_region) -> set of question strings
_CACHED_PAIN_QUESTIONS: dict[tuple[str, str], set[str]] = {}

# Cache for RAG knowledge results - only fetch once during INITIAL phase
# Keyed by (context_key, body_region) -> Dict containing knowledge_results and metadata
_CACHED_RAG_KNOWLEDGE: dict[tuple[str, str], Dict[str, Any]] = {}

_QUERY_QUEUE_TTL_SECONDS = int(os.getenv("PAIN_RAG_QUERY_TTL_SECONDS", "900"))
_QUERY_QUEUE_MAX_LENGTH = int(os.getenv("PAIN_RAG_QUERY_MAX_LENGTH", "50"))

# Questions queue configuration
_QUESTIONS_QUEUE_TTL_SECONDS = int(os.getenv("PAIN_QUESTIONS_QUEUE_TTL_SECONDS", "1800"))  # 30 minutes
_QUESTIONS_QUEUE_MAX_LENGTH = int(os.getenv("PAIN_QUESTIONS_QUEUE_MAX_LENGTH", "10"))  # Max 10 questions in queue

_RAG_MAX_WORKERS = max(1, int(os.getenv("PAIN_RAG_MAX_WORKERS", "4")))
_RAG_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=_RAG_MAX_WORKERS)


def _shutdown_rag_executor():
    try:
        _RAG_THREAD_POOL.shutdown(wait=False)
    except Exception:
        pass


atexit.register(_shutdown_rag_executor)

def _sanitize_for_key(value: str) -> str:
    """Sanitize identifiers for safe Redis key usage."""
    if not value:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _query_queue_key(user_id: str, body_region: str) -> str:
    """Build Redis key for storing agent-generated RAG queries."""
    safe_user = _sanitize_for_key(user_id)
    safe_region = _sanitize_for_key(body_region)
    return f"pain:rag_queries:{safe_user}:{safe_region}"


def _questions_queue_key(user_id: str, body_region: str) -> str:
    """Build Redis key for storing extracted questions queue for pain assessment."""
    safe_user = _sanitize_for_key(user_id)
    safe_region = _sanitize_for_key(body_region)
    return f"pain:questions_queue:{safe_user}:{safe_region}"


def _store_queries_in_queue(
    user_id: str,
    body_region: str,
    queries: List[str],
    hyde_snippet: Optional[str],
    conversation_history: str,
    source: str = "agent",
) -> None:
    """Persist agent-generated queries in Redis for reuse/background processing."""
    if not queries:
        return

    snapshot = conversation_history[:500] if conversation_history else ""
    timestamp = datetime.utcnow().isoformat()

    try:
        redis_client = get_sync_redis()
    except Exception as exc:  # pragma: no cover - Redis optional at runtime
        logger.warning(
            "RAG query cache unavailable (sync Redis client failure): %s", exc
        )
        return

    entries: List[str] = []
    for query in queries:
        if not isinstance(query, str):
            continue
        cleaned = query.strip()
        if not cleaned:
            continue
        entry = {
            "query": cleaned,
            "body_region": body_region,
            "hyde_snippet": (hyde_snippet or "").strip(),
            "conversation_context_preview": snapshot,
            "created_at": timestamp,
            "source": source,
        }
        entries.append(json.dumps(entry, ensure_ascii=False))

    if not entries:
        return

    key = _query_queue_key(user_id or "unknown_user", body_region)
    try:
        pipeline = redis_client.pipeline()
        pipeline.rpush(key, *entries)
        pipeline.ltrim(key, -_QUERY_QUEUE_MAX_LENGTH, -1)
        pipeline.expire(key, _QUERY_QUEUE_TTL_SECONDS)
        pipeline.execute()
        logger.debug(
            "Stored %d RAG queries in Redis queue for user=%s region=%s",
            len(entries),
            user_id,
            body_region,
        )
    except Exception as exc:  # pragma: no cover - Redis errors handled gracefully
        logger.warning(
            "Failed to persist RAG queries for user=%s region=%s: %s",
            user_id,
            body_region,
            exc,
        )


# ============================================================================
# QUESTIONS QUEUE MANAGEMENT (Redis-backed, multi-user support)
# ============================================================================

def add_questions_to_queue(
    user_id: str,
    body_region: str,
    questions: List[str],
    conversation_history: str = "",
) -> None:
    """Add extracted questions to user's queue (Redis-backed, simple list storage).

    Args:
        user_id: User identifier
        body_region: Body region being assessed
        questions: List of question strings to add to queue
        conversation_history: Recent conversation context for deduplication
    """
    if not questions:
        logger.info(f"🚫 [Queue] No questions to add for user={user_id} region={body_region}")
        return

    try:
        redis_client = get_sync_redis()
    except Exception as exc:
        logger.warning(f"⚠️ [Queue] Redis unavailable for user={user_id}: {exc}")
        return

    queue_key = _questions_queue_key(user_id, body_region)

    # Get existing queue to check for duplicates (only check queue, not conversation history)
    try:
        existing_questions = redis_client.lrange(queue_key, 0, -1)
        existing_set = set()
        for item in existing_questions:
            try:
                # Support both old format (dict) and new format (string)
                if item.startswith('{'):
                    data = json.loads(item)
                    existing_set.add(data.get("question", "").strip().lower())
                else:
                    existing_set.add(item.strip().lower())
            except:
                existing_set.add(item.strip().lower())
    except Exception:
        existing_set = set()

    # Filter out duplicates (only within queue and current batch, no conversation history check)
    new_questions = []
    added_count = 0
    skipped_count = 0

    for question in questions:
        if not isinstance(question, str):
            continue

        question_text = question.strip()
        if not question_text:
            continue

        question_lower = question_text.lower()

        # Skip if already in queue
        if question_lower in existing_set:
            logger.debug(f"🔁 [Queue] Skipping duplicate in queue: {question_text[:80]}")
            skipped_count += 1
            continue

        new_questions.append(question_text)  # Just store the string
        existing_set.add(question_lower)  # Prevent duplicates within this batch
        added_count += 1

    if not new_questions:
        logger.info(f"🚫 [Queue] All {len(questions)} questions were duplicates for user={user_id}")
        return

    # Replace entire queue with new questions (no append, always fresh)
    try:
        pipeline = redis_client.pipeline()
        pipeline.delete(queue_key)  # Clear old queue
        pipeline.rpush(queue_key, *new_questions)  # Add new questions
        pipeline.expire(queue_key, _QUESTIONS_QUEUE_TTL_SECONDS)
        pipeline.execute()

        logger.info(
            f"✅ [Queue] Replaced queue with {added_count} questions (skipped {skipped_count} duplicates) "
            f"for user={user_id} region={body_region}"
        )

        # Log current queue state
        _log_queue_state(user_id, body_region)

    except Exception as exc:
        logger.error(f"❌ [Queue] Failed to add questions for user={user_id}: {exc}")


def get_next_question_from_queue(
    user_id: str,
    body_region: str,
) -> Optional[str]:
    """Get next question from queue without removing it (peek).

    Returns None if queue is empty.
    """
    try:
        redis_client = get_sync_redis()
    except Exception as exc:
        logger.warning(f"⚠️ [Queue] Redis unavailable for user={user_id}: {exc}")
        return None

    queue_key = _questions_queue_key(user_id, body_region)

    try:
        # Get first unasked question
        items = redis_client.lrange(queue_key, 0, -1)

        for idx, item in enumerate(items):
            try:
                data = json.loads(item)
                if not data.get("asked", False):
                    question = data.get("question", "")
                    logger.info(
                        f"👀 [Queue] Next question for user={user_id}: '{question[:80]}...'"
                    )
                    return question
            except Exception:
                continue

        logger.info(f"🏁 [Queue] No more questions for user={user_id} region={body_region}")
        return None

    except Exception as exc:
        logger.error(f"❌ [Queue] Failed to get next question for user={user_id}: {exc}")
        return None


def mark_question_as_asked(
    user_id: str,
    body_region: str,
    question: str,
) -> bool:
    """Mark a question as asked in the current queue.

    Returns:
        True if the question was found and marked, False otherwise.
    """
    try:
        redis_client = get_sync_redis()
    except Exception as exc:
        logger.warning(f"⚠️ [Queue] Redis unavailable for user={user_id}: {exc}")
        return False

    queue_key = _questions_queue_key(user_id, body_region)

    try:
        # Find and mark the question as asked
        items = redis_client.lrange(queue_key, 0, -1)
        question_lower = question.strip().lower()

        for idx, item in enumerate(items):
            try:
                data = json.loads(item)
                if data.get("question", "").strip().lower() == question_lower:
                    # Mark as asked
                    data["asked"] = True
                    data["asked_at"] = datetime.utcnow().isoformat()
                    redis_client.lset(queue_key, idx, json.dumps(data, ensure_ascii=False))

                    logger.info(
                        f"✓ [Queue] Marked as asked for user={user_id}: "
                        f"'{question[:80]}...'"
                    )

                    # Log current queue state
                    _log_queue_state(user_id, body_region)

                    return True
            except Exception:
                continue

        logger.warning(f"⚠️ [Queue] Question not found in queue for user={user_id}")
        return False

    except Exception as exc:
        logger.error(f"❌ [Queue] Failed to mark question as asked for user={user_id}: {exc}")
        return False


def clear_questions_queue(user_id: str, body_region: str) -> None:
    """Clear questions queue (called when reloading queue with new RAG results)."""
    try:
        redis_client = get_sync_redis()
        queue_key = _questions_queue_key(user_id, body_region)

        redis_client.delete(queue_key)

        logger.info(f"🧹 [Queue] Cleared queue for user={user_id} region={body_region}")

    except Exception as exc:
        logger.error(f"❌ [Queue] Failed to clear queue for user={user_id}: {exc}")


def _log_queue_state(user_id: str, body_region: str) -> None:
    """Log current state of questions queue for debugging."""
    try:
        redis_client = get_sync_redis()
        queue_key = _questions_queue_key(user_id, body_region)

        items = redis_client.lrange(queue_key, 0, -1)
        total = len(items)

        logger.info(
            f"📊 [Queue State] user={user_id} region={body_region} | "
            f"Total questions: {total}"
        )

        # Log each question in queue (simple string format)
        for idx, item in enumerate(items, 1):
            try:
                # Support both old dict format and new string format
                if isinstance(item, bytes):
                    item = item.decode('utf-8')

                if item.startswith('{'):
                    data = json.loads(item)
                    question = data.get("question", "")
                else:
                    question = item

                logger.info(f"  [{idx}] {question[:100]}")
            except Exception:
                logger.info(f"  [{idx}] {item[:100]}")
                continue

    except Exception as exc:
        logger.debug(f"Failed to log queue state for user={user_id}: {exc}")


def get_full_queue_state(user_id: str, body_region: str) -> Dict[str, Any]:
    """Get full queue state for debugging/monitoring (simplified)."""
    try:
        redis_client = get_sync_redis()
        queue_key = _questions_queue_key(user_id, body_region)

        items = redis_client.lrange(queue_key, 0, -1)

        questions = []
        for item in items:
            try:
                if isinstance(item, bytes):
                    item = item.decode('utf-8')

                # Support both old dict format and new string format
                if item.startswith('{'):
                    data = json.loads(item)
                    questions.append(data.get("question", ""))
                else:
                    questions.append(item)
            except:
                questions.append(str(item))

        return {
            "user_id": user_id,
            "body_region": body_region,
            "total_questions": len(questions),
            "questions": questions,  # Simple list of strings
        }
    except Exception as exc:
        logger.error(f"Failed to get queue state for user={user_id}: {exc}")
        return {
            "user_id": user_id,
            "body_region": body_region,
            "error": str(exc),
            "questions": [],
        }


# Using v2 collection with enhanced metadata (questions, red flags, patient statements)
PRIMARY_HIERARCHICAL_COLLECTION = "pain_hierarchical_chunks_v2"
LEGACY_HIERARCHICAL_COLLECTION = "pain_hierarchical_chunks"  # Fallback if needed

class ClinicalPriority(Enum):
    """Clinical priority levels for knowledge retrieval."""
    RED_FLAG = "red_flag"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EnhancedRAGAssessmentRetriever:
    """Optimized RAG system for fast AI-powered pain assessments with memory integration."""

    def __init__(self):
        """Initialize the optimized assessment retriever."""
        configure_global_embedding()
        self.qdrant_client = QdrantClient(url=QDRANT_URL, timeout=30.0)
        self._indices_cache = {}
        # TEMPORARILY DISABLED: Mem0 client initialization for performance testing
        # self.memory_client = get_mem0_client()
        self.memory_client = None  # Disabled for performance testing
        # Pre-cache commonly used collections
        self._precache_collections()
        print("⚡ OPTIMIZED RAG: EnhancedRAGAssessmentRetriever initialized (Mem0 disabled)")

    def _precache_collections(self):
        """Pre-cache available collections with fallback logic."""
        # Use only the working hierarchical collection
        try:
            self._get_index(PRIMARY_HIERARCHICAL_COLLECTION)
            print(f"✅ Cached collection: {PRIMARY_HIERARCHICAL_COLLECTION}")
            return
        except Exception as e:
            print(f"❌ Could not cache {PRIMARY_HIERARCHICAL_COLLECTION}: {e}")
            print("⚠️ WARNING: No collections available - RAG will be disabled")
    
    def _get_index(self, collection_name: str) -> VectorStoreIndex:
        """Get or create vector index for specified Qdrant collection."""
        if collection_name not in self._indices_cache:
            # pain_hierarchical_chunks uses single unnamed vector (no vector_name needed)
            # pain_hierarchical_chunks_v2 uses named vector "text-dense"
            # Only specify vector_name for v2 collection
            if collection_name == "pain_hierarchical_chunks_v2":
                vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=collection_name,
                    vector_name="text-dense",  # Named vector for v2
                )
            else:
                # Single unnamed vector (v1 and other collections)
                vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=collection_name,
                    # No vector_name parameter for single unnamed vectors
                )
            index = VectorStoreIndex.from_vector_store(vector_store)
            self._indices_cache[collection_name] = index

        return self._indices_cache[collection_name]
    
    def _get_user_memory_context(self, user_id: str, body_region: str) -> Dict[str, Any]:
        """Retrieve user's pain history using semantic search (LLM verification disabled for efficiency)."""
        print(f"🧠 MEMORY DISABLED: Skipping memory retrieval for {user_id} regarding {body_region}")

        # TEMPORARILY DISABLED: Mem0 memory retrieval for performance testing
        # try:
        #     # Step 1: Use targeted semantic queries (FR) with temporal preference
        #     targeted_memories = self._retrieve_with_semantic_queries_fr(user_id, body_region)

        #     # Step 2: Use semantic search suggestions directly
        #     categorized_context = self._fallback_categorization(targeted_memories)

        #     print(f"🧠 MEMORY SEMANTIC: Found {sum(len(v) for v in categorized_context.values())} categorized memories (FR, temporal-aware)")
        #     return categorized_context

        # except Exception as e:
        #     logger.error(f"Error in semantic memory context retrieval: {e}")

        # Return empty memory context (same structure as error fallback)
        return {k: [] for k in ["previous_episodes", "treatments_tried", "medications", "triggers", "improvements", "red_flags_history"]}


    def _fallback_categorization(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback categorization using suggested categories from semantic search."""
        print("🔄 FALLBACK: Using suggested categories from semantic queries")

        final_context = {
            "previous_episodes": [],
            "treatments_tried": [],
            "medications": [],
            "triggers": [],
            "improvements": [],
            "red_flags_history": []
        }

        for memory in memories:
            suggested_category = memory.get("suggested_category", "")
            memory_text = memory.get("text", "")

            if suggested_category in final_context and memory_text not in final_context[suggested_category]:
                final_context[suggested_category].append(memory_text)

        return final_context



    def _retrieve_queries_parallel(self,
                                  queries: List[Dict[str, Any]],
                                  body_region: str,
                                  top_k: int,
                                  user_symptoms: str,
                                  conversation_context: str) -> List[Dict[str, Any]]:
        """Execute RAG queries in parallel using ThreadPoolExecutor for 2-3x speed improvement."""
        print(f"⚡ PARALLEL RAG: Executing {len(queries)} queries in parallel")

        all_results = []
        if not queries:
            return all_results

        # Use shared ThreadPoolExecutor to run queries concurrently
        executor = _RAG_THREAD_POOL
        future_to_query = {}
        for query_info in queries:
            future = executor.submit(
                self._retrieve_with_query,
                query_info["query"],
                body_region,
                query_info["priority"],
                top_k // len(queries) + 1,
                query_info.get("search_for_questions", False),
                user_symptoms,
                conversation_context
            )
            future_to_query[future] = query_info

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_query):
            query_info = future_to_query[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"⚡ PARALLEL RAG: Completed query '{query_info['query'][:50]}...' with {len(results)} results")
            except Exception as e:
                logger.error(f"Error in parallel query execution: {e}")
                continue

        print(f"⚡ PARALLEL RAG: Completed all {len(queries)} queries, total results: {len(all_results)}")
        return all_results

    def _chunk_contains_questions(self, text: str) -> bool:
        """Check if a text chunk contains questions (French guillemets or question marks)."""
        global _COMPILED_PATTERNS
        if _COMPILED_PATTERNS:
            # Use pre-compiled patterns for better performance
            french_questions = _COMPILED_PATTERNS['french_questions'].findall(text)
            question_mark_lines = _COMPILED_PATTERNS['question_marks'].findall(text)
        else:
            # Fallback to on-demand compilation
            import re
            french_questions = re.findall(r'«([^»]+)»', text)
            question_mark_lines = re.findall(r'^.+\?$', text, re.MULTILINE)
        return len(french_questions) > 0 or len(question_mark_lines) > 0

    def _count_questions_in_chunk(self, text: str) -> int:
        """Count the number of questions in a text chunk (French guillemets or question marks)."""
        global _COMPILED_PATTERNS
        if _COMPILED_PATTERNS:
            # Use pre-compiled patterns for better performance
            french_questions = _COMPILED_PATTERNS['french_questions'].findall(text)
            question_mark_lines = _COMPILED_PATTERNS['question_marks'].findall(text)
        else:
            # Fallback to on-demand compilation
            import re
            french_questions = re.findall(r'«([^»]+)»', text)
            question_mark_lines = re.findall(r'^.+\?$', text, re.MULTILINE)
        return len(french_questions) + len(question_mark_lines)

    def _extract_questions_from_text(self, text: str) -> List[str]:
        """Extract candidate question strings from a text chunk for debugging.

        Uses French guillemets (« ») containing text that ends with '?' as heuristics.
        Never raises; returns [] on any error.
        """
        try:
            global _COMPILED_PATTERNS
            if _COMPILED_PATTERNS:
                french_q = _COMPILED_PATTERNS['french_questions'].findall(text)
                question_lines = _COMPILED_PATTERNS['question_marks'].findall(text)
            else:
                import re
                french_q = re.findall(r'«([^»]+)»', text)
                question_lines = re.findall(r'^.+\?$', text, re.MULTILINE)

            seen = set()
            out: List[str] = []

            # Only extract guillemet content that ends with ?
            for q in french_q:
                qn = q.strip()
                if qn and qn.endswith('?') and qn not in seen:
                    seen.add(qn)
                    out.append(qn)

            # Add question-mark lines
            for q in question_lines:
                qn = q.strip()
                if qn and qn not in seen:
                    seen.add(qn)
                    out.append(qn)

            return out
        except Exception:
            return []
    
    def _retrieve_with_query(self,
                           query: str,
                           body_region: str,
                           priority: ClinicalPriority,
                           top_k: int,
                           search_for_questions: bool = False,
                           user_symptoms: str = "",
                           conversation_context: str = "") -> List[Dict[str, Any]]:
        """Pure Semantic RAG: Retrieve from hierarchical chunks with AI-driven understanding."""
        try:
            # Use the primary (stable) hierarchical collection
            index = self._get_index(PRIMARY_HIERARCHICAL_COLLECTION)
            collection_name = PRIMARY_HIERARCHICAL_COLLECTION

            # Build enriched semantic query
            enriched_query = self._build_semantic_query(
                query, body_region, user_symptoms, conversation_context
            )

            # Retrieve more candidates for AI filtering
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k * 3
            )

            nodes = retriever.retrieve(enriched_query)

            # Apply semantic filtering and ranking
            filtered_results = self._apply_semantic_filtering(
                nodes, body_region, search_for_questions, user_symptoms, priority
            )

            # Fallback: if strict region filter yields nothing, relax region constraint
            if not filtered_results and (body_region or "").strip().lower() != "general":
                try:
                    print("⚠️ No candidates after region filter; retrying without region constraint")
                    filtered_results = self._apply_semantic_filtering(
                        nodes, "general", search_for_questions, user_symptoms, priority
                    )
                except Exception as _e:
                    print(f"⚠️ Fallback filtering failed: {_e}")

            print(f"🧠 SEMANTIC RAG: Filtered {len(filtered_results)} from {len(nodes)} hierarchical nodes")

            # Verbose dump of filtered results for this single query
            try:
                print(f"🔎 QUERY DETAILS: '{enriched_query}'")
                for i, r in enumerate(filtered_results, 1):
                    meta = r.get("metadata", {})
                    print(
                        "📄 RESULT [" + str(i) + f"] score={r.get('score'):.4f} "
                        f"priority={r.get('priority')} namespace={r.get('namespace')} "
                        f"content_type={r.get('content_type')} has_questions={meta.get('contains_questions')} "
                        f"question_count={meta.get('question_count')}"
                    )
                    print("—— TEXT:")
                    print(r.get("text", "") or "<no text>")
                    print("—— METADATA:")
                    print(meta)
            except Exception as _e:
                print(f"⚠️ Failed verbose dump for query: {_e}")

            return filtered_results[:top_k]

        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
            return []

    def _build_semantic_query(self,
                            base_query: str,
                            body_region: str,
                            user_symptoms: str = "",
                            conversation_context: str = "") -> str:
        """Build enriched query for better semantic matching."""

        # Start with body region and base query
        enriched = f"{body_region} pain {base_query}"

        # Add user's actual symptoms for semantic matching
        if user_symptoms:
            enriched += f" patient symptoms: {user_symptoms}"

        # Add conversation context for continuity
        if conversation_context:
            context_snippet = conversation_context[-200:]  # Last 200 chars
            enriched += f" context: {context_snippet}"

        return enriched

    def _normalize_body_region_french(self, body_region: str) -> str:
        """Normalize French body region input to match Qdrant metadata (English keys).

        The embeddings use English region labels set at embed time:
        back, knee, shoulder, neck, foot_ankle, hip, general.
        """
        fr_to_en = {
            # shoulder
            "épaule": "shoulder",
            "epaule": "shoulder",
            "épaules": "shoulder",
            "epaules": "shoulder",
            # knee
            "genou": "knee",
            "genoux": "knee",
            # back
            "dos": "back",
            "bas du dos": "back",
            "lombaires": "back",
            # neck
            "cou": "neck",
            # hip
            "hanche": "hip",
            "hanches": "hip",
            # ankle/foot
            "cheville": "foot_ankle",
            "pied": "foot_ankle",
            "cheville/pied": "foot_ankle",
            # already English
            "back": "back",
            "knee": "knee",
            "shoulder": "shoulder",
            "neck": "neck",
            "hip": "hip",
            "foot_ankle": "foot_ankle",
            "general": "general",
        }
        key = (body_region or "").strip().lower()
        normalized = fr_to_en.get(key, body_region)
        print(f"🔤 Body region normalization: '{body_region}' → '{normalized}'")
        return normalized

    def _apply_semantic_filtering(self,
                                nodes: List[Any],
                                body_region: str,
                                search_for_questions: bool,
                                user_symptoms: str,
                                priority: ClinicalPriority) -> List[Dict[str, Any]]:
        """Apply semantic filtering based on hierarchical metadata."""

        results = []

        # Normalize French input to match Qdrant French metadata
        normalized_region = self._normalize_body_region_french(body_region)

        for node in nodes:
            metadata = node.metadata or {}

            # Filter by body region using normalized French values
            node_body_region = metadata.get("body_region", "general")
            if normalized_region != "general" and node_body_region not in [normalized_region, "general"]:
                continue

            # Calculate semantic relevance using all factors
            semantic_priority = self._calculate_semantic_priority(
                metadata, search_for_questions, node.score, user_symptoms, priority
            )

            # Assess content richness
            content_richness = self._assess_content_richness(node.text, metadata)

            result = {
                "text": node.text,
                "score": node.score,
                "metadata": metadata,
                "query": "",  # Set by caller
                "priority": semantic_priority,
                "namespace": node_body_region,
                "content_type": self._classify_semantic_content_type(metadata),
                "content_richness": content_richness,
                "has_questions": metadata.get("contains_questions", False),
                "question_count": metadata.get("question_count", 0)
            }
            results.append(result)

        # Sort by semantic relevance: priority, content richness, similarity score
        priority_weights = {"critical": 5, "high": 4, "medium": 3, "low": 2, "very_low": 1}

        results.sort(
            key=lambda x: (
                priority_weights.get(x["priority"], 1),
                x["content_richness"],
                x["score"]
            ),
            reverse=True
        )

        return results

    def _calculate_semantic_priority(self,
                                   metadata: Dict[str, Any],
                                   search_for_questions: bool,
                                   similarity_score: float,
                                   user_symptoms: str = "",
                                   base_priority: ClinicalPriority = ClinicalPriority.MEDIUM) -> str:
        """Calculate priority based on semantic factors."""

        # High similarity always gets priority
        if similarity_score > 0.8:
            return "high"
        elif similarity_score > 0.7:
            return "medium"

        # Question-rich content when searching for questions
        if search_for_questions and metadata.get("contains_questions"):
            question_count = metadata.get("question_count", 0)
            if question_count >= 3:
                return "high"
            elif question_count >= 1:
                return "medium"

        # Use base priority for fallback
        if similarity_score > 0.6:
            return base_priority.value.lower() if hasattr(base_priority, 'value') else "medium"
        else:
            return "low"

    def _classify_semantic_content_type(self, metadata: Dict[str, Any]) -> str:
        """Classify content type based on semantic metadata."""
        if metadata.get("contains_questions"):
            return "questions"
        elif metadata.get("has_structured_content"):
            return "structured"
        else:
            return "general"

    def _assess_content_richness(self, text: str, metadata: Dict[str, Any]) -> float:
        """Assess content richness using semantic indicators."""
        richness = 0.5  # Base score

        # Length indicates comprehensive content
        if len(text.split()) > 100:
            richness += 0.2

        # Questions indicate assessment content
        if metadata.get("contains_questions"):
            richness += 0.3

        # Structured content (headers) indicates organized information
        if metadata.get("has_structured_content"):
            richness += 0.1

        return min(richness, 1.0)
    
    def _consolidate_results(self,
                           all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimized: Fast consolidation and ranking with minimal logging."""
        # Remove duplicates based on text similarity
        consolidated = []
        seen_texts = set()

        # Sort by priority first, then by score
        priority_order = {
            ClinicalPriority.RED_FLAG.value: 4,
            ClinicalPriority.HIGH.value: 3,
            ClinicalPriority.MEDIUM.value: 2,
            ClinicalPriority.LOW.value: 1
        }

        sorted_results = sorted(
            all_results,
            key=lambda x: (priority_order.get(x["priority"], 0), x["score"]),
            reverse=True
        )

        for result in sorted_results:
            # Simple deduplication check
            text_start = result["text"][:200].lower()
            if text_start not in seen_texts:
                seen_texts.add(text_start)
                consolidated.append(result)

                # Limit total results for performance
                if len(consolidated) >= 10:
                    break

        # Detailed logging of consolidated results
        print(f"⚡ Consolidated {len(consolidated)} results from {len(all_results)} total")
        for idx, item in enumerate(consolidated, 1):
            try:
                meta = item.get("metadata", {})
                print(
                    "📚 CONSOLIDATED [" + str(idx) + f"] score={item.get('score'):.4f} "
                    f"priority={item.get('priority')} namespace={item.get('namespace')} "
                    f"content_type={item.get('content_type')} has_questions={meta.get('contains_questions')} "
                    f"question_count={meta.get('question_count')}"
                )
                print("—— TEXT:")
                print(item.get("text", "") or "<no text>")
                print("—— METADATA:")
                print(meta)
            except Exception as _e:
                print(f"⚠️ Failed to print consolidated item #{idx}: {_e}")
        return consolidated


# Initialize global enhanced retriever
_enhanced_retriever = None

def get_enhanced_assessment_retriever() -> EnhancedRAGAssessmentRetriever:
    """Get global enhanced assessment retriever instance."""
    global _enhanced_retriever
    if _enhanced_retriever is None:
        _enhanced_retriever = EnhancedRAGAssessmentRetriever()
    return _enhanced_retriever


def create_pain_summary_generator_tool() -> FunctionTool:
    """Create flexible tool for generating pain summary adapted to available information."""

    def generate_adaptive_pain_summary(body_region: str,
                                     conversation_summary: str,
                                     clinical_insights: str = "") -> str:
        """Generate flexible pain summary adapted to the specific case and available information.

        Args:
            body_region: Body region assessed
            conversation_summary: Summary of conversation and information gathered
            clinical_insights: Any clinical insights or analysis (optional)

        Returns:
            Adaptive JSON summary structured for the specific case
        """
        # Get user_id programmatically from context variable (set by API layer)
        # Safe for concurrent async requests - each request has isolated context
        user_id = _user_id_context.get()
        safe_user_id = re.sub(r"[^A-Za-z0-9._-]", "_", user_id or "unknown_user")

        print(f"🔥 OPENAI AGENT TOOL: generate_adaptive_pain_summary called")
        logger.warning(f"🔥 OPENAI AGENT TOOL: generate_adaptive_pain_summary called for {body_region}, user: {user_id}")
        print(f"🏁 ASSESSMENT COMPLETE - Summary tailored to available information")

        # Start with essential completion signal and metadata
        timestamp = datetime.now()
        summary = {
            "assessment_complete": True,  # CRITICAL: Always signals completion
            "assessment_date": timestamp.strftime("%Y-%m-%d %H:%M"),
            "body_region": body_region,
            "user_id": user_id
        }

        # Parse available information flexibly
        try:
            if conversation_summary:
                # Add conversation insights
                summary["conversation_insights"] = conversation_summary

            if clinical_insights:
                insights_data = json.loads(clinical_insights) if isinstance(clinical_insights, str) else clinical_insights
                summary["clinical_analysis"] = insights_data

            # Add any additional structured information the AI has gathered
            # This allows for flexible, case-specific summary content

        except json.JSONDecodeError:
            # Handle non-JSON input gracefully
            if conversation_summary:
                summary["assessment_notes"] = conversation_summary
            if clinical_insights:
                summary["clinical_notes"] = clinical_insights

        # Let AI determine what other information to include based on the case
        summary["ai_completion_note"] = "Assessment completed based on available information quality and clinical judgment"

        # Log pain summary to file in raw format
        try:
            # Create pain_summaries directory if it doesn't exist
            log_dir = "pain_summaries"
            os.makedirs(log_dir, exist_ok=True)

            # Generate filename with timestamp and user_id
            filename = (
                "pain_summary_"
                f"{safe_user_id}_"
                f"{body_region.replace(' ', '_').replace('/', '_')}_"
                f"{timestamp.strftime('%Y%m%d_%H%M%S')}"
                ".json"
            )
            log_path = os.path.join(log_dir, filename)

            # Write raw JSON to file
            with open(log_path, 'w') as f:
                f.write(json.dumps(summary, indent=2))

            print(f"📁 Pain summary logged to: {log_path}")

        except Exception as e:
            print(f"⚠️ Failed to log pain summary to file: {e}")
            # Continue anyway - logging failure shouldn't break the assessment

        print(f"✅ Generated adaptive summary with {len(summary)} main components")

        # Clear ALL cached assessment state after summary generation (assessment concluded)
        try:
            _CACHED_PAIN_QUESTIONS.clear()
            _CACHED_RAG_KNOWLEDGE.clear()
            logger.info("🧹 CACHE: Cleared all assessment caches (questions + RAG knowledge) after summary generation")
            print("🧹 ASSESSMENT COMPLETE: Cleared all temporary assessment state")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clear assessment caches: {e}")
            print(f"⚠️ Failed to clear cached state: {e}")

        # Flush pain session state (background RAG buffers, etc.) for this user
        try:
            # Import session state management from chat API
            import sys
            import asyncio

            # Dynamically import to avoid circular dependency
            if 'app.api.chat' in sys.modules:
                chat_module = sys.modules['app.api.chat']
                clear_pain_session_state = getattr(chat_module, 'clear_pain_session_state', None)

                async def _cleanup_async():
                    if clear_pain_session_state:
                        await clear_pain_session_state(user_id, body_region)

                # Run async function in event loop if available
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(_cleanup_async())
                    else:
                        loop.run_until_complete(_cleanup_async())
                    logger.info(f"🧹 Flushed pain session state for user={user_id} region={body_region}")
                except RuntimeError:
                    # If no event loop, try to create one
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(_cleanup_async())
                    new_loop.close()
                    logger.info(f"🧹 Flushed pain session state for user={user_id} region={body_region} (new loop)")
            else:
                logger.debug("app.api.chat module not loaded, skipping session state flush")
        except Exception as e:
            logger.warning(f"⚠️ Failed to flush pain session state: {e}")
            # Don't fail the summary generation if session flush fails

        # Clear the user's question queue so next assessment starts fresh
        try:
            clear_questions_queue(user_id, body_region)
            logger.info("🧹 Queue cleared after summary for user=%s region=%s", user_id, body_region)
        except Exception as e:
            logger.warning("⚠️ Failed to clear questions queue after summary for user=%s region=%s: %s", user_id, body_region, e)

        return json.dumps(summary, indent=2)

    return FunctionTool.from_defaults(
        fn=generate_adaptive_pain_summary,
        name="generate_adaptive_pain_summary",
        description="Generate flexible, adaptive pain summary that adjusts structure and content based on available information - no rigid templates."
    )


def create_pain_session_exit_tool() -> FunctionTool:
    """Create tool allowing the agent to exit pain assessment mode explicitly."""

    def exit_pain_assessment(body_region: str = "general") -> str:
        user_id = _user_id_context.get()
        safe_user = _sanitize_for_key(user_id or "unknown_user")

        print(f"🚪 Pain session exit tool invoked for user={safe_user}, region={body_region}")
        logger.info(
            "🚪 Pain session exit requested by agent for user=%s region=%s",
            user_id,
            body_region,
        )

        # Clear Redis-backed queue so next assessment starts clean
        try:
            clear_questions_queue(user_id, body_region)
            logger.info(
                "🧹 Queue cleared via exit tool for user=%s region=%s",
                user_id,
                body_region,
            )
        except Exception as exc:
            logger.warning(
                "⚠️ Failed to clear queue during exit tool for user=%s region=%s: %s",
                user_id,
                body_region,
                exc,
            )

        # End the pain session flags and clear in-memory session state
        try:
            import sys
            import asyncio

            if 'app.api.chat' in sys.modules:
                chat_module = sys.modules['app.api.chat']
                end_session = getattr(chat_module, 'end_pain_agent_session', None)
                clear_state = getattr(chat_module, 'clear_pain_session_state', None)

                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = None

                async def _cleanup():
                    if end_session:
                        await end_session(user_id)
                        logger.info("🧹 Redis pain session flag cleared for user=%s", user_id)
                    if clear_state:
                        await clear_state(user_id, body_region)
                        logger.info("🧹 In-memory pain session state cleared for user=%s region=%s", user_id, body_region)

                if loop and loop.is_running():
                    asyncio.create_task(_cleanup())
                else:
                    if loop is None:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(_cleanup())
                        finally:
                            loop.close()
                            asyncio.set_event_loop(None)
                    else:
                        loop.run_until_complete(_cleanup())
            else:
                logger.debug(
                    "app.api.chat not in sys.modules; skipping session flag cleanup for user=%s",
                    user_id,
                )
        except Exception as exc:
            logger.warning(
                "⚠️ Failed to flush pain session state during exit tool for user=%s region=%s: %s",
                user_id,
                body_region,
                exc,
            )

        return (
            "Pain assessment session closed. Route the user back to the central assistant and acknowledge the context switch."
        )

    return FunctionTool.from_defaults(
        fn=exit_pain_assessment,
        name="exit_pain_assessment",
        description=(
            "Terminate the current pain assessment session immediately, clear queued questions, "
            "and hand control back to the central router when the user changes topic."
        ),
    )

def initialize_global_assessment_tools():
    """Initialize global assessment tools at startup for performance optimization."""
    global _enhanced_retriever, _COMPILED_PATTERNS

    try:
        # Pre-create retriever with all collections using existing pattern
        print("🔥 Pre-initializing EnhancedRAGAssessmentRetriever...")
        _enhanced_retriever = EnhancedRAGAssessmentRetriever()

        # Pre-compile regex patterns for question detection
        import re
        _COMPILED_PATTERNS = {
            'french_questions': re.compile(r'«([^»]+)»'),
            'question_marks': re.compile(r'^.+\?$', re.MULTILINE)
        }

        print("✅ Global assessment tools initialized at startup")
    except Exception as e:
        logger.error(f"❌ Failed to initialize global assessment tools: {e}")
        # Don't fail startup - tools will initialize on first use
        _enhanced_retriever = None
        _COMPILED_PATTERNS = None

def create_llm_driven_rag_tool() -> FunctionTool:
    """Create LLM-driven RAG tool that generates queries dynamically based on conversation context."""

    def _collect_all_questions(consolidated_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect all extracted questions with priority and context.

        Args:
            consolidated_results: List of consolidated RAG results with metadata

        Returns:
            List of question dicts sorted by priority and relevance
        """
        all_questions: List[Dict[str, Any]] = []

        for doc in consolidated_results:
            metadata = doc.get("metadata", {})
            extracted = metadata.get("assessment_questions")
            if extracted is None:
                extracted = metadata.get("extracted_questions", [])

            for q in extracted:
                q_stripped = q.strip()
                if q_stripped:  # Skip empty questions
                    all_questions.append({
                        "question": q_stripped,
                        "priority": doc.get("priority", "medium"),
                        "relevance_score": doc.get("score", 0.0),
                        "source_context": doc.get("text", "")[:300] + "..."  # First 300 chars of source
                    })

        # Sort by priority (high first) then by relevance score
        priority_order = {"critical": 5, "high": 4, "medium": 3, "low": 2, "very_low": 1}
        all_questions.sort(
            key=lambda x: (
                priority_order.get(x["priority"], 0),
                x["relevance_score"]
            ),
            reverse=True
        )

        # Return top 30 questions for better coverage of technique/training questions
        return all_questions[:30]

    def _collect_examples(
        consolidated_results: List[Dict[str, Any]],
        metadata_key: str,
        limit: int = 20,
    ) -> List[str]:
        """Collect unique examples (e.g., red flags) from consolidated metadata."""

        examples: List[str] = []
        seen: set[str] = set()
        for doc in consolidated_results:
            metadata = doc.get("metadata", {})
            for item in metadata.get(metadata_key, []) or []:
                normalized = item.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                examples.append(normalized)
                if len(examples) >= limit:
                    return examples
        return examples

    def retrieve_with_llm_queries(
        body_region: str,
        conversation_history: str,
        rag_queries: Optional[List[str]] = None,
        hyde_snippet: Optional[str] = None,
    ) -> str:
        """Launch background RAG to refresh the pain questions queue (non-blocking).

        Args:
            body_region: Body region being assessed (e.g., "dos", "epaule")
            conversation_history: Summary of conversation so far
            rag_queries: Optional list of semantic search queries generated by the agent
                        If None, uses predefined initial queries for comprehensive coverage
            hyde_snippet: Optional HyDE-style snippet provided by the agent

        Returns:
            JSON status string indicating that background retrieval was scheduled.
        """
        print("🔥 LLM-DRIVEN RAG TOOL: retrieve_with_llm_queries called (background launch)")
        logger.warning(
            "🔥 LLM-DRIVEN RAG TOOL: retrieve_with_llm_queries called for %s", body_region
        )

        candidate_queries: List[str] = []
        if isinstance(rag_queries, str):
            candidate_queries = [rag_queries]
        elif isinstance(rag_queries, list):
            candidate_queries = rag_queries
        elif rag_queries is None:
            # Use predefined initial queries for comprehensive coverage
            candidate_queries = [
                f"questions d'évaluation {body_region}",
                f"examen clinique {body_region}",
                f"diagnostic différentiel {body_region}",
                f"red flags {body_region}",
                f"tests cliniques {body_region}",
            ]
        else:
            raise ValueError(
                "The 'rag_queries' parameter must be a list of agent-generated search strings."
            )

        normalized_queries: List[str] = []
        for query in candidate_queries:
            if not isinstance(query, str):
                continue
            cleaned = query.strip()
            if cleaned:
                normalized_queries.append(cleaned)

        query_source = "initial_automatic" if rag_queries is None else "agent"
        if not normalized_queries:
            fallback_query = (conversation_history or "").strip()
            if fallback_query:
                fallback_query = fallback_query[:400]
                logger.warning(
                    "No agent-generated queries supplied; using conversation history fallback for retrieval."
                )
                normalized_queries = [fallback_query]
                query_source = "fallback_conversation"
            else:
                raise ValueError(
                    "No valid agent-generated queries supplied to retrieve_with_llm_queries."
                )

        hyde_text = hyde_snippet.strip() if isinstance(hyde_snippet, str) else ""
        user_id = _user_id_context.get()

        retrieval_inputs = normalized_queries[:]
        if hyde_text:
            retrieval_inputs.append(hyde_text)

        print(f"📝 Agent supplied {len(normalized_queries)} queries:")
        for idx, query in enumerate(normalized_queries, 1):
            print(f"   Query {idx}: {query}")
        if hyde_text:
            print(f"📝 Agent supplied HyDE snippet: {hyde_text}")

        def _background_worker():
            """Execute RAG in a separate thread and refresh the queue."""
            try:
                # Store body_region in Redis synchronously (background thread context)
                try:
                    redis_client = get_sync_redis()
                    key = f"pain:body_region:{user_id}"
                    ttl = int(os.getenv("PAIN_SESSION_TTL_SECONDS", "10800"))
                    redis_client.set(key, body_region, ex=ttl)
                    logger.info(f"🔍 Stored body_region '{body_region}' for pain session {user_id}")
                except Exception as e:
                    logger.warning(f"Failed to store body_region in session: {e}")

                # Persist submitted queries for observability - INSIDE background thread
                _store_queries_in_queue(
                    user_id=user_id,
                    body_region=body_region,
                    queries=normalized_queries,
                    hyde_snippet=hyde_text,
                    conversation_history=conversation_history,
                    source=query_source,
                )

                logger.info(
                    "🚀 [retrieve_with_llm_queries] Background RAG started for user=%s region=%s",
                    user_id,
                    body_region,
                )

                retriever = get_enhanced_assessment_retriever()

                all_results: List[Dict[str, Any]] = []
                if not retrieval_inputs:
                    logger.warning("⚠️ No retrieval inputs available; background RAG skipped.")
                    return

                executor = _RAG_THREAD_POOL
                future_to_query = {}
                for query in retrieval_inputs[:5]:
                    future = executor.submit(
                        retriever._retrieve_with_query,
                        query,
                        body_region,
                        ClinicalPriority.HIGH,
                        top_k=3,
                        search_for_questions=True,
                        user_symptoms=conversation_history[:500],
                        conversation_context=conversation_history[:200],
                    )
                    future_to_query[future] = query

                for future in concurrent.futures.as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        print(
                            f"⚡ LLM-RAG (background): Query '{query[:50]}...' returned {len(results)} results"
                        )
                    except Exception as exc:  # pragma: no cover - logging path
                        logger.error(f"❌ Error executing agent-supplied query: {exc}")

                if not all_results:
                    logger.info(
                        "ℹ️ [retrieve_with_llm_queries] No results retrieved for user=%s region=%s",
                        user_id,
                        body_region,
                    )
                    return

                consolidated_results = retriever._consolidate_results(all_results)
                extracted_questions = _collect_all_questions(consolidated_results)
                question_texts = [q["question"] for q in extracted_questions if q.get("question")]

                logger.info(
                    "📚 [retrieve_with_llm_queries] Consolidated %d results, extracted %d questions",
                    len(consolidated_results),
                    len(question_texts),
                )

                if question_texts:
                    add_questions_to_queue(
                        user_id=user_id,
                        body_region=body_region,
                        questions=question_texts,
                        conversation_history=conversation_history,
                    )
                    print(
                        f"✅ Background RAG refreshed queue with {len(question_texts)} questions for user={user_id}"
                    )
                else:
                    logger.info(
                        "ℹ️ [retrieve_with_llm_queries] No new questions extracted for user=%s region=%s",
                        user_id,
                        body_region,
                    )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    "❌ [retrieve_with_llm_queries] Background RAG failed for user=%s region=%s: %s",
                    user_id,
                    body_region,
                    exc,
                    exc_info=True,
                )

        thread_name = f"rag-bg-{_sanitize_for_key(user_id)}-{_sanitize_for_key(body_region)}"
        threading.Thread(target=_background_worker, name=thread_name, daemon=True).start()

        status_payload = {
            "status": "background_rag_launched",
            "body_region": body_region,
            "queries_submitted": normalized_queries[:5],
            "hyde_snippet_included": bool(hyde_text),
            "message": (
                "RAG refresh running in background. QUESTIONS will appear in AVAILABLE_QUESTIONS once ready."
            ),
        }
        return json.dumps(status_payload, ensure_ascii=False, indent=2)

    return FunctionTool.from_defaults(
        fn=retrieve_with_llm_queries,
        name="retrieve_with_llm_queries",
        description=(
            "Launch a background RAG refresh using agent-generated semantic queries "
            "and an optional HyDE snippet tailored to the current conversation."
        )
    )


# Queue management tools removed - queue contents are now fed directly into agent context
# The agent uses questions as inspiration without needing tools to access the queue


def launch_initial_pain_rag(user_id: str, body_region: str, initial_message: str) -> None:
    """Launch initial RAG automatically when pain assessment starts.

    Programmatic wrapper that calls retrieve_with_llm_queries with predefined queries.
    Non-blocking - runs in background thread.

    Args:
        user_id: User identifier (set in context variable before calling)
        body_region: Body region being assessed
        initial_message: User's initial pain message
    """
    logger.info(f"🚀 AUTO-RAG: Launching initial retrieval for user={user_id} region={body_region}")

    # Set user context for the background thread
    set_user_context(user_id)

    # Create tool instance and call it programmatically (rag_queries=None triggers predefined queries)
    tool = create_llm_driven_rag_tool()
    tool.fn(
        body_region=body_region,
        conversation_history=initial_message,
        rag_queries=None,  # None = use predefined initial queries
        hyde_snippet=None
    )


def launch_mem0_retrieval(user_id: str, body_region: str, initial_message: str) -> None:
    """Launch Mem0 memory retrieval asynchronously when pain assessment starts.

    Non-blocking - runs in background thread and stores results in Redis.
    Pain agent will pick up results when available.

    Args:
        user_id: User identifier
        body_region: Body region being assessed
        initial_message: User's initial pain description
    """
    logger.info(f"🧠 AUTO-MEM0: Launching memory retrieval for user={user_id} region={body_region}")

    def _background_mem0_worker():
        """Execute Mem0 retrieval in background thread and store in Redis."""
        try:
            # Import Mem0 client (lazy-loaded)
            from app.core.memory import get_mem0_client

            mem0_client = get_mem0_client()
            if not mem0_client:
                logger.warning(f"⚠️ [Mem0] Client not available (MEM0_API_KEY not set), skipping retrieval")
                return

            logger.info(f"🔍 [Mem0] Retrieving memories for user={user_id} region={body_region}")

            # Use semantic queries mapped to clinical categories
            # Each query's results are semantically relevant to that category
            semantic_queries = {
                "previous_episodes": f"épisodes antérieurs douleur {body_region} historique",
                "treatments_tried": f"traitements essayés thérapies {body_region}",
                "medications": f"médicaments pris anti-inflammatoires antalgiques {body_region}",
                "triggers": f"facteurs déclencheurs aggravation douleur {body_region}",
                "improvements": f"améliorations progrès soulagement {body_region}",
                "red_flags_history": f"signaux d'alarme red flags urgence {body_region}"
            }

            categorized_context = {
                "previous_episodes": [],
                "treatments_tried": [],
                "medications": [],
                "triggers": [],
                "improvements": [],
                "red_flags_history": []
            }

            # Use semantic search for each category - trust Mem0's semantic ranking
            for category, query in semantic_queries.items():
                try:
                    result = mem0_client.search_memories_with_recency(
                        query=query,
                        user_id=user_id,
                        limit=5,
                        recency_weight=0.3
                    )

                    if result.get("success"):
                        memories = result.get("memories", [])

                        for memory in memories:
                            if not isinstance(memory, dict):
                                continue

                            memory_text = memory.get("memory", "") or memory.get("text", "")
                            if memory_text and memory_text not in categorized_context[category]:
                                categorized_context[category].append(memory_text)

                        logger.debug(f"🔍 [Mem0] Category '{category}' found {len(categorized_context[category])} memories")

                except Exception as e:
                    logger.error(f"❌ [Mem0] Category query '{category}' failed: {e}")
                    continue

            # Store in Redis for pain agent (even if empty - signals retrieval is done)
            store_mem0_context(user_id, body_region, categorized_context)

            total_memories = sum(len(v) for v in categorized_context.values())
            logger.info(
                f"✅ [Mem0] Stored {total_memories} categorized memories for user={user_id} region={body_region}"
            )

            print(f"🧠 Mem0 retrieval complete: {total_memories} memories stored in Redis")

        except Exception as exc:
            logger.error(
                f"❌ [Mem0] Background retrieval failed for user={user_id} region={body_region}: {exc}",
                exc_info=True
            )
            # Store empty context on failure so agent doesn't wait forever
            store_mem0_context(user_id, body_region, {
                "previous_episodes": [],
                "treatments_tried": [],
                "medications": [],
                "triggers": [],
                "improvements": [],
                "red_flags_history": []
            })

    # Launch background thread (daemon so it doesn't block shutdown)
    thread_name = f"mem0-bg-{_sanitize_for_key(user_id)}-{_sanitize_for_key(body_region)}"
    threading.Thread(target=_background_mem0_worker, name=thread_name, daemon=True).start()

    logger.info(f"🚀 [Mem0] Background retrieval thread started for user={user_id}")


def get_enhanced_pain_assessment_tools() -> List[FunctionTool]:
    """Get tools for AI-powered pain assessment with LLM-driven RAG.

    Queue contents are fed directly into agent context for inspiration.
    Agent manages questions autonomously based on conversation history.
    """
    from app.tools.date_tool import get_date_tool

    tools = [
        create_llm_driven_rag_tool(),
        create_pain_summary_generator_tool(),
        create_pain_session_exit_tool(),
        get_date_tool()
    ]

    return tools
