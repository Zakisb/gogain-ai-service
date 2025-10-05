"""Enhanced AI-powered assessment tools with memory integration and clinical reasoning."""

from typing import List, Optional, Dict, Any
import logging
import json
import re
import asyncio
import concurrent.futures
import contextvars
from enum import Enum
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from qdrant_client import QdrantClient
from app.core.config import QDRANT_URL
from app.core.embeddings import configure_global_embedding
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

# Global instances for startup optimization
_COMPILED_PATTERNS = None

# Simple in-memory cache for fetched assessment questions across turns
# Keyed by (context_key, body_region) -> set of question strings
_CACHED_PAIN_QUESTIONS: dict[tuple[str, str], set[str]] = {}

# Cache for RAG knowledge results - only fetch once during INITIAL phase
# Keyed by (context_key, body_region) -> Dict containing knowledge_results and metadata
_CACHED_RAG_KNOWLEDGE: dict[tuple[str, str], Dict[str, Any]] = {}

class AssessmentPhase(Enum):
    """Assessment phases for different retrieval strategies."""
    INITIAL = "initial"
    DETAILED = "detailed"  
    DIFFERENTIAL = "differential"
    SUMMARY = "summary"


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
        # Try hierarchical first, fallback to any available collection
        fallback_collections = ["pain_hierarchical_chunks", "questions", "test_questions"]

        for collection in fallback_collections:
            try:
                self._get_index(collection)
                print(f"✓ Cached collection: {collection}")
                return  # Successfully cached at least one
            except Exception as e:
                print(f"⚠ Could not cache {collection}: {e}")
                continue

        print("⚠️ WARNING: No collections available - RAG will be disabled")
    
    def _get_index(self, collection_name: str) -> VectorStoreIndex:
        """Get or create vector index for specified Qdrant collection."""
        if collection_name not in self._indices_cache:
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
                vector_name=None  # Use default/unnamed vector instead of "text-dense"
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

    def _retrieve_with_semantic_queries(self, user_id: str, body_region: str) -> List[Dict[str, Any]]:
        """Optimized: Use minimal targeted semantic queries to retrieve relevant memories."""

        # Simplified semantic queries - only essential categories
        semantic_queries = {
            "previous_episodes": f"previous {body_region} pain episodes history",
            "treatments_tried": f"treatments tried for {body_region} pain",
            "improvements": f"what helped {body_region} pain",
            "red_flags_history": f"concerning symptoms {body_region} pain"
        }

        all_memories = []

        try:
            for category, query in semantic_queries.items():
                result = self.memory_client.search_memories(
                    query=query,
                    user_id=user_id,
                    limit=3  # Reduced limit for faster queries
                )

                if result.get("success") and result.get("memories"):
                    memories_data = result["memories"]
                    if isinstance(memories_data, dict) and "results" in memories_data:
                        memories_list = memories_data["results"]
                    else:
                        memories_list = memories_data

                    if memories_list and isinstance(memories_list, list):
                        for memory in memories_list:
                            if isinstance(memory, dict):
                                memory_text = memory.get("memory", "")
                            elif isinstance(memory, str):
                                memory_text = memory
                            else:
                                memory_text = str(memory)

                            if memory_text:
                                all_memories.append({
                                    "text": memory_text,
                                    "suggested_category": category,
                                    "query_used": query,
                                    "search_score": memory.get("score", 0) if isinstance(memory, dict) else 0
                                })

            return all_memories

        except Exception as e:
            logger.error(f"Error in semantic memory retrieval: {e}")
            return []

    # --- Commented FR/temporal-aware version to use when Mem0 is enabled ---
    # def _retrieve_with_semantic_queries_fr(self, user_id: str, body_region: str) -> List[Dict[str, Any]]:
    #     """Utiliser des requêtes sémantiques ciblées EN FRANÇAIS et privilégier les souvenirs récents.
    #     Aucune logique de mots-clés – uniquement des requêtes sémantiques en français.
    #     """
    #     if not self.memory_client:
    #         return []
    #     semantic_queries_fr = {
    #         "previous_episodes": f"épisodes précédents de douleur {body_region}",
    #         "treatments_tried": f"traitements essayés pour douleur {body_region}",
    #         "improvements": f"ce qui a amélioré douleur {body_region}",
    #         "red_flags_history": f"symptômes inquiétants liés à douleur {body_region}",
    #     }
    #     all_memories: List[Dict[str, Any]] = []
    #     try:
    #         for category, q in semantic_queries_fr.items():
    #             # Prioriser la récence si disponible
    #             if hasattr(self.memory_client, 'search_memories_with_recency'):
    #                 result = self.memory_client.search_memories_with_recency(query=q, user_id=user_id, limit=5, recency_weight=0.5)
    #             else:
    #                 result = self.memory_client.search_memories(query=q, user_id=user_id, limit=5)
    #             if result.get("success") and result.get("memories"):
    #                 for memory in (result["memories"] or []):
    #                     text = memory.get("memory", "") if isinstance(memory, dict) else str(memory)
    #                     if text:
    #                         all_memories.append({
    #                             "text": text,
    #                             "suggested_category": category,
    #                             "query_used": q,
    #                             "search_score": memory.get("score", 0) if isinstance(memory, dict) else 0,
    #                             "timestamp_unix": (memory.get("metadata", {}) or {}).get("timestamp_unix") if isinstance(memory, dict) else None,
    #                         })
    #         # Trier par récence si timestamps présents
    #         all_memories.sort(key=lambda m: (m.get("timestamp_unix") or 0), reverse=True)
    #         return all_memories
    #     except Exception as e:
    #         logger.error(f"Erreur récupération mémoire FR/temporal: {e}")
    #         return []

    # def _apply_temporal_scoring(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """Appliquer un score temporel (récence) au-dessus du score de recherche sémantique."""
    #     import math, time
    #     now = time.time()
    #     scored = []
    #     for m in memories:
    #         base = float(m.get("search_score", 0.0))
    #         ts = m.get("timestamp_unix")
    #         if isinstance(ts, (int, float)):
    #             age = max(0.0, now - float(ts))
    #             tau = 7 * 24 * 3600
    #             decay = math.exp(-(age / tau))
    #         else:
    #             decay = 0.0
    #         combined = 0.7 * base + 0.3 * decay
    #         m2 = dict(m)
    #         m2["temporal_score"] = combined
    #         scored.append(m2)
    #     scored.sort(key=lambda x: x.get("temporal_score", 0.0), reverse=True)
    #     return scored


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



    def retrieve_contextual_knowledge(self,
                                    query: str,
                                    body_region: str,
                                    assessment_phase: AssessmentPhase = AssessmentPhase.INITIAL,
                                    previous_responses: Optional[List[str]] = None,
                                    top_k: int = 5,
                                    memory_context: Optional[Dict[str, Any]] = None,
                                    context_key: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve knowledge with clinical reasoning and optional caller-provided memory context.

        Notes:
        - This function is user-identity agnostic. It never reads or infers a user_id.
        - Callers may pass an opaque `context_key` (e.g., a user/session id) if they want
          per-user caching behavior. If omitted, caching by user is skipped.
        - Callers may also pass a pre-fetched `memory_context`. If omitted, an empty
          memory context is used.
        """
        print(f"\n{'='*80}")
        print(f"⚡ OPTIMIZED RAG: Starting {assessment_phase.value.upper()} phase retrieval")
        print(f"   Body Region: {body_region}")
        print(f"   Query: {query[:100]}...")
        print(f"   Previous Responses: {len(previous_responses) if previous_responses else 0}")
        print(f"{'='*80}\n")

        try:
            # Use provided memory context or fallback to empty structure
            if memory_context is None:
                memory_context = {k: [] for k in [
                    "previous_episodes", "treatments_tried", "medications",
                    "triggers", "improvements", "red_flags_history"
                ]}
            
            # Simple analysis structure - no complex clinical reasoning
            clinical_analysis = {
                "user_responses": " ".join(previous_responses) if previous_responses else "",
                "body_region": body_region,
                "priority_level": ClinicalPriority.MEDIUM
            } if previous_responses else {}
            
            # Build dynamic queries based on assessment phase and context
            queries = self._build_dynamic_queries(
                query, body_region, assessment_phase, clinical_analysis, memory_context
            )
            
            # Retrieve knowledge for each query with user context - PARALLELIZED
            all_results = []
            user_symptoms = " ".join(previous_responses) if previous_responses else ""

            # Execute all queries in parallel for 2-3x speed improvement
            all_results = self._retrieve_queries_parallel(
                queries, body_region, top_k, user_symptoms, query
            )
            
            # Consolidate and rank results
            consolidated_results = self._consolidate_results(all_results)

            # Extract any explicit questions from consolidated text chunks for caching
            try:
                extracted_questions: list[str] = []
                for item in consolidated_results:
                    text = item.get("text", "") or ""
                    if text:
                        extracted_questions.extend(self._extract_questions_from_text(text))

                # Cache by (context_key, body_region) only if a context_key is provided
                if context_key is not None:
                    cache_key = (str(context_key), str(body_region))
                    if cache_key not in _CACHED_PAIN_QUESTIONS:
                        _CACHED_PAIN_QUESTIONS[cache_key] = set()
                    for q in extracted_questions:
                        _CACHED_PAIN_QUESTIONS[cache_key].add(q)
                    print(
                        f"🗂️ CACHE: Stored {len(extracted_questions)} questions (total {len(_CACHED_PAIN_QUESTIONS[cache_key])}) "
                        f"for {cache_key}"
                    )
                else:
                    print("🗂️ CACHE: Skipped per-user caching (no context_key provided)")
            except Exception as e:
                logger.warning(f"Failed to cache extracted questions: {e}")
            
            return {
                "knowledge_results": consolidated_results,
                "memory_context": memory_context,
                "clinical_analysis": clinical_analysis,
                "assessment_phase": assessment_phase.value,
                "recommendations": self._generate_knowledge_recommendations(
                    assessment_phase
                )
            }
            
        except Exception as e:
            logger.error(f"Error in contextual knowledge retrieval: {e}")
            return {
                "knowledge_results": [],
                "memory_context": {},
                "clinical_analysis": {},
                "error": str(e)
            }

    def _retrieve_queries_parallel(self,
                                  queries: List[Dict[str, Any]],
                                  body_region: str,
                                  top_k: int,
                                  user_symptoms: str,
                                  conversation_context: str) -> List[Dict[str, Any]]:
        """Execute RAG queries in parallel using ThreadPoolExecutor for 2-3x speed improvement."""
        print(f"⚡ PARALLEL RAG: Executing {len(queries)} queries in parallel")

        all_results = []

        # Use ThreadPoolExecutor to run queries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all query tasks
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

    def _retrieve_context_questions_parallel(self,
                                            context_aware_queries: List[Dict[str, Any]],
                                            index: VectorStoreIndex,
                                            top_k: int) -> List[Dict[str, Any]]:
        """Execute context-aware question queries in parallel."""
        print(f"⚡ PARALLEL QUESTIONS: Executing {len(context_aware_queries)} question queries in parallel")

        all_question_results = []

        # Use ThreadPoolExecutor to run question queries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all query tasks
            future_to_query = {}
            for query_info in context_aware_queries:
                future = executor.submit(
                    self._execute_single_question_query,
                    query_info, index, top_k // len(context_aware_queries) + 2
                )
                future_to_query[future] = query_info

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_query):
                query_info = future_to_query[future]
                try:
                    results = future.result()
                    all_question_results.extend(results)
                    print(f"⚡ PARALLEL QUESTIONS: Completed query with {len(results)} question results")
                except Exception as e:
                    logger.error(f"Error in parallel question query execution: {e}")
                    continue

        # Sort by context relevance, then by question count and score
        all_question_results.sort(
            key=lambda x: (x["context_relevance"], x["question_count"], x["score"]),
            reverse=True
        )

        # Remove duplicates
        unique_results = []
        seen_content = set()
        for result in all_question_results:
            content_key = result["text"][:200]  # Use first 200 chars as dedup key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break

        print(f"⚡ PARALLEL QUESTIONS: Completed all {len(context_aware_queries)} queries, unique results: {len(unique_results)}")
        return unique_results

    def _execute_single_question_query(self,
                                     query_info: Dict[str, Any],
                                     index: VectorStoreIndex,
                                     similarity_top_k: int) -> List[Dict[str, Any]]:
        """Execute a single question query (for parallel execution)."""
        results = []

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k
        )

        nodes = retriever.retrieve(query_info["query"])

        for node in nodes:
            # Only include chunks that actually contain questions
            if self._chunk_contains_questions(node.text):
                result = {
                    "text": node.text,
                    "score": node.score,
                    "metadata": node.metadata or {},
                    "query": query_info["query"],
                    "context_relevance": query_info["relevance"],
                    "priority": query_info["priority"],
                    "namespace": "questions",  # Default namespace
                    "question_count": self._count_questions_in_chunk(node.text)
                }
                results.append(result)

        return results

    def _build_dynamic_queries(self,
                             base_query: str,
                             body_region: str,
                             phase: AssessmentPhase,
                             clinical_analysis: Dict[str, Any],
                             memory_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build dynamic queries based on assessment context."""
        print(f"🔍 QUERY BUILDER: Phase={phase.value.upper()}, body_region={body_region}")
        queries = []

        if phase == AssessmentPhase.INITIAL:
            # Initial assessment queries - prioritize finding existing questions
            print(f"   📋 Building INITIAL phase queries (expect 4 queries)")
            queries.extend([
                {
                    "query": f"{body_region} pain assessment protocol clinical examination questions",
                    "priority": ClinicalPriority.HIGH,
                    "search_for_questions": True
                },
                {
                    "query": f"{body_region} pain history taking essential questions",
                    "priority": ClinicalPriority.HIGH,
                    "search_for_questions": True
                },
                {
                    "query": f"comment décrirais-tu douleur {body_region}",  # Search for French questions
                    "priority": ClinicalPriority.HIGH,
                    "search_for_questions": True
                },
                {
                    "query": f"{base_query} {body_region} symptoms evaluation",
                    "priority": ClinicalPriority.MEDIUM
                }
            ])

        elif phase == AssessmentPhase.DETAILED:
            # Detailed examination queries
            print(f"   📋 Building DETAILED phase queries (expect 2 queries)")
            queries.extend([
                {
                    "query": f"{body_region} physical examination techniques tests",
                    "priority": ClinicalPriority.HIGH
                },
                {
                    "query": f"{base_query} detailed assessment {body_region}",
                    "priority": ClinicalPriority.MEDIUM
                }
            ])

        elif phase == AssessmentPhase.DIFFERENTIAL:
            # Differential diagnosis queries
            print(f"   📋 Building DIFFERENTIAL phase queries")
            suspected_conditions = clinical_analysis.get("suspected_conditions", [])
            if suspected_conditions:
                print(f"      Found {len(suspected_conditions)} suspected conditions, using top 3")
                for condition_info in suspected_conditions[:3]:  # Top 3 conditions
                    condition = condition_info["condition"]
                    queries.append({
                        "query": f"{body_region} {condition} differential diagnosis assessment",
                        "priority": ClinicalPriority.HIGH
                    })
            else:
                print(f"      No suspected conditions, using general differential query")
                queries.append({
                    "query": f"{body_region} pain differential diagnosis common conditions",
                    "priority": ClinicalPriority.HIGH
                })

        elif phase == AssessmentPhase.SUMMARY:
            # Summary and treatment queries
            print(f"   📋 Building SUMMARY phase queries (expect 2 queries)")
            queries.extend([
                {
                    "query": f"{body_region} pain management recommendations treatment",
                    "priority": ClinicalPriority.HIGH
                },
                {
                    "query": f"{body_region} assessment summary clinical documentation",
                    "priority": ClinicalPriority.MEDIUM
                }
            ])
        
        # Skip red flag queries - simplified approach

        # Add memory-informed queries
        if memory_context.get("treatments_tried"):
            print(f"   📋 Adding memory-informed query (treatments_tried found)")
            queries.append({
                "query": f"{body_region} alternative treatments options",
                "priority": ClinicalPriority.MEDIUM
            })

        print(f"🔍 QUERY BUILDER: Generated {len(queries)} dynamic queries for {phase.value.upper()} phase")
        for i, q in enumerate(queries, 1):
            print(f"   Query {i}: '{q['query'][:80]}...' [priority={q['priority'].value if hasattr(q['priority'], 'value') else q['priority']}]")
        return queries
    
    def _retrieve_context_aware_questions(self,
                                         body_region: str,
                                         user_symptoms: str = "",
                                         conversation_context: str = "",
                                         top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve questions that are contextually relevant to user's specific situation."""
        try:
            # Use the pain assessment questions collection
            collection_name = "pain_assessment_questions"

            # Try specific collection first, fall back to general if needed
            try:
                index = self._get_index(collection_name)
            except Exception:
                index = self._get_index("pain_assessment_questions")  # Fallback to our existing collection

            # Build context-aware queries based on user's actual situation
            context_aware_queries = self._build_context_aware_question_queries(
                body_region, user_symptoms, conversation_context
            )

            all_question_results = []

            # Parallelize context-aware question retrieval
            all_question_results = self._retrieve_context_questions_parallel(
                context_aware_queries, index, top_k
            )

            print(f"🎯 CONTEXT-AWARE QUESTIONS: Found {len(all_question_results)} relevant question chunks for user situation")

            # DEBUG: show a small sample of extracted questions from retrieved chunks
            try:
                debug_questions: List[str] = []
                for r in all_question_results[:5]:  # limit for performance
                    debug_questions.extend(self._extract_questions_from_text(r.get("text", "")))
                # deduplicate and print sample
                uniq = []
                seen = set()
                for q in debug_questions:
                    if q not in seen:
                        seen.add(q)
                        uniq.append(q)
                sample = uniq[:5]
                if sample:
                    print("🧪 DEBUG QDRANT QUESTIONS:", " | ".join(sample))
                else:
                    print("🧪 DEBUG QDRANT QUESTIONS: (none extracted)")
            except Exception as e:
                print(f"🧪 DEBUG QDRANT QUESTIONS: extraction error: {e}")

            return all_question_results

        except Exception as e:
            logger.error(f"Error retrieving context-aware questions: {e}")
            return []

    def _build_context_aware_question_queries(self,
                                            body_region: str,
                                            user_symptoms: str,
                                            conversation_context: str) -> List[Dict[str, Any]]:
        """Build semantic queries using user's exact words and context - no keyword matching."""
        queries = []

        # Build queries using user's exact language and context
        if user_symptoms:
            # Primary query using user's exact description
            queries.append({
                "query": f"{body_region} pain {user_symptoms} questions assessment evaluation",
                "relevance": 10,
                "priority": "user_exact_words"
            })

        # Context-driven follow-up queries using exact conversation context
        if conversation_context:
            queries.append({
                "query": f"{body_region} {conversation_context} follow-up questions",
                "relevance": 9,
                "priority": "conversation_context"
            })

        # Always include comprehensive assessment queries
        queries.extend([
            {
                "query": f"{body_region} pain assessment questions clinical evaluation",
                "relevance": 8,
                "priority": "general_assessment"
            },
            {
                "query": f"{body_region} pain questions symptoms evaluation",
                "relevance": 7,
                "priority": "general_assessment"
            },
            {
                "query": f"{body_region} pain evaluation questions douleur",
                "relevance": 4,
                "priority": "pain_assessment"
            }
        ])

        print(f"🔍 QUERY BUILDER: Built {len(queries)} user-content-based queries (no keyword matching)")
        return queries

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
            # Try multiple collections with fallback logic
            collection_candidates = ["pain_hierarchical_chunks", "questions", "test_questions"]

            index = None
            collection_name = None
            for candidate in collection_candidates:
                try:
                    index = self._get_index(candidate)
                    collection_name = candidate
                    break
                except Exception:
                    continue

            if index is None:
                print("❌ No RAG collections available - returning empty results")
                return []

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
        """Normalize French body region input to match Qdrant metadata (all French)."""
        # All user input is French - map to Qdrant French values
        french_normalization = {
            "épaule": "epaule",
            "épaules": "epaule",
            "genou": "genou",
            "genoux": "genou",
            "dos": "dos",
            "bas du dos": "dos",  # User says "bas du dos", Qdrant has "dos"
            "cou": "cou",
            "hanche": "hanche",
            "hanches": "hanche",
            "cheville": "cheville_pied",
            "pied": "cheville_pied",
            "cheville_pied": "cheville_pied"
        }
        normalized = french_normalization.get(body_region.lower(), body_region)
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
    
    def _generate_knowledge_recommendations(self,
                                          phase: AssessmentPhase) -> Dict[str, Any]:
        """Generate recommendations based on retrieved knowledge and clinical analysis."""
        recommendations = {
            "next_questions_focus": [],
            "clinical_priorities": [],
            "knowledge_gaps": [],
            "assessment_direction": ""
        }
        
        # Phase-specific recommendations
        if phase == AssessmentPhase.INITIAL:
            recommendations["next_questions_focus"] = [
                "Pain onset and duration",
                "Pain characteristics and location",
                "Aggravating and relieving factors"
            ]
            
        elif phase == AssessmentPhase.DETAILED:
            recommendations["next_questions_focus"] = [
                "Functional limitations",
                "Previous treatments",
                "Associated symptoms"
            ]
        
        # Simple clinical priority recommendations
        recommendations["assessment_direction"] = "Continue with comprehensive assessment"
        
        return recommendations


# Initialize global enhanced retriever
_enhanced_retriever = None

def get_enhanced_assessment_retriever() -> EnhancedRAGAssessmentRetriever:
    """Get global enhanced assessment retriever instance."""
    global _enhanced_retriever
    if _enhanced_retriever is None:
        _enhanced_retriever = EnhancedRAGAssessmentRetriever()
    return _enhanced_retriever


def create_enhanced_pain_assessment_tool(
    context_key: Optional[str] = None,
    memory_context: Optional[Dict[str, Any]] = None,
) -> FunctionTool:
    """Create enhanced AI-powered pain assessment tool with memory integration."""

    def retrieve_contextual_pain_knowledge(query: str, 
                                         body_region: str,
                                         assessment_phase: str = "initial",
                                         previous_responses: Optional[str] = None) -> str:
        """Retrieve contextual pain knowledge with memory and clinical reasoning.
        
        Args:
            query: Current question or assessment need
            body_region: Body region being assessed
            assessment_phase: Phase of assessment (initial, detailed, differential, summary)
            previous_responses: JSON string of previous user responses
            
        Returns:
            Comprehensive knowledge context for AI question generation
        """
        print(f"🔥 OPENAI AGENT TOOL: retrieve_contextual_pain_knowledge called")
        logger.warning(f"🔥 OPENAI AGENT TOOL: retrieve_contextual_pain_knowledge called")
        print(f"   Query: {query}")
        print(f"   Body Region: {body_region}")
        print(f"   Phase: {assessment_phase}")
        
        retriever = get_enhanced_assessment_retriever()
        
        # Parse assessment phase
        try:
            phase = AssessmentPhase(assessment_phase.lower())
        except ValueError:
            phase = AssessmentPhase.INITIAL
        
        # Parse previous responses
        responses_list = []
        if previous_responses:
            try:
                responses_list = json.loads(previous_responses)
            except json.JSONDecodeError:
                responses_list = [previous_responses]
        
        # Retrieve contextual knowledge (identity-agnostic; context provided by caller if any)
        result = retriever.retrieve_contextual_knowledge(
            query=query,
            body_region=body_region,
            assessment_phase=phase,
            previous_responses=responses_list,
            memory_context=memory_context,
            context_key=context_key
        )
        # Verbose: print full raw retrieval result prior to formatting
        try:
            print("🧾 RAW RAG RESULT (pre-format):")
            # Print knowledge_results fully
            kr = result.get("knowledge_results", [])
            print(f"   knowledge_results: {len(kr)} items")
            for idx, doc in enumerate(kr, 1):
                meta = doc.get("metadata", {})
                print(f"   —[{idx}] score={doc.get('score')} priority={doc.get('priority')} namespace={doc.get('namespace')}")
                print("     TEXT:")
                print(doc.get("text", "") or "<no text>")
                print("     METADATA:")
                print(meta)
            # Print memory_context and clinical_analysis
            print("   memory_context:")
            print(result.get("memory_context", {}))
            print("   clinical_analysis:")
            print(result.get("clinical_analysis", {}))
            print("   recommendations:")
            print(result.get("recommendations", {}))
        except Exception as _e:
            print(f"⚠️ Failed to print RAW RAG RESULT: {_e}")
        
        # Format result for AI consumption - keep it simple, let LLM do all analysis
        formatted_result = {
            "medical_knowledge": [
                {
                    "content": doc["text"],
                    "relevance_score": doc["score"],
                    "priority": doc["priority"]
                }
                for doc in result.get("knowledge_results", [])[:10]
            ],
            "user_context": result.get("memory_context", {}),
            "clinical_insights": {
                k: v.value if hasattr(v, 'value') else v
                for k, v in result.get("clinical_analysis", {}).items()
            },
            "recommendations": result.get("recommendations", {})
        }
        try:
            print("🧾 FORMATTED RAG RESULT:")
            print(json.dumps(formatted_result, indent=2))
        except Exception:
            pass
        
        return json.dumps(formatted_result, indent=2)
    
    return FunctionTool.from_defaults(
        fn=retrieve_contextual_pain_knowledge,
        name="retrieve_contextual_pain_knowledge",
        description="Retrieve comprehensive pain assessment knowledge combining medical literature, user history from memory, and clinical reasoning to inform AI-generated questions."
    )


def create_ai_question_generator_tool() -> FunctionTool:
    """Create tool for AI-powered question generation using retrieved knowledge and memory."""
    
    def generate_assessment_questions(knowledge_context: str,
                                    body_region: str) -> str:
        """Generate contextually appropriate assessment questions using AI.
        
        Args:
            knowledge_context: JSON context from retrieve_contextual_pain_knowledge
            body_region: Body region being assessed
            
        Returns:
            JSON formatted AI-generated questions with reasoning
        """
        print(f"🔥 OPENAI AGENT TOOL: generate_assessment_questions called")
        logger.warning(f"🔥 OPENAI AGENT TOOL: generate_assessment_questions called for {body_region}")
        
        try:
            context = json.loads(knowledge_context)
        except json.JSONDecodeError:
            context = {"error": "Invalid knowledge context"}

        # Extract key insights from context
        medical_knowledge = context.get("medical_knowledge", [])
        user_context = context.get("user_context", {})
        recommendations = context.get("recommendations", {})

        # Verbose: dump incoming knowledge context
        try:
            print("🧪 QUESTION GENERATOR INPUT (knowledge_context):")
            print(json.dumps(context, indent=2))
        except Exception:
            pass
        
        # Extract existing questions from medical knowledge for inspiration
        existing_questions = []
        question_examples = []
        
        for doc in medical_knowledge:
            content = doc.get("content", "")
            # Extract French questions using the same pattern as document processor
            import re
            found_questions = re.findall(r'«([^»]+)»', content)
            for q in found_questions:
                existing_questions.append(q.strip())
                question_examples.append({
                    "original_question": q.strip(),
                    "relevance_score": doc.get("relevance_score", 0),
                    "priority": doc.get("priority", "medium")
                })
        
        # Generate contextual questions based on retrieved knowledge
        questions_structure = {
            "generated_questions": [],
            "clinical_reasoning": "",
            "memory_insights": user_context,
            "next_steps": "",
            "priority_areas": recommendations.get("next_questions_focus", []),
            "question_inspiration": question_examples[:5],  # Top 5 examples
            "total_question_examples_found": len(existing_questions)
        }
        
        # ALWAYS generate core assessment questions - even without RAG
        # These are essential clinical questions that should ALWAYS be asked

        core_questions = [
            {
                "question": f"Quand cette douleur au {body_region} a-t-elle commencé?",
                "rationale": "Essential for understanding pain timeline and onset",
                "priority": "critical",
                "type": "onset"
            },
            {
                "question": f"Comment décririez-vous la douleur au {body_region}? (aiguë, sourde, lancinante, brûlante, etc.)",
                "rationale": "Pain character helps differential diagnosis",
                "priority": "critical",
                "type": "character"
            },
            {
                "question": f"Où exactement ressentez-vous la douleur au {body_region}?",
                "rationale": "Precise localization for anatomical diagnosis",
                "priority": "critical",
                "type": "location"
            },
            {
                "question": "Sur une échelle de 1 à 10, quelle est l'intensité de votre douleur?",
                "rationale": "Pain severity assessment",
                "priority": "high",
                "type": "severity"
            },
            {
                "question": "Qu'est-ce qui aggrave votre douleur?",
                "rationale": "Identify aggravating factors and triggers",
                "priority": "high",
                "type": "aggravating_factors"
            },
            {
                "question": "Qu'est-ce qui soulage votre douleur?",
                "rationale": "Identify relieving factors for treatment insights",
                "priority": "high",
                "type": "relieving_factors"
            },
            {
                "question": "Avez-vous d'autres symptômes associés? (engourdissement, faiblesse, raideur, etc.)",
                "rationale": "Associated symptoms for comprehensive assessment",
                "priority": "high",
                "type": "associated_symptoms"
            },
            {
                "question": f"Cette douleur au {body_region} limite-t-elle vos activités quotidiennes?",
                "rationale": "Functional impact assessment",
                "priority": "medium",
                "type": "functional_impact"
            },
            {
                "question": "Avez-vous déjà eu ce type de douleur auparavant?",
                "rationale": "Previous episodes indicate chronicity",
                "priority": "medium",
                "type": "history"
            },
            {
                "question": "Avez-vous essayé des traitements pour cette douleur? Si oui, lesquels et avec quels résultats?",
                "rationale": "Previous treatment response guides recommendations",
                "priority": "medium",
                "type": "treatment_history"
            }
        ]

        questions_structure["generated_questions"].extend(core_questions)

        # Add memory-informed questions if available
        if user_context.get("treatments_tried"):
            questions_structure["generated_questions"].append({
                "question": f"Je vois que vous avez déjà essayé des traitements. Qu'est-ce qui a été le plus/moins efficace pour votre douleur au {body_region}?",
                "rationale": "Building on previous treatment history from memory",
                "priority": "high",
                "type": "treatment_history_followup"
            })

        # Add RAG-inspired questions if available
        if existing_questions:
            for i, example_q in enumerate(existing_questions[:3]):  # Top 3 from RAG
                questions_structure["generated_questions"].append({
                    "question": example_q,
                    "rationale": f"Question from medical literature (example {i+1})",
                    "priority": "high",
                    "type": "literature_inspired",
                    "original_inspiration": example_q
                })

        # Agent should ask these questions progressively, adapting based on user responses
        
        # Set clinical reasoning and next steps
        questions_structure["clinical_reasoning"] = f"Generated {len(questions_structure['generated_questions'])} questions based on retrieved medical knowledge, user memory, and clinical analysis"
        questions_structure["next_steps"] = recommendations.get("assessment_direction", "Continue with comprehensive assessment")
        
        try:
            print("🧪 QUESTION GENERATOR OUTPUT:")
            print(json.dumps(questions_structure, indent=2))
        except Exception:
            pass

        return json.dumps(questions_structure, indent=2)
    
    return FunctionTool.from_defaults(
        fn=generate_assessment_questions,
        name="generate_assessment_questions", 
        description="Generate AI-powered, contextually appropriate assessment questions using medical knowledge and user memory."
    )


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

        from datetime import datetime
        import os

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

        # Clear cached questions after summary generation (simple global clear)
        try:
            _CACHED_PAIN_QUESTIONS.clear()
            print("🧹 CACHE: Cleared cached pain questions after summary generation")
        except Exception as e:
            print(f"⚠️ Failed to clear cached questions: {e}")
        return json.dumps(summary, indent=2)

    return FunctionTool.from_defaults(
        fn=generate_adaptive_pain_summary,
        name="generate_adaptive_pain_summary",
        description="Generate flexible, adaptive pain summary that adjusts structure and content based on available information - no rigid templates."
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
        all_questions = []

        for doc in consolidated_results:
            metadata = doc.get("metadata", {})
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

        # Return top 20 questions to avoid overwhelming the AI
        return all_questions[:20]

    def retrieve_with_llm_queries(
        body_region: str,
        conversation_history: str
    ) -> str:
        """Retrieve pain knowledge using LLM-generated queries and HyDE approach.

        Args:
            body_region: Body region being assessed (e.g., "dos", "epaule")
            conversation_history: Summary of conversation so far

        Returns:
            JSON string with retrieved medical knowledge
        """
        print(f"🔥 LLM-DRIVEN RAG TOOL: retrieve_with_llm_queries called")
        logger.warning(f"🔥 LLM-DRIVEN RAG TOOL: retrieve_with_llm_queries called for {body_region}")

        from llama_index.llms.openai import OpenAI
        import os

        # Use GPT-4o-mini for fast query generation
        llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Prompt LLM to generate queries
        query_gen_prompt = f"""Given this pain assessment conversation history, generate 3-5 specific search queries to retrieve relevant medical knowledge from a pain assessment database.

Body Region: {body_region}
Conversation History: {conversation_history}

Generate queries that will help retrieve:
1. Relevant clinical questions for this specific case to help diagnose the pain
2. Information about the mechanism of injury/pain
3. Red flags or concerning symptoms 
4. Potential causes/sources of the pain


Also create a HyDE (Hypothetical Document Embedding) snippet - a 1-2 sentence summary of what medical document would answer this patient's case.

Respond ONLY with valid JSON in this exact format:
{{
  "queries": [
    "query 1 in French ",
    "query 2 in French ",
    "query 3 in French "
  ],
  "hyde_snippet": "Brief summary of patient case for semantic search"
}}"""

        try:
            # Generate queries with LLM
            response = llm.complete(query_gen_prompt)
            response_text = response.text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            query_data = json.loads(response_text)
            queries = query_data.get("queries", [])
            hyde_snippet = query_data.get("hyde_snippet", "")

            print(f"📝 LLM Generated {len(queries)} queries:")
            for i, q in enumerate(queries, 1):
                print(f"   Query {i}: {q}")
            print(f"📝 HyDE snippet: {hyde_snippet}")

        except Exception as e:
            logger.error(f"Failed to generate LLM queries: {e}")
            # Fallback to simple queries
            queries = [
                f"{body_region} pain assessment",
                f"{body_region} clinical evaluation",
                f"douleur {body_region}"
            ]
            hyde_snippet = conversation_history[:200]

        # Now retrieve with generated queries
        retriever = get_enhanced_assessment_retriever()

        # Build query structure for retrieval
        all_results = []

        # Add HyDE snippet as a query too
        if hyde_snippet:
            queries.append(hyde_snippet)

        # Execute queries in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_query = {}
            for query in queries[:5]:  # Max 5 queries
                future = executor.submit(
                    retriever._retrieve_with_query,
                    query,
                    body_region,
                    ClinicalPriority.HIGH,
                    top_k=3,
                    search_for_questions=True,
                    user_symptoms=conversation_history[:500],
                    conversation_context=conversation_history[:200]
                )
                future_to_query[future] = query

            # Collect results
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"⚡ LLM-RAG: Completed query '{query[:50]}...' with {len(results)} results")
                except Exception as e:
                    logger.error(f"Error executing LLM-generated query: {e}")

        # Consolidate results
        consolidated_results = retriever._consolidate_results(all_results)

        # Collect all extracted questions from the results
        extracted_questions = _collect_all_questions(consolidated_results)

        # Format for AI
        formatted_result = {
            "llm_generated_queries": queries[:5],
            "hyde_snippet": hyde_snippet,

            # NEW: Explicit array of extracted questions from medical documents
            "extracted_questions": extracted_questions,

            "medical_knowledge": [
                {
                    "content": doc["text"],
                    "relevance_score": doc["score"],
                    "priority": doc["priority"],

                    # NEW: Include question metadata for transparency
                    "has_questions": doc.get("has_questions", False),
                    "question_count": doc.get("question_count", 0),
                    "questions_in_chunk": doc.get("metadata", {}).get("extracted_questions", [])
                }
                for doc in consolidated_results[:10]
            ],
            "retrieval_method": "llm_driven_rag"
        }

        print(f"✅ LLM-RAG: Retrieved {len(consolidated_results)} chunks using {len(queries)} LLM-generated queries")
        print(f"🔍 EXTRACTED: {len(extracted_questions)} questions from medical documents")

        # Show top extracted questions in logs
        if extracted_questions:
            print(f"📋 TOP EXTRACTED QUESTIONS:")
            for i, q in enumerate(extracted_questions[:5], 1):
                print(f"   {i}. [{q['priority']}] {q['question'][:100]}...")

        return json.dumps(formatted_result, indent=2)

    return FunctionTool.from_defaults(
        fn=retrieve_with_llm_queries,
        name="retrieve_with_llm_queries",
        description="Retrieve pain assessment knowledge using LLM-generated queries tailored to the specific conversation context. More adaptive than fixed query templates."
    )


def get_enhanced_pain_assessment_tools(use_llm_rag: bool = True) -> List[FunctionTool]:
    """Get simplified tools for AI-powered pain assessment - agent decides when to complete.

    Args:
        use_llm_rag: If True, use LLM-driven RAG (new). If False, use static query RAG (old).
    """
    from app.tools.date_tool import get_date_tool

    # Choose RAG tool based on flag
    if use_llm_rag:
        rag_tool = create_llm_driven_rag_tool()  # NEW: LLM-generated queries
        print("🔧 Using LLM-driven RAG tool (adaptive queries)")
    else:
        rag_tool = create_enhanced_pain_assessment_tool()  # OLD: Static queries
        print("🔧 Using static query RAG tool (legacy)")

    return [
        rag_tool,  # Primary RAG tool (either old or new)
        create_pain_summary_generator_tool(),  # Gets user_id from thread-local storage
        get_date_tool()
    ]
