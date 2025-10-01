"""
Hybrid retrieval for pain assessment over two Qdrant collections:
  - questions_all_regions: atomic questions/red_flags/follow_ups
  - context_all_regions:   explanations/causes windows

STRICT: No keyword-based matching on the user query.
We classify user intent/signals purely via embedding similarity (semantic labels),
then use dense retrieval + optional reranker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import NodeWithScore

from app.core.config import QDRANT_URL
from app.core.embeddings import get_embedding_model


COL_QUESTIONS = "questions_all_regions"
COL_CONTEXT = "context_all_regions"


@dataclass
class RetrievalQuery:
    body_regions: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    context: List[str] = field(default_factory=list)
    red_flag_signals: List[str] = field(default_factory=list)
    text: str = ""


def stringify_rq(rq: RetrievalQuery) -> str:
    def join(xs):
        return ", ".join(xs) if xs else "none"
    return (
        f"Pain: {join(rq.body_regions)}; "
        f"symptoms: {join(rq.symptoms)}; "
        f"modifiers: {join(rq.modifiers)}; "
        f"context: {join(rq.context)}; "
        f"screen red flags."
    )


class SemanticClassifier:
    """Embedding-only classifier for user message → structured retrieval object.

    No keyword checks; uses cosine similarity against label descriptions.
    """

    def __init__(self):
        self.embed = get_embedding_model()
        # Label catalogs with natural-language descriptions
        self.labels = {
            "body_regions": {
                "back": "Lower back (lumbar) pain or lumbar spine symptoms",
                "knee": "Knee joint pain or symptoms around the patella and tibiofemoral joint",
                "shoulder": "Shoulder pain (rotator cuff, impingement, deltoid region)",
                "neck": "Neck pain (cervical spine) or upper trapezius pain",
                "hip": "Hip or groin pain related to the hip joint",
                "foot_ankle": "Foot or ankle pain including heel or Achilles",
            },
            "symptoms": {
                "radiation": "Pain radiating down a limb or along a nerve pathway (sciatica-like)",
                "numbness": "Numbness, tingling, pins and needles (paresthesia)",
                "weakness": "Loss of strength or motor weakness in the limb",
            },
            "modifiers": {
                "worse-flexion": "Symptoms are worse with bending forward or spinal flexion",
                "better-extension": "Symptoms improve with arching the back or spinal extension",
                "morning-stiffness": "Morning stiffness that eases after moving around",
            },
            "context": {
                "deadlift": "Symptoms related to heavy lifting or deadlifts in training",
                "recent-volume-spike": "Recent spike in training volume, load, or activity",
                "sedentary-job": "Sedentary desk job, prolonged sitting contributing to symptoms",
            },
            "red_flag_signals": {
                "night_pain": "Night pain that wakes the person from sleep",
                "fever": "Fever, chills, or signs of systemic infection",
                "bladder_bowel": "Bladder or bowel dysfunction, incontinence, retention",
                "infection": "Recent infection or systemic illness with back pain",
                "neuro_deficit": "Severe or progressive neurological deficit or paralysis",
                "trauma": "Recent significant trauma or accident",
            },
        }
        # Precompute label embeddings
        self.vecs: Dict[str, Dict[str, List[float]]] = {}
        for group, lv in self.labels.items():
            self.vecs[group] = {k: self.embed.get_text_embedding(v) for k, v in lv.items()}

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0 or nb <= 0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    def classify(self, user_msg: str, state: Optional[Dict[str, Any]] = None) -> RetrievalQuery:
        state = state or {}
        q = self.embed.get_text_embedding(user_msg or "")

        def top_labels(group: str, threshold: float, topn: int = 2) -> List[str]:
            pairs = [(lbl, self._cos(q, vec)) for lbl, vec in self.vecs[group].items()]
            pairs.sort(key=lambda t: t[1], reverse=True)
            out: List[str] = []
            for i, (lbl, sc) in enumerate(pairs):
                if sc >= threshold and len(out) < topn:
                    out.append(lbl)
            return out

        def multi_labels(group: str, threshold: float) -> List[str]:
            return [lbl for lbl, vec in self.vecs[group].items() if self._cos(q, vec) >= threshold]

        body_regions = state.get("body_regions") or top_labels("body_regions", threshold=0.22, topn=2) or ["back"]
        symptoms = multi_labels("symptoms", threshold=0.25)
        modifiers = multi_labels("modifiers", threshold=0.23)
        context = multi_labels("context", threshold=0.23)
        red_flags = multi_labels("red_flag_signals", threshold=0.27)

        rq = RetrievalQuery(
            body_regions=body_regions,
            symptoms=symptoms,
            modifiers=modifiers,
            context=context,
            red_flag_signals=red_flags,
        )
        rq.text = stringify_rq(rq)
        return rq


_CLASSIFIER: Optional[SemanticClassifier] = None


def build_retrieval_object(user_msg: str, state: Optional[Dict[str, Any]] = None) -> RetrievalQuery:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = SemanticClassifier()
    return _CLASSIFIER.classify(user_msg, state)


class PainQueryEngine:
    def __init__(self, dense_top_k: int = 50):
        load_dotenv()
        self.client = QdrantClient(url=QDRANT_URL)
        self.dense_top_k = dense_top_k

        # Dense retrievers (Qdrant)
        vs_q = QdrantVectorStore(client=self.client, collection_name=COL_QUESTIONS)
        vs_c = QdrantVectorStore(client=self.client, collection_name=COL_CONTEXT)
        self.idx_q = VectorStoreIndex.from_vector_store(vs_q)
        self.idx_c = VectorStoreIndex.from_vector_store(vs_c)

        # Optional: Voyage reranker
        try:
            import voyageai  # type: ignore

            self.voyage_client = voyageai.Client()
            self._has_voyage = True
        except Exception:
            self.voyage_client = None
            self._has_voyage = False

    def _route_filters(self, rq: RetrievalQuery) -> Dict[str, Any]:
        if rq.red_flag_signals:
            types = ["red_flag", "question"]
            phases = ["triage", "intake"]
        else:
            types = ["question", "follow_up"]
            phases = ["intake", "deep_dive"]
        return {"type": types, "phase": phases}

    def _to_qdrant_filter(self, meta: Dict[str, Any]) -> Filter:
        conds = []
        if br := meta.get("body_region"):
            conds.append(FieldCondition(key="body_region", match=MatchAny(any=br)))
        if types := meta.get("type"):
            conds.append(FieldCondition(key="type", match=MatchAny(any=types)))
        if phases := meta.get("phase"):
            conds.append(FieldCondition(key="phase", match=MatchAny(any=phases)))
        return Filter(must=conds)

    def _rerank_voyage(self, query: str, nodes: List[NodeWithScore], rq: RetrievalQuery, top_k: int = 12) -> List[NodeWithScore]:
        if not self._has_voyage or not nodes:
            return nodes[:top_k]
        try:
            pairs = [n.node.get_content() for n in nodes]
            res = self.voyage_client.rerank(
                query=query,
                documents=pairs,
                model="rerank-2-lite",  # cost-effective, adjust if needed
                top_k=min(top_k, len(nodes)),
            )
            # Map back to NodeWithScore order by rerank score
            scored = []
            for item in res.results:
                idx = item.index
                if 0 <= idx < len(nodes):
                    nw = nodes[idx]
                    nw.score = item.relevance_score
                    scored.append(nw)
            return scored
        except Exception:
            return nodes[:top_k]

    def _boost_and_diversify(self, nodes: List[NodeWithScore], rq: RetrievalQuery, k: int = 10) -> List[NodeWithScore]:
        # Simple boosts: red_flag overlap, tag/signal overlap; then naive MMR-ish diversity by text hash bucket
        def jaccard(a: set, b: set) -> float:
            if not a and not b:
                return 0.0
            inter = len(a & b)
            union = len(a | b)
            return inter / union if union else 0.0

        wanted_signals = set(rq.red_flag_signals + rq.symptoms)
        out = []
        for n in nodes:
            payload = n.node.metadata or {}
            base = n.score or 0.0
            ntype = payload.get("type")
            tags = set(payload.get("tags") or [])
            signals = set([k for k, v in (payload.get("signals") or {}).items() if v])
            # boosts
            if ntype == "red_flag" and any(s in wanted_signals for s in signals):
                base += 2.0
            base += 0.5 * jaccard(tags, set(rq.modifiers + rq.context))
            n.score = base
            out.append(n)

        # naive diversity: keep first per deduped normalized text
        seen = set()
        dedup = []
        for n in sorted(out, key=lambda x: (-(x.score or 0), x.node.get_content()[:40])):
            key = (n.node.get_content()[:120].strip().lower())
            if key in seen:
                continue
            seen.add(key)
            dedup.append(n)
            if len(dedup) >= k:
                break
        return dedup

    def retrieve_for_message(self, user_msg: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rq = build_retrieval_object(user_msg, state)
        routing = self._route_filters(rq)
        meta_filter = {"body_region": rq.body_regions, **routing}
        # Dense retrievers
        r_dense_q = self.idx_q.as_retriever(similarity_top_k=self.dense_top_k)
        r_dense_c = self.idx_c.as_retriever(similarity_top_k=self.dense_top_k)

        # Retrieve
        cand_q: List[NodeWithScore] = r_dense_q.retrieve(rq.text)
        # Apply routing/body-region filters
        def _matches_meta(n: NodeWithScore) -> bool:
            m = n.node.metadata or {}
            if m.get("body_region") not in meta_filter.get("body_region", []):
                return False
            if m.get("type") not in meta_filter.get("type", []):
                return False
            if m.get("phase") not in meta_filter.get("phase", []):
                return False
            return True
        cand_q = [n for n in cand_q if _matches_meta(n)]
        cand_c: List[NodeWithScore] = r_dense_c.retrieve(rq.text)
        # Filter context by body_region only
        cand_c = [n for n in cand_c if (n.node.metadata or {}).get("body_region") in rq.body_regions]

        # Rerank questions + selection
        ranked_q = self._rerank_voyage(rq.text, cand_q, rq, top_k=12)
        final_q = self._boost_and_diversify(ranked_q, rq, k=10)

        # Select red flags to pin (top 1–2)
        red_flags = []
        rest_q = []
        for n in final_q:
            if (n.node.metadata or {}).get("type") == "red_flag" and len(red_flags) < 2:
                red_flags.append(n)
            else:
                rest_q.append(n)

        # Map to output contract
        out_red = [
            {
                "id": (n.node.metadata or {}).get("id") or (n.node.id_ if hasattr(n.node, "id_") else None),
                "text": n.node.get_content(),
                "why": rq.red_flag_signals,
                "action": "triage",
            }
            for n in red_flags
        ]

        # Attach follow-ups via metadata graph if present
        questions = []
        followups = []
        for n in rest_q:
            meta = n.node.metadata or {}
            if meta.get("type") == "follow_up":
                continue  # handled as linked followups
            qid = meta.get("id") or (n.node.id_ if hasattr(n.node, "id_") else None)
            questions.append(
                {
                    "id": qid,
                    "text": n.node.get_content(),
                    "next_if_yes": [],  # can be filled via decision graph
                    "tags": meta.get("tags", []),
                }
            )
            for fid in meta.get("related_ids", []):
                followups.append({"for": qid, "id": fid, "text": ""})

        # Causes/explanations
        ranked_c = self._rerank_voyage(rq.text, cand_c, rq, top_k=5)
        likely_causes = []
        for n in ranked_c[:5]:
            meta = n.node.metadata or {}
            likely_causes.append(
                {
                    "id": meta.get("id") or (n.node.id_ if hasattr(n.node, "id_") else None),
                    "label": ", ".join(meta.get("tags", [])[:1]) or "Cause/Explanation",
                    "evidence": rq.modifiers + rq.symptoms,
                    "explain_text": n.node.get_content(),
                }
            )

        return {
            "red_flags": out_red,
            "questions": questions[:10],
            "followups": followups,
            "likely_causes": likely_causes,
            "_debug": {"query": rq.text, "filters": meta_filter},
        }
