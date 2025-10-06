"""Document processing pipeline for pain assessment knowledge bases."""

from typing import List, Dict, Optional, Iterable
from pathlib import Path
import re
import logging
import math

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser, 
    get_leaf_nodes,
    SentenceSplitter
)
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor, 
    SummaryExtractor,
    KeywordExtractor,
)
from llama_index.core.schema import BaseNode

from app.core.embeddings import EmbeddingConfig
from app.core.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes medical/fitness documents for pain assessment knowledge bases."""
    
    def __init__(self, llm=None):
        """Initialize the document processor.
        
        Args:
            llm: Optional LLM for metadata extraction
        """
        self.llm = llm
        self._setup_parsers()
        self._setup_extractors()
        self._setup_quote_classifier()

    def _setup_quote_classifier(self) -> None:
        """Initialize semantic quote classifier using embedding centroids.

        Falls back gracefully if an embedding model cannot be created.
        """
        try:
            embed_model = get_embedding_model()
            self._quote_classifier = _QuoteClassifier(embed_model)
            logger.info("Quote classifier initialized with embedding-based centroids")
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Quote classifier disabled: %s", exc)
            self._quote_classifier = None
    
    def _setup_parsers(self):
        """Setup hierarchical and sentence parsers."""
        self.hierarchical_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=EmbeddingConfig.HIERARCHICAL_CHUNK_SIZES,
            chunk_overlap=EmbeddingConfig.HIERARCHICAL_OVERLAP,
        )
        
        self.sentence_splitter = SentenceSplitter(
            chunk_size=EmbeddingConfig.DEFAULT_CHUNK_SIZE,
            chunk_overlap=EmbeddingConfig.DEFAULT_CHUNK_OVERLAP,
        )
    
    def _setup_extractors(self):
        """Setup metadata extractors."""
        if self.llm is not None:
            self.metadata_extractors = [
                TitleExtractor(nodes=5, llm=self.llm),
                QuestionsAnsweredExtractor(questions=3, llm=self.llm),
                KeywordExtractor(keywords=10, llm=self.llm),
                SummaryExtractor(summaries=["prev", "self"], llm=self.llm),
            ]
        else:
            # Skip LLM-based extractors when no LLM is provided
            self.metadata_extractors = []
        
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document from file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        try:
            documents = SimpleDirectoryReader(
                input_files=[file_path]
            ).load_data()
            
            # Add metadata about the document source
            for doc in documents:
                doc.metadata.update({
                    "source_file": Path(file_path).name,
                    "document_type": "pain_assessment",
                    "body_region": self._extract_body_region_from_filename(file_path),
                })
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def _extract_body_region_from_filename(self, file_path: str) -> str:
        """Extract body region from filename.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Body region name
        """
        filename = Path(file_path).stem.lower()
        
        # Common body regions mapping
        region_map = {
            "shoulder": "shoulder",
            "knee": "knee", 
            "back": "back",
            "ankle": "ankle",
            "hip": "hip",
            "neck": "neck",
            "elbow": "elbow",
            "wrist": "wrist",
        }
        
        for key, region in region_map.items():
            if key in filename:
                return region
        
        return "general"
    
    def process_documents(self, documents: List[Document]) -> List[BaseNode]:
        """Process documents into hierarchical nodes with metadata.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed nodes with metadata
        """
        logger.info(f"Processing {len(documents)} documents...")
        
        # 1. Apply hierarchical parsing
        hierarchical_nodes = self.hierarchical_parser.get_nodes_from_documents(documents)
        logger.info(f"Created {len(hierarchical_nodes)} hierarchical nodes")
        
        # 2. Get leaf nodes for additional processing
        leaf_nodes = get_leaf_nodes(hierarchical_nodes)
        logger.info(f"Extracted {len(leaf_nodes)} leaf nodes")
        
        # 3. Apply metadata extraction to leaf nodes
        processed_nodes = leaf_nodes
        for extractor in self.metadata_extractors:
            try:
                processed_nodes = extractor.process_nodes(processed_nodes)
                logger.info(f"Applied {extractor.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Error with {extractor.__class__.__name__}: {e}")
                continue
        
        # 4. Combine hierarchical and processed leaf nodes
        all_nodes = hierarchical_nodes + processed_nodes
        
        # 5. Add custom metadata for pain assessment
        for node in all_nodes:
            self._add_pain_assessment_metadata(node)
        
        logger.info(f"Final processing complete: {len(all_nodes)} total nodes")
        return all_nodes
    
    def _add_pain_assessment_metadata(self, node: BaseNode):
        """Add minimal metadata for pure semantic retrieval - NO KEYWORD MATCHING.

        Args:
            node: Node to add metadata to
        """
        original_text = node.get_content()

        # Only detect French questions structurally (no keyword classification)
        quoted_segments = re.findall(r'«([^»]+)»', original_text) if ("«" in original_text and "»" in original_text) else []

        question_quotes: List[str] = []
        red_flag_quotes: List[str] = []
        patient_quotes: List[str] = []

        if quoted_segments:
            if self._quote_classifier is not None:
                labels = self._quote_classifier.classify_batch(quoted_segments)
            else:
                labels = [_QuoteClassifier.LABEL_QUESTION] * len(quoted_segments)

            for segment, label in zip(quoted_segments, labels):
                normalized = segment.strip()
                if not normalized:
                    continue
                if label == _QuoteClassifier.LABEL_QUESTION:
                    question_quotes.append(normalized)
                elif label == _QuoteClassifier.LABEL_RED_FLAG:
                    red_flag_quotes.append(normalized)
                else:
                    patient_quotes.append(normalized)

        if question_quotes:
            node.metadata["assessment_questions"] = question_quotes
            node.metadata["extracted_questions"] = question_quotes
            node.metadata["contains_questions"] = True
            node.metadata["question_count"] = len(question_quotes)
        else:
            node.metadata["assessment_questions"] = []
            node.metadata["extracted_questions"] = []
            node.metadata["contains_questions"] = False
            node.metadata["question_count"] = 0

        if red_flag_quotes:
            node.metadata["red_flag_examples"] = red_flag_quotes
            node.metadata["contains_red_flags"] = True
        else:
            node.metadata["red_flag_examples"] = []
            node.metadata["contains_red_flags"] = False

        if patient_quotes:
            node.metadata["patient_statement_examples"] = patient_quotes
        else:
            node.metadata["patient_statement_examples"] = []

        if quoted_segments:
            node.metadata["quoted_segment_count"] = len(quoted_segments)

        # Add basic structural information only
        node.metadata["chunk_length"] = len(original_text)
        node.metadata["has_structured_content"] = bool(re.search(r'^#+\s', original_text, re.MULTILINE))

        # Remove all keyword-based classification - let embeddings handle semantic understanding
    
    def process_file(self, file_path: str) -> List[BaseNode]:
        """Complete processing pipeline for a single file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed nodes
        """
        documents = self.load_document(file_path)
        return self.process_documents(documents)


class _QuoteClassifier:
    """Embedding-based semantic classifier for quoted spans."""

    LABEL_QUESTION = "assessment_question"
    LABEL_RED_FLAG = "red_flag_example"
    LABEL_PATIENT_STATEMENT = "patient_statement"
    LABEL_AMBIGUOUS = "ambiguous"

    _MIN_CONFIDENCE = 0.25
    _MARGIN = 0.07

    def __init__(self, embed_model) -> None:
        self._embed_model = embed_model
        self._centroids = self._build_centroids()

    def classify_batch(self, texts: Iterable[str]) -> List[str]:
        return [self._classify_single(text) for text in texts]

    def _build_centroids(self) -> Dict[str, List[float]]:
        prototypes: Dict[str, List[str]] = {
            self.LABEL_QUESTION: [
                "Depuis combien de temps as-tu cette douleur ?",
                "La douleur est-elle constante ou par épisodes ?",
                "Qu'est-ce qui aggrave ou soulage ta douleur ?",
                "As-tu déjà consulté pour ce problème ?",
            ],
            self.LABEL_RED_FLAG: [
                "Je ne peux plus du tout me redresser.",
                "Je suis resté bloqué, impossible de bouger pendant plusieurs heures.",
                "J'ai du mal à contrôler mes urines depuis que la douleur est apparue.",
                "Je ne sens plus bien ma selle.",
            ],
            self.LABEL_PATIENT_STATEMENT: [
                "La douleur descend dans ma jambe quand je marche.",
                "J'ai mal surtout le matin au réveil.",
                "Je ressens un coup de poignard quand je fais un squat.",
                "Le repos me soulage un peu mais la douleur revient dès que je bouge.",
            ],
        }

        centroids: Dict[str, List[float]] = {}
        for label, examples in prototypes.items():
            vectors = [self._embed(example) for example in examples]
            centroids[label] = self._mean_vector(vectors)
        return centroids

    def _classify_single(self, text: str) -> str:
        vec = self._embed(text)
        if not vec:
            return self.LABEL_AMBIGUOUS

        scores: Dict[str, float] = {}
        for label, centroid in self._centroids.items():
            scores[label] = self._cosine_similarity(vec, centroid)

        best_label, best_score = max(scores.items(), key=lambda item: item[1])
        sorted_scores = sorted(scores.values(), reverse=True)
        second_best = sorted_scores[1] if len(sorted_scores) > 1 else -1.0

        if best_score < self._MIN_CONFIDENCE or (best_score - second_best) < self._MARGIN:
            return self.LABEL_AMBIGUOUS

        return best_label

    def _embed(self, text: str) -> List[float]:
        return self._embed_model.get_text_embedding(text)

    @staticmethod
    def _mean_vector(vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return []
        length = len(vectors[0])
        accumulator = [0.0] * length
        for vec in vectors:
            for idx, value in enumerate(vec):
                accumulator[idx] += value
        return [value / len(vectors) for value in accumulator]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(a, b):
            dot += x * y
            norm_a += x * x
            norm_b += y * y
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
