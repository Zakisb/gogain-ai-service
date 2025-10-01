"""Document processing pipeline for pain assessment knowledge bases."""

from typing import List, Dict, Optional
from pathlib import Path
import re
import logging

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
        if "«" in original_text and "»" in original_text:
            questions = re.findall(r'«([^»]+)»', original_text)
            if questions:
                node.metadata["contains_questions"] = True
                node.metadata["question_count"] = len(questions)
                node.metadata["extracted_questions"] = questions[:3]
                # Let semantic search handle question type classification

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