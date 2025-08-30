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
        self.metadata_extractors = [
            TitleExtractor(nodes=5),
            QuestionsAnsweredExtractor(questions=3),
            KeywordExtractor(keywords=10),
            SummaryExtractor(summaries=["prev", "self"]),
        ]
        
    
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
        """Add custom metadata for pain assessment context.
        
        Args:
            node: Node to add metadata to
        """
        text = node.get_content().lower()
        
        # Identify question nodes
        if "«" in text and "»" in text:
            node.metadata["contains_questions"] = True
            questions = re.findall(r'«([^»]+)»', text)
            node.metadata["question_count"] = len(questions)
        
        # Identify red flag sections
        if any(term in text for term in ["urgence", "médical", "consultation", "fracture", "luxation"]):
            node.metadata["contains_red_flags"] = True
        
        # Identify exercise criteria sections
        if any(term in text for term in ["exercice", "renforcement", "étirement", "mobilisation"]):
            node.metadata["contains_exercise_info"] = True
        
        # Identify assessment criteria
        if any(term in text for term in ["évaluation", "diagnostic", "symptôme", "douleur"]):
            node.metadata["contains_assessment_info"] = True
    
    def process_file(self, file_path: str) -> List[BaseNode]:
        """Complete processing pipeline for a single file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed nodes
        """
        documents = self.load_document(file_path)
        return self.process_documents(documents) 