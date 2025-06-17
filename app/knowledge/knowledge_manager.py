"""High-level knowledge management for pain assessment and training guidance."""

from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode

from .document_processor import DocumentProcessor
from .embeddings_pipeline import EmbeddingsPipeline

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """High-level manager for all knowledge bases in the GoGain system."""
    
    def __init__(self, llm=None, collection_prefix: str = "gogain"):
        """Initialize the knowledge manager.
        
        Args:
            llm: Optional LLM for metadata extraction
            collection_prefix: Prefix for vector store collections
        """
        self.llm = llm
        self.document_processor = DocumentProcessor(llm=llm)
        self.embeddings_pipeline = EmbeddingsPipeline(collection_prefix=collection_prefix)
        self._indices_cache: Dict[str, VectorStoreIndex] = {}
    
    def process_and_embed_document(self, 
                                 file_path: str,
                                 body_region: Optional[str] = None,
                                 knowledge_type: str = "pain",
                                 force_recreate: bool = False) -> VectorStoreIndex:
        """Complete pipeline: process document and create embeddings.
        
        Args:
            file_path: Path to the document file
            body_region: Body region (auto-detected if not provided)
            knowledge_type: Type of knowledge (pain, training, nutrition)
            force_recreate: Force recreation even if exists
            
        Returns:
            VectorStoreIndex for the processed document
        """
        logger.info(f"Processing document: {file_path}")
        
        # Process document into nodes
        nodes = self.document_processor.process_file(file_path)
        
        # Extract body region if not provided
        if not body_region:
            if nodes and "body_region" in nodes[0].metadata:
                body_region = nodes[0].metadata["body_region"]
            else:
                body_region = self.document_processor._extract_body_region_from_filename(file_path)
        
        logger.info(f"Identified body region: {body_region}")
        
        # Create embeddings and index
        index = self.embeddings_pipeline.get_or_create_index(
            nodes=nodes,
            body_region=body_region,
            knowledge_type=knowledge_type,
            force_recreate=force_recreate
        )
        
        # Cache the index
        cache_key = f"{knowledge_type}_{body_region}"
        self._indices_cache[cache_key] = index
        
        logger.info(f"Successfully processed and embedded document for {body_region} {knowledge_type}")
        return index
    
    def get_knowledge_base(self, 
                          body_region: str, 
                          knowledge_type: str = "pain") -> Optional[VectorStoreIndex]:
        """Get knowledge base for a specific body region and type.
        
        Args:
            body_region: Body region identifier
            knowledge_type: Type of knowledge
            
        Returns:
            VectorStoreIndex if available
        """
        cache_key = f"{knowledge_type}_{body_region}"
        
        # Check cache first
        if cache_key in self._indices_cache:
            return self._indices_cache[cache_key]
        
        # Try to load from vector store
        index = self.embeddings_pipeline.load_existing_index(body_region, knowledge_type)
        if index:
            self._indices_cache[cache_key] = index
        
        return index
    
    def create_retriever(self, 
                       body_region: str,
                       knowledge_type: str = "pain",
                       similarity_top_k: int = 5,
                       **kwargs):
        """Create a retriever for a specific knowledge base.
        
        Args:
            body_region: Body region identifier
            knowledge_type: Type of knowledge
            similarity_top_k: Number of similar chunks to retrieve
            **kwargs: Additional retriever arguments
            
        Returns:
            Retriever instance
        """
        index = self.get_knowledge_base(body_region, knowledge_type)
        if not index:
            raise ValueError(f"No knowledge base found for {body_region} {knowledge_type}")
        
        return index.as_retriever(
            similarity_top_k=similarity_top_k,
            **kwargs
        )
    
    def create_query_engine(self, 
                          body_region: str,
                          knowledge_type: str = "pain",
                          response_mode: str = "tree_summarize",
                          **kwargs):
        """Create a query engine for a specific knowledge base.
        
        Args:
            body_region: Body region identifier
            knowledge_type: Type of knowledge
            response_mode: Response mode for the query engine
            **kwargs: Additional query engine arguments
            
        Returns:
            Query engine instance
        """
        index = self.get_knowledge_base(body_region, knowledge_type)
        if not index:
            raise ValueError(f"No knowledge base found for {body_region} {knowledge_type}")
        
        return index.as_query_engine(
            response_mode=response_mode,
            **kwargs
        )
    
    def batch_process_documents(self, 
                              documents_dir: str,
                              knowledge_type: str = "pain",
                              file_pattern: str = "*.md") -> Dict[str, VectorStoreIndex]:
        """Process multiple documents in batch.
        
        Args:
            documents_dir: Directory containing documents
            knowledge_type: Type of knowledge
            file_pattern: File pattern to match
            
        Returns:
            Dictionary mapping body regions to indices
        """
        documents_path = Path(documents_dir)
        files = list(documents_path.glob(file_pattern))
        
        if not files:
            logger.warning(f"No files found matching {file_pattern} in {documents_dir}")
            return {}
        
        results = {}
        
        for file_path in files:
            try:
                logger.info(f"Processing {file_path}")
                index = self.process_and_embed_document(
                    str(file_path),
                    knowledge_type=knowledge_type
                )
                
                # Extract body region for results
                body_region = self.document_processor._extract_body_region_from_filename(str(file_path))
                results[body_region] = index
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Batch processing complete: {len(results)} documents processed")
        return results
    
    def get_available_knowledge_bases(self) -> List[Dict[str, str]]:
        """Get list of available knowledge bases.
        
        Returns:
            List of knowledge base information
        """
        collections = self.embeddings_pipeline.list_collections()
        
        knowledge_bases = []
        for collection in collections:
            # Parse collection name: prefix_knowledge_type_body_region
            parts = collection.split('_')
            if len(parts) >= 3:
                knowledge_type = parts[1]
                body_region = '_'.join(parts[2:])  # Handle multi-word regions
                
                stats = self.embeddings_pipeline.get_collection_stats(body_region, knowledge_type)
                knowledge_bases.append({
                    "body_region": body_region,
                    "knowledge_type": knowledge_type,
                    "collection_name": collection,
                    "status": stats.get("status") if stats else "unknown",
                    "points_count": stats.get("points_count", 0) if stats else 0,
                })
        
        return knowledge_bases
    
    def delete_knowledge_base(self, body_region: str, knowledge_type: str = "pain") -> bool:
        """Delete a knowledge base.
        
        Args:
            body_region: Body region identifier
            knowledge_type: Type of knowledge
            
        Returns:
            True if successful
        """
        # Remove from cache
        cache_key = f"{knowledge_type}_{body_region}"
        if cache_key in self._indices_cache:
            del self._indices_cache[cache_key]
        
        # Delete from vector store
        return self.embeddings_pipeline.delete_collection(body_region, knowledge_type)
    
    def clear_cache(self):
        """Clear the indices cache."""
        self._indices_cache.clear()
        logger.info("Knowledge base cache cleared")
    
    def get_knowledge_base_stats(self) -> Dict[str, Dict]:
        """Get statistics for all knowledge bases.
        
        Returns:
            Statistics for each knowledge base
        """
        knowledge_bases = self.get_available_knowledge_bases()
        stats = {}
        
        for kb in knowledge_bases:
            key = f"{kb['knowledge_type']}_{kb['body_region']}"
            stats[key] = {
                "body_region": kb["body_region"],
                "knowledge_type": kb["knowledge_type"], 
                "status": kb["status"],
                "points_count": kb["points_count"],
            }
        
        return stats 