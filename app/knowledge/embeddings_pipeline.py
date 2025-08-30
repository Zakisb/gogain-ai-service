"""Embeddings pipeline for creating and managing vector stores."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.qdrant import get_qdrant_client
from app.core.embeddings import get_embedding_model, configure_global_embedding, EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingsPipeline:
    """Pipeline for creating and managing embeddings and vector stores."""
    
    def __init__(self, collection_prefix: str = "gogain"):
        """Initialize the embeddings pipeline.
        
        Args:
            collection_prefix: Prefix for Qdrant collection names
        """
        self.collection_prefix = collection_prefix
        self.qdrant_client = get_qdrant_client()
        self.embed_model = get_embedding_model()
        configure_global_embedding()
        
    def _get_collection_name(self, body_region: str, knowledge_type: str = "pain") -> str:
        """Generate collection name for a specific body region and knowledge type.
        
        Args:
            body_region: Body region (e.g., 'shoulder', 'knee')
            knowledge_type: Type of knowledge (e.g., 'pain', 'training', 'nutrition')
            
        Returns:
            Collection name
        """
        return f"{self.collection_prefix}_{knowledge_type}_{body_region}"
    
    def create_collection(self, collection_name: str, vector_size: Optional[int] = None) -> bool:
        """Create a new Qdrant collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of embedding vectors (auto-detected from config if not provided)
            
        Returns:
            True if successful
        """
        # Use configured vector dimension if not specified
        if vector_size is None:
            vector_size = EmbeddingConfig.DEFAULT_VECTOR_DIM
        try:
            # Check if collection already exists
            collections = self.qdrant_client.get_collections().collections
            existing_names = [col.name for col in collections]
            
            if collection_name in existing_names:
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Create new collection
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"Created collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            return False
    
    def create_vector_store(self, collection_name: str) -> QdrantVectorStore:
        """Create a QdrantVectorStore for the specified collection.
        
        Args:
            collection_name: Name of the Qdrant collection
            
        Returns:
            QdrantVectorStore instance
        """
        return QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name
        )
    
    def embed_nodes(self, 
                   nodes: List[BaseNode], 
                   body_region: str,
                   knowledge_type: str = "pain") -> VectorStoreIndex:
        """Embed nodes and create vector index.
        
        Args:
            nodes: List of nodes to embed
            body_region: Body region identifier
            knowledge_type: Type of knowledge
            
        Returns:
            VectorStoreIndex
        """
        collection_name = self._get_collection_name(body_region, knowledge_type)
        
        # Create collection if it doesn't exist
        self.create_collection(collection_name)
        
        # Create vector store
        vector_store = self.create_vector_store(collection_name)
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create and return index
        logger.info(f"Embedding {len(nodes)} nodes for {body_region} {knowledge_type}")
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
        logger.info(f"Successfully created index for {collection_name}")
        return index
    
    def load_existing_index(self, 
                          body_region: str, 
                          knowledge_type: str = "pain") -> Optional[VectorStoreIndex]:
        """Load an existing vector index.
        
        Args:
            body_region: Body region identifier
            knowledge_type: Type of knowledge
            
        Returns:
            VectorStoreIndex if exists, None otherwise
        """
        collection_name = self._get_collection_name(body_region, knowledge_type)
        
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            existing_names = [col.name for col in collections]
            
            if collection_name not in existing_names:
                logger.warning(f"Collection {collection_name} does not exist")
                return None
            
            # Create vector store and load index
            vector_store = self.create_vector_store(collection_name)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            logger.info(f"Loaded existing index for {collection_name}")
            return index
            
        except Exception as e:
            logger.error(f"Error loading index for {collection_name}: {e}")
            return None
    
    def get_or_create_index(self, 
                          nodes: List[BaseNode], 
                          body_region: str,
                          knowledge_type: str = "pain",
                          force_recreate: bool = False) -> VectorStoreIndex:
        """Get existing index or create new one.
        
        Args:
            nodes: Nodes to embed (used if creating new)
            body_region: Body region identifier
            knowledge_type: Type of knowledge
            force_recreate: Force recreation even if exists
            
        Returns:
            VectorStoreIndex
        """
        if not force_recreate:
            existing_index = self.load_existing_index(body_region, knowledge_type)
            if existing_index:
                return existing_index
        
        # Create new index
        return self.embed_nodes(nodes, body_region, knowledge_type)
    
    def delete_collection(self, body_region: str, knowledge_type: str = "pain") -> bool:
        """Delete a collection.
        
        Args:
            body_region: Body region identifier
            knowledge_type: Type of knowledge
            
        Returns:
            True if successful
        """
        collection_name = self._get_collection_name(body_region, knowledge_type)
        
        try:
            self.qdrant_client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections with the configured prefix.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.qdrant_client.get_collections().collections
            filtered_collections = [
                col.name for col in collections 
                if col.name.startswith(self.collection_prefix)
            ]
            return filtered_collections
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def get_collection_stats(self, body_region: str, knowledge_type: str = "pain") -> Optional[Dict[str, Any]]:
        """Get statistics for a collection.
        
        Args:
            body_region: Body region identifier
            knowledge_type: Type of knowledge
            
        Returns:
            Collection statistics
        """
        collection_name = self._get_collection_name(body_region, knowledge_type)
        
        try:
            info = self.qdrant_client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Error getting stats for {collection_name}: {e}")
            return None 