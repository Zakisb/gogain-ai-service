"""Embedding configuration and models for the GoGain AI service."""

from typing import Optional
from llama_index.core import Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
# from llama_index.embeddings.openai import OpenAIEmbedding
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EmbeddingConfig:
    """Configuration for embedding models."""
    
    # Default embedding model settings - using Voyage AI
    DEFAULT_MODEL = "voyage-3-large"  # Voyage AI's 1024-dim generalist model
    DEFAULT_VECTOR_DIM = 1024  # Voyage AI vector dimension
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50
    
    # Hierarchical parsing settings
    HIERARCHICAL_CHUNK_SIZES = [2048, 512, 128]
    HIERARCHICAL_OVERLAP = 20
    
    # Voyage AI specific settings
    VOYAGE_INPUT_TYPE = "document"  # For indexing corpus documents


def get_embedding_model(model_name: Optional[str] = None) -> object:
    """Get the configured embedding model.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        Embedding model instance
    """
    model_name = model_name or EmbeddingConfig.DEFAULT_MODEL
    
    # Check for Voyage AI API key first (primary choice)
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    
    if model_name.startswith("voyage-") and voyage_api_key:
        # Voyage AI models
        return VoyageEmbedding(
            voyage_api_key=voyage_api_key,
            model_name=model_name,
            input_type=EmbeddingConfig.VOYAGE_INPUT_TYPE
        )
    elif model_name.startswith("text-embedding"):
        # OpenAI models (fallback) - temporarily disabled
        raise ValueError("OpenAI embedding support temporarily disabled")
    else:
        # Default behavior: try Voyage AI first, then fallback to OpenAI
        if voyage_api_key:
            return VoyageEmbedding(
                voyage_api_key=voyage_api_key,
                model_name=EmbeddingConfig.DEFAULT_MODEL,
                input_type=EmbeddingConfig.VOYAGE_INPUT_TYPE
            )
        else:
            # Fallback disabled - only support Voyage AI for now
            raise ValueError("VOYAGE_API_KEY required for embedding functionality")


def configure_global_embedding():
    """Configure global LlamaIndex embedding settings."""
    Settings.embed_model = get_embedding_model() 