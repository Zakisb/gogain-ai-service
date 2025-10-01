# app/knowledge/__init__.py
"""Knowledge base management for pain assessment and training guidance."""

from .document_processor import DocumentProcessor
# Removed embeddings_pipeline - functionality migrated to other modules

__all__ = [
    "DocumentProcessor",
] 