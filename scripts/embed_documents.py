#!/usr/bin/env python3
# scripts/embed_documents.py
"""CLI script for embedding documents and managing knowledge bases."""

import argparse
import sys
import logging
from pathlib import Path

# Add app to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.knowledge import KnowledgeManager
from app.core.embeddings import configure_global_embedding
from llama_index.llms.gemini import Gemini


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_knowledge_manager() -> KnowledgeManager:
    """Create and configure knowledge manager."""
    # Configure embeddings
    configure_global_embedding()
    
    # Check embedding model configuration
    import os
    if os.getenv("VOYAGE_API_KEY"):
        print("✓ Using Voyage AI for embeddings (1024-dim)")
    elif os.getenv("OPENAI_API_KEY"):
        print("✓ Using OpenAI for embeddings (1536-dim)")
    else:
        print("⚠ Warning: No embedding API key found (VOYAGE_API_KEY or OPENAI_API_KEY)")
    
    # Initialize LLM for metadata extraction (optional)
    try:
        llm = Gemini(model="models/gemini-2.0-flash")
        print("✓ Using Gemini for metadata extraction")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize LLM: {e}")
        print("  Proceeding without LLM-based metadata extraction")
        llm = None
    
    return KnowledgeManager(llm=llm)


def embed_single_document(args):
    """Embed a single document."""
    km = create_knowledge_manager()
    
    try:
        index = km.process_and_embed_document(
            file_path=args.file,
            body_region=args.body_region,
            knowledge_type=args.knowledge_type,
            force_recreate=args.force
        )
        
        print(f"✓ Successfully embedded document: {args.file}")
        
        # Show some stats
        stats = km.embeddings_pipeline.get_collection_stats(
            body_region=args.body_region or "auto-detected",
            knowledge_type=args.knowledge_type
        )
        if stats:
            print(f"  Collection: {stats['name']}")
            print(f"  Points: {stats['points_count']}")
            print(f"  Status: {stats['status']}")
            
    except Exception as e:
        print(f"✗ Error embedding document: {e}")
        return 1
    
    return 0


def batch_embed_documents(args):
    """Embed multiple documents from a directory."""
    km = create_knowledge_manager()
    
    try:
        results = km.batch_process_documents(
            documents_dir=args.directory,
            knowledge_type=args.knowledge_type,
            file_pattern=args.pattern
        )
        
        if results:
            print(f"✓ Successfully processed {len(results)} documents:")
            for body_region, index in results.items():
                print(f"  - {body_region}")
        else:
            print("⚠ No documents were processed")
            
    except Exception as e:
        print(f"✗ Error in batch processing: {e}")
        return 1
    
    return 0


def list_knowledge_bases(args):
    """List all available knowledge bases."""
    km = create_knowledge_manager()
    
    try:
        knowledge_bases = km.get_available_knowledge_bases()
        
        if not knowledge_bases:
            print("No knowledge bases found")
            return 0
        
        print(f"Found {len(knowledge_bases)} knowledge bases:")
        print("-" * 80)
        print(f"{'Body Region':<15} {'Type':<12} {'Status':<10} {'Points':<8} {'Collection'}")
        print("-" * 80)
        
        for kb in knowledge_bases:
            print(f"{kb['body_region']:<15} {kb['knowledge_type']:<12} {kb['status']:<10} {kb['points_count']:<8} {kb['collection_name']}")
            
    except Exception as e:
        print(f"✗ Error listing knowledge bases: {e}")
        return 1
    
    return 0


def delete_knowledge_base(args):
    """Delete a knowledge base."""
    km = create_knowledge_manager()
    
    # Confirm deletion
    if not args.yes:
        response = input(f"Are you sure you want to delete {args.knowledge_type}_{args.body_region}? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted")
            return 0
    
    try:
        success = km.delete_knowledge_base(
            body_region=args.body_region,
            knowledge_type=args.knowledge_type
        )
        
        if success:
            print(f"✓ Successfully deleted {args.knowledge_type}_{args.body_region}")
        else:
            print(f"✗ Failed to delete {args.knowledge_type}_{args.body_region}")
            return 1
            
    except Exception as e:
        print(f"✗ Error deleting knowledge base: {e}")
        return 1
    
    return 0


def test_retrieval(args):
    """Test retrieval from a knowledge base."""
    km = create_knowledge_manager()
    
    try:
        # Get the knowledge base
        index = km.get_knowledge_base(
            body_region=args.body_region,
            knowledge_type=args.knowledge_type
        )
        
        if not index:
            print(f"✗ No knowledge base found for {args.knowledge_type}_{args.body_region}")
            return 1
        
        # Create query engine
        query_engine = km.create_query_engine(
            body_region=args.body_region,
            knowledge_type=args.knowledge_type
        )
        
        # Test query
        query = args.query or f"What are the main causes of {args.body_region} pain?"
        print(f"Query: {query}")
        print("-" * 50)
        
        response = query_engine.query(query)
        print(response)
        
    except Exception as e:
        print(f"✗ Error testing retrieval: {e}")
        return 1
    
    return 0


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="GoGain Knowledge Base Management")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Embed single document
    embed_parser = subparsers.add_parser("embed", help="Embed a single document")
    embed_parser.add_argument("file", help="Path to document file")
    embed_parser.add_argument("--body-region", help="Body region (auto-detected if not provided)")
    embed_parser.add_argument("--knowledge-type", default="pain", help="Knowledge type (default: pain)")
    embed_parser.add_argument("--force", action="store_true", help="Force recreation if exists")
    embed_parser.set_defaults(func=embed_single_document)
    
    # Batch embed documents
    batch_parser = subparsers.add_parser("batch", help="Batch embed documents from directory")
    batch_parser.add_argument("directory", help="Directory containing documents")
    batch_parser.add_argument("--knowledge-type", default="pain", help="Knowledge type (default: pain)")
    batch_parser.add_argument("--pattern", default="*.md", help="File pattern (default: *.md)")
    batch_parser.set_defaults(func=batch_embed_documents)
    
    # List knowledge bases
    list_parser = subparsers.add_parser("list", help="List all knowledge bases")
    list_parser.set_defaults(func=list_knowledge_bases)
    
    # Delete knowledge base
    delete_parser = subparsers.add_parser("delete", help="Delete a knowledge base")
    delete_parser.add_argument("body_region", help="Body region")
    delete_parser.add_argument("--knowledge-type", default="pain", help="Knowledge type (default: pain)")
    delete_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=delete_knowledge_base)
    
    # Test retrieval
    test_parser = subparsers.add_parser("test", help="Test retrieval from knowledge base")
    test_parser.add_argument("body_region", help="Body region")
    test_parser.add_argument("--knowledge-type", default="pain", help="Knowledge type (default: pain)")
    test_parser.add_argument("--query", help="Test query (default: auto-generated)")
    test_parser.set_defaults(func=test_retrieval)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.verbose)
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main()) 