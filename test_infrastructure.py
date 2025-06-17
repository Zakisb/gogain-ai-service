#!/usr/bin/env python3
# test_infrastructure.py
"""Test basic infrastructure components before building the full system."""

import sys
from pathlib import Path

# Add app to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import qdrant_client
        print("✅ qdrant_client imported successfully")
    except ImportError as e:
        print(f"❌ qdrant_client import failed: {e}")
        return False
    
    try:
        from llama_index.core import VectorStoreIndex
        print("✅ llama_index.core imported successfully")
    except ImportError as e:
        print(f"❌ llama_index.core import failed: {e}")
        return False
        
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
        print("✅ OpenAI embeddings imported successfully")
    except ImportError as e:
        print(f"❌ OpenAI embeddings import failed: {e}")
        return False
        
    try:
        from llama_index.llms.gemini import Gemini
        print("✅ Gemini LLM imported successfully")
    except ImportError as e:
        print(f"❌ Gemini LLM import failed: {e}")
        return False
    
    return True


def test_qdrant_connection():
    """Test connection to Qdrant vector database."""
    print("\n🔍 Testing Qdrant connection...")
    
    try:
        from app.core.qdrant import get_qdrant_client
        client = get_qdrant_client()
        
        # Test basic connection
        collections = client.get_collections()
        print(f"✅ Qdrant connected successfully")
        print(f"   Found {len(collections.collections)} existing collections")
        
        # List existing collections
        if collections.collections:
            print("   Existing collections:")
            for col in collections.collections:
                print(f"     - {col.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("💡 Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        return False


def test_embedding_model():
    """Test embedding model initialization."""
    print("\n🔍 Testing embedding model...")
    
    try:
        from app.core.embeddings import get_embedding_model
        
        embed_model = get_embedding_model()
        print("✅ Embedding model initialized successfully")
        print(f"   Model: {embed_model}")
        
        # Check which API key is being used
        import os
        if os.getenv("VOYAGE_API_KEY"):
            print("   Using: Voyage AI (primary)")
        elif os.getenv("OPENAI_API_KEY"):
            print("   Using: OpenAI (fallback)")
        else:
            print("   Warning: No API keys detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding model initialization failed: {e}")
        print("💡 Make sure VOYAGE_API_KEY or OPENAI_API_KEY is set in your environment")
        return False


def test_document_exists():
    """Test that the shoulder pain document exists."""
    print("\n🔍 Testing document availability...")
    
    doc_path = Path("documents/shoulder_pain_final_report.md")
    
    if doc_path.exists():
        size = doc_path.stat().st_size
        print(f"✅ Document found: {doc_path}")
        print(f"   Size: {size:,} bytes")
        return True
    else:
        print(f"❌ Document not found: {doc_path}")
        print("💡 Make sure the shoulder pain document is in the documents/ directory")
        return False


def main():
    """Run all infrastructure tests."""
    print("🧪 INFRASTRUCTURE TESTING")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_qdrant_connection,
        test_embedding_model,
        test_document_exists,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("📊 RESULTS")
    print("=" * 20)
    print(f"Passed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("🎉 All infrastructure tests passed! Ready to proceed.")
        return 0
    else:
        print("⚠️  Some tests failed. Fix these issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 