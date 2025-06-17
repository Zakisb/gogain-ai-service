#!/usr/bin/env python3
# test_voyage_config.py
"""Test script to verify Voyage AI embedding configuration."""

import sys
import os
from pathlib import Path

# Add app to Python path
sys.path.append(str(Path(__file__).parent))


def test_voyage_config():
    """Test Voyage AI embedding configuration."""
    print("🧪 Testing Voyage AI Configuration")
    print("=" * 40)
    
    # Check environment variables
    print("🔍 Checking environment variables...")
    voyage_key = os.getenv("VOYAGE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if voyage_key:
        print(f"✅ VOYAGE_API_KEY found: {voyage_key[:8]}...")
    else:
        print("❌ VOYAGE_API_KEY not found")
    
    if openai_key:
        print(f"✅ OPENAI_API_KEY found: {openai_key[:8]}...")
    else:
        print("❌ OPENAI_API_KEY not found")
    
    # Test embedding configuration
    print("\n🔍 Testing embedding configuration...")
    try:
        from app.core.embeddings import EmbeddingConfig, get_embedding_model
        
        print(f"✅ Default model: {EmbeddingConfig.DEFAULT_MODEL}")
        print(f"✅ Default vector dimension: {EmbeddingConfig.DEFAULT_VECTOR_DIM}")
        print(f"✅ Voyage input type: {EmbeddingConfig.VOYAGE_INPUT_TYPE}")
        
        # Test model initialization
        print("\n🔍 Testing model initialization...")
        embed_model = get_embedding_model()
        print(f"✅ Embedding model created: {type(embed_model).__name__}")
        
        # Check if it's using Voyage AI
        if hasattr(embed_model, 'model_name'):
            print(f"   Model name: {embed_model.model_name}")
        if hasattr(embed_model, 'input_type'):
            print(f"   Input type: {embed_model.input_type}")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_dimension():
    """Test that vector dimension is correctly configured."""
    print("\n🔍 Testing vector dimension configuration...")
    
    try:
        from app.knowledge.embeddings_pipeline import EmbeddingsPipeline
        
        pipeline = EmbeddingsPipeline()
        print("✅ EmbeddingsPipeline initialized")
        
        # Test collection creation with auto-detected dimensions
        test_collection = "test_voyage_dimensions"
        success = pipeline.create_collection(test_collection)
        
        if success:
            print(f"✅ Test collection '{test_collection}' created with auto-detected dimensions")
            
            # Check collection info
            try:
                info = pipeline.qdrant_client.get_collection(test_collection)
                vector_size = info.config.params.vectors.size
                print(f"✅ Vector size: {vector_size}")
                
                if vector_size == 1024:
                    print("✅ Correct dimension for Voyage AI (1024)")
                elif vector_size == 1536:
                    print("⚠️  Using OpenAI dimensions (1536) - check if Voyage AI is properly configured")
                else:
                    print(f"⚠️  Unexpected vector size: {vector_size}")
                
                # Clean up
                pipeline.qdrant_client.delete_collection(test_collection)
                print(f"🧹 Test collection '{test_collection}' cleaned up")
                
            except Exception as e:
                print(f"⚠️  Could not get collection info: {e}")
        else:
            print("❌ Failed to create test collection")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Vector dimension test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Voyage AI configuration tests."""
    print("🚀 VOYAGE AI CONFIGURATION TEST")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_voyage_config),
        ("Vector Dimensions", test_vector_dimension),
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {name} test PASSED")
        else:
            print(f"❌ {name} test FAILED")
    
    print(f"\n📊 RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All Voyage AI configuration tests passed!")
        print("\n💡 Next steps:")
        print("1. Run: python test_infrastructure.py")
        print("2. Run: python test_components.py")
        print("3. Start embedding documents!")
        return 0
    else:
        print("⚠️  Some tests failed. Check configuration above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 