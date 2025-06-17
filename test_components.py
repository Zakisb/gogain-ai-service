#!/usr/bin/env python3
# test_components.py
"""Test individual knowledge system components in isolation."""

import sys
from pathlib import Path

# Add app to Python path
sys.path.append(str(Path(__file__).parent))


def test_document_processor():
    """Test document processing component."""
    print("🔍 Testing DocumentProcessor...")
    
    try:
        from app.knowledge.document_processor import DocumentProcessor
        
        # Initialize processor
        processor = DocumentProcessor()
        print("✅ DocumentProcessor initialized")
        
        # Test document loading
        doc_path = "documents/shoulder_pain_final_report.md"
        documents = processor.load_document(doc_path)
        print(f"✅ Document loaded: {len(documents)} document(s)")
        
        # Test document processing
        nodes = processor.process_documents(documents)
        print(f"✅ Document processed: {len(nodes)} nodes created")
        
        # Analyze nodes
        hierarchical_nodes = [n for n in nodes if hasattr(n, 'parent_node') and n.parent_node]
        leaf_nodes = [n for n in nodes if not hasattr(n, 'parent_node') or not n.parent_node]
        
        print(f"   - Hierarchical nodes: {len(hierarchical_nodes)}")
        print(f"   - Leaf nodes: {len(leaf_nodes)}")
        
        # Check metadata
        nodes_with_questions = [n for n in nodes if n.metadata.get('contains_questions')]
        nodes_with_red_flags = [n for n in nodes if n.metadata.get('contains_red_flags')]
        nodes_with_exercises = [n for n in nodes if n.metadata.get('contains_exercise_info')]
        
        print(f"   - Nodes with questions: {len(nodes_with_questions)}")
        print(f"   - Nodes with red flags: {len(nodes_with_red_flags)}")
        print(f"   - Nodes with exercise info: {len(nodes_with_exercises)}")
        
        # Show sample node content
        if nodes:
            sample_node = nodes[0]
            content_preview = sample_node.get_content()[:200] + "..."
            print(f"   - Sample content: {content_preview}")
            print(f"   - Sample metadata keys: {list(sample_node.metadata.keys())}")
        
        return True, nodes
        
    except Exception as e:
        print(f"❌ DocumentProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_embeddings_pipeline(test_nodes=None):
    """Test embeddings pipeline component."""
    print("\n🔍 Testing EmbeddingsPipeline...")
    
    try:
        from app.knowledge.embeddings_pipeline import EmbeddingsPipeline
        
        # Initialize pipeline
        pipeline = EmbeddingsPipeline()
        print("✅ EmbeddingsPipeline initialized")
        
        # Test collection creation
        test_collection = "test_pain_shoulder"
        success = pipeline.create_collection(test_collection)
        print(f"✅ Test collection created: {success}")
        
        # List collections
        collections = pipeline.list_collections()
        print(f"✅ Collections listed: {len(collections)} found")
        for col in collections:
            print(f"   - {col}")
        
        # If we have test nodes, try embedding them
        if test_nodes and len(test_nodes) > 0:
            # Use just a few nodes for testing
            sample_nodes = test_nodes[:5]
            print(f"📝 Testing embedding with {len(sample_nodes)} sample nodes...")
            
            index = pipeline.embed_nodes(sample_nodes, "test_shoulder", "pain")
            print("✅ Nodes embedded successfully")
            
            # Test loading the index back
            loaded_index = pipeline.load_existing_index("test_shoulder", "pain")
            if loaded_index:
                print("✅ Index loaded successfully")
            else:
                print("⚠️  Index loading failed")
            
            # Get stats
            stats = pipeline.get_collection_stats("test_shoulder", "pain")
            if stats:
                print(f"✅ Collection stats: {stats['points_count']} points")
            
            # Clean up test collection
            cleanup = pipeline.delete_collection("test_shoulder", "pain")
            print(f"🧹 Test collection cleanup: {cleanup}")
            
            return True, index
        else:
            print("⚠️  No test nodes provided, skipping embedding test")
            return True, None
        
    except Exception as e:
        print(f"❌ EmbeddingsPipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_knowledge_manager():
    """Test knowledge manager component."""
    print("\n🔍 Testing KnowledgeManager...")
    
    try:
        from app.knowledge.knowledge_manager import KnowledgeManager
        
        # Initialize knowledge manager
        km = KnowledgeManager()
        print("✅ KnowledgeManager initialized")
        
        # Test available knowledge bases
        available = km.get_available_knowledge_bases()
        print(f"✅ Available knowledge bases: {len(available)}")
        for kb in available:
            print(f"   - {kb['body_region']} ({kb['knowledge_type']}): {kb['points_count']} points")
        
        # Test stats
        stats = km.get_knowledge_base_stats()
        print(f"✅ Knowledge base stats retrieved: {len(stats)} entries")
        
        return True
        
    except Exception as e:
        print(f"❌ KnowledgeManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_processing():
    """Test complete document processing pipeline."""
    print("\n🔍 Testing End-to-End Processing...")
    
    try:
        from app.knowledge.knowledge_manager import KnowledgeManager
        
        km = KnowledgeManager()
        
        # Process document with a unique name for testing
        doc_path = "documents/shoulder_pain_final_report.md"
        print(f"📝 Processing document: {doc_path}")
        
        index = km.process_and_embed_document(
            file_path=doc_path,
            body_region="test_shoulder_e2e",
            knowledge_type="pain",
            force_recreate=True
        )
        
        print("✅ End-to-end processing successful")
        
        # Test retrieval
        query_engine = km.create_query_engine("test_shoulder_e2e", "pain")
        test_query = "What are the main causes of shoulder pain?"
        
        print(f"🔍 Testing query: {test_query}")
        response = query_engine.query(test_query)
        
        print(f"✅ Query successful")
        print(f"   Response length: {len(str(response))} characters")
        print(f"   Response preview: {str(response)[:300]}...")
        
        # Clean up
        cleanup = km.delete_knowledge_base("test_shoulder_e2e", "pain")
        print(f"🧹 Cleanup successful: {cleanup}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all component tests."""
    print("🧪 COMPONENT TESTING")
    print("=" * 50)
    
    # Test 1: Document Processor
    processor_ok, test_nodes = test_document_processor()
    
    # Test 2: Embeddings Pipeline
    embeddings_ok, test_index = test_embeddings_pipeline(test_nodes)
    
    # Test 3: Knowledge Manager
    manager_ok = test_knowledge_manager()
    
    # Test 4: End-to-End
    e2e_ok = test_end_to_end_processing()
    
    # Results
    tests = [
        ("DocumentProcessor", processor_ok),
        ("EmbeddingsPipeline", embeddings_ok), 
        ("KnowledgeManager", manager_ok),
        ("End-to-End", e2e_ok),
    ]
    
    passed = sum(1 for _, ok in tests if ok)
    
    print("\n📊 COMPONENT TEST RESULTS")
    print("=" * 30)
    for name, ok in tests:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{name:<20} {status}")
    
    print(f"\nPassed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("🎉 All component tests passed! Ready for agent integration.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 