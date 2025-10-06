#!/usr/bin/env python3
"""Enhanced hierarchical embedding with medical-aware metadata."""

import sys
import os
import glob
from pathlib import Path
sys.path.append(os.path.dirname(__file__))

from app.knowledge.document_processor import DocumentProcessor
#from app.knowledge.embeddings_pipeline import EmbeddingsPipeline
from app.core.embeddings import configure_global_embedding
# Skip LLM imports - will work without Gemini
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import json

def embed_with_enhanced_hierarchical():
    """Embed pain documents using enhanced hierarchical chunker."""

    print("🏗️ Enhanced Hierarchical Embedding with Medical-Aware Metadata")
    print("=" * 70)

    # Configure embeddings
    configure_global_embedding()

    # Skip LLM for now - focus on medical-aware metadata
    llm = None
    print("✓ Using medical-aware metadata extraction (no LLM needed)")

    # Initialize enhanced document processor
    doc_processor = DocumentProcessor(llm=llm)

    # Initialize Qdrant client
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    # Pain documents to process
    pain_documents = [
        "documents/lower_back_pain_final_report.md",
        "documents/knees_pain_final_report.md",
        "documents/shoulder_pain_final_report.md",
        "documents/neck_pain_final_report.md",
        "documents/foot_ankle_final_pain_report.md",
        "documents/hips_final_pain_report.md",
        "documents/Pain report.md"
    ]

    # Body region mapping
    body_region_map = {
        "lower_back_pain_final_report.md": "back",
        "knees_pain_final_report.md": "knee",
        "shoulder_pain_final_report.md": "shoulder",
        "neck_pain_final_report.md": "neck",
        "foot_ankle_final_pain_report.md": "foot_ankle",
        "hips_final_pain_report.md": "hip",
        "Pain report.md": "general"
    }

    total_stats = {
        "total_nodes": 0,
        "question_nodes": 0,
        "red_flag_nodes": 0,
        "cause_nodes": 0,
        "documents_processed": 0,
        "collections_created": []
    }

    # Create main collection for hierarchical chunks
    collection_name = "pain_hierarchical_chunks_v2"

    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text-dense": VectorParams(size=1024, distance=Distance.COSINE)
            },
        )
        print(f"✓ Created collection: {collection_name}")
        total_stats["collections_created"].append(collection_name)
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"⚠️ Collection {collection_name} already exists – recreating to ensure vector configuration")
            qdrant_client.delete_collection(collection_name)
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text-dense": VectorParams(size=1024, distance=Distance.COSINE)
                },
            )
            print(f"✓ Recreated collection: {collection_name}")
            total_stats["collections_created"].append(collection_name)
        else:
            print(f"❌ Error creating collection: {e}")
            return

    # Create vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        vector_name="text-dense"
    )

    # Process each document
    all_nodes = []

    for doc_path in pain_documents:
        if not os.path.exists(doc_path):
            print(f"⚠️ Document not found: {doc_path}")
            continue

        filename = os.path.basename(doc_path)
        body_region = body_region_map.get(filename, "unknown")

        print(f"\n📚 Processing: {filename} -> {body_region}")
        print("-" * 50)

        try:
            # Process document with enhanced hierarchical chunker
            nodes = doc_processor.process_file(doc_path)

            # Add body region to all nodes
            for node in nodes:
                node.metadata["body_region"] = body_region
                node.metadata["source_document"] = filename

            print(f"✅ Created {len(nodes)} hierarchical nodes:")

            # Analyze node types
            question_count = sum(1 for n in nodes if n.metadata.get("contains_questions"))
            red_flag_count = sum(1 for n in nodes if n.metadata.get("contains_red_flags"))
            cause_count = sum(1 for n in nodes if n.metadata.get("contains_causes"))
            exercise_count = sum(1 for n in nodes if n.metadata.get("contains_exercise_info"))

            print(f"   Question nodes: {question_count}")
            print(f"   Red flag nodes: {red_flag_count}")
            print(f"   Cause nodes: {cause_count}")
            print(f"   Exercise nodes: {exercise_count}")

            # Update totals
            total_stats["total_nodes"] += len(nodes)
            total_stats["question_nodes"] += question_count
            total_stats["red_flag_nodes"] += red_flag_count
            total_stats["cause_nodes"] += cause_count
            total_stats["documents_processed"] += 1

            all_nodes.extend(nodes)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue

    if not all_nodes:
        print("❌ No nodes to embed!")
        return

    print(f"\n🔗 Embedding {len(all_nodes)} nodes into Qdrant...")
    print("-" * 50)

    # Create storage context and embed all nodes
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        # Create vector index with all nodes
        index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context
        )

        print(f"✅ Successfully embedded all nodes!")

    except Exception as e:
        print(f"❌ Error during embedding: {e}")
        return

    print(f"\n🎉 HIERARCHICAL EMBEDDING COMPLETE!")
    print("=" * 70)
    print(f"Documents processed: {total_stats['documents_processed']}/7")
    print(f"Total nodes: {total_stats['total_nodes']}")
    print(f"Question-rich nodes: {total_stats['question_nodes']}")
    print(f"Red flag nodes: {total_stats['red_flag_nodes']}")
    print(f"Cause nodes: {total_stats['cause_nodes']}")

    # Verify collection
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"\n🗄️ Collection Status:")
        print(f"   Name: {collection_name}")
        print(f"   Points: {collection_info.points_count}")
        print(f"   Status: {collection_info.status}")
    except Exception as e:
        print(f"⚠️ Could not get collection info: {e}")

    # Test retrieval
    print(f"\n🧪 Quick Test - Enhanced Hierarchical Retrieval:")
    try:
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query("lower back pain assessment questions")

        print(f"Query: 'lower back pain assessment questions'")
        print(f"Response: {str(response)[:200]}...")

        # Show metadata from source nodes
        if hasattr(response, 'source_nodes') and response.source_nodes:
            print(f"\nSource Node Metadata Examples:")
            for i, node in enumerate(response.source_nodes[:2], 1):
                metadata = node.metadata
                print(f"  Node {i}:")
                print(f"    Contains questions: {metadata.get('contains_questions', False)}")
                print(f"    Question count: {metadata.get('question_count', 0)}")
                print(f"    Clinical priority: {metadata.get('clinical_priority', 'unknown')}")
                print(f"    Body region: {metadata.get('body_region', 'unknown')}")

    except Exception as e:
        print(f"❌ Error in test query: {e}")

    print(f"\n🎊 SUCCESS: Enhanced hierarchical chunks with rich medical metadata!")
    print(f"📊 Context preserved with hierarchical chunking + medical awareness!")

if __name__ == "__main__":
    embed_with_enhanced_hierarchical()
