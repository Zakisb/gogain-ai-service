# GoGain Knowledge Management System

This document describes the knowledge management system that powers the AI agents with domain-specific expertise for pain assessment, training guidance, and nutrition planning.

## Overview

The knowledge system is built around three core components:

1. **DocumentProcessor**: Processes medical/fitness documents using hierarchical parsing
2. **EmbeddingsPipeline**: Manages vector embeddings and Qdrant collections
3. **KnowledgeManager**: High-level orchestration and caching

## Architecture

```
documents/
├── shoulder_pain_final_report.md
├── knee_pain_report.md
└── back_pain_report.md

                    ↓ (DocumentProcessor)

Hierarchical Nodes with Metadata
├── Level 1: Main sections
├── Level 2: Subsections
└── Level 3: Detailed content

                    ↓ (EmbeddingsPipeline)

Qdrant Vector Store
├── gogain_pain_shoulder
├── gogain_pain_knee
└── gogain_pain_back

                    ↓ (KnowledgeManager)

Agent Tools & Query Engines
└── Pain Assessment Agent + Knowledge Tools
```

## Key Features

### Hierarchical Document Processing

- **Preserves Structure**: Maintains document hierarchy (H1, H2, H3)
- **Rich Metadata**: Extracts questions, red flags, exercise criteria
- **Context Preservation**: Parent-child relationships maintained
- **Multi-level Retrieval**: Can retrieve specific details or broader context

### Smart Embedding Strategy

- **HierarchicalNodeParser**: Primary splitting strategy for structured medical documents
- **Metadata Extraction**: Automatic extraction of medical/fitness concepts
- **Body Region Detection**: Auto-detects body regions from filenames
- **Flexible Storage**: Separate collections per body region and knowledge type

### Vector Store Management

- **Qdrant Integration**: Persistent vector storage with collections
- **Collection Naming**: `gogain_{knowledge_type}_{body_region}`
- **Automatic Creation**: Creates collections as needed
- **Statistics Tracking**: Monitor embedding counts and status

## Quick Start

### 1. Install Dependencies

```bash
pip install llama-index
pip install llama-index-vector-stores-qdrant
pip install llama-index-embeddings-openai
pip install qdrant-client
```

### 2. Start Qdrant

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or install locally
# See: https://qdrant.tech/documentation/quick-start/
```

### 3. Set Environment Variables

```bash
export OPENAI_API_KEY="your_openai_key"
export GOOGLE_API_KEY="your_google_key"  # For Gemini LLM
```

### 4. Embed Your First Document

```bash
# Embed the shoulder pain document
python scripts/embed_documents.py embed documents/shoulder_pain_final_report.md

# Or batch process all documents
python scripts/embed_documents.py batch documents/
```

### 5. Test the System

```bash
# List available knowledge bases
python scripts/embed_documents.py list

# Test retrieval
python scripts/embed_documents.py test shoulder --query "What causes shoulder impingement?"

# Run the demo
python examples/pain_agent_with_knowledge.py
```

## Usage Examples

### Basic Usage

```python
from app.knowledge import KnowledgeManager

# Initialize knowledge manager
km = KnowledgeManager()

# Process and embed a document
index = km.process_and_embed_document(
    file_path="documents/shoulder_pain_final_report.md",
    body_region="shoulder",
    knowledge_type="pain"
)

# Get knowledge base
index = km.get_knowledge_base("shoulder", "pain")

# Create query engine
query_engine = km.create_query_engine("shoulder", "pain")
response = query_engine.query("What are signs of shoulder impingement?")
```

### Integration with Agents

```python
from app.knowledge import KnowledgeManager
from app.agents.pain import build as build_pain_agent
from llama_index.core.tools import QueryEngineTool

# Create knowledge-enhanced agent
km = KnowledgeManager()
llm = Gemini(model="models/gemini-2.0-flash")

# Create query engine tool
query_engine = km.create_query_engine("shoulder", "pain")
knowledge_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="shoulder_pain_knowledge",
    description="Shoulder pain assessment knowledge base"
)

# Create agent with knowledge tool
pain_agent = build_pain_agent(llm)
pain_agent.tools = [knowledge_tool]
```

### Batch Processing

```python
# Process all documents in a directory
results = km.batch_process_documents(
    documents_dir="documents/",
    knowledge_type="pain",
    file_pattern="*pain*.md"
)

# Check what was processed
for body_region, index in results.items():
    print(f"Processed: {body_region}")
```

## CLI Commands

### Embed Documents

```bash
# Single document
python scripts/embed_documents.py embed documents/shoulder_pain_final_report.md

# With specific body region
python scripts/embed_documents.py embed documents/report.md --body-region knee

# Force recreation
python scripts/embed_documents.py embed documents/shoulder_pain_final_report.md --force
```

### Batch Processing

```bash
# Process all markdown files
python scripts/embed_documents.py batch documents/

# Specific pattern
python scripts/embed_documents.py batch documents/ --pattern "*_pain_*.md"

# Different knowledge type
python scripts/embed_documents.py batch documents/ --knowledge-type training
```

### Management

```bash
# List all knowledge bases
python scripts/embed_documents.py list

# Delete a knowledge base
python scripts/embed_documents.py delete shoulder

# Test retrieval
python scripts/embed_documents.py test shoulder --query "What are shoulder red flags?"
```

## Document Structure Requirements

For optimal results, structure your documents as follows:

```markdown
# Main Topic (H1)

## Category 1 (H2)

Content with specific information...

### Subcategory 1.1 (H3)

- Detailed points
- Assessment questions in « quotes »
- **Bold** for emphasis

### Subcategory 1.2 (H3)

More detailed content...

## Category 2 (H2)

More content...
```

### Recommended Metadata

The system automatically extracts:

- **Questions**: Text in « French quotes » or "English quotes"
- **Red flags**: Medical urgency terms
- **Exercise info**: Exercise-related content
- **Assessment info**: Diagnostic/evaluation content

## Configuration

### Embedding Settings

```python
# In app/core/embeddings.py
class EmbeddingConfig:
    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50
    HIERARCHICAL_CHUNK_SIZES = [2048, 512, 128]
    HIERARCHICAL_OVERLAP = 20
```

### Collection Naming

Collections are named: `{prefix}_{knowledge_type}_{body_region}`

Examples:

- `gogain_pain_shoulder`
- `gogain_training_upper_body`
- `gogain_nutrition_weight_loss`

## Troubleshooting

### Common Issues

1. **"No knowledge bases found"**

   - Make sure Qdrant is running
   - Check that documents were embedded successfully
   - Verify OpenAI API key is set

2. **"Collection already exists"**

   - Use `--force` flag to recreate
   - Or delete existing: `python scripts/embed_documents.py delete {body_region}`

3. **"Error loading document"**

   - Check file path and permissions
   - Ensure file is in supported format (markdown, text)

4. **Retrieval returning poor results**
   - Check document structure (use headers)
   - Verify metadata extraction worked
   - Try different chunk sizes in config

### Debugging

Enable verbose logging:

```bash
python scripts/embed_documents.py -v embed documents/shoulder_pain_final_report.md
```

Check knowledge base stats:

```python
km = KnowledgeManager()
stats = km.get_knowledge_base_stats()
print(stats)
```

## Next Steps

1. **Add More Body Regions**: Create documents for knee, back, ankle, etc.
2. **Training Knowledge**: Add exercise selection and programming knowledge
3. **Nutrition Knowledge**: Add meal planning and dietary guidance
4. **Integration**: Connect knowledge tools to all agents
5. **Optimization**: Fine-tune chunk sizes and metadata extraction

## API Reference

See the source code for detailed API documentation:

- `app/knowledge/knowledge_manager.py`
- `app/knowledge/document_processor.py`
- `app/knowledge/embeddings_pipeline.py`
