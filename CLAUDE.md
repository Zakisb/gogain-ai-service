# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoGain AI Coach is a Python microservice built with FastAPI that provides AI-powered fitness coaching through conversational agents. The service acts as a virtual coach offering personalized training programs, nutrition plans, pain management guidance, and health insights.

## Core Architecture

### Technology Stack

- **Framework**: FastAPI for REST API
- **AI Orchestration**: LlamaIndex for RAG and agent workflows
- **LLM Integration**: LiteLLM supporting OpenAI, Anthropic, Google Gemini
- **Vector Database**: Qdrant for knowledge embeddings
- **Embeddings**: Voyage AI (primary) or OpenAI (fallback)
- **Memory**: Persistent chat memory with JSON storage
- **Containerization**: Docker with docker-compose for local development

### Agent System

The service implements a multi-agent architecture with:

- **CentralRouterAgent**: Main coordinator that routes requests to specialist agents (`app/agents/central.py`)
- **Specialist Agents**: Domain-specific agents for pain, training, nutrition, health insights, and general Q&A
- **Knowledge Integration**: RAG-powered agents with access to embedded medical/fitness documents

### Knowledge Management System

Located in `app/knowledge/`:

- **DocumentProcessor**: Hierarchical parsing of medical/fitness documents
- **EmbeddingsPipeline**: Vector embedding and Qdrant collection management
- **KnowledgeManager**: High-level orchestration and caching

## Common Development Commands

### Environment Setup

```bash
# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Service

```bash
# Start Qdrant vector database
docker-compose up qdrant -d

# Run FastAPI development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or run with full Docker setup
docker-compose up
```

### Knowledge Base Management

```bash
# Embed documents into vector database
python scripts/embed_documents.py embed documents/shoulder_pain_final_report.md

# Batch process all documents
python scripts/embed_documents.py batch documents/

# List available knowledge bases
python scripts/embed_documents.py list

# Test knowledge retrieval
python scripts/embed_documents.py test shoulder --query "What causes shoulder pain?"
```

### Testing

```bash
# Test individual knowledge system components
python test_components.py

# Test infrastructure (Qdrant connectivity)
python test_infrastructure.py

# Test embedding configuration
python test_voyage_config.py

# Interactive chat testing
python interactive_chat.py
```

## Configuration

### Environment Variables (.env file)

Required environment variables:

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `GOOGLE_API_KEY`: Google Gemini API key
- `VOYAGE_API_KEY`: Voyage AI embeddings (recommended)
- `DEFAULT_MODEL`: Default LLM model (e.g., "gemini/gemini-pro")

### Model Configuration

The service uses LiteLLM for model abstraction. Default model is configured in `app/core/config.py`. Supported models include:

- OpenAI: "gpt-3.5-turbo", "gpt-4"
- Anthropic: "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
- Google: "gemini/gemini-pro", "gemini/gemini-2.0-flash"

## Key Directories

- `app/agents/`: Agent implementations for different domains
- `app/api/`: FastAPI router definitions
- `app/core/`: Core configuration and utilities (config, embeddings, Qdrant)
- `app/knowledge/`: Knowledge management system
- `documents/`: Medical/fitness knowledge documents for RAG
- `scripts/`: CLI tools for document embedding and management
- `examples/`: Usage examples and demos

## Development Workflow

1. **Adding New Knowledge**: Place medical/fitness documents in `documents/` and run the embedding script
2. **Agent Development**: Implement new agents in `app/agents/` following existing patterns
3. **API Endpoints**: Add new routes in `app/api/` and include in main FastAPI app
4. **Testing**: Use component tests and interactive chat for validation

## Important Notes

- The service requires Qdrant to be running for vector storage operations
- Knowledge documents should be structured with proper headings (H1, H2, H3) for optimal RAG performance
- Agent system uses LlamaIndex's FunctionAgent and AgentWorkflow for orchestration
- Chat memory is stored in `app/chat_memory/` as JSON files per user
- Collection naming follows pattern: `gogain_{knowledge_type}_{body_region}`

## Troubleshooting

- **Qdrant Connection Issues**: Ensure Qdrant is running on port 6333
- **Embedding Errors**: Check API keys and network connectivity
- **Agent Failures**: Verify LLM model availability and API quotas
- **Knowledge Retrieval**: Use `scripts/embed_documents.py list` to verify knowledge bases exist
