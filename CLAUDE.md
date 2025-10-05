# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoGain AI Coach is a Python microservice built with FastAPI that provides AI-powered fitness coaching through conversational agents. The service acts as a virtual coach offering personalized training programs, nutrition plans, pain management guidance, and health insights.

## Core Architecture

### Technology Stack

- **Framework**: FastAPI for REST API
- **AI Orchestration**: LlamaIndex for RAG and agent workflows
- **LLM Integration**: LiteLLM supporting OpenAI, Anthropic, Google Gemini
- **Vector Database**: Pinecone for knowledge embeddings
- **Embeddings**: Voyage AI (primary) or OpenAI (fallback)
- **Memory**: Persistent chat memory with JSON storage (migrating to Mem0 for enhanced AI agent memory)
- **Containerization**: Docker with docker-compose for local development

### Agent System

The service implements a multi-agent architecture with:

- **CentralRouterAgent**: Main coordinator that routes requests to specialist agents (`app/agents/central.py`)
- **Specialist Agents**: Domain-specific agents for pain, training, nutrition, health insights, and general Q&A
- **Knowledge Integration**: RAG-powered agents with access to embedded medical/fitness documents

### Knowledge Management System

Located in `app/knowledge/`:

- **DocumentProcessor**: Hierarchical parsing of medical/fitness documents
- **EmbeddingsPipeline**: Vector embedding and Pinecone index management
- **KnowledgeManager**: High-level orchestration and caching

## Common Development Commands

### Environment Setup

```bash
# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

### Running the Service

```bash
# Run FastAPI development server (Pinecone is cloud-hosted)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or run with Docker setup
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

# Test infrastructure (Pinecone connectivity)
python test_pinecone.py

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
- `PINECONE_API_KEY`: Pinecone vector database API key
- `DEFAULT_MODEL`: Default LLM model (e.g., "gemini/gemini-pro")
- `MEM0_API_KEY`: Mem0 API key for enhanced agent memory (optional)

### Model Configuration

The service uses LiteLLM for model abstraction. Default model is configured in `app/core/config.py`. Supported models include:

- OpenAI: "gpt-3.5-turbo", "gpt-4"
- Anthropic: "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
- Google: "gemini/gemini-pro", "gemini/gemini-2.0-flash"

## Key Directories

- `app/agents/`: Agent implementations for different domains
- `app/api/`: FastAPI router definitions
- `app/core/`: Core configuration and utilities (config, embeddings, Pinecone)
- `app/knowledge/`: Knowledge management system
- `documents/`: Medical/fitness knowledge documents for RAG
- `scripts/`: CLI tools for document embedding and management
- `examples/`: Usage examples and demos

## Development Workflow

1. **Adding New Knowledge**: Place medical/fitness documents in `documents/` and run the embedding script
2. **Agent Development**: Implement new agents in `app/agents/` following existing patterns
3. **API Endpoints**: Add new routes in `app/api/` and include in main FastAPI app
4. **Testing**: Use component tests and interactive chat for validation

## Pain Assessment AI Enhancement Requirements

The pain assessment workflow is being enhanced to be fully AI-driven with the following specifications:

###  Target State


**Target**: Fully AI-based pain assessment that:
- Uses RAG system to fetch relevant medical knowledge based on user queries
- Leverages Mem0 long-term memory for personalized, tailored questioning
- Dynamically generates questions based on AI analysis rather than hardcoded templates
- Outputs structured pain_summary in JSON format at workflow completion
- Use the virtual-coach/chat endpoint WITHOUT REMOVING THE OLD ENDPOINT, BUT WE WANT THIS ENDPOINT TO ALWAYS BE WORKING WITH STREAMIGN + TOOL CALLS 
- EVERYTIME WE TEST WE ONLY WANT TO HIT THE VIRTUAL-COACH ENDPOINT this is the one that will go into production

### Key Changes Required

1. **Enhanced RAG Integration**: Improve knowledge retrieval to inform question generation
2. **Mem0 Memory Integration**: Use persistent memory to ask context-aware, personalized questions
3. **Dynamic Question Generation**: AI determines necessary questions based on patient context and medical knowledge
4. **Structured Output**: Generate comprehensive pain_summary JSON at assessment completion

### Implementation Goals

- Pain agent should intelligently determine what questions to ask based on:
  - Initial user pain description
  - Retrieved medical knowledge from RAG system
  - Historical patient data from Mem0 memory
  - Clinical assessment best practices embedded in knowledge base
- Assessment should feel natural and conversational while maintaining clinical rigor
- Final output should provide structured, actionable pain assessment summary

## Important Notes

- The service uses Pinecone cloud-hosted vector database for knowledge storage
- Knowledge documents should be structured with proper headings (H1, H2, H3) for optimal RAG performance
- Agent system uses LlamaIndex's FunctionAgent and AgentWorkflow for orchestration
- Chat memory is currently stored as JSON files per user, with plans to migrate to Mem0 for enhanced AI agent memory capabilities
- Index naming follows pattern: `gogain-{knowledge_type}-{body_region}`

## Critical Development Rules

**NEVER USE KEYWORD-BASED MATCHING IN THIS CODEBASE**
- Do not implement keyword detection (e.g., `if "pain" in query.lower()`)
- Do not use hardcoded string matching for routing or logic decisions
- All routing and decision-making must be AI-driven through the agent system
- Use semantic understanding via embeddings and LLM reasoning instead of keywords
- This ensures the system remains intelligent, context-aware, and maintainable

## Troubleshooting

- **Pinecone Connection Issues**: Check PINECONE_API_KEY and network connectivity
- **Embedding Errors**: Verify API keys (Voyage AI/OpenAI) and network connectivity
- **Agent Failures**: Verify LLM model availability and API quotas
- **Knowledge Retrieval**: Use `scripts/embed_documents.py list` to verify Pinecone indexes exist

