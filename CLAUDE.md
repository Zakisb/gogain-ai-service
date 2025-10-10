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
- **Containerization**: Docker with docker compose up --build for local development

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
# Run FastAPI development server (Qdrant is locally-hosted)
docker compose up
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

# Test embedding configuration
python test_voyage_config.py

# Interactive chat testing
python coach.py
```

## Configuration

### Environment Variables (.env file)

Required environment variables:

- `OPENAI_API_KEY`: OpenAI API key
- `GOOGLE_API_KEY`: Google Gemini API key
- `VOYAGE_API_KEY`: Voyage AI embeddings (recommended)
- `DEFAULT_MODEL`: Default LLM model (e.g., "gemini/gemini-pro")
- `MEM0_API_KEY`: Mem0 API key for enhanced agent memory (optional)

### Model Configuration

The service uses LiteLLM for model abstraction. Default model is configured in `app/core/config.py`. Supported models include:

- OpenAI: "gpt-3.5-turbo", "gpt-4"
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
- Uses RAG system to fetch relevant medical knowledge AND pain assessment questions  based on user queries
- Leverages Mem0 long-term memory for personalized, tailored questioning
- Dynamically generates questions based on AI analysis AND retrieved data from the RAG
- Outputs structured pain_summary in JSON format at workflow completion
- Use the virtual-coach/chat endpoint WITHOUT REMOVING THE OLD ENDPOINT, BUT WE WANT THIS ENDPOINT TO ALWAYS BE WORKING WITH STREAMIGN + TOOL CALLS , do not edit the old endpoint since we dont need it 
- We want the pain agent to always respond in a json format like shown in the prompt, so that it always lists a question, along with a list of options to answer that question, in the event where we cannot put those questions into any categories theere should always be an option called "autre" so that the user can answer by text if their specific case does not exist in the list of options listed into the AI, so the pain agent workflow stays the same, but the output format changes, its no longer asking questions to the user in any format, its asking questions to the user with a list of possible answers to make the user experience better.
- EVERYTIME WE TEST WE ONLY WANT TO HIT THE VIRTUAL-COACH ENDPOINT this is the one that will go into production

### Key Changes Required

1. **Enhanced RAG Integration**: Improve knowledge retrieval to inform question generation
2. **Mem0 Memory Integration**: Use persistent memory to ask context-aware, personalized questions
3. **Dynamic Question Generation**: AI determines necessary questions based on patient context and medical knowledge
4. **Fast Workflow**: We want no delay, we use gpt4o-mini for everything, the RAG should never run while the user is waitign it should always run while the user is chatting with the chatbot and the chatbot has quesitons to ask in its queue, the user SHOULD NEVER have to wait for the chatbot to retrieve data from qdrant to ask questions, lets say we fetch 10  questions from the rag, the ai decides to ask 4 , when the chatbot arrives to the 3rd question, the agent should launch the rag queries with newly gathered data, it should never wait till the queue is empty to start the RAG

### Implementation Goals

- Pain agent should intelligently determine what questions to ask based on:
  - Initial user pain description
  - Retrieved medical knowledge from RAG system
  - Historical patient data from Mem0 memory
  - Clinical assessment best practices embedded in knowledge base
- Assessment should feel natural and conversational while maintaining clinical rigor
- We also want to make sure to have NO delay, the user experience needs to be good, so the user never has to wait longer than 4s for a response
- Final output should provide structured, actionable pain assessment summary

## Important Notes

- The service uses Qdrant  self-hosted vector database for knowledge storage
- Knowledge documents should be structured with proper headings (H1, H2, H3) for optimal RAG performance
- Agent system uses LlamaIndex's FunctionAgent and AgentWorkflow for orchestration
- Index naming follows pattern: `gogain-{knowledge_type}-{body_region}`

## Critical Development Rules

**NEVER USE KEYWORD-BASED MATCHING IN THIS CODEBASE**
- Do not implement keyword detection (e.g., `if "pain" in query.lower()`)
- Do not use hardcoded string matching for routing or logic decisions
- All routing and decision-making must be AI-driven through the agent system
- Use semantic understanding via embeddings and LLM reasoning instead of keywords
- This ensures the system remains intelligent, context-aware, and maintainable

## Troubleshooting

- **Qdrant Connection Issues**: Check QDRANT_API_KEY and connectivity
- **Embedding Errors**: Verify API keys (Voyage AI/OpenAI) and network connectivity
- **Agent Failures**: Verify LLM model availability and API quotas
- **Knowledge Retrieval**: Use `python3 analyze_qdrant.py` to verify qdrant  indexes exist

