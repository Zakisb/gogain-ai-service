# GoGain AI Coach

An AI-powered fitness coaching microservice built with FastAPI that provides personalized training programs, nutrition plans, pain management guidance, and health insights through conversational AI agents.

## Architecture Overview

GoGain AI Coach implements a multi-agent architecture with RAG (Retrieval-Augmented Generation) capabilities:

- **CentralRouterAgent**: Coordinates and routes requests to specialist agents, NOTE: WILL BE DITCHED WHEN WE SWITCH TO LANGCHAIN
- **Specialist Agents**: Domain-specific agents for pain assessment, training, nutrition, health insights, and general Q&A (nutrition and health insights agents dont work yet)
- **Knowledge Integration**: RAG-powered system with medical/fitness document embeddings (fitness agent does not use RAG, not sure if its necessary)
- **Memory System**: Persistent chat memory with Mem0 integration for personalized interactions 
- mem0 retrieval runs before conversation starts and gets stored in redis for performance
- all agents receive profile data, 
- fitness goals get stored in mem0 (qdrant) ,
- user info automatically gets stored in mem0 if relevant 
- the informations the user mentions  are already  stored in mem0 so there is no information loss, TODO: feed pain/medical data from memory into pain agent during pain assessments


## Technology Stack

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| AI Orchestration | LlamaIndex |
| LLM Integration | LiteLLM (OpenAI, Anthropic, Google Gemini) |
| Vector Database | Qdrant (self-hosted) |
| Embeddings | Voyage AI (primary), OpenAI (fallback) |
| Memory | Mem0 for enhanced AI agent memory |
| Containerization | Docker + Docker Compose |

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- API Keys for:
  - OpenAI
  - Voyage AI (for embeddings)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# LLM API Keys
OPENAI_API_KEY=your_openai_api_key
VOYAGE_API_KEY=your_voyage_api_key
MEM0_API_KEY=your_mem0_api_key  # Optional

# Model Configuration

# Qdrant Configuration (handled by docker-compose)
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### 3. Start the Service

```bash
docker compose up --build
```

The service will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Knowledge Base Management

### Embedding Documents into Qdrant

The service uses a hierarchical document processing system to embed medical and fitness knowledge into the vector database.

#### Embedding from Within the Docker Container

1. **Access the running container**:
```bash
docker compose exec -it ai_service bash
```

2. **Run the hierarchical embedding script**:
```bash
# Embed pain docs
python3 embed_hierarchical_enhanced.py

# For fitness questions
Not needed, the fitness prompt already has them
```

The hierarchical embedding system will:
- Parse documents with hierarchical structure (H1, H2, H3 headings)
- Create intelligent chunks with context preservation
- Generate embeddings using Voyage AI
- Store vectors in Qdrant with metadata




## API Endpoints

### Main Endpoints

- `POST /virtual-coach/chat` - Primary streaming chat endpoint with tool calls (production)

### Testing the API

- run the coach.py file for testing, will route to any agent, supports all features rag/memory/agents


# Install dependencies
- dependencies are in minreqs.txt


# Run the service (requires local Qdrant)
docker compose up
```

### Project Structure

```
gogain-ai-service-v1/
├── app/
│   ├── agents/           # AI agent implementations
│   │   ├── qa.py    	  # General router agent
│   │   ├── pain.py       # Pain assessment agent
│   │   ├── training.py   # Training program agent 
│   │   └── ...
│   ├── api/              # FastAPI route definitions
│   ├── core/             # Core configuration and utilities
│   ├── knowledge/        # Knowledge management system.
│   └── models/           # Pydantic models
├── documents/            # Medical/fitness knowledge documents
├── scripts/              # CLI tools for management
├── docker-compose.yml    # Docker orchestration
├── Dockerfile           # Container definition
└── minreqs.txt     # Python dependencies
```

### Adding New Agents

1. Create a new agent file in `app/agents/`
2. Implement the agent following the existing pattern
3. Register the agent in the central router
4. Add corresponding API endpoints in `app/api/`

### Testing

```bash
# Test individual components
python test_components.py

# Test embedding configuration
python test_voyage_config.py

# Interactive chat testing
python coach.py

#fetches all memories of a specific user (takes user_id as argument)
python scripts/get_user_memories.py 

# Analyze Qdrant collections
python3 analyze_qdrant.py
MUST BE USED TO CHECK IF THE EMBEDDING RAN CORRECTLY , HAS TO BE CALLED FROM INSIDE THE GOGAIN CONTAINER

python3 connectivity_qdrant.py to check connectivity to qdrant
python3 connectivity_postgres.py to check connectivity ti postgres
```

## Pain Assessment Workflow

The pain assessment system is fully AI-driven with:

- **RAG Integration**: Retrieves relevant medical knowledge and assessment questions
- **Mem0 Memory**: Personalizes questioning based on patient history
- **Dynamic Generation**: AI determines necessary questions based on context
- **Structured Output**: Returns JSON-formatted pain summaries

### Pain Assessment Response Format

```json
{
  "question": "Can you describe the location of your pain?",
  "options": [
    "Front of shoulder",
    "Side of shoulder",
    "Back of shoulder",
    "Entire shoulder area",
    "autre"
  ]
}
```

The "autre" (other) option allows users to provide custom text responses when their specific case isn't covered by the predefined options.

## Configuration

### Model Selection

Default model is configured in `app/core/config.py`. Supported models:

- **OpenAI**: `gpt-3.5-turbo`, `gpt-4`, `gpt-4o-mini`
- **Google**: `gemini/gemini-pro`, `gemini/gemini-2.0-flash` (NOT COMPATIBLE WITH LLAMA_INDEX BUT WILL BE AN OPTION WHEN WE SWITCH TO LANGCHAIN)

### Vector Database

The service uses self-hosted Qdrant. Index naming follows:
```
gogain-{knowledge_type}-{body_region}
```

Example: `gogain-pain-shoulder`



```

### Performance Requirements

- **Response Time**: Maximum 4 seconds per user interaction
- **RAG Preloading**: Fetch questions while user is interacting, not during wait time, Never run the RAG while user is waiting always start rag and dont wait for the response, it will populate the queue with pain questions.
- **Question Queue**: When 10 questions are fetched and AI asks 4, trigger new RAG queries at question 3

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Qdrant connection fails | Verify `docker compose` is running and ports are accessible |
| Embedding errors | Check API keys (Voyage AI/OpenAI) and network connectivity |
| Agent failures | Verify LLM model availability and API quotas |
| Knowledge retrieval issues | Run `python3 analyze_qdrant.py` to verify indexes exist |

## API Documentation

Once the service is running, visit http://localhost:8000/docs for interactive API documentation powered by Swagger UI.


### Future improvements:

- Currently the pain agent fetches questions but does not follow the workflow mentioned in the documents, its good enough but GPT4o is just not smart enough to know what questions are the most important, Solution:Either rephrase the prompt, if it doesnt work --> change the chunking technique + new prompt

- Ditch llama_index and use langchain, will save router  costs and langchain is easier to work with.
- Add diet agent (easy)
- Health insights (not enough data yet)
- Tell the frontend to ping the backend once user opens the app, to populate redis with user data from database/memory and make the response time shorter




