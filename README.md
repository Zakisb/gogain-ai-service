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
| AI Orchestration | langchain|
| Vector Database | Qdrant (self-hosted) |
| Embeddings | Voyage AI (RAG), OpenAI (memory) |
| Memory | Mem0 for enhanced AI agent memory |
| Containerization | Docker |

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

# Audio transcription (optional overrides)
TRANSCRIPTION_MODEL=whisper-1
TRANSCRIPTION_UPLOAD_DIR=uploads

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
- `POST /virtual-coach/transcribe-audio` - Legacy-compatible audio transcription (multipart/form-data)

#### Virtual Coach: user-id-only flow

This endpoint is designed for frontend clients that only know the authenticated user id and the latest user message. The server fetches and injects all relevant context (profile, memories, training snapshot) automatically, then routes to the right agent and streams the answer.

- Path: `/virtual-coach/chat` (also available at `/api/v1/chat/virtual-coach/chat` when mounted under the API prefix)
- Method: `POST`
- Content-Type: `application/json`
- Request body:
  - `userId` (string | number, required): The numeric app user id. Non‑numeric identifiers (emails) are rejected for profile/training lookups.
  - `message` (string, required): The user’s latest utterance.
  - `conversationHistory` (array, optional): Optional prior turns for UI compatibility; server relies on its own session memory.

Minimal example request:

```bash
curl -N -X POST http://localhost:8000/chat/virtual-coach/chat \
  -H 'Content-Type: application/json' \
  -d '{"userId": 54, "message": "J\u2019ai mal \u00e0 l\u2019\u00e9paule depuis 2 semaines"}'
```

#### Virtual Coach: Audio Transcription

- Path: `/virtual-coach/transcribe-audio`
- Method: `POST`
- Content-Type: `multipart/form-data`
- Field name: `audio`
- Response: `{ "transcription": "<text>" }`

Example request:

```bash
curl -X POST http://localhost:8000/virtual-coach/transcribe-audio \
  -F "audio=@/path/to/sample.wav"
```

Streaming response format is Server-Sent Events (SSE) with frames shaped as:

```text
data: {"type":"text_delta","content":"..."}

data: {"type":"final","content":"..."}

```

Key behaviors (from `app/api/chat.py`):
- Agent routing: The GeneralCoachAgent decides whether to answer directly or route to PainAssessmentAgent or TrainingAgent.
- Automatic context: On each turn the server pulls and caches user profile, Mem0 memories, and a condensed training program snapshot; these are injected as system context for better personalization.
- Memory: Mem0 memories are fetched once per window and cached; new useful facts from general QA are written back to Mem0.
- Fast start: First token is streamed quickly; tool results are not echoed to the user stream.
- Completion handling: Specialist sessions (pain/training) auto‑close when completion is detected, resetting that specialist context.

Optional prefetch to ensure a fast first turn:

```bash
curl -X POST http://localhost:8000/chat/prefetch-user-context \
  -H 'Content-Type: application/json' \
  -d '{"userId": 54}'
```

Notes:
- Always call the virtual‑coach endpoint for production testing; legacy endpoints remain but should not be modified.
- The server streams with proper SSE headers and chunked transfer encoding; keep the HTTP connection open on the client and append `text_delta` chunks to your UI while waiting for `final`.

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


### Testing

```bash
# Test individual components
python test_components.py

# Test embedding configuration
python test_voyage_config.py

# Interactive chat testing
python coach.py



# Analyze Qdrant collections
python3 analyze_qdrant.py
MUST BE USED TO CHECK IF THE EMBEDDING RAN CORRECTLY , HAS TO BE CALLED FROM INSIDE THE GOGAIN CONTAINER

python3 connectivity_qdrant.py to check connectivity to qdrant
python3 connectivity_postgres.py to check connectivity ti postgres
python3 get_mems.py [user_id] ex : get_mems.py 54       to see all memories from a certain user
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

- **OpenAI**:  `gpt-4o-mini`, `gpt-4o`
- We use gpt4o for responses, and 4o-mini for llm based metadata enriching

### Vector Database

The service uses self-hosted Qdrant


### Performance Requirements

- **Response Time**: Maximum 3 seconds per user interaction
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


- Add diet agent (easy)
- Health insights (no data yet)
- Tell the frontend to ping the backend once user opens the app, to populate redis with user data from database/memory and make the response time shorter (we already have an API endpoint in this backend that the frontend can ping to trigger fetching & caching)

### Notes:
- The user_id in mem0 is the same user_id we get via the API
- The rag is always called in the background so it doesnt cause delays or ruin the user experience
- fitness goals and pain assessments are stored in memory
- the .env file has a  ENABLE_MEM0_MESSAGE_RETRIEVAL it enables search and retrieval from memory everytime the user sends a message, by default its set to false for performance but it can be set to TRUE if there is no major delays caused by it, (the advantage of it is when the conversations get long mem0 gets a lot of context so its a way to filter out what we retrieve rather than dumping all the memories into the agent's context.
