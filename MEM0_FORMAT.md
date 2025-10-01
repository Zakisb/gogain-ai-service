# Mem0 Memory Storage and Retrieval Format

## Overview
This document shows the exact format used for storing and fetching memories from Mem0 in the GoGain AI service.

## Storage Format (`add_memory`)

### Location
`app/core/memory.py:35-40`

### Function Signature
```python
def add_memory(self, message: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
```

### Parameters
- **message** (str): The actual content to store (user message, assistant response, or summary)
- **user_id** (str): Unique identifier for the user
- **metadata** (dict, optional): Additional context information

### Example Usage (from chat.py:421-425)
```python
get_mem0_client().add_memory(
    message=request.message,                    # The user's message text
    user_id=request.user_id,                    # User identifier (e.g., "user123")
    metadata={
        "type": "user_message",                 # Type of memory
        "timestamp": str(asyncio.get_event_loop().time())  # When it was stored
    }
)
```

### Full Example
```python
# Storing a user message
mem0_client.add_memory(
    message="J'ai mal au genou depuis 2 semaines",
    user_id="user_456",
    metadata={
        "type": "user_message",
        "timestamp": "1709234567.123",
        "body_region": "genou",          # Optional: body region discussed
        "session_id": "sess_789"         # Optional: conversation session
    }
)

# Storing assistant response
mem0_client.add_memory(
    message="Quand cette douleur a-t-elle commencé?",
    user_id="user_456",
    metadata={
        "type": "assistant_message",
        "timestamp": "1709234570.456",
        "agent": "EnhancedPainAssessmentAgent"
    }
)
```

### Return Format
```python
{
    "success": True,
    "result": {
        # Mem0 API response with memory ID and confirmation
    }
}
# OR on error:
{
    "success": False,
    "error": "Error message here"
}
```

---

## Retrieval Format (`get_memories`)

### Location
`app/core/memory.py:42-48`

### Function Signature
```python
def get_memories(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
```

### Parameters
- **user_id** (str): Unique identifier for the user
- **limit** (int): Maximum number of memories to retrieve (default: 10)

### Example Usage
```python
result = mem0_client.get_memories(
    user_id="user_456",
    limit=20
)

if result["success"]:
    memories = result["memories"]
    for memory in memories:
        print(f"Memory: {memory['memory']}")
        print(f"Created: {memory['created_at']}")
        print(f"Metadata: {memory['metadata']}")
```

### Return Format
```python
{
    "success": True,
    "memories": [
        {
            "id": "mem_abc123xyz",
            "memory": "J'ai mal au genou depuis 2 semaines",
            "user_id": "user_456",
            "metadata": {
                "type": "user_message",
                "timestamp": "1709234567.123",
                "body_region": "genou"
            },
            "created_at": "2024-03-01T10:30:00Z",
            "updated_at": "2024-03-01T10:30:00Z"
        },
        # ... more memories
    ]
}
# OR on error:
{
    "success": False,
    "error": "Error message here"
}
```

---

## Search Format (`search_memories`)

### Location
`app/core/memory.py:50-56`

### Function Signature
```python
def search_memories(self, query: str, user_id: str, limit: int = 5) -> Dict[str, Any]:
```

### Parameters
- **query** (str): Semantic search query
- **user_id** (str): Unique identifier for the user
- **limit** (int): Maximum number of results (default: 5)

### Example Usage
```python
# Semantic search for pain-related memories
result = mem0_client.search_memories(
    query="douleur au genou exercices",
    user_id="user_456",
    limit=5
)

if result["success"]:
    relevant_memories = result["memories"]
    for memory in relevant_memories:
        print(f"Memory: {memory['memory']}")
        print(f"Relevance score: {memory.get('score', 'N/A')}")
```

### Return Format
```python
{
    "success": True,
    "memories": [
        {
            "id": "mem_xyz789",
            "memory": "Douleur au genou lors de la montée d'escaliers",
            "score": 0.92,  # Relevance score
            "user_id": "user_456",
            "metadata": {
                "type": "user_message",
                "body_region": "genou"
            },
            "created_at": "2024-03-01T10:30:00Z"
        },
        # ... more relevant memories sorted by score
    ]
}
```

---

## Metadata Best Practices

### Common Metadata Fields
```python
metadata = {
    # Required/Recommended
    "type": "user_message" | "assistant_message" | "pain_summary" | "training_plan",
    "timestamp": "1709234567.123",  # Unix timestamp as string

    # Optional but useful
    "agent": "EnhancedPainAssessmentAgent",  # Which agent created this
    "body_region": "genou" | "dos" | "épaule",  # For pain-related memories
    "session_id": "sess_789",  # Link memories in same conversation
    "assessment_phase": "initial" | "detailed" | "differential",  # Pain assessment stage
    "language": "fr",  # Language of the memory

    # Domain-specific
    "pain_intensity": "7/10",  # For pain assessments
    "training_goal": "hypertrophy",  # For training plans
    "diet_type": "vegetarian"  # For nutrition plans
}
```

---

## Current Status in Codebase

**⚠️ IMPORTANT**: Mem0 memory storage is **currently DISABLED** in the codebase for performance testing.

See:
- `app/api/chat.py:418-430` - Memory storage is commented out
- `app/core/memory.py:55-57` - Memory client disabled
- `app/knowledge/enhanced_assessment_tools.py:55-57` - Memory retrieval disabled

To re-enable:
1. Uncomment memory code in `chat.py`
2. Set `self.memory_client = get_mem0_client()` in `enhanced_assessment_tools.py`
3. Ensure `MEM0_API_KEY` is set in `.env`

---

## Configuration

### Environment Variables
```bash
# Required for Mem0
OPENAI_API_KEY=sk-...        # Mem0 uses OpenAI embeddings
MEM0_API_KEY=mem0_key_...    # Optional, if using Mem0 cloud

# Qdrant (vector store for Mem0)
QDRANT_URL=http://localhost:6333
```

### Mem0 Configuration (memory.py:22-31)
```python
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0",
            "url": qdrant_url,
            "timeout": 30.0
        }
    }
}
```

---

## Integration Example

### Full Pain Assessment with Memory
```python
from app.core.memory import get_mem0_client

mem0 = get_mem0_client()

# 1. Store user's initial complaint
mem0.add_memory(
    message="J'ai mal au genou quand je cours",
    user_id="user_123",
    metadata={
        "type": "user_message",
        "timestamp": str(time.time()),
        "body_region": "genou",
        "session_id": "pain_assessment_001"
    }
)

# 2. Later: Search for relevant history
history = mem0.search_memories(
    query="genou douleur course",
    user_id="user_123",
    limit=5
)

if history["success"]:
    past_injuries = [m["memory"] for m in history["memories"]]
    print(f"Found {len(past_injuries)} relevant past memories")

# 3. Store assessment summary
mem0.add_memory(
    message="Patient rapporte douleur antéro-latérale genou depuis 3 semaines...",
    user_id="user_123",
    metadata={
        "type": "pain_summary",
        "timestamp": str(time.time()),
        "body_region": "genou",
        "assessment_complete": True,
        "session_id": "pain_assessment_001"
    }
)
```