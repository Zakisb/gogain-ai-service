# GoGain AI Coach — Service Overview

GoGain AI Coach is a FastAPI microservice that powers AI-driven fitness coaching via multi-agent orchestration (general coach plus specialist agents). It integrates LlamaIndex for RAG, LiteLLM for LLM abstraction, Pinecone/Qdrant for vector search, and Mem0 for long‑term memory.

## Audio → Chat Flow (New)

The audio pathway has been streamlined so the frontend can upload audio once and immediately receive both:
- the transcribed user message (first Server‑Sent Event), and
- the normal streaming chat response from the virtual coach.

### Endpoint
- POST `/virtual-coach/transcribe-audio`
  - accepts `multipart/form-data`
  - fields:
    - `audio` (or `file`): uploaded audio file
    - `userId`: required when you want the service to forward the transcript into chat
    - `conversationHistory` (optional): JSON array of prior messages (role/content)

### Streaming Behavior
This endpoint now returns a streaming response (SSE). The stream begins with a transcript event, followed by the usual chat stream:
- First frame: `{"type":"transcription","content":"<transcript>"}`
- Subsequent frames: `text_delta` chunks and a final `message_stop` with the full conversation messages (user and assistant), same as `/virtual-coach/chat`.

This allows the frontend to:
1) Append the user’s transcribed message to local conversation history immediately upon the first event.
2) Continue consuming the remaining SSE frames to render the assistant reply. On `message_stop`, you receive the final conversation snapshot (excluding system messages), which you can also persist if desired.

No extra client round trip is needed. The backend does transcription and immediately streams chat, preserving latency and UX.

### cURL Test
```
curl -N -v \
  -F "audio=@/path/to/sample.wav" \
  -F "userId=12345" \
  -F 'conversationHistory=[]' \
  http://localhost:8000/virtual-coach/transcribe-audio
```
Expected stream order:
- `data: {"type":"transcription","content":"..."}`
- `data: {"type":"text_delta","content":"..."}` (many)
- `data: {"type":"message_stop","content":[ ... messages ... ]}`

## Related Endpoints
- Text chat (legacy-compatible):
  - POST `/virtual-coach/chat`
  - POST `/api/v1/chat/virtual-coach/chat` (namespaced)
- Audio transcription (no chat handoff):
  - POST `/virtual-coach/transcribe-audio` without `userId` — returns `{ "transcription": "..." }` JSON.

## Notes
- Transcription uses OpenAI Whisper‑compatible models (`TRANSCRIPTION_MODEL`), configured via `.env`.
- When `userId` is included, the service forwards the transcript into the virtual coach chat pipeline and streams the combined SSE output.
- The initial `transcription` frame is intended for client storage of the user’s message in local conversation history.

