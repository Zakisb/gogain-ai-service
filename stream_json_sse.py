#!/usr/bin/env python3
"""
stream_json_sse.py  (server)

FastAPI SSE server for streaming JSON responses.
Compatible with your stack: llama-index==0.12.52, llama-index-llms-openai==0.4.7, openai==1.x

- Uses LlamaIndex ChatMessage/MessageRole (required by 0.12.x).
- Streams tokens to the client (SSE "message" events).
- Buffers full text and emits a final "parsed" event with *raw JSON* (no HTML escaping).
"""

import os, json
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse

# .env loader (optional)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except Exception:
    pass

# LlamaIndex (0.12.x)
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("ERROR: OPENAI_API_KEY not set (env or .env)")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
llm = OpenAI(model=MODEL_NAME, temperature=0.0, api_key=OPENAI_API_KEY)

app = FastAPI(title="Streaming JSON SSE (LlamaIndex 0.12.52)")

SYSTEM_PROMPT = """
Tu dois répondre UNIQUEMENT avec UN SEUL objet JSON strictement valide, sans texte additionnel ni backticks.

Schéma attendu :
{
  "type": "single_choice",
  "options": [
    {"id": "string", "label": "string", "showWhen": "string"}
  ]
}

Contraintes :
- "type" = "single_choice"
- "options" est une liste d'objets id/label/showWhen
- showWhen ∈ {"always","noTrainingProgram","hasTrainingProgram","hasNoDietPlan","hasDietPlan"}
- Ne jamais ajouter un second objet JSON ni du texte autour.
"""

FEW_SHOT = """Exemple :
Utilisateur : "Je veux commencer un programme."
Assistant :
{
  "type": "single_choice",
  "options": [
    {"id": "fitness", "label": "Faire du sport", "showWhen": "noTrainingProgram"},
    {"id": "autre", "label": "Autre question de santé ou fitness", "showWhen": "always"}
  ]
}
"""

INDEX_HTML = """<!doctype html>
<html>
<head><meta charset="utf-8"/><title>LlamaIndex Streaming JSON SSE</title></head>
<body>
  <h2>LlamaIndex Streaming JSON SSE (0.12.x)</h2>
  <textarea id="message">Salut</textarea><br/>
  <button id="start">Start stream</button>
  <pre id="raw"></pre>
  <pre id="parsed">—</pre>
  <script>
    let es;
    document.getElementById('start').addEventListener('click', () => {
      const raw = document.getElementById('raw');
      const parsed = document.getElementById('parsed');
      raw.textContent = ""; parsed.textContent = "—";
      const msg = encodeURIComponent(document.getElementById('message').value || "");
      es = new EventSource(`/stream?message=${msg}`);
      es.addEventListener('message', ev => raw.textContent += ev.data);
      es.addEventListener('parsed', ev => {
        try { parsed.textContent = JSON.stringify(JSON.parse(ev.data), null, 2); }
        catch (e) { parsed.textContent = "Parse error: " + e + "\\nPayload: " + ev.data; }
        es.close();
      });
      es.addEventListener('error_json', ev => { parsed.textContent = "Final parse error: " + ev.data; es.close(); });
    });
  </script>
</body>
</html>
"""

def sse_event_stream(user_message: str) -> Generator[bytes, None, None]:
    buffer = ""
    try:
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=FEW_SHOT),
            ChatMessage(role=MessageRole.USER, content=user_message),
        ]

        stream = llm.stream_chat(messages=messages)

        for event in stream:
            if getattr(event, "delta", None):
                chunk = event.delta
                yield f"data: {chunk}\n\n".encode("utf-8")
                buffer += chunk

        # Final: parse & emit raw JSON (no HTML escaping)
        try:
            obj = json.loads(buffer)
            yield f"event: parsed\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
        except Exception as e:
            # Send the raw buffer so a terminal tool can still salvage it if needed
            err = f"{e.__class__.__name__}: {e}"
            yield f"event: error_json\ndata: {err}\n\n".encode("utf-8")
    finally:
        yield b"event: done\ndata: true\n\n"

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)

@app.get("/stream")
def stream(message: str = Query(..., description="User message (French)")):
    return StreamingResponse(sse_event_stream(message), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}
