"""Basic chat endpoint using AgentWorkflow with CentralRouterAgent and PainAgent."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import asyncio
import json
from typing import List, Dict, Any, Union

from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI

from app.agents.central import build_workflow
from app.core.memory import get_mem0_client
from app.core.config import DEFAULT_MODEL
from app.tools.date_tool import current_datetime_context_line
from app.core.cache import (
    get_conversation_history as cache_get_conversation_history,
    add_conversation_message as cache_add_conversation_message,
    start_pain_agent_session as cache_start_pain_agent_session,
    add_to_pain_agent_session as cache_add_to_pain_agent_session,
    get_pain_agent_session_history as cache_get_pain_agent_session_history,
    end_pain_agent_session as cache_end_pain_agent_session,
    is_pain_agent_session_active as cache_is_pain_agent_session_active,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global agent cache (initialized at startup, reused for all users)
# Agents are stateless - user context comes from messages array per-request
_CACHED_AGENTS = {
    "pain": None,
    "training": None,
    "qa": None,
}
_AGENT_CACHE_LOCK = asyncio.Lock()


# NOTE: Workflow is NOT cached because it has mutable instance state (_current_agent, _user_id, etc)
# that would cause race conditions between concurrent users. Only agents are cached.

# Create API Router
router = APIRouter(prefix="/chat")
legacy_router = APIRouter()




async def get_cached_conversation_history(user_id: str, limit: int = 3):
    """Get conversation history from Redis-backed cache (TTL-bound)."""
    history = await cache_get_conversation_history(user_id, limit)
    if history:
        logger.info(
            f"🎯 Cache HIT - Using cached history for {user_id} ({len(history)} messages)"
        )
    else:
        logger.info(
            f"💨 Cache MISS - Starting new conversation for {user_id} (no API calls)"
        )
    return history


async def add_to_conversation_cache(user_id: str, message: str, role: str):
    """Add message to Redis cache (keeps last N, refreshes TTL)."""
    await cache_add_conversation_message(user_id, message, role)


async def start_pain_agent_session(user_id: str):
    """Start a fresh isolated pain agent session for a user (Redis-backed)."""
    await cache_start_pain_agent_session(user_id)
    logger.info(f"🔴 Started fresh pain agent session for {user_id}")


async def add_to_pain_agent_session(user_id: str, message: str, role: str):
    """Add message to isolated pain agent session (Redis-backed)."""
    await cache_add_to_pain_agent_session(user_id, message, role)
    logger.info(f"🔴 Added {role} message to pain session for {user_id}")


async def get_pain_agent_session_history(user_id: str) -> List[Dict[str, str]]:
    """Get isolated pain agent session history (Redis-backed)."""
    return await cache_get_pain_agent_session_history(user_id)


async def end_pain_agent_session(user_id: str):
    """Mark pain agent session as ended (Redis-backed)."""
    await cache_end_pain_agent_session(user_id)
    logger.info(f"🔴 Ended pain agent session for {user_id}")


async def is_pain_agent_session_active(user_id: str) -> bool:
    """Check if pain agent session is active for user (Redis-backed)."""
    return await cache_is_pain_agent_session_active(user_id)


def build_message_array_context(history, current_message, user_memories=None, limit=3, include_memories=True):
    """Build universal message array context that works with all LLM providers."""
    messages = []

    # Add user memories as system context (if enabled and available)
    if include_memories and user_memories:
        memory_lines = []
        for memory in user_memories:
            if isinstance(memory, dict) and 'memory' in memory:
                memory_lines.append(f"- {memory['memory']}")
            elif isinstance(memory, str):
                memory_lines.append(f"- {memory}")

        if memory_lines:
            system_context = f"User background/history:\n" + "\n".join(memory_lines)
            messages.append({
                "role": "system",
                "content": system_context
            })

    # Add conversation history (respecting limit)
    if history:
        recent_history = history[-limit:] if limit else history
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # Add current user message
    messages.append({
        "role": "user",
        "content": current_message
    })

    return messages


async def build_pain_agent_context(user_id: str, current_message: str, body_region: str = "general", user_memories=None, include_memories=True):
    """Build isolated context for pain agent using only pain session messages.

    This ensures the pain agent sees only messages from the current pain assessment
    session, not the full conversation history. Messages are still stored in the
    global history for the router to see.

    Also injects available questions from the Redis queue into the context.
    """
    messages = []

    # Add user memories as system context (if enabled and available)
    if include_memories and user_memories:
        memory_lines = []
        for memory in user_memories:
            if isinstance(memory, dict) and 'memory' in memory:
                memory_lines.append(f"- {memory['memory']}")
            elif isinstance(memory, str):
                memory_lines.append(f"- {memory}")

        if memory_lines:
            system_context = f"User background/history:\n" + "\n".join(memory_lines)
            messages.append({
                "role": "system",
                "content": system_context
            })

    # Inject questions queue into context (if available)
    try:
        from app.knowledge.enhanced_assessment_tools import get_full_queue_state
        queue_state = get_full_queue_state(user_id, body_region)
        questions = queue_state.get("questions", [])

        if questions:
            queue_context = "QUESTIONS SUGGÉRÉES (inspirations pour ton prochain question):\n"
            for idx, q in enumerate(questions[:15], 1):  # Limit to top 15 for better coverage
                queue_context += f"{idx}. {q}\n"

            messages.append({
                "role": "system",
                "content": queue_context
            })
            logger.info(f"📋 Injected {len(questions[:15])} suggested questions into pain agent context")
    except Exception as e:
        logger.warning(f"⚠️ Failed to inject queue into context: {e}")

    # Get isolated pain session history
    pain_history = await get_pain_agent_session_history(user_id)

    # Add pain session messages (all of them, no limit - it's a fresh context)
    for msg in pain_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Add current user message
    messages.append({
        "role": "user",
        "content": current_message
    })

    logger.info(f"🔴 Built pain agent context with {len(pain_history)} session messages + current message")
    return messages


async def warmup_agents(llm):
    """Build and cache essential agents at startup.

    This eliminates the 20-second first-request delay by pre-building agents.
    Workflow is NOT cached (has mutable state, must be per-request for multi-user safety).
    """
    async with _AGENT_CACHE_LOCK:
        logger.info("🔥 Building essential agents for cache...")

        # Import agent builders
        from app.agents.pain import build as build_pain
        from app.agents.training import build as build_training
        from app.agents.qa import build as build_qa

        # Build only the 3 essential agents
        _CACHED_AGENTS["pain"] = build_pain(llm)
        _CACHED_AGENTS["training"] = build_training(llm)
        _CACHED_AGENTS["qa"] = build_qa(llm)

        logger.info("✅ Cached 3 essential agents (Pain, Training, QA - multi-user safe)")


def get_cached_or_build_agents(llm):
    """Return cached agents if available, otherwise build them.

    Returns list of agents in order: [Pain, Training, QA]
    """
    if all(_CACHED_AGENTS.values()):
        logger.info("⚡ Using cached agents (fast path)")
        return [
            _CACHED_AGENTS["pain"],
            _CACHED_AGENTS["training"],
            _CACHED_AGENTS["qa"],
        ]
    else:
        # Fallback: build if cache not ready (shouldn't happen after warmup)
        logger.warning("⚠️ Agent cache not ready, building on-demand (slow path)")
        from app.agents.pain import build as build_pain
        from app.agents.training import build as build_training
        from app.agents.qa import build as build_qa

        return [
            build_pain(llm),
            build_training(llm),
            build_qa(llm),
        ]


def get_or_build_workflow(llm):
    """Build workflow instance per-request (uses cached agents).

    Workflow has mutable state (_current_agent, _user_id, etc) so must be created
    per-request to avoid race conditions between concurrent users.

    Agents are cached globally (safe: they're stateless).
    """
    specialist_agents = get_cached_or_build_agents(llm)
    workflow = build_workflow(
        llm,
        specialist_agents,
        initial_state={}
    )
    logger.info("🏗️ Built workflow for request (using cached agents)")
    return workflow


# ==========================
# Node-Compatible VC endpoint
# ==========================

class VirtualCoachChatRequest(BaseModel):
    userId: Union[str, int]
    message: str
    conversationHistory: List[Dict[str, Any]] = []


def emit_vc_event(event_type: str, content: Any) -> str:
    """Emit a Node-compatible SSE frame with proper newlines for immediate transmission.

    Format: data: {"type":"<type>","content":<content>}\n\n
    Using double newline to ensure immediate transmission in Docker.
    """
    payload = json.dumps({"type": event_type, "content": content}, ensure_ascii=False)
    return f"data: {payload}\n\n"


def unwrap_pain_agent_response(response_text: str) -> str:
    """Parse pain agent structured JSON response and transform to frontend-compatible format.

    Frontend expects questions in this format:
    <question_text>
    ```json
    {"id": "option1", "label": "First option"}
    {"id": "option2", "label": "Second option"}
    ```

    If response_type is "conclusion", extract the plain text from the conclusion field.
    If response_type is "question", transform to markdown format with newline-separated JSON objects.

    Args:
        response_text: The raw response from pain agent (possibly JSON)

    Returns:
        Either transformed markdown format (for questions) or plain text (for conclusions)
    """
    try:
        # Try to parse as JSON
        parsed = json.loads(response_text.strip())

        # Check if it's a structured pain agent response
        if isinstance(parsed, dict) and "response_type" in parsed:
            response_type = parsed.get("response_type")

            if response_type == "conclusion":
                # Unwrap conclusion to plain text
                conclusion_text = parsed.get("conclusion", "")
                logger.info(f"🔓 Unwrapped pain agent conclusion ({len(conclusion_text)} chars)")
                return conclusion_text
            elif response_type == "question":
                # Transform to frontend format: question_text\n```json\n{options}\n```
                question_text = parsed.get("question", "")
                options = parsed.get("options", [])

                if not question_text or not options:
                    logger.warning("⚠️ Malformed question JSON, returning as-is")
                    return response_text  # Fallback to original if malformed

                # Build frontend-compatible format
                output = question_text + "\n```json\n"
                for option in options:
                    if isinstance(option, dict) and "id" in option and "label" in option:
                        # One JSON object per line, no commas between objects
                        output += json.dumps(option, ensure_ascii=False) + "\n"
                output += "```"

                logger.info(f"❓ Transformed pain agent question to frontend format ({len(options)} options)")
                return output

        # Not a structured response, return as-is
        return response_text

    except (json.JSONDecodeError, ValueError):
        # Not JSON, return as-is
        return response_text


@legacy_router.post("/virtual-coach/chat")
async def handle_virtual_coach_chat(request: VirtualCoachChatRequest) -> StreamingResponse:
    """Compatibility endpoint that mirrors the legacy Node /virtual-coach/chat contract.

    - Accepts: { userId, message, conversationHistory }
    - Streams: text_delta chunks, optional tool_result notifications, and a final message_stop
      with the updated conversation (no system messages).
    """

    async def stream_generator():
        try:
            user_id = str(request.userId)
            user_message = request.message
            history = request.conversationHistory or []

            logger.info(f"[VC Compat] Starting chat for user {user_id}")

            # Initialize LLM
            temp = 1.0 if DEFAULT_MODEL.startswith('gpt-5') else 0.7
            llm = OpenAI(model=DEFAULT_MODEL, temperature=temp)

            # Set user_id in thread-local storage for tools to access programmatically
            from app.knowledge.enhanced_assessment_tools import set_user_context
            set_user_context(user_id)

            # Build a fresh workflow per request
            workflow = get_or_build_workflow(llm)

            # Build router/agent contexts from provided conversationHistory for better continuity
            def to_history_pairs(conv: List[Dict[str, Any]]) -> List[Dict[str, str]]:
                out = []
                for m in conv:
                    role = m.get("role")
                    content = m.get("content")
                    if not role or content is None:
                        continue
                    out.append({"role": role, "content": content})
                return out

            prior_history = to_history_pairs(history)

            # Build router context with minimal history for fast routing decisions
            routing_messages = build_message_array_context(
                history=prior_history[-2:] if len(prior_history) > 2 else prior_history,
                current_message=user_message,
                user_memories=[],
                limit=3,  # Router needs last 2 turns for context (handles follow-ups like "oui")
                include_memories=False,
            )
            routing_messages.insert(0, {"role": "system", "content": f"Current user_id: {user_id}"})

            # DUAL HISTORY SYSTEM: Check if pain agent session is active
            # If yes, build agent context from ISOLATED pain session (fresh context)
            # If no, build from full conversation history
            # NOTE: We DON'T add the current message to the pain session yet - we wait until
            # after workflow execution to confirm pain agent actually handled it
            if await is_pain_agent_session_active(user_id):
                logger.info(f"🔴 Pain session ACTIVE - building agent context from isolated session for {user_id}")
                # Get body_region from pain session (set by RAG tool calls)
                from app.core.cache import get_pain_session_body_region
                body_region = await get_pain_session_body_region(user_id)
                logger.info(f"🔍 Retrieved body_region '{body_region}' for pain session context")

                # Build context from pain session only (isolated, fresh context)
                # The current user message will be added by build_pain_agent_context
                agent_messages = await build_pain_agent_context(
                    user_id=user_id,
                    current_message=user_message,
                    body_region=body_region,
                    user_memories=[],
                    include_memories=False,
                )
            else:
                logger.info(f"🟢 Pain session INACTIVE - building agent context from full history for {user_id}")
                # Build context from full conversation history (normal flow)
                agent_messages = build_message_array_context(
                    history=prior_history,
                    current_message=user_message,
                    user_memories=[],
                    limit=20,  # Reduced from 40 for faster processing
                    include_memories=False,
                )

            agent_messages.insert(0, {"role": "system", "content": f"Current user_id: {user_id}"})

            final_response_content = ""
            collected_response = ""

            # Prefer streaming when available
            if hasattr(workflow, 'run_with_streaming'):
                def coerce_chunk_to_text(chunk) -> str:
                    if chunk is None:
                        return ""
                    if isinstance(chunk, str):
                        return chunk
                    if isinstance(chunk, (list, tuple)):
                        return "".join(coerce_chunk_to_text(item) for item in chunk)
                    if isinstance(chunk, dict):
                        if 'content' in chunk:
                            return coerce_chunk_to_text(chunk['content'])
                        if 'delta' in chunk:
                            return coerce_chunk_to_text(chunk['delta'])
                        if 'text' in chunk:
                            return coerce_chunk_to_text(chunk['text'])
                        return ""
                    text_attr = getattr(chunk, 'text', None)
                    if text_attr is not None:
                        return coerce_chunk_to_text(text_attr)
                    content_attr = getattr(chunk, 'content', None)
                    if content_attr is not None:
                        return coerce_chunk_to_text(content_attr)
                    delta_attr = getattr(chunk, 'delta', None)
                    if delta_attr is not None:
                        return coerce_chunk_to_text(delta_attr)
                    return str(chunk)

                try:
                    async for response_chunk in workflow.run_with_streaming(
                        user_msg=user_message,
                        router_context=routing_messages,
                        agent_context=agent_messages,
                        user_id=user_id,
                    ):
                        token = coerce_chunk_to_text(response_chunk)
                        if not token:
                            continue
                        # Normalize like Node: strip assistant: prefixes
                        t = token.replace("\r", "")
                        if t.startswith("assistant: "):
                            t = t[len("assistant: "):].lstrip()
                        elif t.startswith("assistant:\n"):
                            t = t[len("assistant:\n"):].lstrip()
                        elif t.startswith("assistant:"):
                            t = t[len("assistant:"):].lstrip()
                        if not t:
                            continue
                        collected_response += t
                        yield emit_vc_event("text_delta", t) + "\n\n"  # Extra padding for Docker flushing
                        # Force immediate flush
                        await asyncio.sleep(0)
                except Exception as e:
                    logger.warning(f"[VC Compat] Streaming failed, will fallback: {e}")

            # If nothing streamed, or as completion
            if not collected_response:
                # Fallback to non-streaming
                if hasattr(workflow, 'run'):
                    result = await workflow.run(
                        user_msg=user_message,
                        router_context=routing_messages,
                        agent_context=agent_messages,
                        user_id=user_id,
                    )
                elif hasattr(workflow, 'arun'):
                    result = await workflow.arun(
                        user_msg=user_message,
                        router_context=routing_messages,
                        agent_context=agent_messages,
                        user_id=user_id,
                    )
                else:
                    result = ""
                final_response_content = (
                    str(getattr(result, 'response', result)).strip() if result is not None else ""
                )
            else:
                final_response_content = collected_response

            if final_response_content.startswith("assistant: "):
                final_response_content = final_response_content[len("assistant: "):].strip()
            elif final_response_content.startswith("assistant:\n"):
                final_response_content = final_response_content[len("assistant:\n"):].strip()
            elif final_response_content.startswith("assistant:"):
                final_response_content = final_response_content[len("assistant:"):].strip()

            if not final_response_content:
                final_response_content = "I'm sorry, I couldn't generate a response. Please try again."

            # DUAL HISTORY SYSTEM: Check which agent handled the request
            handling_agent = workflow.get_current_agent() if hasattr(workflow, 'get_current_agent') else None
            logger.info(f"🎯 Request handled by agent: {handling_agent}")

            # Unwrap pain agent structured JSON responses if needed
            if handling_agent == "EnhancedPainAssessmentAgent":
                final_response_content = unwrap_pain_agent_response(final_response_content)

            # Store messages based on which agent handled the request
            if handling_agent == "EnhancedPainAssessmentAgent":
                logger.info(f"🔴 Pain agent handled request - storing in pain session for {user_id}")

                # START pain session if not already active (first pain agent interaction)
                if not await is_pain_agent_session_active(user_id):
                    logger.info(f"🔴 Starting new pain session for {user_id}")
                    await start_pain_agent_session(user_id)

                # ALWAYS add both user message and assistant response to the pain session storage
                # (the user message was used in context building but not yet persisted to session)
                await add_to_pain_agent_session(user_id, user_message, "user")
                await add_to_pain_agent_session(user_id, final_response_content, "assistant")
                # Pain session remains active for next turn
            elif handling_agent and handling_agent != "CentralRouterAgent":
                # Different specialist agent handled request - end pain session if active
                if await is_pain_agent_session_active(user_id):
                    logger.info(f"🟢 Different agent ({handling_agent}) handled request - ending pain session for {user_id}")
                    await end_pain_agent_session(user_id)
            elif handling_agent == "CentralRouterAgent":
                # Router handled request directly (shouldn't happen normally, but handle it)
                # Check if user message suggests starting a pain session (router will handoff next turn)
                # For now, don't start a session preemptively
                pass

            # Build final conversation like Node: history + current user + assistant
            conversation_messages = list(prior_history)
            conversation_messages.append({"role": "user", "content": user_message})
            conversation_messages.append({"role": "assistant", "content": final_response_content})

            filtered_messages = [m for m in conversation_messages if m.get("role") != "system"]
            yield emit_vc_event("message_stop", filtered_messages)

        except Exception as e:
            logger.error(f"[VC Compat] Error: {e}", exc_info=True)
            # Emit a best-effort message_stop so clients can finalize state
            fallback_conv = list(request.conversationHistory or [])
            fallback_conv.append({"role": "user", "content": request.message})
            fallback_conv.append({"role": "assistant", "content": "An error occurred. Please try again."})
            filtered = [m for m in fallback_conv if m.get("role") != "system"]
            yield emit_vc_event("message_stop", filtered)

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Transfer-Encoding": "chunked",  # Critical for Docker streaming
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


# Register the same handler on the namespaced router so /api/v1/chat/virtual-coach/chat still works
router.add_api_route(
    "/virtual-coach/chat",
    handle_virtual_coach_chat,
    methods=["POST"],
)


__all__ = ["router", "legacy_router", "get_or_build_workflow", "warmup_agents"]

# user memories 
@router.get("/memories/{user_id}")
async def get_user_memories(user_id: str, limit: int = 10):
    """Retrieve user memories from Mem0."""
    try:
        result = get_mem0_client().get_memories(user_id, limit)
        if result["success"]:
            return {"memories": result["memories"], "user_id": user_id}
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories: {str(e)}")

@router.get("/memories/{user_id}/search")
async def search_user_memories(user_id: str, query: str, limit: int = 5):
    """Search user memories in Mem0."""
    try:
        result = get_mem0_client().search_memories(query, user_id, limit)
        if result["success"]:
            return {"memories": result["memories"], "user_id": user_id, "query": query}
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search memories: {str(e)}")

@router.delete("/memories/{user_id}")
async def clear_user_memories(user_id: str):
    """Clear all memories for a specific user."""
    try:
        result = get_mem0_client().delete_all_memories(user_id)
        if result["success"]:
            return {
                "message": f"Successfully cleared all memories for user {user_id}",
                "deleted_count": result.get("deleted_count", 0),
                "user_id": user_id
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memories: {str(e)}")
