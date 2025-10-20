"""LangChain-powered chat endpoint orchestrated by the general coach agent."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import time
import asyncio
import json
from typing import List, Dict, Any, Union, Optional, Tuple

from app.agents.qa import GeneralCoachAgent, RouterDecision
from app.agents.pain import PainAssessmentAgent
from app.agents.training import TrainingAgent
from app.core.memory import get_mem0_client
from app.core.session_manager import SessionManager
from app.core.config import ENABLE_MEM0_MESSAGE_RETRIEVAL
from app.core.cache import (
    get_conversation_history as cache_get_conversation_history,
    add_conversation_message as cache_add_conversation_message,
    start_pain_agent_session as cache_start_pain_agent_session,
    add_to_pain_agent_session as cache_add_to_pain_agent_session,
    get_pain_agent_session_history as cache_get_pain_agent_session_history,
    end_pain_agent_session as cache_end_pain_agent_session,
    is_pain_agent_session_active as cache_is_pain_agent_session_active,
    get_cached_user_memories,
    cache_user_memories,
    get_cached_user_profile,
    cache_user_profile,
    get_cached_training_program,
    cache_user_training_program,
)
from app.core.database import get_user_profile, get_user_training_program_overview

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global LangChain agents (initialised lazily)
_GENERAL_AGENT: Optional[GeneralCoachAgent] = None
_PAIN_AGENT: Optional[PainAssessmentAgent] = None
_TRAINING_AGENT: Optional[TrainingAgent] = None
_AGENT_INIT_LOCK = asyncio.Lock()
_LAST_AGENT_BY_USER: Dict[str, str] = {}


async def get_agents() -> Tuple[GeneralCoachAgent, PainAssessmentAgent, TrainingAgent]:
    """Lazily instantiate and return shared agent instances.

    Ensures the three core agents (general, pain, training) are available for routing.
    """
    global _GENERAL_AGENT, _PAIN_AGENT, _TRAINING_AGENT
    if _GENERAL_AGENT is None or _PAIN_AGENT is None or _TRAINING_AGENT is None:
        async with _AGENT_INIT_LOCK:
            # First initialize the specialist agents (without tools wrapping)
            if _PAIN_AGENT is None:
                logger.info("🧠 Initialising PainAssessmentAgent")
                _PAIN_AGENT = PainAssessmentAgent()
            if _TRAINING_AGENT is None:
                logger.info("🧠 Initialising TrainingAgent")
                _TRAINING_AGENT = TrainingAgent()

            # Then initialize the QA agent (no specialist tools; routing handled server-side)
            if _GENERAL_AGENT is None:
                logger.info("🧠 Initialising GeneralCoachAgent")
                _GENERAL_AGENT = GeneralCoachAgent()
                logger.info("✅ GeneralCoachAgent initialized")

    # mypy help: the tuple will not contain None here
    return _GENERAL_AGENT, _PAIN_AGENT, _TRAINING_AGENT  # type: ignore

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


# REMOVED: extract_body_region_from_message - keyword matching violates CLAUDE.md principle
# The LLM now intelligently chooses body_region via tool parameters based on user's description


async def get_or_fetch_user_memories(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve user memories from Redis cache, or fetch from Mem0 and cache if not available.

    This function is called once at request start to avoid repeated Mem0 API calls during the conversation.
    Memories are cached in Redis for 10 minutes by default.
    """
    # Try cache first
    cached_memories = await get_cached_user_memories(user_id)
    if cached_memories:
        logger.info(f"🧠 Using {len(cached_memories)} cached memories for {user_id}")
        return cached_memories

    # Cache miss - fetch from Mem0
    try:
        mem0_client = get_mem0_client()
        if not mem0_client:
            logger.debug(f"🧠 Mem0 client not available for {user_id}")
            return []

        result = await asyncio.to_thread(
            mem0_client.get_memories,
            user_id=user_id,
            limit=limit,
        )
        if result.get("success"):
            memories = result.get("memories", [])
            logger.info(f"🧠 Fetched {len(memories)} memories from Mem0 for {user_id}")

            # Cache for future requests
            if memories:
                await cache_user_memories(user_id, memories)

            return memories
        else:
            logger.warning(f"🧠 Failed to fetch memories from Mem0: {result.get('error')}")
            return []
    except Exception as e:
        logger.error(f"🧠 Error fetching user memories: {e}")
        return []


async def get_or_fetch_user_profile(user_id: str) -> Dict[str, Any]:
    """Retrieve user profile from Redis cache, or fetch from database and cache if not available.

    This function is called once at request start to avoid repeated database queries.
    Profile is cached in Redis for 1 hour by default.
    """
    # Validate that user_id is numeric
    try:
        numeric_user_id = int(user_id)
    except (ValueError, TypeError):
        logger.error(f"👤 Invalid user_id format: '{user_id}' - must be a numeric ID, not an email")
        return {}

    # Try cache first
    cached_profile = await get_cached_user_profile(user_id)
    if cached_profile:
        logger.info(f"👤 Using cached profile for user {user_id}")
        return cached_profile

    # Cache miss - fetch from database
    try:
        profile = await asyncio.to_thread(get_user_profile, numeric_user_id)
        if profile:
            logger.info(f"👤 Fetched profile from database for user {user_id}")
            # Cache for future requests
            await cache_user_profile(user_id, profile)
            return profile
        else:
            logger.warning(f"👤 No profile found in database for user {user_id}")
            return {}
    except Exception as e:
        logger.error(f"👤 Error fetching user profile: {e}")
        return {}


async def get_or_fetch_user_training_program(user_id: str) -> Dict[str, Any]:
    """Retrieve user training program summary via cache or database."""
    # Validate that user_id is numeric
    try:
        numeric_user_id = int(user_id)
    except (ValueError, TypeError):
        logger.error(f"🏋️ Invalid user_id format: '{user_id}' - must be a numeric ID, not an email")
        return {"hasProgram": False}

    cached_program = await get_cached_training_program(user_id)
    if cached_program is not None:
        if "snapshot" not in cached_program:
            snapshot = _format_training_program_context(cached_program)
            if snapshot:
                cached_program["snapshot"] = snapshot
        logger.info(f"🏋️ Using cached training snapshot for user {user_id}")
        return cached_program

    try:
        program = await asyncio.to_thread(get_user_training_program_overview, numeric_user_id)
        summary: Dict[str, Any] = {"hasProgram": False}
        if program:
            summary = {"hasProgram": True, "program": program}

        snapshot = _format_training_program_context(summary)
        if snapshot:
            summary["snapshot"] = snapshot

        await cache_user_training_program(user_id, summary)
        logger.info(f"🏋️ Cached fresh training snapshot for user {user_id}")
        return summary
    except Exception as e:
        logger.error(f"🏋️ Error fetching training program: {e}")
        return {"hasProgram": False}


def _format_user_profile_context(user_profile: Dict[str, Any]) -> Optional[str]:
    """Convert user profile dict into a readable system context string."""
    if not user_profile:
        return None

    lines = []

    first_name = user_profile.get("firstName") or user_profile.get("firstname")
    last_name = user_profile.get("lastName") or user_profile.get("lastname")
    if first_name and last_name:
        lines.append(f"Nom complet: {first_name} {last_name}")
        lines.append(f"Prénom: {first_name}")
    elif first_name:
        lines.append(f"Prénom: {first_name}")

    email = user_profile.get("email")
    if email:
        lines.append(f"Email: {email}")

    birth_date = user_profile.get("birthDate") or user_profile.get("birthdate")
    if birth_date:
        # Calculate age from birth date
        from datetime import datetime
        try:
            if isinstance(birth_date, str):
                birth_date_obj = datetime.fromisoformat(birth_date.replace('Z', '+00:00'))
            else:
                birth_date_obj = birth_date
            today = datetime.now()
            age = today.year - birth_date_obj.year - ((today.month, today.day) < (birth_date_obj.month, birth_date_obj.day))
            lines.append(f"Âge: {age} ans")
            lines.append(f"Date de naissance: {birth_date}")
        except:
            lines.append(f"Date de naissance: {birth_date}")

    gender = user_profile.get("gender")
    if gender:
        lines.append(f"Genre: {gender}")

    height = user_profile.get("height")
    height_unit = user_profile.get("heightUnit") or user_profile.get("heightunit")
    if height is not None:
        lines.append(
            f"Taille: {height}{f' {height_unit}' if height_unit else ' cm'}".strip()
        )

    weight = user_profile.get("weight")
    weight_unit = user_profile.get("weightUnit") or user_profile.get("weightunit")
    if weight is not None:
        lines.append(
            f"Poids: {weight}{f' {weight_unit}' if weight_unit else ' kg'}".strip()
        )

    goal = user_profile.get("goal")
    if goal:
        lines.append(f"Objectif principal: {goal}")

    fitness_level = user_profile.get("fitnessLevel") or user_profile.get("fitnesslevel")
    if fitness_level:
        lines.append(f"Niveau de fitness: {fitness_level}")

    desired_freq = user_profile.get("desiredTrainingFrequency") or user_profile.get(
        "desiredtrainingfrequency"
    )
    if desired_freq:
        lines.append(f"Fréquence d'entraînement souhaitée: {desired_freq}")

    equipments = user_profile.get("availableEquipments") or user_profile.get(
        "availableequipments"
    )
    if equipments:
        if isinstance(equipments, (list, tuple)):
            equipments_str = ", ".join(str(item) for item in equipments)
        else:
            equipments_str = str(equipments)
        lines.append(f"Équipement disponible: {equipments_str}")

    if not lines:
        return None

    return "PROFIL UTILISATEUR (INFORMATIONS PERSONNELLES À UTILISER DIRECTEMENT):\n" + "\n".join(lines)


def _format_training_program_context(summary: Dict[str, Any]) -> Optional[str]:
    """Convert training program summary into a condensed snapshot for the agent."""
    if not summary:
        return None

    def _fmt(value: Any, default: str = "Non spécifié") -> str:
        if value is None or value == "":
            return default
        return str(value)

    lines: List[str] = ["TRAINING PROGRAM SNAPSHOT:"]

    if not summary.get("hasProgram"):
        lines.append("Aucun programme d'entraînement actif n'est enregistré pour cet utilisateur.")
        return "\n".join(lines)

    program = summary.get("program") or {}
    lines.append(f"Nom: {_fmt(program.get('name'))}")
    lines.append(f"Objectif: {_fmt(program.get('focus'))}")
    status_value = program.get("status")
    if status_value:
        lines.append(f"Statut: {_fmt(status_value)}")
    lines.append(f"Durée: {_fmt(program.get('durationWeeks'))} semaines")
    lines.append(f"Fréquence: {_fmt(program.get('frequencyPerWeek'))} séances par semaine")

    start_date = program.get("startDate")
    end_date = program.get("endDate")
    if start_date or end_date:
        lines.append(f"Période: {_fmt(start_date)} → {_fmt(end_date)}")

    training_days = program.get("trainingDays") or []
    if not training_days:
        lines.append("Aucun jour d'entraînement planifié pour le moment.")
    else:
        max_days = 4
        for day in training_days[:max_days]:
            day_title = f"Jour {day.get('dayNumber')}: {_fmt(day.get('name'), 'Sans nom')}"
            date_part = day.get('date')
            if date_part:
                day_title += f" ({date_part})"
            lines.append(day_title)

            exercises = day.get("exercises") or []
            for exercise in exercises[:4]:
                exercise_line = f"  • {_fmt(exercise.get('name'))}"
                sets = exercise.get('sets')
                reps = exercise.get('reps')
                rest = exercise.get('restSeconds')
                details: List[str] = []
                if sets:
                    details.append(f"{sets} séries")
                if reps:
                    details.append(f"{reps} reps")
                if rest:
                    details.append(f"repos {rest}s")
                if details:
                    exercise_line += f" ({', '.join(details)})"
                lines.append(exercise_line)

            if len(exercises) > 4:
                lines.append(f"  • … {len(exercises) - 4} autres exercices")

        if len(training_days) > max_days:
            lines.append(f"… {len(training_days) - max_days} autres jours d'entraînement non listés")

    return "\n".join(lines)


def build_message_array_context(
    history,
    current_message,
    user_profile=None,
    user_memories=None,
    limit=3,
    include_memories=True,
    training_snapshot=None,
    message_based_memories=None,
):
    """Build universal message array context that works with all LLM providers."""
    messages = []

    # Add user profile as system context (if available)
    if user_profile:
        profile_context = _format_user_profile_context(user_profile)
        if profile_context:
            messages.append({
                "role": "system",
                "content": profile_context
            })

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

    # Add message-based contextual memories (if available)
    if message_based_memories:
        contextual_memory_lines = []
        for memory in message_based_memories:
            if isinstance(memory, dict):
                mem_text = memory.get('memory') or memory.get('content') or memory.get('text', '')
                if mem_text:
                    contextual_memory_lines.append(f"- {mem_text}")
            elif isinstance(memory, str):
                contextual_memory_lines.append(f"- {memory}")

        if contextual_memory_lines:
            contextual_context = f"Relevant memories for current topic:\n" + "\n".join(contextual_memory_lines)
            messages.append({
                "role": "system",
                "content": contextual_context
            })

    # Add training program snapshot (if available)
    if training_snapshot:
        messages.append({
            "role": "system",
            "content": training_snapshot
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


async def build_pain_agent_context(user_id: str, current_message: str, body_region: str = "general", user_memories=None, include_memories=True, user_profile=None):
    """Build isolated context for pain agent using only pain session messages.

    This ensures the pain agent sees only messages from the current pain assessment
    session, not the full conversation history. Messages are still stored in the
    global history for the router to see.

    Also injects available questions from the Redis queue into the context.
    """
    messages = []

    # Add user profile context if provided
    if user_profile:
        profile_context = _format_user_profile_context(user_profile)
        if profile_context:
            messages.append({
                "role": "system",
                "content": profile_context
            })

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

    # Inject Mem0 memory context (if available from background retrieval)
    try:
        from app.knowledge.enhanced_assessment_tools import get_mem0_context
        mem0_context = get_mem0_context(user_id, body_region)

        if mem0_context:
            # Build memory context text
            memory_text = "HISTORIQUE MÉDICAL DU PATIENT (depuis Mem0):\n\n"

            if mem0_context.get("previous_episodes"):
                memory_text += "Épisodes précédents:\n"
                for episode in mem0_context["previous_episodes"][:5]:
                    memory_text += f"- {episode}\n"
                memory_text += "\n"

            if mem0_context.get("treatments_tried"):
                memory_text += "Traitements essayés:\n"
                for treatment in mem0_context["treatments_tried"][:5]:
                    memory_text += f"- {treatment}\n"
                memory_text += "\n"

            if mem0_context.get("medications"):
                memory_text += "Médicaments:\n"
                for med in mem0_context["medications"][:5]:
                    memory_text += f"- {med}\n"
                memory_text += "\n"

            if mem0_context.get("triggers"):
                memory_text += "Facteurs déclencheurs connus:\n"
                for trigger in mem0_context["triggers"][:5]:
                    memory_text += f"- {trigger}\n"
                memory_text += "\n"

            if mem0_context.get("improvements"):
                memory_text += "Ce qui a aidé dans le passé:\n"
                for improvement in mem0_context["improvements"][:5]:
                    memory_text += f"- {improvement}\n"
                memory_text += "\n"

            if mem0_context.get("red_flags_history"):
                memory_text += "⚠️ Alertes historiques:\n"
                for flag in mem0_context["red_flags_history"][:5]:
                    memory_text += f"- {flag}\n"
                memory_text += "\n"

            messages.append({
                "role": "system",
                "content": memory_text.strip()
            })

            total_memories = sum(len(v) for v in mem0_context.values())
            logger.info(f"🧠 Injected {total_memories} Mem0 memories into pain agent context")
        else:
            logger.debug(f"🔍 No Mem0 context available yet for user={user_id} region={body_region}")

    except Exception as e:
        logger.warning(f"⚠️ Failed to inject Mem0 context: {e}")

    # Inject questions queue into context (if available)
    try:
        from app.knowledge.enhanced_assessment_tools import get_full_queue_state
        queue_state = get_full_queue_state(user_id, body_region)
        questions = queue_state.get("questions", [])

        if questions:
            queue_context = "QUESTIONS SUGGÉRÉES (inspirations pour ton prochain question):\n"
            top_questions = questions[:15]
            for idx, q in enumerate(top_questions, 1):  # Limit to top 15 for better coverage
                queue_context += f"{idx}. {q}\n"

            messages.append({
                "role": "system",
                "content": queue_context
            })
            logger.info(
                "📋 Injected %d suggested questions into pain agent context",
                len(top_questions)
            )
            logger.info(
                "🗂️ Queue snapshot for user=%s region=%s: %s",
                user_id,
                body_region,
                top_questions,
            )
    except Exception as e:
        logger.warning(f"⚠️ Failed to inject queue into context: {e}")

    # Get isolated pain session history from LangChain memory (NOT old cache)
    from app.core.langchain_memory import get_conversation_history
    pain_langchain_history = get_conversation_history(user_id, session_type="pain")
    pain_messages = pain_langchain_history.messages

    # Convert LangChain messages to dict format
    for lc_msg in pain_messages:
        if hasattr(lc_msg, 'type'):
            if lc_msg.type == 'human':
                messages.append({"role": "user", "content": lc_msg.content})
            elif lc_msg.type == 'ai':
                messages.append({"role": "assistant", "content": lc_msg.content})
            elif lc_msg.type == 'system':
                messages.append({"role": "system", "content": lc_msg.content})

    # Add current user message
    messages.append({
        "role": "user",
        "content": current_message
    })

    logger.info(f"🔴 Built pain agent context with {len(pain_messages)} session messages + current message")
    return messages


async def build_training_agent_context(
    user_id: str,
    current_message: str,
    user_profile: Dict[str, Any],
    user_memories: List[Any],
    training_snapshot: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Build context payload for the training agent with isolated history."""
    # Retrieve conversation history from LangChain memory
    training_history = await SessionManager.get_conversation_for_agent(
        user_id=user_id,
        agent_type="training",
        limit=None,
    )

    # Start with system context (profile, memories, snapshot)
    context_messages = build_message_array_context(
        history=[],
        current_message=current_message,
        user_profile=user_profile,
        user_memories=user_memories,
        include_memories=True,
        training_snapshot=training_snapshot,
        limit=10,
    )

    # Remove placeholder current message, will re-append after history
    if context_messages:
        context_messages = context_messages[:-1]

    # Attach training-specific conversation history
    for msg in training_history:
        if hasattr(msg, "type"):
            role = "user" if msg.type == "human" else "assistant" if msg.type == "ai" else "system"
            content = msg.content if hasattr(msg, "content") else str(msg)
            context_messages.append({"role": role, "content": content})

    # Finally, add current user message
    context_messages.append({"role": "user", "content": current_message})

    return context_messages, training_history


def build_session_summary(
    active_agent: str,
    last_agent: Optional[str],
    pain_metadata: Optional[Dict[str, Any]],
    training_snapshot: Optional[str],
) -> str:
    """Create a concise session summary for router context."""
    lines: List[str] = []
    lines.append(f"Active agent: {active_agent}")
    if pain_metadata:
        body_region = pain_metadata.get("body_region")
        if body_region:
            lines.append(f"Pain session body region: {body_region}")
    if last_agent:
        lines.append(f"Dernier agent utilisé: {last_agent}")
    else:
        lines.append("Dernier agent utilisé: GeneralQAAgent")
    if training_snapshot:
        lines.append("TRAINING PROGRAM SNAPSHOT:")
        lines.append(training_snapshot)
    return "\n".join(lines)


async def warmup_agents() -> None:
    """Initialize the LangChain agents at startup."""
    await get_agents()


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


def _split_for_frontend_chunks(text: str, just_emitted_backticks: bool) -> (List[str], bool):
    """Split outgoing text into UI-friendly text_delta chunks.

    Ensures the opening fence appears as two consecutive chunks: first "```", then "json".
    """
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        # If last emission was raw backticks, peel off leading 'json' as a standalone chunk
        if just_emitted_backticks:
            if text.startswith("json", i):
                chunks.append("json")
                i += 4
                just_emitted_backticks = False
                continue
            else:
                just_emitted_backticks = False

        # Normalize combined tokens like '```json' or '```\njson'
        if text.startswith("```json", i):
            chunks.append("```")
            chunks.append("json")
            i += len("```json")
            continue
        if text.startswith("```\njson", i):
            chunks.append("```")
            chunks.append("json")
            i += len("```\njson")
            continue

        # Plain opening fence
        if text.startswith("```", i):
            chunks.append("```")
            just_emitted_backticks = True
            i += 3
            continue

        # Aggregate non-special text until next fence start
        start = i
        while i < n and not text.startswith("```", i):
            i += 1
        if i > start:
            chunks.append(text[start:i])

    return chunks, just_emitted_backticks


def unwrap_pain_agent_response(response_text: str) -> str:
    """Pass-through function for pain agent responses.

    The pain agent now outputs frontend-compatible format directly, so no transformation is needed.
    This function is kept for backwards compatibility and potential future processing needs.

    Frontend expects questions in this format:
    <question_text>
    ```json
    {"id": "option1", "label": "First option"}
    {"id": "option2", "label": "Second option"}
    ```

    Args:
        response_text: The response from pain agent (already in frontend format)

    Returns:
        The response text as-is (already formatted correctly)
    """
    # Pain agent now outputs frontend format directly - no transformation needed
    logger.info(f"✅ Pain agent response already in frontend format ({len(response_text)} chars)")
    return response_text.strip()


@legacy_router.post("/virtual-coach/chat")
async def handle_virtual_coach_chat(request: VirtualCoachChatRequest) -> StreamingResponse:
    """Compatibility endpoint that mirrors the legacy Node /virtual-coach/chat contract."""

    async def stream_generator():
        req_start_ts = time.perf_counter()
        user_id = str(request.userId)
        user_message = request.message
        history = request.conversationHistory or []
        prior_history = [
            {"role": entry.get("role"), "content": entry.get("content")}
            for entry in history
            if entry.get("role") and entry.get("content") is not None
        ]

        try:
            general_agent, pain_agent, training_agent = await get_agents()
            logger.info(f"[VC Compat] Starting chat for user {user_id}")

            # Check for manual agent switches (user wants to cancel current specialist)
            manual_switch_target = await SessionManager.should_switch_agent(user_id, user_message)
            if manual_switch_target:
                logger.info(f"🔄 Manual agent switch detected for user {user_id} -> {manual_switch_target}")
                # Get current active agent
                current_agent = await SessionManager.get_active_agent(user_id)
                if current_agent in ("pain", "training"):
                    # End the current specialist session
                    await SessionManager.end_specialist_session(
                        user_id=user_id,
                        agent_type=current_agent,
                        clear_history=True
                    )
                    logger.info(f"✅ Ended {current_agent} session, switched to general")

            # Get current active agent for routing
            active_agent_type = await SessionManager.get_active_agent(user_id)
            logger.info(f"🎯 Active agent for user {user_id}: {active_agent_type}")

            try:
                user_profile = await get_or_fetch_user_profile(user_id)
                user_memories = await get_or_fetch_user_memories(user_id, limit=20)
                training_program_summary = await get_or_fetch_user_training_program(user_id)
                training_snapshot = training_program_summary.get("snapshot")

                # Search Mem0 using current user message for contextual memory retrieval
                # This block can be disabled by setting ENABLE_MEM0_MESSAGE_RETRIEVAL=false in .env
                retrieval_start = time.perf_counter()
                message_based_memories = []
                if ENABLE_MEM0_MESSAGE_RETRIEVAL:
                    try:
                        mem0_client = get_mem0_client()
                        if mem0_client:
                            logger.info(f"🔍 [Mem0 Search] Searching with query: '{user_message[:100]}...'")
                            result = await asyncio.to_thread(
                                mem0_client.search_memories,
                                query=user_message,
                                user_id=user_id,
                                limit=5,
                            )
                            retrieval_elapsed = time.perf_counter() - retrieval_start

                            if result.get("success"):
                                message_based_memories = result.get("memories", [])
                                logger.info(
                                    f"✅ [Mem0 Search] Retrieved {len(message_based_memories)} contextual memories in {retrieval_elapsed:.3f}s"
                                )
                            else:
                                logger.warning(f"⚠️ [Mem0 Search] Failed: {result.get('error')}")
                    except Exception as mem_exc:
                        retrieval_elapsed = time.perf_counter() - retrieval_start
                        logger.error(f"❌ [Mem0 Search] Error after {retrieval_elapsed:.3f}s: {mem_exc}")
                else:
                    logger.info("⊘ [Mem0 Search] Disabled via ENABLE_MEM0_MESSAGE_RETRIEVAL config")

            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(f"[VC Compat] Context preload failed: {exc}", exc_info=True)
                user_profile = {}
                user_memories = []
                training_program_summary = {}
                training_snapshot = None
                message_based_memories = []

            # SMART ROUTING: determine which agent should handle this turn
            handling_agent_type = active_agent_type
            handling_agent_obj: Optional[Any] = None
            handling_agent = ""
            context_messages: List[Dict[str, Any]] = []
            general_context: Optional[List[Dict[str, Any]]] = None
            training_history: List[Any] = []
            body_region = "general"

            current_metadata: Dict[str, Any] = {}
            if active_agent_type in ("pain", "training"):
                current_metadata = await SessionManager.get_session_metadata(user_id)

            if active_agent_type == "general":
                logger.info("🧭 Routing decision required (currently with General agent)")
                general_context = build_message_array_context(
                    history=prior_history,
                    current_message=user_message,
                    user_profile=user_profile,
                    user_memories=user_memories,
                    include_memories=True,
                    training_snapshot=training_snapshot,
                    message_based_memories=message_based_memories,
                    limit=10,
                )

                session_summary = build_session_summary(
                    active_agent=active_agent_type,
                    last_agent=_LAST_AGENT_BY_USER.get(user_id),
                    pain_metadata=current_metadata if current_metadata else {},
                    training_snapshot=training_snapshot,
                )
                router_conversation = list(prior_history)
                router_conversation.append({"role": "user", "content": user_message})

                try:
                    decision = await general_agent.decide_route(
                        router_conversation,
                        user_message,
                        session_summary=session_summary,
                    )
                    logger.info(
                        "🧭 Router decision user=%s -> %s (confidence=%.2f): %s",
                        user_id,
                        decision.agent,
                        decision.confidence,
                        decision.reason,
                    )
                except Exception as routing_error:
                    logger.error(
                        f"❌ Router decision failed for user {user_id}: {routing_error}",
                        exc_info=True,
                    )
                    decision = RouterDecision(
                        agent="general",
                        reason="Routing fallback due to error",
                        confidence=0.0,
                    )

                if decision.agent == "pain":
                    handling_agent_type = "pain"
                    body_region = (decision.body_region or "general").strip()
                    try:
                        await SessionManager.start_specialist_session(
                            user_id=user_id,
                            agent_type="pain",
                            metadata={"body_region": body_region},
                        )
                        logger.info(
                            "🚀 Started pain session for user %s (region=%s)",
                            user_id,
                            body_region,
                        )
                        try:
                            from app.knowledge.enhanced_assessment_tools import launch_mem0_retrieval

                            launch_mem0_retrieval(user_id, body_region, user_message)
                        except Exception as launch_error:
                            logger.warning(f"⚠️ Failed to launch Mem0 retrieval: {launch_error}")
                    except Exception as start_error:
                        logger.error(f"❌ Failed to start pain session: {start_error}", exc_info=True)
                        handling_agent_type = "general"
                        general_context = general_context or []
                        current_metadata = {}
                    else:
                        current_metadata = {"body_region": body_region}

                elif decision.agent == "training":
                    handling_agent_type = "training"
                    try:
                        await SessionManager.start_specialist_session(
                            user_id=user_id,
                            agent_type="training",
                            metadata={},
                        )
                        logger.info("🚀 Started training session for user %s", user_id)
                    except Exception as start_error:
                        logger.error(f"❌ Failed to start training session: {start_error}", exc_info=True)
                        handling_agent_type = "general"
                        general_context = general_context or []
                        current_metadata = {}
                    else:
                        current_metadata = {}
                else:
                    handling_agent_type = "general"
                    current_metadata = {}

            # Update active agent type based on routing decision
            active_agent_type = handling_agent_type

            if handling_agent_type == "pain":
                handling_agent_obj = pain_agent
                handling_agent = "PainAssessmentAgent"

                if not current_metadata:
                    current_metadata = await SessionManager.get_session_metadata(user_id)
                body_region = current_metadata.get("body_region", body_region)

                # Store user message BEFORE building context to include in memory
                await SessionManager.store_message(
                    user_id=user_id,
                    message=user_message,
                    role="user",
                    agent_context="pain"
                )

                context_messages = await build_pain_agent_context(
                    user_id=user_id,
                    current_message=user_message,
                    body_region=body_region,
                    user_memories=user_memories,
                    include_memories=False,
                    user_profile=user_profile
                )

            elif handling_agent_type == "training":
                handling_agent_obj = training_agent
                handling_agent = "TrainingAgent"

                # Store user message BEFORE building context
                await SessionManager.store_message(
                    user_id=user_id,
                    message=user_message,
                    role="user",
                    agent_context="training"
                )

                context_messages, training_history = await build_training_agent_context(
                    user_id=user_id,
                    current_message=user_message,
                    user_profile=user_profile,
                    user_memories=user_memories,
                    training_snapshot=training_snapshot,
                )

                logger.info(
                    "🏋️ Training agent context prepared: %d messages (%d from history)",
                    len(context_messages),
                    len(training_history),
                )
                for idx, msg in enumerate(context_messages[-5:]):
                    logger.debug(f"  [{idx}] {msg.get('role')}: {msg.get('content', '')[:100]}")

            else:
                handling_agent_obj = general_agent
                handling_agent = "GeneralQAAgent"
                if general_context is None:
                    general_context = build_message_array_context(
                        history=prior_history,
                        current_message=user_message,
                        user_profile=user_profile,
                        user_memories=user_memories,
                        include_memories=True,
                        training_snapshot=training_snapshot,
                        message_based_memories=message_based_memories,
                        limit=10,
                    )
                context_messages = general_context

            # Set user context for tools (so they can access user_id programmatically)
            from app.knowledge.enhanced_assessment_tools import set_user_context
            set_user_context(user_id)

            collected_response_parts: List[str] = []
            final_payload: Optional[Dict[str, Any]] = None
            just_emitted_backticks = False
            token_chunks = 0
            first_token_ts: Optional[float] = None

            try:
                logger.info(f"🚀 Starting astream_response for {handling_agent}")
                event_count = 0
                async for agent_event in handling_agent_obj.astream_response(
                    context_messages,
                    user_message,
                ):
                    event_count += 1
                    event_type = agent_event.get("type")
                    logger.debug(f"📦 Event #{event_count}: {event_type}")

                    if event_type == "tool_result":
                        tool_name = agent_event.get("tool_name") or "tool"
                        tool_payload = {
                            "function_name": tool_name,
                            "status": "ok",
                        }
                        yield emit_vc_event("tool_result", tool_payload) + ""
                        await asyncio.sleep(0)
                        continue

                    if event_type == "text":
                        text = (agent_event.get("content") or "").replace("", "")
                        pieces, just_emitted_backticks = _split_for_frontend_chunks(text, just_emitted_backticks)
                        for piece in pieces:
                            if not piece:
                                continue
                            if first_token_ts is None:
                                first_token_ts = time.perf_counter()
                            collected_response_parts.append(piece)
                            token_chunks += 1
                            yield emit_vc_event("text_delta", piece) + ""
                            await asyncio.sleep(0)
                        continue

                    if event_type == "final":
                        final_payload = agent_event
                        break

                logger.info(f"✅ Completed astream_response: {event_count} events processed")
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(f"[VC Compat] Streaming failed, falling back: {exc}", exc_info=True)

            final_response_content = ""
            if final_payload and final_payload.get("content"):
                final_response_content = str(final_payload["content"]).strip()
            if not final_response_content:
                final_response_content = "".join(collected_response_parts).strip()

            # FALLBACK: If streaming failed (no response), try non-streaming ainvoke
            if not final_response_content and handling_agent in ("TrainingAgent", "PainAssessmentAgent"):
                logger.warning(f"⚠️ Streaming failed for {handling_agent}, falling back to ainvoke")
                try:
                    result = await handling_agent_obj.ainvoke(context_messages, user_message)
                    final_response_content = result.get("output") or result.get("final_output") or str(result)
                    logger.info(f"✅ Fallback ainvoke succeeded: {len(final_response_content)} chars")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback ainvoke also failed: {fallback_error}", exc_info=True)

            if not final_response_content:
                final_response_content = "Je suis désolé, je n'ai pas pu générer de réponse. Peux-tu reformuler ?"

            total_elapsed = time.perf_counter() - req_start_ts
            if first_token_ts is not None:
                ttfb = first_token_ts - req_start_ts
                logger.info(
                    "⏱️ Turn timings | user=%s agent=%s streamed=%s ttfb=%.2fs total=%.2fs chunks=%d",
                    user_id,
                    handling_agent,
                    bool(collected_response_parts),
                    ttfb,
                    total_elapsed,
                    token_chunks,
                )
            else:
                logger.info(
                    "⏱️ Turn timings | user=%s agent=%s streamed=%s total=%.2fs",
                    user_id,
                    handling_agent,
                    False,
                    total_elapsed,
                )

            # Store messages using SessionManager (handles both general and specialist contexts)
            # BUT: Don't store if already stored (direct routing stored user message early, tools store everything)
            try:
                # Check if we need to store messages
                # - Direct routing (pain/training): Only store ASSISTANT message (user already stored early)
                # - General agent WITHOUT tools: Store BOTH messages
                if handling_agent in ("PainAssessmentAgent", "TrainingAgent"):
                    # Direct routing: User message was stored BEFORE calling agent
                    # Only store assistant response here
                    await SessionManager.store_message(
                        user_id=user_id,
                        message=final_response_content,
                        role="assistant",
                        agent_context=active_agent_type
                    )
                    logger.debug(f"📝 Stored assistant response for {handling_agent} (user message stored early)")

                else:
                    # General agent: store both user and assistant messages
                    await SessionManager.store_message(
                        user_id=user_id,
                        message=user_message,
                        role="user"
                    )
                    await SessionManager.store_message(
                        user_id=user_id,
                        message=final_response_content,
                        role="assistant"
                    )
                    logger.debug(f"📝 Stored both messages for {handling_agent}")
                # Also update old cache for backwards compatibility
                await add_to_conversation_cache(user_id, user_message, "user")
                await add_to_conversation_cache(user_id, final_response_content, "assistant")
            except Exception as exc:
                logger.warning(f"⚠️ Failed to update conversation storage: {exc}")

            # Handle completion detection for directly-routed specialists
            # Check if specialist session should end based on response content
            try:
                if handling_agent == "PainAssessmentAgent":
                    # Check if pain assessment completed
                    if '"assessment_complete": true' in final_response_content or '"assessment_complete":true' in final_response_content:
                        logger.info(f"🏁 Pain assessment completed for user {user_id} (direct route)")
                        await SessionManager.end_specialist_session(
                            user_id=user_id,
                            agent_type="pain",
                            clear_history=True
                        )
                    elif "pain assessment session closed" in final_response_content.lower():
                        logger.info(f"🚪 Pain assessment manually exited for user {user_id} (direct route)")
                        await SessionManager.end_specialist_session(
                            user_id=user_id,
                            agent_type="pain",
                            clear_history=True
                        )

                elif handling_agent == "TrainingAgent":
                    # Check if training program created
                    if "programme créé avec succès" in final_response_content.lower() or "✅ programme créé" in final_response_content.lower():
                        logger.info(f"🏁 Training program completed for user {user_id} (direct route)")
                        await SessionManager.end_specialist_session(
                            user_id=user_id,
                            agent_type="training",
                            clear_history=True
                        )
            except Exception as detection_error:
                logger.warning(f"⚠️ Could not detect specialist completion: {detection_error}")

            # Track last handling agent
            if handling_agent == "PainAssessmentAgent":
                _LAST_AGENT_BY_USER[user_id] = "PainAssessmentAgent"
            elif handling_agent == "TrainingAgent":
                _LAST_AGENT_BY_USER[user_id] = "TrainingAgent"
            elif handling_agent == "GeneralQAAgent":
                # General QA handled the query directly
                _LAST_AGENT_BY_USER[user_id] = "GeneralQAAgent"

                # Store general QA conversations to Mem0
                try:
                    mem0_client = get_mem0_client()
                    if mem0_client:
                        result = await asyncio.to_thread(
                            mem0_client.add_memory,
                            message=user_message,
                            user_id=user_id,
                            metadata={
                                "type": "qa_conversation",
                                "role": "user",
                                "agent": "GeneralQAAgent",
                            },
                        )
                        # Log outcome and invalidate cache on success so next turn sees fresh data
                        if isinstance(result, dict) and result.get("success"):
                            logger.info("🧠 [Mem0] Saved QA user message (%s chars)", len(user_message))
                            try:
                                from app.core.cache import invalidate_user_memories_cache
                                invalidate_user_memories_cache(user_id)
                            except Exception as cache_exc:
                                logger.debug(f"🧠 [Mem0] Cache invalidate skipped: {cache_exc}")
                        else:
                            err = (result or {}).get("error") if isinstance(result, dict) else "unknown error"
                            logger.warning(f"⚠️ [Mem0] Save returned unsuccessful result: {err}")
                except Exception as exc:
                    logger.warning(f"⚠️ Failed to store QA message to Mem0: {exc}")

            conversation_messages = list(prior_history)
            conversation_messages.append({"role": "user", "content": user_message})
            conversation_messages.append({"role": "assistant", "content": final_response_content})
            filtered_messages = [m for m in conversation_messages if m.get("role") != "system"]
            yield emit_vc_event("message_stop", filtered_messages)

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"[VC Compat] Error: {exc}", exc_info=True)
            fallback_conv = list(prior_history)
            fallback_conv.append({"role": "user", "content": user_message})
            fallback_conv.append({"role": "assistant", "content": "Une erreur est survenue. Merci de réessayer."})
            filtered = [m for m in fallback_conv if m.get("role") != "system"]
            yield emit_vc_event("message_stop", filtered)

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Transfer-Encoding": "chunked",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
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


__all__ = ["router", "legacy_router", "warmup_agents"]

# user memories 
@router.get("/memories/{user_id}")
async def get_user_memories(user_id: str, limit: int = 10):
    """Retrieve user memories from Mem0."""
    try:
        mem0_client = get_mem0_client()
        if not mem0_client:
            raise HTTPException(status_code=503, detail="Mem0 client not available")
        result = await asyncio.to_thread(mem0_client.get_memories, user_id, limit)
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
        mem0_client = get_mem0_client()
        if not mem0_client:
            raise HTTPException(status_code=503, detail="Mem0 client not available")
        result = await asyncio.to_thread(
            mem0_client.search_memories, query, user_id, limit
        )
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
        mem0_client = get_mem0_client()
        if not mem0_client:
            raise HTTPException(status_code=503, detail="Mem0 client not available")
        result = await asyncio.to_thread(mem0_client.delete_all_memories, user_id)
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


# ===============
# VC compat routes
# ===============

@legacy_router.put("/virtual-coach/create-openai-thread")
async def create_openai_thread():
    """Return a thread id for frontend bootstrap.

    Returns { data: { threadId } } per contract.
    """
    import uuid
    thread_id = f"vc_{uuid.uuid4().hex}"
    return {"data": {"threadId": thread_id}}


@legacy_router.post("/virtual-coach/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """Accept audio and return a transcription string (compat stub)."""
    try:
        name = getattr(file, "filename", "audio") or "audio"
        return {"transcription": f"[stub] Received {name}. Configure ASR to enable transcription."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
