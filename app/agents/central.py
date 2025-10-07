from __future__ import annotations

"""Central Router Agent and Workflow for specialist agents."""

import logging
import asyncio
from typing import List, Any, Dict
import os

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.tools import FunctionTool


logger = logging.getLogger(__name__)


_DESCRIPTION = (
    "Central routing coordinator that analyzes user messages and routes them to the appropriate "
    "specialist agent (pain, training, nutrition, health insights, or general Q&A)."
)


_SYSTEM_PROMPT = """
Tu es l'Agent Routeur Central pour GoGain AI Coach.

Ta mission est de décider quel spécialiste doit traiter le message de l'utilisateur.
Tu ne dois JAMAIS répondre directement à la place d'un spécialiste.

SALUTATIONS :
- Toute salutation ("salut", "hey", "bonjour", "hello", "hi") → handoff_GeneralQAAgent

PRIORITÉ AU CONTEXTE EN COURS (CRITIQUE) :
- Tu peux et DOIS analyser tout l'historique de discussion pour deviner le flux en cours.
- Si l'utilisateur est manifestement dans un parcours d'évaluation de la douleur, poursuis avec l'EnhancedPainAssessmentAgent tant qu'il ne change pas clairement de sujet.
- MAIS si l'utilisateur pose une question COMPLÈTEMENT DIFFÉRENTE (ex: "C'est quoi le NEAT?", "Tapis de course ou dehors?"), BASCULE vers l'agent approprié (GeneralQAAgent).

CHANGEMENT DE SUJET EXPLICITE :
- Si l'utilisateur pose une question qui N'EST PAS liée à sa douleur actuelle → CHANGE D'AGENT immédiatement
- Exemples de changement de sujet:
  * "C'est quoi le NEAT?" → GeneralQAAgent
  * "Tapis de course ou dehors?" → GeneralQAAgent
  * "Comment faire un programme muscu?" → PersonalTrainingAgent
  * "Que manger pour perdre du poids?" → NutritionAgent

CONTINUITÉ AVANT CHANGEMENT (pour réponses courtes seulement):
- Si le dernier agent actif était l'EnhancedPainAssessmentAgent et que le message actuel ressemble à une réponse à une question sur la douleur, reste avec l'agent douleur.
- Considère comme réponses de suivi douleur : durée/début (« depuis 2 mois »), intensité (« 6/10 »), localisation (« à l'avant du genou », « côté interne »), caractère (« douleur aiguë », « sensation de brûlure »), déclencheurs (« quand je m'accroupis »), facteurs calmants (« le repos aide »), évolution (« pire le matin »), confirmations oui/non à des questions de douleur, ou toute précision sur les symptômes.
- Si l'utilisateur affirme ne plus ressentir de douleur ou demande clairement un plan sans lien avec la douleur, tu peux diriger vers un autre agent approprié.

Utilise immédiatement l'outil de handoff adapté :
- Toute mention de douleur OU toute réponse de suivi douleur → handoff_EnhancedPainAssessmentAgent
- Création d'un plan d'entraînement uniquement → handoff_PersonalTrainingAgent
- Création d'un plan alimentaire uniquement → handoff_NutritionAgent
- Questions générales de santé/fitness NON liées à une douleur → handoff_GeneralQAAgent
- Si tu ne sais pas quel agent choisir ET que ce n'est pas lié à une douleur → handoff_GeneralQAAgent

RÈGLES DE ROUTAGE (INTERPRÈTE LE MESSAGE ACTUEL DANS SON CONTEXTE) :
- Agent douleur : dès que le message mentionne une douleur/blessure/gêne OU répond à une question de l'agent douleur (cf. exemples ci-dessus) → envoie à l'agent douleur.
- Agent entraînement : uniquement si le message demande explicitement « crée-moi un programme », « construis une routine », etc., et que l'utilisateur n'est plus en parcours douleur.
- Agent nutrition : uniquement si le message demande explicitement un plan alimentaire et qu'il n'y a pas de suivi douleur en cours.
- Agent General QA : pour les conseils généraux lorsque le parcours douleur est clairement terminé ou que le sujet change.

RÈGLES DE DÉCISION SENSIBLES AU CONTEXTE :
- Maintiens le fil : ne coupe pas une évaluation douleur en plein milieu.
- Vérifie les dernières questions de l'assistant : si elles concernaient la douleur, suppose que la réponse actuelle en est le suivi.
- En cas de doute raisonnable (ex. message court comme « à l'avant », « depuis 2 mois », « oui »), privilégie l'agent douleur plutôt que de changer.
- Si l'utilisateur change explicitement de sujet (« Au fait, peux-tu me faire un programme muscu ? ») ou déclare ne plus être gêné, bascule vers l'agent correspondant.

GESTION DES CAS LIMITE :
- Réponses brèves (« oui », « non », « parfois », « 5/10 », « à l'avant », « il y a 2 mois », etc.) → vérifie la dernière question posée. Si elle venait de l'agent douleur, reste sur l'agent douleur.
- Si l'utilisateur affirme que la douleur a disparu ou n'est plus un problème, tu peux diriger vers entraînement/nutrition/général selon le besoin exprimé.
"""

# --- Temporal Awareness (commented, non-intrusive) ---
# NOTE: The chat layer can auto-inject a system message like:
#   current_datetime = "Current date and time: 2025-09-30T14:32:15Z (mardi 30 septembre 2025, 14h32)"
# Guidance for routing (keep commented to avoid changing behavior):
# - Utilise la date/heure actuelle si elle aide à interpréter la récence d'un sujet.
# - Si un message fait référence à un évènement récent, favorise la continuité avec l'agent actif.


def build_router(llm: BaseLLM) -> FunctionAgent:
    """Factory that returns a configured Central Router Agent instance."""

    agent = FunctionAgent(
        name="CentralRouterAgent",
        description=_DESCRIPTION,
        system_prompt=_SYSTEM_PROMPT,
        llm=llm,
        tools=[],  # tools will be attached later via workflow
        verbose=True,
    )

    # Define explicit handoff targets for downstream workflow wiring
    agent.can_handoff_to = [
        "EnhancedPainAssessmentAgent",
        "PersonalTrainingAgent",
      #  "NutritionAgent",
      #  "HealthInsightsAgent",
        "GeneralQAAgent",
    ]

    logger.info("✅ CentralRouterAgent created with handoff capability")
    return agent


def create_handoff_function(target_agent: str, workflow: "EnhancedAgentWorkflow"):
    """Return a callable that performs a handoff to target_agent."""

    def handoff(message: str) -> str:
        logger.info(f"🤝 Handoff to {target_agent} with message: {message[:100]}...")
        workflow._set_current_agent(target_agent)
        return f"Handoff to {target_agent} complete."

    return handoff


class EnhancedAgentWorkflow(AgentWorkflow):
    """Workflow manager with explicit handoff support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_agent = self.root_agent
        self._user_id = None  # Store user_id for tools to access
        self._initialize_handoff_tools()

    def _set_current_agent(self, agent_name: str):
        if agent_name in self.agents:
            self._current_agent = agent_name
            logger.info(f"🔄 Current agent set to {agent_name}")
        else:
            logger.error(f"❌ Agent {agent_name} not found in workflow")

    def get_current_agent(self) -> str:
        """Get the name of the agent that handled the last request."""
        return self._current_agent

    def _initialize_handoff_tools(self):
        """Attach handoff tools to only the CentralRouterAgent."""
        for agent_name, agent_config in self.agents.items():
            # Only the CentralRouterAgent should have handoff tools
            if agent_name == "CentralRouterAgent" and agent_config.can_handoff_to:
                tools = []
                for target_agent in agent_config.can_handoff_to:
                    if target_agent in self.agents:
                        tool_name = f"handoff_{target_agent}"
                        tool = FunctionTool.from_defaults(
                            fn=create_handoff_function(target_agent, self),
                            name=tool_name,
                            description=f"Hand off conversation to {target_agent}",
                        )
                        tools.append(tool)
                        logger.info(f"✅ Created handoff tool {tool_name} for {agent_name}")

                if agent_config.tools is None:
                    agent_config.tools = tools
                else:
                    agent_config.tools.extend(tools)

                logger.info(f"📦 Agent {agent_name} now has {len(agent_config.tools)} tools")
            else:
                # Ensure specialist agents have NO handoff capabilities
                if hasattr(agent_config, 'can_handoff_to'):
                    agent_config.can_handoff_to = []
                # Clear any existing handoff tools
                if agent_config.tools:
                    agent_config.tools = [tool for tool in agent_config.tools
                                        if not tool.metadata.name.startswith("handoff_")]
                logger.info(f"🚫 Cleared handoff tools for specialist agent {agent_name}")

    async def get_tools(self, agent_name: str, input_str: str = None):
        """Override to prevent automatic handoff tool creation by parent class."""
        # CRITICAL: Add loop prevention to prevent infinite get_tools calls
        if not hasattr(self, '_get_tools_call_count'):
            self._get_tools_call_count = {}

        call_key = f"{agent_name}_{self._current_agent}"
        self._get_tools_call_count[call_key] = self._get_tools_call_count.get(call_key, 0) + 1

        # Prevent infinite loops - max 15 calls per agent combination (allow extensive tool usage)
        if self._get_tools_call_count[call_key] > 15:
            # IMPORTANT: Do NOT return an empty list.
            # Returning [] causes downstream LLM clients to send tool_choice without tools,
            # which triggers provider errors (e.g., OpenAI 400: tool_choice requires tools).
            logger.warning(
                f"⚠️ LOOP PREVENTION: get_tools called {self._get_tools_call_count[call_key]} times for {call_key}, "
                "returning existing non-handoff tools instead of empty list"
            )

            current_agent_name = self._current_agent
            current_agent_config = self.agents[current_agent_name]

            # Return the agent's existing tools (filtering out handoff_ for specialists),
            # and skip dynamic retrieval to avoid perpetuating the loop.
            if current_agent_name != "CentralRouterAgent":
                agent_tools = [
                    tool for tool in (current_agent_config.tools or [])
                    if not tool.metadata.name.startswith("handoff_")
                ]
            else:
                agent_tools = current_agent_config.tools or []

            logger.info(
                f"🔧 LOOP PREVENTION: returning {len(agent_tools)} tools for {current_agent_name}"
            )
            return self._ensure_tools_are_async(agent_tools)

        # Use the current agent, not the agent_name parameter (which might be stale)
        current_agent_name = self._current_agent

        logger.info(f"🔧 get_tools called for agent_name='{agent_name}', but using current_agent='{current_agent_name}' (call #{self._get_tools_call_count[call_key]})")

        # Get the current agent's regular tools (excluding cached handoff tools)
        current_agent_config = self.agents[current_agent_name]

        # For specialist agents, only use their original tools (no handoff tools)
        if current_agent_name != "CentralRouterAgent":
            # Specialist agents should have NO handoff tools
            agent_tools = []
            if current_agent_config.tools:
                # Filter out any handoff tools that might have been added
                agent_tools = [tool for tool in current_agent_config.tools
                             if not tool.metadata.name.startswith("handoff_")]
            tools = [*agent_tools]
            logger.info(f"🔧 Specialist agent {current_agent_name} has {len(tools)} non-handoff tools")
        else:
            # CentralRouterAgent uses all its tools (including handoff tools)
            agent_tools = current_agent_config.tools or []
            tools = [*agent_tools]
            logger.info(f"🔧 Central router agent has {len(tools)} tools (including handoff)")

        # Dynamic tool retriever not used in current setup; intentionally omitted

        logger.info(f"🔧 Final tool count for {current_agent_name}: {len(tools)}")
        return self._ensure_tools_are_async(tools)

    async def run(self, user_msg: str, router_context=None, agent_context=None, user_id=None, **kwargs):
        """Custom run method that handles agent transitions with separate contexts."""
        logger.info(f"🚀 EnhancedAgentWorkflow starting with root agent: {self.root_agent}")

        # Store user_id for tools to access programmatically
        self._user_id = user_id or "unknown_user"

        # CRITICAL: Reset get_tools call counter for new workflow run
        self._get_tools_call_count = {}

        # Handle both string and message array contexts
        if isinstance(router_context, list):
            # Message array - extract last user message for routing
            self._router_message = next((msg["content"] for msg in reversed(router_context) if msg["role"] == "user"), user_msg)
            self._router_context_array = router_context
        else:
            # String context (legacy)
            self._router_message = router_context or user_msg
            self._router_context_array = None

        if isinstance(agent_context, list):
            # Message array - extract last user message for agent execution
            self._agent_message = next((msg["content"] for msg in reversed(agent_context) if msg["role"] == "user"), user_msg)
            self._agent_context_array = agent_context
        else:
            # String context (legacy)
            self._agent_message = agent_context or user_msg
            self._agent_context_array = None

        logger.info(f"🎯 Router context: {len(str(self._router_message))} chars (array: {self._router_context_array is not None})")
        logger.info(f"🤖 Agent context: {len(str(self._agent_message))} chars (array: {self._agent_context_array is not None})")
        self._current_agent = self.root_agent

        max_iterations = 5  # Prevent infinite loops
        iteration = 0

        # Initial message selection based on agent type
        user_msg = self._router_message if self._current_agent == "CentralRouterAgent" else self._agent_message

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 Iteration {iteration}: Current agent is {self._current_agent}")

            # Store current agent before execution
            agent_before_execution = self._current_agent

            # CRITICAL FIX: Set the workflow's current agent for parent class
            # This ensures parent AgentWorkflow.run() uses the right agent
            original_root = self.root_agent
            self.root_agent = self._current_agent

            try:
                # Execute the current agent using the parent's run method
                # Pass message array if available, otherwise use string
                if self._current_agent == "CentralRouterAgent" and self._router_context_array:
                    # Convert message array to memory for parent workflow
                    from llama_index.core.memory import ChatMemoryBuffer
                    from llama_index.core.llms import ChatMessage
                    memory = ChatMemoryBuffer.from_defaults()
                    for msg in self._router_context_array[:-1]:  # All except last message
                        chat_msg = ChatMessage(role=msg["role"], content=msg["content"])
                        memory.put(chat_msg)
                    result = await super().run(user_msg=user_msg, memory=memory, **kwargs)
                elif self._current_agent != "CentralRouterAgent" and self._agent_context_array:
                    # Convert message array to memory for agents
                    from llama_index.core.memory import ChatMemoryBuffer
                    from llama_index.core.llms import ChatMessage
                    memory = ChatMemoryBuffer.from_defaults()
                    for msg in self._agent_context_array[:-1]:  # All except last message
                        chat_msg = ChatMessage(role=msg["role"], content=msg["content"])
                        memory.put(chat_msg)
                    result = await super().run(user_msg=user_msg, memory=memory, **kwargs)
                else:
                    # Fallback to string-based execution
                    result = await super().run(user_msg=user_msg, **kwargs)
            finally:
                # Restore original root agent
                self.root_agent = original_root

            # Check if handoff occurred (agent changed)
            if self._current_agent != agent_before_execution:
                logger.info(f"🤝 Handoff detected! Switching from {agent_before_execution} to {self._current_agent}")

                # Continue with the new agent - use appropriate message based on agent type
                user_msg = self._router_message if self._current_agent == "CentralRouterAgent" else self._agent_message
                continue
            else:
                # No handoff, we're done
                logger.info(f"✅ Agent {self._current_agent} completed processing")
                # Attach agent metadata to result for dual history tracking
                if hasattr(result, '__dict__'):
                    result._handling_agent = self._current_agent
                return result

        logger.warning(f"⚠️ Max iterations reached ({max_iterations}), returning last result")
        # Attach agent metadata to result for dual history tracking
        if hasattr(result, '__dict__'):
            result._handling_agent = self._current_agent
        return result

    async def run_with_streaming(self, user_msg: str, router_context=None, agent_context=None, user_id=None, **kwargs):
        """Custom streaming run method that handles agent transitions with separate contexts."""
        logger.info(f"🚀 EnhancedAgentWorkflow starting with STREAMING for root agent: {self.root_agent}")

        # Store user_id for tools to access programmatically
        self._user_id = user_id or "unknown_user"

        # CRITICAL: Reset get_tools call counter for new workflow run
        self._get_tools_call_count = {}

        # Prefer message arrays; fall back to strings
        if isinstance(router_context, list):
            self._router_message = next((m.get("content", "") for m in reversed(router_context) if m.get("role") == "user"), user_msg)
        else:
            self._router_message = router_context or user_msg
        # Store arrays so streaming can use full context
        self._router_context_array = router_context if isinstance(router_context, list) else None

        if isinstance(agent_context, list):
            self._agent_message = next((m.get("content", "") for m in reversed(agent_context) if m.get("role") == "user"), user_msg)
        else:
            self._agent_message = agent_context or user_msg
        # Store arrays so streaming can use full context
        self._agent_context_array = agent_context if isinstance(agent_context, list) else None

        logger.info(f"🎯 STREAMING Router context length: {len(str(self._router_message))}")
        logger.info(f"🤖 STREAMING Agent context length: {len(str(self._agent_message))}")
        self._current_agent = self.root_agent

        max_iterations = 5  # Prevent infinite loops
        iteration = 0

        # Initial message selection based on agent type
        user_msg = self._router_message if self._current_agent == "CentralRouterAgent" else self._agent_message

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 STREAMING Iteration {iteration}: Current agent is {self._current_agent}")

            # Store current agent before execution
            agent_before_execution = self._current_agent

            # Check if this is the router (no streaming) or specialist agent (with streaming)
            if self._current_agent == "CentralRouterAgent":
                logger.info("🎯 Router execution - NO STREAMING")

                # Set the workflow's current agent for parent class
                original_root = self.root_agent
                self.root_agent = self._current_agent

                try:
                    # Execute router without streaming (routing decisions should be fast)
                    # IMPORTANT: include prior router context as memory so routing can
                    # consider the last few turns (e.g., short replies like "oui").
                    if getattr(self, "_router_context_array", None):
                        try:
                            from llama_index.core.memory import ChatMemoryBuffer
                            from llama_index.core.llms import ChatMessage
                            memory = ChatMemoryBuffer.from_defaults()
                            for msg in self._router_context_array[:-1]:  # All except last user msg
                                chat_msg = ChatMessage(role=msg.get("role", "user"), content=msg.get("content", ""))
                                memory.put(chat_msg)
                            result = await super().run(user_msg=user_msg, memory=memory, **kwargs)
                        except Exception:
                            # Fallback if memory construction fails
                            result = await super().run(user_msg=user_msg, **kwargs)
                    else:
                        result = await super().run(user_msg=user_msg, **kwargs)
                finally:
                    # Restore original root agent
                    self.root_agent = original_root

                # Check if handoff occurred (agent changed)
                if self._current_agent != agent_before_execution:
                    logger.info(f"🤝 STREAMING Handoff detected! Switching from {agent_before_execution} to {self._current_agent}")

                    # Continue with the new agent - use appropriate message based on agent type
                    user_msg = self._router_message if self._current_agent == "CentralRouterAgent" else self._agent_message
                    continue
                else:
                    # Router completed without handoff - shouldn't happen but handle it
                    logger.info(f"✅ STREAMING Agent {self._current_agent} completed processing")
                    yield result  # Yield final result
                    return

            else:
                logger.info(f"🎯 Specialist agent execution - WITH STREAMING for {self._current_agent}")

                # TEMPORARY: Check if this is the Pain Assessment Agent (needs tool calls)
                if self._current_agent == "EnhancedPainAssessmentAgent":
                    logger.info("🔧 Pain Assessment Agent detected - enabling final-response streaming after tool execution")

                    # For Pain Assessment Agent, perform tool-enabled execution first (non-streaming),
                    # then stream ONLY the final, user-visible message.
                    original_root = self.root_agent
                    self.root_agent = self._current_agent

                    try:
                        # Run with tools to completion (non-streaming)
                        # CRITICAL: include full agent context as memory so the specialist
                        # can see the prior turns (prevents repeated questions like
                        # re-asking the body region when it was given earlier).
                        # FORCE tool calling for pain agent to ensure RAG retrieval
                        tool_choice_param = {"tool_choice": "auto"}  # Force at least one tool call
                        if getattr(self, "_agent_context_array", None):
                            try:
                                from llama_index.core.memory import ChatMemoryBuffer
                                from llama_index.core.llms import ChatMessage
                                memory = ChatMemoryBuffer.from_defaults()
                                for msg in self._agent_context_array[:-1]:  # All except last user msg
                                    chat_msg = ChatMessage(role=msg.get("role", "user"), content=msg.get("content", ""))
                                    memory.put(chat_msg)
                                result = await super().run(user_msg=user_msg, memory=memory, **tool_choice_param, **kwargs)
                            except asyncio.CancelledError:
                                logger.warning("⚠️ Pain agent execution cancelled by client disconnect")
                                raise  # Re-raise to properly handle cancellation
                            except Exception as e:
                                # Fallback if memory construction fails
                                logger.warning(f"⚠️ Memory construction failed: {e}, falling back to basic execution")
                                result = await super().run(user_msg=user_msg, **tool_choice_param, **kwargs)
                        else:
                            result = await super().run(user_msg=user_msg, **tool_choice_param, **kwargs)

                        # Extract final text in a robust way
                        final_text = str(result.response if hasattr(result, 'response') else result)
                        final_text = final_text or ""

                        # Strip "assistant:" prefix if present (FunctionAgent sometimes includes role prefix)
                        if final_text.startswith("assistant:"):
                            final_text = final_text[len("assistant:"):].strip()

                        # Try true streaming via the underlying LLM for the final response
                        # If anything goes wrong, fall back to safe chunked streaming.
                        streamed = False
                        try:
                            from openai import AsyncOpenAI

                            current_agent_config = self.agents[self._current_agent]
                            agent_instance = current_agent_config

                            if final_text:
                                logger.info("🌊 Streaming final pain-agent response via OpenAI native streaming")
                                async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                                model_name = getattr(getattr(agent_instance, "llm", None), "model", "gpt-4o-mini")
                                openai_messages = [
                                    {
                                        "role": "system",
                                        "content": (
                                            "Répète EXACTEMENT le texte du message utilisateur, sans guillemets, "
                                            "sans ajout ni omission, en respectant les sauts de ligne et l'espacement."
                                        ),
                                    },
                                    {"role": "user", "content": final_text},
                                ]

                                try:
                                    stream = await async_client.chat.completions.create(
                                        model=model_name,
                                        messages=openai_messages,
                                        stream=True,
                                    )
                                    async for event in stream:
                                        try:
                                            delta = event.choices[0].delta
                                            chunk = getattr(delta, "content", None) or ""
                                        except Exception:
                                            chunk = ""
                                        if chunk:
                                            yield chunk
                                    streamed = True
                                except asyncio.CancelledError:
                                    logger.warning("⚠️ Streaming cancelled by client disconnect")
                                    raise  # Re-raise to properly handle cancellation
                                except Exception as stream_err:
                                    logger.warning(
                                        f"⚠️ Final-response OpenAI streaming failed: {stream_err}"
                                    )
                        except Exception as stream_err:  # pragma: no cover - defensive fallback
                            logger.warning(
                                f"⚠️ Final-response LLM streaming failed, falling back to chunked streaming: {stream_err}"
                            )

                        if not streamed:
                            # Do not simulate streaming; delegate to API layer fallback
                            logger.info("⛔ Skipping simulated streaming for pain-agent; delegating to API fallback")
                            return

                    finally:
                        # Always restore original root agent
                        self.root_agent = original_root

                    # Pain agent turn completed
                    return

                # For other specialist agents, use direct LLM streaming
                logger.info(f"🎯 Regular specialist agent execution - WITH STREAMING for {self._current_agent}")

                # CRITICAL FIX: Set the workflow's current agent for parent class
                original_root = self.root_agent
                self.root_agent = self._current_agent

                try:
                    # Execute specialist agent with streaming support via direct LLM access
                    current_agent_config = self.agents[self._current_agent]
                    agent_instance = current_agent_config

                    stream_success = False
                    stream_error: Exception | None = None

                    try:
                        # Build message array for streaming if available; fallback to single user message
                        messages_for_stream = None
                        if getattr(self, "_agent_context_array", None):
                            try:
                                from llama_index.core.llms import ChatMessage
                                messages_for_stream = [
                                    ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
                                    for m in self._agent_context_array
                                ]
                            except Exception:
                                # Fallback: pass through the raw dicts (some clients accept dict format)
                                messages_for_stream = list(self._agent_context_array)
                        else:
                            # Minimal message array
                            try:
                                from llama_index.core.llms import ChatMessage
                                messages_for_stream = [ChatMessage(role="user", content=user_msg)]
                            except Exception:
                                messages_for_stream = [{"role": "user", "content": user_msg}]

                        logger.info(f"🌊 Using OpenAI native streaming for {self._current_agent}")
                        try:
                            from openai import AsyncOpenAI

                            # Convert message array to OpenAI format
                            openai_messages: list[dict] = []
                            for m in (messages_for_stream or [{"role": "user", "content": user_msg}]):
                                if hasattr(m, "role") and hasattr(m, "content"):
                                    openai_messages.append({"role": m.role, "content": m.content})
                                elif isinstance(m, dict):
                                    openai_messages.append({
                                        "role": m.get("role", "user"),
                                        "content": m.get("content", ""),
                                    })

                            # CRITICAL: Extract and prepend agent's system prompt
                            agent_system_prompt = getattr(agent_instance, 'system_prompt', None)
                            if agent_system_prompt:
                                openai_messages.insert(0, {"role": "system", "content": agent_system_prompt})
                                logger.info(f"✅ Injected system prompt for {self._current_agent}")
                            else:
                                logger.warning(f"⚠️ No system prompt found for {self._current_agent}")

                            async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                            model_name = getattr(getattr(agent_instance, "llm", None), "model", "gpt-4o-mini")
                            stream = await async_client.chat.completions.create(
                                model=model_name,
                                messages=openai_messages,
                                stream=True,
                            )
                            async for event in stream:
                                try:
                                    delta = event.choices[0].delta
                                    chunk = getattr(delta, "content", None) or ""
                                except Exception:
                                    chunk = ""
                                if chunk:
                                    yield chunk
                            stream_success = True
                        except Exception as err:  # pragma: no cover - streaming path
                            stream_error = err
                            logger.warning(
                                f"⚠️ OpenAI native streaming failed for {self._current_agent}, falling back: {err}"
                            )
                    except Exception as err:  # pragma: no cover - streaming path
                        stream_error = err
                        logger.warning(
                            f"⚠️ Streaming failed for {self._current_agent}, falling back to regular execution: {err}"
                        )

                    if not stream_success:
                        if stream_error:
                            logger.debug(f"Streaming error detail: {stream_error}")
                        # Do not simulate streaming; delegate to API layer fallback
                        logger.info("⛔ Skipping simulated streaming for specialist agent; delegating to API fallback")
                        return

                finally:
                    # Restore original root agent
                    self.root_agent = original_root

                # Specialist agent completed
                logger.info(f"✅ STREAMING Specialist agent {self._current_agent} completed processing")
                return

        logger.warning(f"⚠️ STREAMING Max iterations reached ({max_iterations})")
        return


def build_workflow(
    llm: BaseLLM, specialist_agents: List[Any], initial_state: dict | None = None
) -> EnhancedAgentWorkflow:
    """Build AgentWorkflow with CentralRouterAgent as root and specialists.

    Uses a faster/cheaper model for routing decisions to minimize latency.
    """
    # Use fast, cheap model for router (routing is simple decision-making)
    from llama_index.llms.openai import OpenAI
    router_llm = OpenAI(model="gpt-4o-mini", temperature=0.3)

    central_router = build_router(router_llm)
    all_agents = [central_router] + list(specialist_agents)

    workflow = EnhancedAgentWorkflow(
        agents=all_agents,
        root_agent="CentralRouterAgent",
        initial_state=initial_state or {},
    )

    logger.info("🤖 Workflow initialized with fast router (gpt-4o-mini) and specialists")
    return workflow
