from __future__ import annotations

"""Central Router Agent and Workflow for specialist agents."""

import logging
import asyncio
from typing import List, Any, Dict, Optional
import os

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.tools import FunctionTool
from app.core.config import ROUTER_MODEL, DEFAULT_MODEL


logger = logging.getLogger(__name__)


# Global OpenAI client (reused across requests for connection pooling)
_OPENAI_CLIENT = None

def get_openai_client():
    """Get or create global AsyncOpenAI client for connection reuse."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import AsyncOpenAI
        _OPENAI_CLIENT = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _OPENAI_CLIENT


_DESCRIPTION = (
    "Central routing coordinator that analyzes user messages and routes them to the appropriate "
    "specialist agent (pain, training, nutrition, health insights, or general Q&A)."
)


_SYSTEM_PROMPT = """
Tu es l'Agent Routeur Central pour GoGain AI Coach.

Ta mission est de décider quel spécialiste doit traiter le message de l'utilisateur.
Tu ne dois JAMAIS répondre directement à la place d'un spécialiste.

--------------------------------------------------------------------------------
ORDRE D'ACTION (OBLIGATOIRE, AVANT TOUTE AUTRE RÈGLE)
--------------------------------------------------------------------------------
1. Tu dois TOUJOURS appeler exactement UN outil de handoff parmi: handoff_EnhancedPainAssessmentAgent, handoff_PersonalTrainingAgent, handoff_GeneralQAAgent.
2. Si une session douleur est active (le contexte système l'indique) OU si le message actuel ressemble à une réponse à une question douleur récente (ex: intensité 0-10, "hier", "modérée", précisions de symptômes, confirmations de suivi douleur), ALORS tu dois APPELER IMMÉDIATEMENT `handoff_EnhancedPainAssessmentAgent`. Ne produis aucun texte assistant.
3. Si la demande concerne la création, la modification ou la consultation d'un programme d'entraînement (objectifs, séances, programme, fréquence, équipement) ou s'il s'agit d'une réponse à une question de l'agent entraînement, ALORS tu dois APPELER IMMÉDIATEMENT `handoff_PersonalTrainingAgent`.
4. Si tu n'es pas certain après avoir analysé les 8 derniers messages (utilisateur et assistant) ou si le message est hors douleur/entraînement, ALORS tu dois APPELER `handoff_GeneralQAAgent`.
5. Interdiction absolue: ne produis JAMAIS de contenu assistant, ni de JSON, ni de texte. Ton seul rôle est de choisir et d'appeler le bon handoff.
6. Pour chaque décision, analyse toujours les 8 derniers messages de la conversation pour deviner le workflow en cours. Considère les réponses courtes ("oui", "non", "modérée", "4-6", "hier", etc.) comme des suivis du dernier spécialiste qui a parlé et handoff vers ce même spécialiste.

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
  * "Crée-moi un programme d'entraînement" → PersonalTrainingAgent
  * "Question au sujet d'un programme d'entrainement" -> PersonalTrainingAgent
  * "Est ce que j'ai un programme d'entrainement?" -> PersonalTrainingAgent
  * "Je veux un plan d'entraînement" → PersonalTrainingAgent
  * "Que manger pour perdre du poids?" → NutritionAgent

CONTINUITÉ AVANT CHANGEMENT (pour réponses courtes seulement):
- **RÈGLE CRITIQUE**: Si le dernier agent actif était PersonalTrainingAgent et que la dernière question posée concernait un programme d'entraînement (objectif, niveau, fréquence, équipement, etc.), TOUTE réponse courte doit rester avec PersonalTrainingAgent. NE JAMAIS router vers Pain Agent pendant une création de programme.
- Si le dernier agent actif était l'EnhancedPainAssessmentAgent et que le message actuel ressemble à une réponse à une question sur la douleur, reste avec l'agent douleur.
- Considère comme réponses de suivi douleur : durée/début (« depuis 2 mois »), intensité (« 6/10 »), localisation (« à l'avant du genou », « côté interne »), caractère (« douleur aiguë », « sensation de brûlure »), déclencheurs (« quand je m'accroupis »), facteurs calmants (« le repos aide »), évolution (« pire le matin »), confirmations oui/non à des questions de douleur, ou toute précision sur les symptômes.
- Si l'utilisateur affirme ne plus ressentir de douleur ou demande clairement un plan sans lien avec la douleur, tu peux diriger vers un autre agent approprié.

Utilise immédiatement l'outil de handoff adapté :
- Toute mention de douleur OU toute réponse de suivi douleur → handoff_EnhancedPainAssessmentAgent
- Demande explicite de création/modification d'un programme d'entraînement/fitness → handoff_PersonalTrainingAgent
- Création d'un plan alimentaire uniquement → handoff_NutritionAgent
- Questions générales de santé/fitness NON liées à une douleur → handoff_GeneralQAAgent
- Si tu ne sais pas quel agent choisir ET que ce n'est pas lié à une douleur → handoff_GeneralQAAgent

RÈGLES DE ROUTAGE (INTERPRÈTE LE MESSAGE ACTUEL DANS SON CONTEXTE) :
- Agent douleur : dès que le message mentionne une douleur/blessure/gêne OU répond à une question de l'agent douleur (cf. exemples ci-dessus) → envoie à l'agent douleur.
  * Vérifie qu'il existe un symptôme, gêne ou suivi d'une question douleur avant de router.
- Agent entraînement (PersonalTrainingAgent) : UNIQUEMENT si l'utilisateur demande EXPLICITEMENT la création, modification ou consultation d'un programme d'entraînement/fitness (ex: "crée-moi un programme", "fais-moi un plan d'entraînement", "je veux un programme muscu", "montre-moi mon programme"). NE PAS router si c'est juste une question générale sur l'entraînement.
  * Inclut toutes les demandes d'amélioration de performance sans douleur: "développer ma force", "nouveau cycle d'entraînement", "programme de musculation", "adapter mon plan". Tant que l'intention est de planifier ou ajuster des séances, choisis PersonalTrainingAgent.
- Agent nutrition : uniquement si le message demande explicitement un plan alimentaire et qu'il n'y a pas de suivi douleur en cours.
- Agent General QA : pour les conseils généraux sur l'entraînement/santé/fitness, questions techniques, ou lorsque le parcours douleur est clairement terminé.

⚠️ NE ROUTE JAMAIS VERS L'AGENT DOULEUR si le message porte sur un objectif d'entraînement (force, hypertrophie, condition physique, "nouveau programme") SANS mention de douleur/blessure. Ces requêtes vont au PersonalTrainingAgent, même si l'utilisateur est en pleine forme et ne parle que d'amélioration de performance.

RÈGLES DE DÉCISION SENSIBLES AU CONTEXTE :
- Maintiens le fil : ne coupe pas une évaluation douleur en plein milieu.
- Vérifie les dernières questions de l'assistant : si elles concernaient la douleur, suppose que la réponse actuelle en est le suivi.
- En cas de doute raisonnable (ex. message court comme « à l'avant », « depuis 2 mois », « oui »), privilégie l'agent douleur plutôt que de changer.
- Si l'utilisateur change explicitement de sujet (« Au fait, peux-tu me faire un programme muscu ? ») ou déclare ne plus être gêné, bascule vers l'agent correspondant.

GESTION DES CAS LIMITE :
- Réponses brèves (« oui », « non », « parfois », « 5/10 », « 3 », « 4 », « à l'avant », « il y a 2 mois », etc.) → vérifie la dernière question posée ET l'agent qui l'a posée.
  * Si elle venait du PersonalTrainingAgent (questions sur programme d'entraînement) → reste avec PersonalTrainingAgent
  * Si elle venait de l'agent douleur (questions sur la douleur) → reste sur l'agent douleur
  * En cas de doute, privilégie l'agent qui a posé la dernière question
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

    def _emit_training_context(self) -> Optional[str]:
        """Return training context once when PersonalTrainingAgent is active."""
        if (
            getattr(self, "_training_context", None)
            and self._current_agent == "PersonalTrainingAgent"
            and not getattr(self, "_training_context_emitted", False)
        ):
            self._training_context_emitted = True
            return self._training_context
        return None

    async def run(self, user_msg: str, router_context=None, agent_context=None, user_id=None, training_context=None, **kwargs):
        """Custom run method that handles agent transitions with separate contexts."""
        logger.info(f"🚀 EnhancedAgentWorkflow starting with root agent: {self.root_agent}")

        # Store user_id for tools to access programmatically
        self._user_id = user_id or "unknown_user"
        self._training_context = training_context
        self._training_context_emitted = False

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
                    training_snapshot = self._emit_training_context()
                    if training_snapshot:
                        training_msg = ChatMessage(role="system", content=training_snapshot)
                        memory.put(training_msg)
                    for msg in self._agent_context_array[:-1]:  # All except last message
                        chat_msg = ChatMessage(role=msg["role"], content=msg["content"])
                        memory.put(chat_msg)
                    result = await super().run(user_msg=user_msg, memory=memory, **kwargs)
                else:
                    training_snapshot = self._emit_training_context()
                    if training_snapshot:
                        from llama_index.core.memory import ChatMemoryBuffer
                        from llama_index.core.llms import ChatMessage
                        memory = ChatMemoryBuffer.from_defaults()
                        memory.put(ChatMessage(role="system", content=training_snapshot))
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

    async def run_with_streaming(self, user_msg: str, router_context=None, agent_context=None, user_id=None, training_context=None, **kwargs):
        """Custom streaming run method that handles agent transitions with separate contexts."""
        logger.info(f"🚀 EnhancedAgentWorkflow starting with STREAMING for root agent: {self.root_agent}")

        # Store user_id for tools to access programmatically
        self._user_id = user_id or "unknown_user"
        self._training_context = training_context
        self._training_context_emitted = False

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

                    # Continue to next iteration - specialist agent will stream in else block
                    # Update user message for specialist
                    user_msg = self._agent_message
                    continue
                else:
                    # Router completed without handoff - shouldn't happen but handle it
                    logger.info(f"✅ STREAMING Agent {self._current_agent} completed processing")
                    yield result  # Yield final result
                    return

            else:
                logger.info(f"🎯 Specialist agent execution - REAL STREAMING for {self._current_agent}")

                # Real streaming with tool support via OpenAI SDK
                original_root = self.root_agent
                self.root_agent = self._current_agent

                try:
                    from openai import AsyncOpenAI
                    import json as json_lib

                    # Build message history
                    messages = []

                    # Add system prompt
                    current_agent_config = self.agents[self._current_agent]
                    agent_instance = current_agent_config
                    system_prompt = getattr(agent_instance, 'system_prompt', None)
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})

                    training_snapshot = self._emit_training_context()
                    if training_snapshot:
                        messages.append({"role": "system", "content": training_snapshot})

                    # Add conversation history
                    if getattr(self, "_agent_context_array", None):
                        for msg in self._agent_context_array[:-1]:
                            messages.append({
                                "role": msg.get("role", "user"),
                                "content": msg.get("content", "")
                            })

                    # Add current user message
                    messages.append({"role": "user", "content": user_msg})

                    # Convert tools to OpenAI format
                    agent_tools = getattr(agent_instance, 'tools', None) or []
                    openai_tools = []
                    tool_map = {}  # Map tool names to actual tool functions

                    for tool in agent_tools:
                        if hasattr(tool, 'metadata'):
                            tool_map[tool.metadata.name] = tool
                            # Use LlamaIndex's built-in OpenAI conversion
                            if hasattr(tool.metadata, 'to_openai_tool'):
                                tool_def = tool.metadata.to_openai_tool()
                            else:
                                # Fallback to manual construction
                                tool_def = {
                                    "type": "function",
                                    "function": {
                                        "name": tool.metadata.name,
                                        "description": tool.metadata.description or "",
                                        "parameters": tool.metadata.get_parameters_dict() if hasattr(tool.metadata, 'get_parameters_dict') else {}
                                    }
                                }
                            openai_tools.append(tool_def)

                    logger.info(f"🌊 Real streaming for {self._current_agent} with {len(openai_tools)} tools")

                    # Execute streaming with tool support (reuse global client)
                    client = get_openai_client()
                    model = getattr(getattr(agent_instance, "llm", None), "model", DEFAULT_MODEL)

                    # Loop to handle multiple tool calls
                    max_iterations = 10
                    for iteration in range(max_iterations):
                        stream = await client.chat.completions.create(
                            model=model,
                            messages=messages,
                            tools=openai_tools if openai_tools else None,
                            stream=True,
                        )

                        # Accumulate response and tool calls
                        content_chunks = []
                        tool_calls = {}
                        finish_reason = None

                        async for chunk in stream:
                            if not chunk.choices:
                                continue

                            delta = chunk.choices[0].delta
                            finish_reason = chunk.choices[0].finish_reason

                            # Stream text content to user
                            if delta.content:
                                content_chunks.append(delta.content)
                                yield delta.content

                            # Accumulate tool calls
                            if delta.tool_calls:
                                for tc in delta.tool_calls:
                                    idx = tc.index
                                    if idx not in tool_calls:
                                        tool_calls[idx] = {
                                            "id": "",
                                            "name": "",
                                            "arguments": ""
                                        }
                                    if tc.id:
                                        tool_calls[idx]["id"] = tc.id
                                    if tc.function and tc.function.name:
                                        tool_calls[idx]["name"] = tc.function.name
                                    if tc.function and tc.function.arguments:
                                        tool_calls[idx]["arguments"] += tc.function.arguments

                        # If no tool calls, we're done
                        if finish_reason == "stop" or not tool_calls:
                            break

                        # Execute tool calls
                        if finish_reason == "tool_calls":
                            logger.info(f"🔧 Executing {len(tool_calls)} tool calls")

                            # Add assistant message with tool calls to history
                            full_content = "".join(content_chunks)
                            assistant_msg = {"role": "assistant", "content": full_content or None}

                            # Add tool call info
                            assistant_msg["tool_calls"] = []
                            for tc_data in tool_calls.values():
                                assistant_msg["tool_calls"].append({
                                    "id": tc_data["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc_data["name"],
                                        "arguments": tc_data["arguments"]
                                    }
                                })
                            messages.append(assistant_msg)

                            # Execute each tool and add results
                            for tc_data in tool_calls.values():
                                tool_name = tc_data["name"]
                                tool_args_str = tc_data["arguments"]

                                try:
                                    tool_args = json_lib.loads(tool_args_str)
                                    tool_fn = tool_map.get(tool_name)

                                    if tool_fn:
                                        logger.info(f"⚙️ Calling tool: {tool_name}")

                                        # Inject user_id programmatically for training tools
                                        if tool_name in ['create_training_program', 'save_fitness_goal_assessment']:
                                            tool_args['user_id'] = str(self._user_id)
                                            logger.info(f"💉 Injected user_id={self._user_id} into {tool_name}")

                                        result = await asyncio.to_thread(tool_fn, **tool_args)
                                        tool_result = str(result)
                                    else:
                                        tool_result = f"Error: Unknown tool {tool_name}"
                                        logger.error(f"❌ Unknown tool: {tool_name}")

                                    # Add tool result to messages
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tc_data["id"],
                                        "content": tool_result
                                    })

                                    # Emit a lightweight event for the SSE layer to notify UI caches
                                    try:
                                        yield {"event": "tool_result", "function_name": tool_name, "status": "ok"}
                                    except Exception:
                                        # If yielding from here fails for any reason, continue gracefully
                                        pass

                                except Exception as e:
                                    logger.error(f"❌ Tool execution failed: {e}")
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tc_data["id"],
                                        "content": f"Error: {str(e)}"
                                    })
                                    try:
                                        yield {"event": "tool_result", "function_name": tool_name, "status": "error"}
                                    except Exception:
                                        pass

                            # Continue to next iteration to get final response
                            continue

                        # Unknown finish reason
                        break

                finally:
                    self.root_agent = original_root

                logger.info(f"✅ REAL STREAMING completed for {self._current_agent}")
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
    router_llm = OpenAI(model=ROUTER_MODEL, temperature=0.3)

    central_router = build_router(router_llm)
    all_agents = [central_router] + list(specialist_agents)

    workflow = EnhancedAgentWorkflow(
        agents=all_agents,
        root_agent="CentralRouterAgent",
        initial_state=initial_state or {},
    )

    logger.info("🤖 Workflow initialized with fast router (%s) and specialists", ROUTER_MODEL)
    return workflow
