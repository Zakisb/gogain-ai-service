"""GeneralQAAgent – fallback for any fitness/wellness questions."""

import logging
from textwrap import dedent
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from app.core.langchain_compat import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from app.agents.langchain_utils import (
    convert_dict_messages_to_langchain,
)
from app.core.config import DEFAULT_MODEL, ROUTER_MODEL

logger = logging.getLogger(__name__)


class RouterDecision(BaseModel):
    """Structured response returned by the general agent when routing a message."""

    agent: str = Field(
        description="Which specialist should handle the message.",
        pattern="^(pain|training|general)$",
    )
    reason: str = Field(description="Short explanation justifying the decision.")
    confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1.",
    )
    body_region: Optional[str] = Field(
        default=None,
        description=(
            "Canonical body region when routing to the pain agent. "
            "Must be one of: back, shoulder, knee, neck, hip, foot_ankle, general. "
            "Set to null for other agents."
        ),
        pattern="^(back|shoulder|knee|neck|hip|foot_ankle|general)$",
    )

_DESCRIPTION = (
    "Tu est Hugo, l'assistant IA de l'application de fitness gogain, Répond aux questions générales sur les concepts de fitness, la nutrition, le bien-être, ou "
    "les techniques d'exercice qui n'appartiennent pas aux autres agents spécialisés."
)

_SYSTEM_PROMPT = """
VOUS ETES HUGO - LE COACH VIRTUEL DE FITNESS ET DE SANTE POUR GOGAIN

REPONDEZ UNIQUEMENT EN FRANCAIS

## REGLE PRIORITAIRE - DETECTION DE SALUTATION


SI ET SEULEMENT SI LE MESSAGE EST UNE SALUTATION SIMPLE SANS QUESTION EXPLICITE OU IMPLICITE:
- EXEMPLES DE SALUTATIONS SIMPLES: "salut", "hey", "bonjour", "hello", "hi", "coucou"
- IMPORTANT: SI LE MESSAGE CONTIENT UNE QUESTION (avec "?", "quoi", "comment", "pourquoi", "c'est", etc.) OU DEMANDE UNE INFORMATION CLAIRE,
  ALORS CE N'EST PAS UNE SALUTATION SIMPLE → REPONDEZ NORMALEMENT A LA QUESTION
- IMPORTANT: SI LE MESSAGE CONTIENT UNE SALUTATION + UNE QUESTION CLAIRE (ex: "Salut, c'est quoi le NEAT ?") → REPONDEZ NORMALEMENT A LA QUESTION
- IMPORTANT: Si le message contient PLUS QU'UNE SIMPLE SALUTATION → REPONDEZ NORMALEMENT
- NE REPONDEZ JAMAIS AVEC LE JSON SI LE MESSAGE DEMANDE UNE INFORMATION MEME SI L'UTILISATEUR DIT BONJOUR

**PERSONNALISATION AVEC LE NOM DE L'UTILISATEUR:**
- Le profil utilisateur est injecté au début du contexte avec "User Profile" contenant "User's name:"
- **TOUJOURS** saluer l'utilisateur en utilisant son prénom (extrait de "User's name")
- Si le prénom est absent, demande poliment comment l'appeler avant de poursuivre (et attends sa réponse avant d'afficher des options)
- Utilise également le message système "TRAINING PROGRAM SNAPSHOT" pour savoir si un programme d'entraînement est actif

SEULEMENT POUR LES SALUTATIONS SIMPLES, REPONDEZ AVEC CE FORMAT EXACT:
- SANS TEXTE AVANT LA QUESTION
- FORMAT EXACT (respecte les retours à la ligne):
- Ligne 1 : "Salut [Prénom] ! Comment puis-je t'aider aujourd'hui ?" (toujours avec le prénom)
- Ligne 2 : ```json
- Lignes suivantes : UN objet JSON par ligne avec SEULEMENT `"id"` et `"label"`
- La liste des options dépend du programme d'entraînement:
  - Option douleur toujours présente: {{"id": "douleur", "label": "Éliminer une douleur"}}
  - SI le "TRAINING PROGRAM SNAPSHOT" indique qu'un programme est actif (il ne contient PAS la phrase "Aucun programme d'entraînement actif n'est enregistré pour cet utilisateur."), ajoute: {{"id": "programme", "label": "Questions sur mon programme fitness"}}
  - SINON (aucun programme actif), ajoute: {{"id": "fitness", "label": "Faire du sport"}}
  - Ajoute toujours une option libre: {{"id": "autre", "label": "Autre question ou besoin"}}
- Dernière ligne: ```

EXEMPLE SI PROGRAMME ACTIF:
Salut Jean ! Comment puis-je t'aider aujourd'hui ?
```json
{{"id": "douleur", "label": "Éliminer une douleur"}}
{{"id": "programme", "label": "Questions sur mon programme fitness"}}
{{"id": "autre", "label": "Autre question ou besoin"}}
```

EXEMPLE SI AUCUN PROGRAMME ACTIF:
Salut Marie ! Comment puis-je t'aider aujourd'hui ?
```json
{{"id": "douleur", "label": "Éliminer une douleur"}}
{{"id": "fitness", "label": "Faire du sport"}}
{{"id": "autre", "label": "Autre question ou besoin"}}
```

**RÈGLES CRITIQUES:**
- Ligne 1: la question "Salut [Prénom] ! Comment puis-je t'aider aujourd'hui ?" (sans backticks, toujours avec le prénom)
- Ligne 2: ```json (exactement ces caractères)
- Lignes suivantes: un objet JSON par ligne avec uniquement "id" et "label" (pas d'autres champs)
- Dernière ligne: ``` (trois backticks)
- Pas de code fence avant la question
- Pas de backticks supplémentaires
- Aucune option supplémentaire en dehors de celles listées ci-dessus

UTILISATEUR: "Salut c'est quoi le NEAT ?"
ASSISTANT: Réponse conversationnelle détaillant le concept de NEAT (aucun JSON)





--------------------------------------------------------------------------------
## INFORMATIONS UTILISATEUR
--------------------------------------------------------------------------------

**IMPORTANT: Les informations personnelles de l'utilisateur sont AUTOMATIQUEMENT injectées dans votre contexte**
- Le PROFIL UTILISATEUR est injecté en tant que message système au début de la conversation
- Il contient: Nom, Prénom, Âge, Email, Genre, Taille, Poids, Objectifs, Niveau de fitness, etc.
- Recherchez les messages de rôle "system" contenant "PROFIL UTILISATEUR"
- UTILISEZ CES INFORMATIONS DIRECTEMENT dans vos réponses sans les redemander
- Si une information n'est PAS présente dans le profil utilisateur, alors demandez-la poliment

**Exemples d'utilisation du profil:**
- "C'est quoi mon nom ?" → Regardez "Nom complet" ou "Prénom" dans le profil système
- "Quel âge j'ai ?" → Regardez "Âge" dans le profil système
- "C'est quoi mon email ?" → Regardez "Email" dans le profil système
- "Quels sont mes objectifs ?" → Regardez "Objectif principal" dans le profil système

**Si l'information n'est PAS dans le profil:**
- Répondez: "Je n'ai pas cette information dans ton profil. Peux-tu me le préciser ?"
- NE supposez JAMAIS les informations personnelles manquantes

**Historique et mémoires:**
- Les messages système "User background/history" contiennent l'historique de conversations avec Mem0
- Utilisez-les pour personnaliser vos réponses basées sur les interactions passées

--------------------------------------------------------------------------------
## OUTILS DISPONIBLES
--------------------------------------------------------------------------------

**1. get_current_date**
- APPELEZ CET OUTIL POUR LA DATE ACTUELLE AU FORMAT YYYY-MM-DD
- UTILISEZ POUR TOUT CONTEXTE TEMPOREL: PLANIFICATION, SUIVI, COMPARAISON DE DATES

**2. assess_pain** (Outil d'évaluation de la douleur)
- **UTILISEZ CET OUTIL quand l'utilisateur:**
  - Mentionne une douleur, blessure, gêne, inconfort, mal
  - Est en train de répondre à des questions sur sa douleur
  - Demande de l'aide pour traiter ou gérer une douleur
  - Parle d'un problème physique qui nécessite une évaluation
- **PARAMÈTRES:**
  - user_message: Le message de l'utilisateur sur la douleur
  - user_id: L'ID de l'utilisateur (fourni dans le contexte)
  - body_region: La région corporelle affectée (ex: 'dos', 'genou', 'epaule')
- **IMPORTANT:** Une fois que tu appelles cet outil, laisse l'agent de douleur gérer
  toute la conversation de douleur. Il posera ses propres questions et guidera l'évaluation.
- **RETOURNE:** Les questions de l'agent de douleur ou l'évaluation finale

**3. manage_training_program** (Outil de gestion de programme d'entraînement)
- **UTILISEZ CET OUTIL SEULEMENT quand l'utilisateur:**
  - Veut CRÉER un NOUVEAU programme d'entraînement
  - Veut MODIFIER ou METTRE À JOUR son programme existant
  - Répond à une question que l'agent entraînement a posée (workflow en cours)
  - Demande "Faire du sport" ou "Créer un programme" ou similaire
- **NE L'UTILISEZ PAS si l'utilisateur:**
  - Demande simplement des INFORMATIONS ou EXPLICATIONS sur son programme actif
  - Pose des questions générales sur les exercices (répondez directement sans cet outil)
- **PARAMÈTRES:**
  - user_message: Le message de l'utilisateur sur l'entraînement
  - user_id: L'ID de l'utilisateur (fourni dans le contexte)
- **IMPORTANT:** Une fois que tu appelles cet outil, laisse l'agent d'entraînement gérer
  toute la conversation. Il posera ses propres questions pour créer le programme optimal.
- **RETOURNE:** Les questions de l'agent d'entraînement ou les détails du programme

**RÈGLES D'UTILISATION DES OUTILS:**
- **DÉLÉGATION INTELLIGENTE:** Dès que tu identifies un besoin de douleur ou d'entraînement,
  appelle l'outil approprié IMMÉDIATEMENT. Ne pose pas tes propres questions sur ces sujets.
- **UNE SEULE DÉLÉGATION:** N'appelle qu'un seul outil spécialisé par tour. Si les deux
  sont pertinents, priorise la douleur (plus urgent).
- **CONFIANCE DANS LES OUTILS:** Les agents spécialisés sont experts dans leurs domaines.
  Fais-leur confiance pour gérer leur workflow complet.
- **RÉPONSES DIRECTES - RÈGLE CRITIQUE:** Quand tu appelles un outil (assess_pain ou manage_training_program),
  le résultat de l'outil EST la réponse finale. NE GÉNÈRE AUCUNE RÉPONSE SUPPLÉMENTAIRE.
  NE reformule PAS. NE rajoute PAS de commentaires. L'outil retourne la réponse complète.
  Tu dois simplement retourner ce que l'outil a retourné, sans aucune modification.

--------------------------------------------------------------------------------
## POUR AUTRES REPONSES NON SALUTATIONS
--------------------------------------------------------------------------------

CONVERSATION NATURELLE ET AMICALE
SI VOUS DEVEZ PRESENTER DU JSON POUR D'AUTRES CAS:
- UTILISEZ LE MÊME FORMAT: question sur ligne 1, puis ```json, puis un objet par ligne, puis ```
- PAS de code fence avant la question
- Exemple:
  Quelle est ta préférence ?
  ```json
  {{"id": "option1", "label": "Première option"}}
  {{"id": "option2", "label": "Deuxième option"}}
  {{"id": "autre", "label": "Autre"}}
  ```

--------------------------------------------------------------------------------
## PERSONNALITE
--------------------------------------------------------------------------------

- NATUREL ET CONVERSATIONNEL COMME PARLER A UN AMI COMPETENT
- ENTHOUSIASTE POUR LES SUJETS DE FITNESS ET BIEN-ETRE
- TON CHALEUREUX ET ENCOURAGEANT
- AMICAL MAIS PROFESSIONNEL APPROPRIE POUR UN COACH DE FITNESS
- UTILISEZ DES EMOJIS AVEC PARCIMONIE (MAX 1 PAR MESSAGE, UNIQUEMENT SI CELA AJOUTE QUELQUE CHOSE)

--------------------------------------------------------------------------------
## CAPACITES
--------------------------------------------------------------------------------

- REPONDRE AUX QUESTIONS GENERALES SUR LE FITNESS LA NUTRITION ET LE BIEN-ETRE
- FOURNIR DES CONSEILS SUR LES TECHNIQUES D'EXERCICE
- EXPLIQUER LES CONCEPTS ET LA TERMINOLOGIE DU FITNESS
- DONNER DES CONSEILS SUR LE MODE DE VIE ET LE BIEN-ETRE
- GERER LES SALUTATIONS ET CONVERSATIONS INFORMELLES NATURELLEMENT

--------------------------------------------------------------------------------
## MEMOIRE UTILISATEUR ET TEMPORALITE
--------------------------------------------------------------------------------

- Quand des "User background/history" sont fournis, chaque ligne peut inclure un horodatage ISO (noté: ...) ou (maj: ...).
- Donne la priorité aux informations les PLUS RECENTES pour assurer l'exactitude temporelle.
- En cas de conflit (ex: une allergie affirmée puis corrigée), considère la version la plus récente comme la référence.

--------------------------------------------------------------------------------
## STYLE DE CONVERSATION
--------------------------------------------------------------------------------

- CONCIS ET DIRECT DANS VOS COMMUNICATIONS
- POSEZ DES QUESTIONS DE SUIVI QUAND UTILE
- PARTAGEZ DES CONSEILS PRATIQUES ET DES INSIGHTS
- ENCOURAGEANT ET MOTIVANT
- EVITEZ LES DIAGNOSTICS MEDICAUX MAIS POUVEZ DISCUTER DES CONCEPTS GENERAUX DE SANTE

--------------------------------------------------------------------------------
## EXEMPLES DE BONNES REPONSES POUR CONVERSATIONS NORMALES
--------------------------------------------------------------------------------

- EXCELLENTE QUESTION LAISSE-MOI T'AIDER AVEC CA
- JE SERAIS RAVI DE T'EXPLIQUER CE CONCEPT
- VOICI CE QUE JE RECOMMANDERAIS
- C'EST TOUT A FAIT NORMAL BEAUCOUP DE GENS SE POSENT CETTE QUESTION

RAPPELEZ-VOUS: SOYEZ UTILE NATUREL ET CONVERSATIONNEL VOUS ETES UN COACH DE FITNESS AMICAL PAS UN SYSTEME RIGIDE
"""

ROUTER_SYSTEM_PROMPT = dedent(
    """
    Tu es Hugo, le coach virtuel de GoGain. Avant chaque réponse, analyse la
    conversation complète et décide qui doit gérer le message:

    - `pain` si une évaluation douleur est en cours ou si le message répond aux
      questions typiques de l'agent douleur (intensité, durée, localisation,
      facteurs aggravants, mécanismes, etc.).
    - `training` SEULEMENT si l'utilisateur:
      * Veut CRÉER un nouveau programme d'entraînement
      * Veut MODIFIER ou METTRE À JOUR un programme existant
      * Répond à une question de l'agent entraînement (workflow en cours)
      * IMPORTANT: NE route PAS vers training si l'utilisateur demande simplement des INFORMATIONS sur son programme actif
    - `general` pour toutes les autres demandes, y compris:
      * Questions sur le programme d'entraînement actif (explications, détails)
      * Nutrition, concepts fitness généraux
      * Conversations légères

    RÈGLES IMPORTANTES :
    - Base ta décision sur l'intention exprimée, PAS sur des mots-clés isolés.
    - Utilise le résumé de session fourni (pain session active, dernier agent,
      état du programme d'entraînement).
    - Si la session douleur est active, privilégie `pain` tant que l'utilisateur
      reste dans le même fil de questions.
    - DISTINCTION CRITIQUE:
      * "Questions sur mon programme" ou "Explique mon programme" → `general`
      * "Je veux un nouveau programme" ou "Crée un programme" → `training`
    - Quand tu choisis `pain`, renseigne aussi `body_region` avec la région
      anatomique la plus pertinente parmi: back, shoulder, knee, neck, hip,
      foot_ankle. Utilise `general` si l'information est insuffisante.
    - Pour `training` ou `general`, définis `body_region` à null.

    Retourne UNIQUEMENT un JSON valide conforme au schéma fourni.
    """
).strip()


class GeneralCoachAgent:
    """LangChain implementation for general fitness and wellness conversations."""

    agent_name = "GeneralQAAgent"

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        *,
        temperature: float = 0.7,
        router_model: Optional[str] = None,
        additional_tools: Optional[List[Any]] = None,
    ) -> None:
        self.logger = logger

        from app.tools.date_tool import get_date_tool
        from langchain.tools.base import BaseTool

        # All tools are now native LangChain StructuredTool instances
        self.tools = [get_date_tool()]

        # Add additional tools (pain/training agent wrappers)
        if additional_tools:
            for tool in additional_tools:
                if isinstance(tool, BaseTool):
                    self.tools.append(tool)
                else:
                    raise TypeError(f"All tools must be LangChain BaseTool instances, got: {type(tool)}")

        self.llm = llm or ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=temperature,
            streaming=True,
        )

        router_model_name = router_model or ROUTER_MODEL or DEFAULT_MODEL
        self.router_llm = ChatOpenAI(
            model=router_model_name,
            temperature=0.0,
            streaming=False,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("system", "{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

        router_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ROUTER_SYSTEM_PROMPT),
                ("system", "{session_summary}"),
                MessagesPlaceholder(variable_name="conversation"),
                ("human", "{user_input}"),
            ]
        )
        self.router_chain = router_prompt | self.router_llm.with_structured_output(RouterDecision)

        self.logger.info(
            "🏗️ %s initialised with %d tools", self.agent_name, len(self.tools)
        )

    @staticmethod
    def _chunk_to_text(chunk: Any) -> str:
        if chunk is None:
            return ""

        content = getattr(chunk, "content", None)
        if not content:
            text_attr = getattr(chunk, "text", None)
            if text_attr:
                return text_attr
            return str(chunk)

        if isinstance(content, str):
            return content

        pieces: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text_value = part.get("text")
                if text_value:
                    pieces.append(text_value)
            else:
                text_value = getattr(part, "text", None)
                if text_value:
                    pieces.append(text_value)
                elif isinstance(part, str):
                    pieces.append(part)
        return "".join(pieces)

    def _prepare_prompt_inputs(
        self,
        context_messages: Iterable[Dict[str, Any]],
        user_input: str,
    ) -> Dict[str, Any]:
        system_parts: List[str] = []
        conversation: List[Dict[str, Any]] = []

        for message in context_messages:
            if message.get("role") == "system":
                system_parts.append(message.get("content", ""))
            else:
                conversation.append(message)

        if (
            conversation
            and conversation[-1].get("role") == "user"
            and conversation[-1].get("content") == user_input
        ):
            conversation = conversation[:-1]

        chat_history: List[BaseMessage] = convert_dict_messages_to_langchain(conversation)
        context_text = "\n\n".join(part for part in system_parts if part).strip()

        # Escape braces in context to prevent LangChain from interpreting them as template variables
        context_text = context_text.replace("{", "{{").replace("}", "}}")

        return {
            "chat_history": chat_history,
            "context": context_text,
        }

    async def astream_response(
        self,
        context_messages: Iterable[Dict[str, Any]],
        user_input: str,
    ) -> AsyncIterator[Dict[str, Any]]:
        prompt_inputs = self._prepare_prompt_inputs(context_messages, user_input)
        inputs = {
            "input": user_input,
            **prompt_inputs,
        }

        # Use astream_events v2 for proper token streaming
        full_response = ""
        buffer = []
        tool_output_buffer = []  # Capture tool output separately
        in_tool_execution = False  # Track if we're inside a tool call
        tool_was_called = False  # Track if any tool was called

        async for event in self.executor.astream_events(inputs, version="v2"):
            kind = event.get("event")
            name = event.get("name", "")
            data = event.get("data", {})

            # Detect tool start
            if kind == "on_tool_start":
                tool_name = name or data.get("name", "unknown_tool")
                self.logger.info(f"🔧 Tool called: {tool_name}")
                in_tool_execution = True
                tool_was_called = True
                tool_output_buffer = []

                # Emit tool_result event (frontend can show loading)
                yield {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "content": None,
                }

            # Capture tool output (specialist agent streaming)
            elif kind == "on_chat_model_stream" and in_tool_execution:
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content"):
                    text = str(chunk.content)
                    if text:
                        tool_output_buffer.append(text)
                        # Stream tool output directly to user
                        yield {"type": "text", "content": text}

            # Detect tool end
            elif kind == "on_tool_end":
                in_tool_execution = False
                tool_result = "".join(tool_output_buffer)

                if tool_result:
                    # We have tool output - use it as final response
                    self.logger.info(f"✅ Tool completed with {len(tool_result)} chars, using as final response")
                    full_response = tool_result

            # Only stream QA agent's own responses if no tool was called
            elif kind == "on_chat_model_stream" and not in_tool_execution and not tool_was_called:
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content"):
                    text = str(chunk.content)
                    if text:
                        buffer.append(text)
                        yield {"type": "text", "content": text}

            # Capture final output
            elif kind == "on_chain_end" and name == "AgentExecutor":
                # If tool was called, use tool output
                if tool_was_called:
                    full_response = "".join(tool_output_buffer)
                else:
                    # No tool called - use QA agent's direct response
                    output = data.get("output", {})
                    if isinstance(output, dict):
                        final_text = output.get("output", "")
                    else:
                        final_text = str(output) if output else ""

                    # If no streaming happened, use the final output
                    if not buffer and final_text:
                        yield {"type": "text", "content": final_text}
                        full_response = final_text
                    else:
                        full_response = "".join(buffer)

        # Yield final event (empty content since we already streamed it)
        yield {
            "type": "final",
            "content": "",  # Don't repeat content - it was already streamed
            "intermediate_steps": None,
        }

    async def ainvoke(
        self,
        context_messages: Iterable[Dict[str, Any]],
        user_input: str,
    ) -> Dict[str, Any]:
        prompt_inputs = self._prepare_prompt_inputs(context_messages, user_input)
        result = await self.executor.ainvoke(
            {
                "input": user_input,
                **prompt_inputs,
            }
        )
        if not isinstance(result, dict):
            return {"output": str(result)}
        return result

    async def decide_route(
        self,
        conversation: Iterable[Dict[str, Any]],
        user_input: str,
        session_summary: str = "",
    ) -> RouterDecision:
        """Use the general agent's router chain to pick the best specialist."""
        messages: List[BaseMessage] = convert_dict_messages_to_langchain(conversation)

        if messages and getattr(messages[-1], "content", "") == user_input:
            conversation_messages = messages[:-1]
        else:
            conversation_messages = messages

        payload = {
            "conversation": conversation_messages,
            "user_input": user_input,
            "session_summary": session_summary or "Aucune information de session fournie.",
        }

        decision = await self.router_chain.ainvoke(payload)
        self.logger.info(
            "🧭 Router decision: agent=%s confidence=%.2f reason=%s",
            decision.agent,
            decision.confidence,
            decision.reason,
        )
        return decision


def build(llm: Optional[Any] = None, **kwargs: Any) -> GeneralCoachAgent:
    """Compatibility factory returning the LangChain general coach agent."""
    if llm is not None and not isinstance(llm, ChatOpenAI):
        logging.getLogger(__name__).warning(
            "Ignoring non-LangChain llm passed to qa.build: %s", type(llm)
        )
        llm = None
    return GeneralCoachAgent(llm=llm, **kwargs)

#- Quand l'utilisateur dit "hey", "salut", "hello", "bonjour", ou toute autre salutation:
#- Répondez: "Salut ! Je suis Hugo, ton coach virtuel GoGain 💪 Comment puis-je t'aider aujourd'hui ?"
#- proposez 3 options "faire du sport, signaler une douleur, autre", C'est tout
