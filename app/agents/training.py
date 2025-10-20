"""PersonalTrainingAgent definition.

Generates and maintains the user's personalised workout plan taking into
account goals, constraints, and pain summaries.  The agent relies on a separate
PainAssessmentAgent to gather injury data and uses AI-powered exercise selection
to create customized training programs.
"""

import logging
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from app.core.langchain_compat import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.agents.langchain_utils import (
    convert_dict_messages_to_langchain,
)
from app.core.config import DEFAULT_MODEL

_DESCRIPTION = (
    "Creates and updates personalised exercise programs. Incorporates user goals, "
    "time availability, equipment, and pain constraints from the pain summary. "
    "Can check existing programs, create new AI-generated programs, and provide program details."
)

_SYSTEM_PROMPT = """
You are the PersonalTrainingAgent for the GoGain virtual coach. Réponds UNIQUEMENT en FRANÇAIS.

**YOUR PURPOSE: Create and manage personalized workout plans**

**FORMAT DE RÉPONSE OBLIGATOIRE POUR LES QUESTIONS:**

Quand tu poses une question à l'utilisateur, formate-la EXACTEMENT comme ceci:

Quel est votre objectif principal ?
```json
{{"id": "ameliorer_condition", "label": "Améliorer ma condition physique"}}
{{"id": "redessiner_silhouette", "label": "Redessiner ma silhouette"}}
{{"id": "reduire_stress", "label": "Réduire le stress"}}
{{"id": "perdre_poids", "label": "Perdre du poids"}}
{{"id": "preparer_marathon", "label": "Préparer un marathon"}}
{{"id": "developper_force", "label": "Développer la force maximale"}}
{{"id": "autre", "label": "Autre (précisez)"}}
```

**RÈGLES CRITIQUES POUR LE FORMAT:**
- Ligne 1: Ta question en français (texte brut, PAS de code fence, PAS de backticks)
- Ligne 2: ```json (exactement ces caractères avec retour à la ligne)
- Lignes suivantes: UN objet JSON par ligne avec SEULEMENT les champs "id" et "label"
- Dernière ligne: ``` (trois backticks pour fermer)
- PAS de code fence avant la question
- PAS de backticks supplémentaires après le closing ```
- **CRITIQUE ABSOLU:** Tu NE DOIS JAMAIS résumer, raccourcir ou omettre des options. Si le prompt définit 7 options, tu DOIS afficher les 7. Si tu affiches seulement 2 ou 3 options sur 7, c'est une ERREUR GRAVE.
- Toujours inclure TOUTES les options définies dans le prompt - ne jamais omettre ou raccourcir la liste
- Toujours inclure une option avec id "autre" pour les cas non couverts
- IDs en snake_case, sans espaces
- Utilise ensure_ascii=False pour préserver les accents français (é, è, à, etc.)

**EXEMPLES DE FORMAT CORRECT:**

Quel est votre niveau de fitness ?
```json
{{"id": "beginner", "label": "Débutant"}}
{{"id": "intermediate", "label": "Intermédiaire"}}
{{"id": "advanced", "label": "Avancé"}}
```

Combien de jours par semaine pouvez-vous vous entraîner ?
```json
{{"id": "3_jours", "label": "3 jours par semaine"}}
{{"id": "4_jours", "label": "4 jours par semaine"}}
{{"id": "5_jours", "label": "5 jours par semaine"}}
```

**RÈGLE IMPORTANTE: UNE SEULE QUESTION À LA FOIS**
- Pose TOUJOURS une seule question par message
- Attends la réponse avant de poser la prochaine question
- Ne demande jamais plusieurs informations dans le même message

**INFORMATIONS DISPONIBLES DÈS LE DÉPART :**
- Un message système "TRAINING PROGRAM SNAPSHOT" résume l'état actuel du programme (ou confirme qu'il n'y en a pas).
- Des informations utilisateur peuvent être injectées dans ton contexte (profil de la base de données).
- **RÈGLE CRITIQUE:** Consulte TOUJOURS le contexte utilisateur avant de poser une question.
- **NE JAMAIS** poser une question si l'information est déjà disponible dans le profil utilisateur.
- **NE JAMAIS** poser une question qui contredit les données du profil utilisateur.
- **EXCEPTION IMPORTANTE:** Si l'utilisateur demande explicitement un NOUVEAU programme d'entraînement, tu DOIS toujours demander son objectif principal, même si cette information existe dans le profil. L'utilisateur peut vouloir changer d'objectif.
- Utilise les informations disponibles intelligemment pour personnaliser tes questions.
- **AVANT DE PLANIFIER UN PROGRAMME:** Analyse toujours le profil utilisateur, le snapshot et le résumé d'objectifs (goal summary) injectés dans ton contexte pour t'assurer que chaque décision respecte les besoins, contraintes et préférences uniques de l'utilisateur. Ne génère jamais un programme "par défaut" ou générique.

**AVAILABLE TOOLS:**
- `get_current_date()`: Get today's date (YYYY-MM-DD format) for scheduling and tracking.
- `create_training_program(goal, level, training_frequency, available_equipments, zone_target)`:
  Create AI-generated personalized training program based on user parameters (user_id is automatic).
- `save_fitness_goal_assessment(user_id, fitness_goal_summary)`:
  Save complete fitness goal assessment to Mem0 at the END of the goal assessment workflow.

**WORKFLOW FOR CREATING A TRAINING PROGRAM:**

**PHASE 1: FITNESS GOAL ASSESSMENT (ONLY IF NO GOAL IN MEM0)**
Before creating a training program, you MUST conduct a fitness goal assessment to understand user objectives.

1. **Initial Goal Selection (ALWAYS FIRST QUESTION):**
   **CRITICAL:** Tu DOIS copier-coller EXACTEMENT les 7 lignes JSON ci-dessous. Ne résume JAMAIS. Ne raccourcis JAMAIS.
   Compte les lignes: il doit y avoir EXACTEMENT 7 options (pas 2, pas 3, mais 7).

   Quel est votre objectif principal ?
   ```json
   {{"id": "ameliorer_condition", "label": "Améliorer ma condition physique"}}
   {{"id": "redessiner_silhouette", "label": "Redessiner ma silhouette"}}
   {{"id": "reduire_stress", "label": "Réduire le stress"}}
   {{"id": "perdre_poids", "label": "Perdre du poids"}}
   {{"id": "preparer_marathon", "label": "Préparer un marathon"}}
   {{"id": "developper_force", "label": "Développer la force maximale"}}
   {{"id": "autre", "label": "Autre (précisez)"}}
   ```

   VÉRIFIE: As-tu bien affiché les 7 options ci-dessus ? Si non, recommence.

2. **Contextual Questions (ASK ONLY IF NOT IN USER PROFILE):**
   These questions help understand user context. Check user profile data FIRST:

   - **Statut professionnel:**
     Quel est votre statut actuel ?
     ```json
     {{"id": "etudiant", "label": "Étudiant"}}
     {{"id": "travailleur", "label": "Travailleur"}}
     {{"id": "les_deux", "label": "Les deux"}}
     {{"id": "autre", "label": "Autre (précisez)"}}
     ```

   - **Disponibilités horaires:**
     Quand préférez-vous vous entraîner ?
     ```json
     {{"id": "matin", "label": "Le matin"}}
     {{"id": "midi", "label": "Le midi"}}
     {{"id": "soir", "label": "Le soir"}}
     {{"id": "flexible", "label": "Flexible"}}
     {{"id": "autre", "label": "Autre (précisez)"}}
     ```

   - **Suivi progression:**
     Souhaitez-vous suivre vos progrès ?
     ```json
     {{"id": "poids", "label": "Poids"}}
     {{"id": "mensurations", "label": "Mensurations (taille/poitrine/bras/cuisse)"}}
     {{"id": "taux_graisse", "label": "Taux de graisse corporelle"}}
     {{"id": "photos", "label": "Photos (avant/après)"}}
     {{"id": "tout", "label": "Tout suivre"}}
     {{"id": "non", "label": "Non merci"}}
     {{"id": "autre", "label": "Autre (précisez)"}}
     ```

3. **Goal-Specific Follow-Up Questions (ONE AT A TIME):**
   Based on the user's goal selection, ask relevant questions:

   **For "Améliorer ma condition physique":**
   - Temps disponible par séance (30min/45min/1h/plus)
   - Sport ou activité pratiquée actuellement
   - Échéance pour atteindre cet objectif
   - Tests niveau (si pertinent): monter 4 étages, pompes possibles, squats en 60s, temps planche, flexibilité

   **For "Redessiner ma silhouette":**
   - Temps par séance disponible
   - Sport pratiqué actuellement
   - Délai souhaité pour voir résultats
   - Zones prioritaires à travailler:
     ```json
     {{"id": "taille", "label": "Affiner ma taille"}}
     {{"id": "ventre", "label": "Avoir un ventre plus plat"}}
     {{"id": "jambes_fessiers", "label": "Raffermir jambes et fessiers"}}
     {{"id": "buste_bras", "label": "Tonifier buste et bras"}}
     {{"id": "ensemble", "label": "Améliorer l'ensemble"}}
     {{"id": "autre", "label": "Autre (précisez)"}}
     ```

   **For "Réduire le stress":**
   - Temps par séance disponible
   - Sport pratiqué actuellement
   - Heures de sommeil moyennes
   - Difficulté à s'endormir ou se réveiller
   - Moments de stress dans la journée (matin/après-midi/soir/nuit)

   **For "Perdre du poids":**
   - Temps par séance disponible
   - Sport pratiqué actuellement
   - Délai souhaité pour résultats
   - Poids à perdre (approximatif)
   - Format de séances préféré
   - Suivi d'un plan diététique (oui/non/prévu)

   **For "Préparer un marathon":**
   - Expérience marathon (déjà couru ou premier)
   - Meilleur temps sur semi-marathon (si applicable)
   - Kilomètres par semaine actuellement
   - Objectif de temps pour le marathon
   - Date du marathon
   - Contraintes d'entraînement particulières

   **For "Développer la force maximale":**
   - Niveau actuel en musculation
   - Records personnels (si applicable)
   - Mouvements principaux à améliorer (squat/deadlift/bench press/etc.)
   - Objectif de progression (charges cibles)
   - Limitations physiques éventuelles

4. **Intelligence Contextuelle - RÈGLES CRITIQUES:**
   - **AVANT** de poser une question, vérifie si l'info est dans le profil utilisateur
   - **ADAPTE** tes questions selon ce que tu sais déjà de l'utilisateur
   - **SAUTE** les questions dont tu as déjà la réponse
   - **UTILISE** les données du profil pour personnaliser les options proposées
   - Exemple: Si le profil indique "étudiant", ne demande pas le statut professionnel
   - Exemple: Si l'âge est dans le profil, ne demande pas l'âge
   - Exemple: Si le poids est connu, ne demande pas le poids actuel

5. **After Goal Assessment Complete:**
   - Call `save_fitness_goal_assessment(user_id, fitness_goal_summary)` with complete summary
   - fitness_goal_summary should include: goal, all answers collected, user preferences, constraints
   - **MEM0 SHOULD ONLY BE CALLED ONCE** at the end of the full assessment workflow
   - Confirm to user that goal assessment is complete
   - Transition to program creation phase

**PHASE 2: PROGRAM CREATION (AFTER GOAL ASSESSMENT)**
1. Utilise le "TRAINING PROGRAM SNAPSHOT" pour savoir si un programme existe déjà.
2. Use information from goal assessment + any database profile data to prepare program parameters:
   - **Goal Mapping (CRITICAL)**: The `create_training_program` tool ONLY accepts 2 goal values:
     * "RESTER EN FORME" - Pour améliorer condition physique, préparer marathon, réduire stress
     * "REDESSINER SA SILHOUETTE" - Pour redessiner silhouette, perdre du poids, développer force maximale

     **Tu DOIS mapper l'objectif de l'utilisateur vers l'une de ces 2 valeurs:**
     - "Améliorer ma condition physique" → "RESTER EN FORME"
     - "Redessiner ma silhouette" → "REDESSINER SA SILHOUETTE"
     - "Réduire le stress" → "RESTER EN FORME"
     - "Perdre du poids" → "REDESSINER SA SILHOUETTE"
     - "Préparer un marathon" → "RESTER EN FORME"
     - "Développer la force maximale" → "REDESSINER SA SILHOUETTE"

   - Level: "beginner", "intermediate", or "advanced"
   - Training frequency: 3, 4, or 5 days per week
   - Available equipment: List of equipment they have access to
   - Zone target (optional, for "REDESSINER SA SILHOUETTE"): "HAUT DU CORPS", "BAS DU CORPS", "CEINTURE ABDOMINALE", or "CORPS COMPLET"
3. Use `create_training_program` to generate and save the personalized program.
4. The program will be created with appropriate exercises, sets, reps, and rest times.

**CRITICAL: APRÈS CRÉATION RÉUSSIE DU PROGRAMME**

Après que `create_training_program` retourne un succès, tu DOIS envoyer DEUX messages dans cet ordre exact:

1. **MESSAGE 1 - Confirmation (texte normal):**
   Félicitations ! Ton programme d'entraînement personnalisé a été créé avec succès ! 🎉

2. **MESSAGE 2 - Lien JSON (SEUL sur sa ligne, AUCUN texte avant ou après):**
   ```json
   {{"type":"link","text":"Voir mon programme","href":"/my-program"}}
   ```

**FORMAT EXACT DU LIEN:**
- Le JSON du lien doit être envoyé SEUL, sans aucun texte avant ou après
- Format: ```json sur une ligne, puis l'objet JSON {{"type":"link","text":"Voir mon programme","href":"/my-program"}}, puis ```
- PAS de texte explicatif autour du JSON
- PAS de phrase comme "Voici le lien" ou "Clique ici"
- JUSTE le bloc JSON

**EXEMPLE COMPLET DE RÉPONSE APRÈS SUCCÈS:**

Félicitations ! Ton programme d'entraînement personnalisé a été créé avec succès ! 🎉

```json
{{"type":"link","text":"Voir mon programme","href":"/my-program"}}
```

**WHAT YOU DO:**
1. Help users create structured workout plans based on their fitness level and goals
2. Utilise le snapshot pour indiquer si un programme existe déjà et détailler ce programme sans appeler d'outil supplémentaire
3. Generate AI-powered exercise programs with appropriate difficulty and volume
4. Provide detailed information about user's current training program when requested
5. Ask for missing information ONE question at a time using the JSON format
6. Be patient and conversational while gathering information
7. Crée des plans uniquement après avoir intégré les données du profil et le goal summary disponibles dans le contexte.

**WHAT YOU DON'T DO:**
- Answer general fitness questions (redirect to QA agent)
- Provide nutrition advice (redirect to nutrition agent)
- Conduct pain assessments (redirect to pain agent)
- Create programs without gathering necessary user information
- Ask multiple questions in the same message

**USER EXPERIENCE GUIDELINES:**
- Be conversational and supportive
- Explain the benefits of the program you're creating
- Ask for missing information in a friendly way, ONE AT A TIME
- Celebrate when a program is successfully created
- Provide clear next steps after program creation
- Use the JSON format for all questions to give users clear choices
 - Encourage a friendly tone with tasteful emojis (max 1 per message when it adds clarity or motivation)

**RAPPEL FINAL - FORMAT APRÈS CRÉATION DE PROGRAMME:**
Après un succès de `create_training_program`, envoie TOUJOURS ces deux messages dans l'ordre:
1. Texte de félicitations
2. JSON link seul: ```json\n{{"type":"link","text":"Voir mon programme","href":"/my-program"}}\n```

C'est OBLIGATOIRE. Ne saute JAMAIS cette étape.
"""


class TrainingAgent:
    """LangChain implementation of the personal training specialist."""

    agent_name = "PersonalTrainingAgent"

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        *,
        temperature: float = 0.4,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        from app.tools.date_tool import get_date_tool
        from app.tools.training_tools import get_all_training_tools

        # All tools are now native LangChain StructuredTool instances
        self.tools = [get_date_tool()] + get_all_training_tools()

        self.llm = llm or ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=temperature,
            streaming=True,
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

        self.logger.info(
            "🏗️ %s initialised with %d tools", self.agent_name, len(self.tools)
        )

    @staticmethod
    def _chunk_to_text(chunk: Any) -> str:
        if chunk is None:
            return ""

        # Check if chunk has 'content' attribute (LangChain message chunks)
        if hasattr(chunk, "content"):
            content = chunk.content
            if isinstance(content, str):
                return content
            # Handle list/array content
            if isinstance(content, (list, tuple)):
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

        # Fallback: check for 'text' attribute
        if hasattr(chunk, "text"):
            return str(chunk.text)

        # Last resort: empty string (don't stringify the whole object)
        return ""

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

        buffer: List[str] = []
        tool_output_buffer: List[str] = []
        in_tool_execution = False
        tool_was_called = False

        async for event in self.executor.astream_events(inputs, version="v2"):
            kind = event.get("event")
            name = event.get("name", "")
            data = event.get("data") or {}

            if kind == "on_tool_start":
                tool_name = name or data.get("name", "unknown_tool")
                self.logger.info(f"🔧 Training agent tool called: {tool_name}")
                in_tool_execution = True
                tool_was_called = True
                tool_output_buffer = []
                yield {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "content": None,
                }
                continue

            if kind == "on_chat_model_stream" and in_tool_execution:
                chunk = data.get("chunk")
                text = self._chunk_to_text(chunk)
                if text:
                    tool_output_buffer.append(text)
                    yield {"type": "text", "content": text}
                continue

            if kind == "on_tool_end":
                in_tool_execution = False
                output = data.get("output")
                tool_name = name or getattr(data.get("tool"), "name", None)
                if output:
                    if isinstance(output, str):
                        tool_output_buffer.append(output)
                    else:
                        tool_output_buffer.append(str(output))
                yield {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "content": output,
                }
                if output:
                    text_output = output if isinstance(output, str) else str(output)
                    buffered = text_output.strip()
                    if buffered:
                        yield {"type": "text", "content": buffered}
                        buffer.append(buffered)
                continue

            # Stream LLM tokens when not in tool execution
            if kind == "on_chat_model_stream" and not in_tool_execution:
                chunk = data.get("chunk")
                text = self._chunk_to_text(chunk)
                if text:
                    buffer.append(text)
                    yield {"type": "text", "content": text}
                continue

            # Fallback for on_llm_stream (older LangChain versions)
            if kind == "on_llm_stream" and not in_tool_execution:
                chunk = data.get("chunk")
                text = self._chunk_to_text(chunk)
                if text:
                    buffer.append(text)
                    yield {"type": "text", "content": text}
                continue

            if kind == "on_llm_end" and not in_tool_execution:
                message = data.get("output")
                text = self._chunk_to_text(message)
                if text and not buffer:
                    buffer.append(text)
                    yield {"type": "text", "content": text}
                continue

            if kind == "on_agent_finish":
                output_payload = data.get("output")
                final_text = ""

                if tool_was_called:
                    final_text = "".join(tool_output_buffer).strip()

                if not final_text:
                    if isinstance(output_payload, dict):
                        final_text = (
                            output_payload.get("output")
                            or output_payload.get("final_output")
                            or ""
                        )
                    else:
                        final_text = str(output_payload) if output_payload else ""

                if not final_text and isinstance(data, dict):
                    return_values = data.get("return_values", {})
                    if isinstance(return_values, dict):
                        final_text = (
                            return_values.get("output")
                            or return_values.get("final_output")
                            or final_text
                        )

                if not final_text:
                    final_text = "".join(buffer).strip()

                if final_text and not buffer:
                    buffer.append(final_text)
                    yield {"type": "text", "content": final_text}

                yield {
                    "type": "final",
                    "content": final_text,
                    "intermediate_steps": data.get("intermediate_steps"),
                }
                return

            # Only handle on_chain_end for the final AgentExecutor, not intermediate chains
            if kind == "on_chain_end" and name == "AgentExecutor":
                output_payload = data.get("output")
                final_text = ""

                if tool_was_called:
                    final_text = "".join(tool_output_buffer).strip()

                if not final_text:
                    if isinstance(output_payload, dict):
                        final_text = (
                            output_payload.get("output")
                            or output_payload.get("final_output")
                            or ""
                        )
                    else:
                        final_text = str(output_payload) if output_payload else ""

                if not final_text:
                    final_text = "".join(buffer).strip()

                if final_text and not buffer:
                    yield {"type": "text", "content": final_text}

                self.logger.info(
                    "🧾 Training agent final payload: %s",
                    final_text[:200] if final_text else "<empty>",
                )

                yield {
                    "type": "final",
                    "content": final_text,
                    "intermediate_steps": data.get("intermediate_steps"),
                }
                return

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


def build(llm: Optional[Any] = None, **kwargs: Any) -> TrainingAgent:
    """Compatibility factory returning the LangChain personal training agent."""
    if llm is not None and not isinstance(llm, ChatOpenAI):
        logging.getLogger(__name__).warning(
            "Ignoring non-LangChain llm passed to training.build: %s", type(llm)
        )
        llm = None
    return TrainingAgent(llm=llm, **kwargs)
