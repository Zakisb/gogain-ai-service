"""PersonalTrainingAgent definition.

Generates and maintains the user's personalised workout plan taking into
account goals, constraints, and pain summaries.  The agent relies on a separate
PainAssessmentAgent to gather injury data and on RAG tools (to be added later)
for exercise selection.
"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.base import BaseLLM
# Training agent tools temporarily disabled (tools.py removed)

_DESCRIPTION = (
    "Creates and updates personalised exercise programs. Incorporates user goals, "
    "time availability, equipment, and pain constraints from the pain summary."
)

_SYSTEM_PROMPT = """
You are the PersonalTrainingAgent for the GoGain virtual coach.

**YOUR ONLY PURPOSE: Create and modify workout plans**

**AVAILABLE TOOL:**
- `get_current_date()`: Call this tool whenever you need the current date (YYYY-MM-DD format).
  Use it for scheduling, tracking plan start dates, checking pain_summary recency (<30 days), etc.

WHAT YOU DO:
1. Create structured workout plans (days per week, session length, exercise list with sets/reps)
2. Always check for existing pain_summary (recent < 30 days) before creating plans
3. If no pain_summary exists, instruct user to get pain assessment first
4. Exclude/modify exercises that aggravate existing pain conditions
5. Output plans as Markdown with JSON metadata block: ```json training_plan```

WHAT YOU DON'T DO:
- Answer general fitness questions
- Provide exercise advice without creating a plan
- Handle nutrition topics
- Conduct pain assessments
"""


def build(llm: BaseLLM) -> FunctionAgent:
    import logging
    logger = logging.getLogger(__name__)

    # Training agent tools temporarily disabled (tools.py removed)
    from app.tools.date_tool import get_date_tool
    tools = [get_date_tool()]

    logger.info("🏗️ Building PersonalTrainingAgent with %d tools", len(tools))

    agent = FunctionAgent(
        name="PersonalTrainingAgent",
        description=_DESCRIPTION,
        system_prompt=_SYSTEM_PROMPT,
        llm=llm,
        tools=tools,
        verbose=True,
    )

    logger.info("✅ PersonalTrainingAgent created successfully")
    return agent 
