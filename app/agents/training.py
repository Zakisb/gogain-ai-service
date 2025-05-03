"""PersonalTrainingAgent definition.

Generates and maintains the user's personalised workout plan taking into
account goals, constraints, and pain summaries.  The agent relies on a separate
PainAssessmentAgent to gather injury data and on RAG tools (to be added later)
for exercise selection.
"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.litellm import LiteLLM

_DESCRIPTION = (
    "Creates and updates personalised exercise programs. Incorporates user goals, "
    "time availability, equipment, and pain constraints from the pain summary."
)

_SYSTEM_PROMPT = """
You are the PersonalTrainingAgent for the GoGain virtual coach.
Your tasks:
1. Given the user's goals and constraints, craft a structured workout plan
   (days per week, session length, exercise list with sets/reps).
2. Always check for an existing pain_summary (recent < 30 days). If missing,
   instruct the CentralRouter to initiate PainAssessmentAgent first.
3. When pain_summary is present, exclude or modify exercises that aggravate
   those areas.
4. Output plans as Markdown with a JSON metadata block labelled
   ```json training_plan``` for downstream storage.
5. Do not handle nutrition – that belongs to NutritionAgent.
"""


def build(llm: LiteLLM):
    return FunctionAgent(
        name="PersonalTrainingAgent",
        description=_DESCRIPTION,
        system_prompt=_SYSTEM_PROMPT,
        llm=llm,
        tools=[],  # Exercise-search tool will be plugged in later
        can_handoff_to=[],
    ) 