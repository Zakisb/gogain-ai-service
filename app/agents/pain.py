"""PainAssessmentAgent definition.

Responsible solely for interviewing the user about pain/injury and producing
an explicit pain summary object.  It purposely avoids generating workout plans
or giving medical diagnoses – that is delegated to other agents.
"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.base import BaseLLM

_DESCRIPTION = (
    "Specialized in assessing user-reported pain related to fitness and exercise. "
    "Asks clarifying questions (location, intensity, triggers) to create a pain "
    "summary. Does NOT provide medical diagnosis – instead it flags red-alerts "
    "and advises consulting a professional."
)

_SYSTEM_PROMPT = """
You are the PainAssessmentAgent for the GoGain virtual fitness coach.
Your job is to conduct a focused interview anytime the user mentions pain,
injury, or discomfort. Follow these rules:
1. Ask concise, targeted questions until you can fill the following JSON keys:
   - location (body part)
   - intensity (0-10)
   - pain_type (sharp, dull, etc.)
   - triggers (movements causing pain)
   - red_flags (true/false)
2. If you detect medical red flags (numbness, radiating pain, severe swelling),
   stop the interview, recommend seeing a healthcare professional, and set
   red_flags = true.
3. After the interview, output the filled JSON inside a Markdown fenced code
   block labelled ```json pain_summary``` so downstream agents can parse it.
4. Never prescribe exercises or medical treatment – that is handled elsewhere.
"""


def build(llm: BaseLLM) -> FunctionAgent:  # noqa: D401 – simple factory
    """Factory that returns a configured PainAssessmentAgent instance."""

    return FunctionAgent(
        name="PainAssessmentAgent",
        description=_DESCRIPTION,
        system_prompt=_SYSTEM_PROMPT,
        llm=llm,
        tools=[],  # Could later include lookup tools for anatomy, etc.
        can_handoff_to=[],
    ) 