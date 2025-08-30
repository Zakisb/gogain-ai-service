"""HealthInsightsAgent – interprets wearable data and provides actionable advice."""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.litellm import LiteLLM

_DESCRIPTION = (
    "Analyses wearable tracker data (sleep, HRV, steps) to offer recovery and "
    "load-adjustment insights. Flags days that might require lighter sessions or "
    "additional rest."
)

_SYSTEM_PROMPT = """
You are the HealthInsightsAgent for the GoGain virtual coach.
Tasks:
1. Interpret time-series health metrics (sleep hours, HRV score, resting HR,
   step count) provided as JSON.
2. Generate concise, motivating feedback plus training-load recommendations.
3. Set a flag `deload_required` = true when recovery scores are poor.
4. Output advice as plain text followed by a ```json health_insight``` block.
"""


def build(llm: LiteLLM):
    return FunctionAgent.from_defaults(
        fn_handler=None,
        name="HealthInsightsAgent",
        description=_DESCRIPTION,
        system_prompt=_SYSTEM_PROMPT,
        llm=llm,
        tools=[],
    ) 