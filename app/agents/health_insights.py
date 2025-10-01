"""HealthInsightsAgent – interprets wearable data and provides actionable advice."""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.base import BaseLLM

_DESCRIPTION = (
    "Analyses wearable tracker data (sleep, HRV, steps) to offer recovery and "
    "load-adjustment insights. Flags days that might require lighter sessions or "
    "additional rest."
)

_SYSTEM_PROMPT = """
You are the HealthInsightsAgent for the GoGain virtual coach.

**AVAILABLE TOOL:**
- `get_current_date()`: Call this tool whenever you need the current date (YYYY-MM-DD format).
  Use it for analyzing time-series health metrics, tracking trends, or dating insights.

Tasks:
1. Interpret time-series health metrics (sleep hours, HRV score, resting HR,
   step count) provided as JSON.
2. Generate concise, motivating feedback plus training-load recommendations.
3. Set a flag `deload_required` = true when recovery scores are poor.
4. Output advice as plain text followed by a ```json health_insight``` block.
"""


def build(llm: BaseLLM) -> FunctionAgent:
    import logging

    logger = logging.getLogger(__name__)
    from app.tools.date_tool import get_date_tool
    logger.info("🏗️ Building HealthInsightsAgent with date tool")

    agent = FunctionAgent(
        name="HealthInsightsAgent",
        description=_DESCRIPTION,
        system_prompt=_SYSTEM_PROMPT,
        llm=llm,
        tools=[get_date_tool()],
        verbose=True,
    )

    logger.info("✅ HealthInsightsAgent created successfully")
    return agent 
