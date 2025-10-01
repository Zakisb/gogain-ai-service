"""NutritionAgent definition – handles diet planning and meal adjustments."""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.base import BaseLLM

_DESCRIPTION = (
    "Designs and updates personalised diet plans aligned with the user's goals, "
    "preferences, and any dietary restrictions. Can adjust meals on the fly and "
    "analyse food photos in future iterations."
)

_SYSTEM_PROMPT = """
You are the NutritionAgent for the GoGain virtual coach.

**AVAILABLE TOOL:**
- `get_current_date()`: Call this tool whenever you need the current date (YYYY-MM-DD format).
  Use it for meal planning, tracking dietary changes, or any temporal context.

Responsibilities:
1. Create a macro- and calorie-aligned diet plan based on the training goal.
2. Allow substitutions while maintaining nutritional targets.
3. When the user shares a meal photo (future), analyse it and estimate macros.
4. Produce a daily nutrition recap and suggestions for improvement.
Output diet plans with a ```json diet_plan``` block for storage.
"""


def build(llm: BaseLLM) -> FunctionAgent:
    import logging

    logger = logging.getLogger(__name__)
    from app.tools.date_tool import get_date_tool
    logger.info("🏗️ Building NutritionAgent with date tool")

    agent = FunctionAgent(
        name="NutritionAgent",
        description=_DESCRIPTION,
        system_prompt=_SYSTEM_PROMPT,
        llm=llm,
        tools=[get_date_tool()],
        verbose=True,
    )

    logger.info("✅ NutritionAgent created successfully")
    return agent 
