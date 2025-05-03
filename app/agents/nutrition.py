"""NutritionAgent definition – handles diet planning and meal adjustments."""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.litellm import LiteLLM

_DESCRIPTION = (
    "Designs and updates personalised diet plans aligned with the user's goals, "
    "preferences, and any dietary restrictions. Can adjust meals on the fly and "
    "analyse food photos in future iterations."
)

_SYSTEM_PROMPT = """
You are the NutritionAgent for the GoGain virtual coach.
Responsibilities:
1. Create a macro- and calorie-aligned diet plan based on the training goal.
2. Allow substitutions while maintaining nutritional targets.
3. When the user shares a meal photo (future), analyse it and estimate macros.
4. Produce a daily nutrition recap and suggestions for improvement.
Output diet plans with a ```json diet_plan``` block for storage.
"""


def build(llm: LiteLLM):
    return FunctionAgent.from_defaults(
        fn_handler=None,
        name="NutritionAgent",
        description=_DESCRIPTION,
        system_prompt=_SYSTEM_PROMPT,
        llm=llm,
        tools=[],
    ) 