"""GeneralQAAgent – fallback for any fitness/wellness questions."""

from llama_index.core.agent import FunctionAgent
from llama_index.llms.litellm import LiteLLM

_DESCRIPTION = (
    "Answers general questions about fitness concepts, nutrition, wellness, or "
    "exercise techniques that don't belong to other specialised agents."
)

_SYSTEM_PROMPT = """
You are the GeneralQAAgent for the GoGain virtual coach.
Provide clear, evidence-based answers in a friendly tone.
Avoid medical diagnoses; link to reputable sources when useful.
"""


def build(llm: LiteLLM):
    return FunctionAgent.from_defaults(
        fn_handler=None,
        name="GeneralQAAgent",
        description=_DESCRIPTION,
        system_prompt=_SYSTEM_PROMPT,
        llm=llm,
        tools=[],
    ) 