"""CentralRouterAgent builder.

Keeps the long system prompt out of the FastAPI layer so `app/api/chat.py`
remains clean.  Receives the list of specialist agents (tools) at call time
and returns a configured `FunctionAgent` instance ready to route messages.
"""

from typing import List
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.base.llms.base import BaseLLM

_DESCRIPTION = (
    "Primary agent responsible for understanding user intent, routing requests "
    "to specialist agents, and synthesising final responses."
)

_SYSTEM_PROMPT = """
You are the central AI coordinator for the GoGain virtual fitness coach.
Your primary role is to understand the user's message and delegate the task to
the appropriate specialist agent:
- PainAssessmentAgent: Handle pain, discomfort, injury discussions.

For pain-related queries, delegate to PainAssessmentAgent. For simple greetings 
or general conversation, respond directly. Keep your responses friendly and supportive.
If medical topics come up, remind users to consult healthcare professionals for 
proper diagnosis.
"""

  

def build_router(llm: BaseLLM, tools: List[FunctionAgent]) -> FunctionAgent:
    """Create the CentralRouterAgent (FunctionAgent)."""

    # Build the router with explicit constructor so we can control advanced knobs
    return FunctionAgent(
        name="CentralRouterAgent",
        description=_DESCRIPTION,
        system_prompt=_SYSTEM_PROMPT,
        llm=llm,
        tools=[],
        # Allow the router to pass control to any specialist agent explicitly
        can_handoff_to=[tool.name for tool in tools],
    )


# ----------------------------------------------------------------------------
# Helper to construct a full AgentWorkflow in one call
# ----------------------------------------------------------------------------


def build_workflow(
    llm: BaseLLM, tools: List[FunctionAgent], initial_state: dict | None = None
) -> AgentWorkflow:
    """Return an `AgentWorkflow` with the central router as root.

    Parameters
    ----------
    llm : LiteLLM
        Shared LLM instance.
    tools : list[FunctionAgent]
        Specialist agents to be used as tools.
    initial_state : dict | None
        Optional initial shared state across the workflow.
    """

    # build_router requires the LLM to instantiate the CentralRouterAgent
    central_agent = build_router(llm, tools)
    return AgentWorkflow(
        agents=[central_agent, *tools],
        root_agent=central_agent.name,
        initial_state=initial_state or {},
    ) 