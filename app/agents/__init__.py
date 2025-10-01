from __future__ import annotations

"""Agent registry.

This package exposes a single helper – `build_all` – that receives an LLM
instance and returns a list of *specialist* agents ready to be wired into the
CentralRouterAgent.  Each agent is implemented in its own module to keep
responsibilities isolated and code easy to navigate.
"""

from typing import List, Callable

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.agent.workflow import FunctionAgent

# Import *factory* functions (NOT agents) from sub-modules.
from .pain import build as _build_pain  # noqa: F401
from .training import build as _build_training  # noqa: F401
from .nutrition import build as _build_nutrition  # noqa: F401
from .health_insights import build as _build_health  # noqa: F401
from .qa import build as _build_qa  # noqa: F401

# Ordered list of factories – easier to iterate.
_AGENT_FACTORIES: List[Callable[[BaseLLM], FunctionAgent]] = [
    _build_pain,
    _build_training,
    _build_nutrition,
    _build_health,
    _build_qa,
]


def build_all(llm: BaseLLM) -> List[FunctionAgent]:
    """Instantiate and return every specialist agent.

    Parameters
    ----------
    llm : BaseLLM
        The LLM instance to pass to each agent so they all share the same
        backend for this request (cost & context efficiency).
    """
    return [factory(llm) for factory in _AGENT_FACTORIES] 
