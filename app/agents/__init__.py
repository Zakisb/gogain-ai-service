from __future__ import annotations

"""Agent registry.

This package exposes a single helper – `build_all` – that receives an LLM
instance and returns a list of *specialist* agents ready to be wired into the
CentralRouterAgent.  Each agent is implemented in its own module to keep
responsibilities isolated and code easy to navigate.
"""

from typing import List

from .pain import PainAssessmentAgent  # noqa: F401
from .training import TrainingAgent  # noqa: F401
from .nutrition import NutritionAgent  # noqa: F401
from .health_insights import HealthInsightsAgent  # noqa: F401
from .qa import GeneralCoachAgent  # noqa: F401


def build_all(*_args, **_kwargs) -> List[object]:
    """Legacy helper returning instantiated specialist agents."""
    return [
        PainAssessmentAgent(),
        TrainingAgent(),
        NutritionAgent(),
        HealthInsightsAgent(),
        GeneralCoachAgent(),
    ]
