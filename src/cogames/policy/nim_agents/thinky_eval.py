"""Legacy module path for Thinky eval helpers.

The implementation moved to `cogames_agents.policy.nim_agents.thinky_eval`.
"""

from __future__ import annotations

import importlib
from typing import Any

try:
    _thinky_eval = importlib.import_module("cogames_agents.policy.nim_agents.thinky_eval")
except ModuleNotFoundError as exc:
    if exc.name and (exc.name == "cogames_agents" or exc.name.startswith("cogames_agents.")):
        raise ModuleNotFoundError(
            "Legacy import `cogames.policy.nim_agents.thinky_eval` requires optional dependency "
            "`cogames-agents` (install `cogames[agents]`)."
        ) from exc
    raise

AGENT_PATH = _thinky_eval.AGENT_PATH
EVALS = _thinky_eval.EVALS
MAX_STEPS = _thinky_eval.MAX_STEPS
NUM_COGS = _thinky_eval.NUM_COGS
SEED = _thinky_eval.SEED
main = _thinky_eval.main
run_eval = _thinky_eval.run_eval


def __getattr__(name: str) -> Any:
    return getattr(_thinky_eval, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_thinky_eval)))


__all__ = [
    "AGENT_PATH",
    "EVALS",
    "MAX_STEPS",
    "NUM_COGS",
    "SEED",
    "main",
    "run_eval",
]
