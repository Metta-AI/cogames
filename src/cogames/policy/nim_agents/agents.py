"""Legacy module path for Nim-based agent policies.

The implementation moved to `cogames_agents.policy.nim_agents.agents`. Keep this
wrapper so older checkpoints and policy bundles that reference the old import
path can still load.
"""

from __future__ import annotations

from collections.abc import Callable

from mettagrid.policy.policy import MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


def _raise_missing_nim_agents() -> None:
    raise ModuleNotFoundError(
        "Nim scripted agents are not available (missing `nim_agents` bindings). "
        "If you're developing locally, build them with: "
        "`cd packages/cogames-agents/src/cogames_agents/policy/nim_agents && nim c nim_agents.nim`."
    )


start_measure: Callable[[], None]
end_measure: Callable[[], None]
ThinkyAgentsMultiPolicy: type[MultiAgentPolicy]
RandomAgentsMultiPolicy: type[MultiAgentPolicy]
RaceCarAgentsMultiPolicy: type[MultiAgentPolicy]
LadyBugAgentsMultiPolicy: type[MultiAgentPolicy]
CogsguardAgentsMultiPolicy: type[MultiAgentPolicy]
CogsguardAlignAllAgentsMultiPolicy: type[MultiAgentPolicy]

try:
    from cogames_agents.policy.nim_agents.agents import (
        CogsguardAgentsMultiPolicy as _CogsguardAgentsMultiPolicy,
    )
    from cogames_agents.policy.nim_agents.agents import (
        CogsguardAlignAllAgentsMultiPolicy as _CogsguardAlignAllAgentsMultiPolicy,
    )
    from cogames_agents.policy.nim_agents.agents import (
        LadyBugAgentsMultiPolicy as _LadyBugAgentsMultiPolicy,
    )
    from cogames_agents.policy.nim_agents.agents import (
        RaceCarAgentsMultiPolicy as _RaceCarAgentsMultiPolicy,
    )
    from cogames_agents.policy.nim_agents.agents import (
        RandomAgentsMultiPolicy as _RandomAgentsMultiPolicy,
    )
    from cogames_agents.policy.nim_agents.agents import (
        ThinkyAgentsMultiPolicy as _ThinkyAgentsMultiPolicy,
    )
    from cogames_agents.policy.nim_agents.agents import (
        end_measure as _end_measure,
    )
    from cogames_agents.policy.nim_agents.agents import (
        start_measure as _start_measure,
    )
except (ModuleNotFoundError, OSError) as exc:
    if (
        isinstance(exc, ModuleNotFoundError)
        and exc.name
        and (exc.name == "cogames_agents" or exc.name.startswith("cogames_agents."))
    ):
        raise ModuleNotFoundError(
            "Legacy import `cogames.policy.nim_agents.agents` requires optional dependency "
            "`cogames-agents` (install `cogames[agents]`)."
        ) from exc
    # Fall back to stubs if the optional Nim bindings are missing.

    def _missing_start_measure() -> None:
        _raise_missing_nim_agents()

    def _missing_end_measure() -> None:
        _raise_missing_nim_agents()

    class _MissingThinkyAgentsMultiPolicy(MultiAgentPolicy):
        def __init__(self, policy_env_info: PolicyEnvInterface, **_: object):
            _raise_missing_nim_agents()

    class _MissingRandomAgentsMultiPolicy(MultiAgentPolicy):
        def __init__(self, policy_env_info: PolicyEnvInterface, **_: object):
            _raise_missing_nim_agents()

    class _MissingRaceCarAgentsMultiPolicy(MultiAgentPolicy):
        def __init__(self, policy_env_info: PolicyEnvInterface, **_: object):
            _raise_missing_nim_agents()

    class _MissingLadyBugAgentsMultiPolicy(MultiAgentPolicy):
        def __init__(self, policy_env_info: PolicyEnvInterface, **_: object):
            _raise_missing_nim_agents()

    class _MissingCogsguardAgentsMultiPolicy(MultiAgentPolicy):
        def __init__(self, policy_env_info: PolicyEnvInterface, **_: object):
            _raise_missing_nim_agents()

    class _MissingCogsguardAlignAllAgentsMultiPolicy(MultiAgentPolicy):
        def __init__(self, policy_env_info: PolicyEnvInterface, **_: object):
            _raise_missing_nim_agents()

    start_measure = _missing_start_measure
    end_measure = _missing_end_measure
    ThinkyAgentsMultiPolicy = _MissingThinkyAgentsMultiPolicy
    RandomAgentsMultiPolicy = _MissingRandomAgentsMultiPolicy
    RaceCarAgentsMultiPolicy = _MissingRaceCarAgentsMultiPolicy
    LadyBugAgentsMultiPolicy = _MissingLadyBugAgentsMultiPolicy
    CogsguardAgentsMultiPolicy = _MissingCogsguardAgentsMultiPolicy
    CogsguardAlignAllAgentsMultiPolicy = _MissingCogsguardAlignAllAgentsMultiPolicy
else:
    start_measure = _start_measure
    end_measure = _end_measure
    ThinkyAgentsMultiPolicy = _ThinkyAgentsMultiPolicy
    RandomAgentsMultiPolicy = _RandomAgentsMultiPolicy
    RaceCarAgentsMultiPolicy = _RaceCarAgentsMultiPolicy
    LadyBugAgentsMultiPolicy = _LadyBugAgentsMultiPolicy
    CogsguardAgentsMultiPolicy = _CogsguardAgentsMultiPolicy
    CogsguardAlignAllAgentsMultiPolicy = _CogsguardAlignAllAgentsMultiPolicy

__all__ = [
    "start_measure",
    "end_measure",
    "ThinkyAgentsMultiPolicy",
    "RandomAgentsMultiPolicy",
    "RaceCarAgentsMultiPolicy",
    "LadyBugAgentsMultiPolicy",
    "CogsguardAgentsMultiPolicy",
    "CogsguardAlignAllAgentsMultiPolicy",
]
