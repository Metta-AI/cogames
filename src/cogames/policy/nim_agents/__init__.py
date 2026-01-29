"""Legacy import shim for Nim-based agents.

Historically, Nim agents lived at `cogames.policy.nim_agents`. They were moved
to the optional `cogames-agents` package under `cogames_agents.policy.nim_agents`.

Old policy bundles may still import the legacy path; this module preserves
backwards compatibility by re-exporting the new implementation when available.
"""

from __future__ import annotations

try:
    import cogames.policy.nim_agents.agents as agents  # noqa: F401
    from cogames.policy.nim_agents.agents import (  # noqa: F401
        CogsguardAlignAllAgentsMultiPolicy,
        LadyBugAgentsMultiPolicy,
        RaceCarAgentsMultiPolicy,
        RandomAgentsMultiPolicy,
        ThinkyAgentsMultiPolicy,
    )
except ModuleNotFoundError as exc:
    if exc.name and (exc.name == "cogames_agents" or exc.name.startswith("cogames_agents.")):
        raise ModuleNotFoundError(
            "Legacy import `cogames.policy.nim_agents` requires optional dependency "
            "`cogames-agents` (install `cogames[agents]`)."
        ) from exc
    raise

__all__ = [
    "RandomAgentsMultiPolicy",
    "ThinkyAgentsMultiPolicy",
    "RaceCarAgentsMultiPolicy",
    "LadyBugAgentsMultiPolicy",
    "CogsguardAlignAllAgentsMultiPolicy",
]
