from __future__ import annotations

from cogames.policy.starter_agent import BaseStarterPolicy


class MinerRolePolicy(BaseStarterPolicy):
    short_names = ["miner"]
    _role = "miner"


class ScoutRolePolicy(BaseStarterPolicy):
    short_names = ["scout"]
    _role = "scout"


class AlignerRolePolicy(BaseStarterPolicy):
    short_names = ["aligner"]
    _role = "aligner"


class ScramblerRolePolicy(BaseStarterPolicy):
    short_names = ["scrambler"]
    _role = "scrambler"
