"""Machina-1 variant bundle: pulls in the full energy -> solar -> days chain."""

from __future__ import annotations

from typing import override

from cogames.cogs_vs_clips.days import DaysVariant
from cogames.core import CoGameMissionVariant, Deps


class CvCMachina1Variant(CoGameMissionVariant):
    name: str = "machina_1"

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[DaysVariant])
