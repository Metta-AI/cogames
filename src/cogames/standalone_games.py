"""Canonical install metadata for standalone CoGames packages."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GitSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    git: str


class StandaloneGameInstall(BaseModel):
    model_config = ConfigDict(frozen=True)

    module_name: str
    package_name: str
    source: GitSource


STANDALONE_GAMES: dict[str, StandaloneGameInstall] = {
    "diplomacog": StandaloneGameInstall(
        module_name="diplomacog.cogame",
        package_name="diplomacog",
        source=GitSource(
            git="https://github.com/Metta-AI/cogame-diplomacog.git",
        ),
    ),
    "hungercog": StandaloneGameInstall(
        module_name="hungercog.game",
        package_name="hungercog",
        source=GitSource(
            git="https://github.com/Metta-AI/cogame-hungercog.git",
        ),
    ),
    "overcogged": StandaloneGameInstall(
        module_name="overcogged.game.game",
        package_name="overcogged",
        source=GitSource(
            git="https://github.com/Metta-AI/cogame-overcogged.git",
        ),
    ),
    "tribal-village": StandaloneGameInstall(
        module_name="tribal_village_env.recipe",
        package_name="tribal-village",
        source=GitSource(
            git="https://github.com/Metta-AI/cogame-tribal_village.git",
        ),
    ),
}
