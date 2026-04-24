"""Canonical install metadata for standalone CoGames packages."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GitSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    git: str
    rev: str | None = None


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
    "tribalcog": StandaloneGameInstall(
        module_name="tribal_village_env.recipe",
        package_name="tribalcog",
        source=GitSource(
            git="https://github.com/Metta-AI/cogame-tribal.git",
        ),
    ),
    "amongcogs": StandaloneGameInstall(
        module_name="amongcogs.game.game",
        package_name="amongcogs",
        source=GitSource(
            git="https://github.com/Metta-AI/cogame-amongcogs.git",
        ),
    ),
    "werecog": StandaloneGameInstall(
        module_name="werecog.cogame",
        package_name="werecog",
        source=GitSource(
            git="https://github.com/Metta-AI/cogame-werecog.git",
        ),
    ),
    "cogsguard": StandaloneGameInstall(
        module_name="cogsguard.game.game",
        package_name="cogsguard",
        source=GitSource(
            git="https://github.com/Metta-AI/cogame-cogsguard.git",
            rev="e088553fdb5753404327a12ba8412412999fa558",
        ),
    ),
}
