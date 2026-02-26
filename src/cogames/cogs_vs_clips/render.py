from collections.abc import Sequence

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.team import TeamConfig
from mettagrid.config.render_config import (
    RenderAsset,
    RenderConfig,
    RenderHudConfig,
    RenderStatusBarConfig,
)


def render_config(team_objs: Sequence[TeamConfig]) -> RenderConfig:
    render_assets = {
        "junction": [
            RenderAsset(asset="junction.working", tags=["team:cogs"]),
            RenderAsset(asset="junction.clipped1", tags=["team:clips"]),
            RenderAsset(asset="junction"),
        ],
        **{
            f"{team.short_name}:{gear}": [RenderAsset(asset=f"{gear}_station")]
            for team in team_objs
            for gear in CvCConfig.GEAR
        },
    }

    return RenderConfig(
        agent_huds={
            "hp": RenderHudConfig(resource="hp", max=100, rank=0),
            "energy": RenderHudConfig(resource="energy", max=100, rank=1),
        },
        object_status={
            "agent": {
                "hp": RenderStatusBarConfig(resource="hp", max=100, divisions=10, rank=0),
                "energy": RenderStatusBarConfig(
                    resource="energy",
                    short_name="E",
                    max=100,
                    divisions=20,
                    rank=1,
                ),
            },
        },
        assets=render_assets,
    )
