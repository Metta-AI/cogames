from __future__ import annotations

from cogames.core import CoGameMission
from mettagrid.config.action_config import ActionsConfig, MoveActionConfig, NoopActionConfig
from mettagrid.config.mettagrid_config import AgentConfig, GameConfig, MettaGridConfig, WallConfig
from mettagrid.config.obs_config import GlobalObsConfig, ObsConfig
from mettagrid.config.render_config import RenderConfig


class CvCMission(CoGameMission):
    """Mission configuration for CvC game mode."""

    default_variant: str | None = "machina_1"
    max_steps: int = 10000
    num_agents: int = 8

    def variant_module_prefixes(self) -> tuple[str, ...]:
        return ("cogames.games.cogs_vs_clips.",)

    def make_base_env(self) -> MettaGridConfig:
        return MettaGridConfig(
            game=GameConfig(
                map_builder=self.map_builder,
                max_steps=self.max_steps,
                num_agents=self.num_agents,
                resource_names=[],
                obs=ObsConfig(
                    global_obs=GlobalObsConfig(
                        local_position=True,
                        last_action_move=True,
                    ),
                    aoe_mask=True,
                ),
                actions=ActionsConfig(
                    move=MoveActionConfig(),
                    noop=NoopActionConfig(),
                ),
                agents=[AgentConfig() for _ in range(self.num_agents)],
                objects={"wall": WallConfig(name="wall")},
                render=RenderConfig(symbols={"wall": "⬛"}),
                events={},
            )
        )
