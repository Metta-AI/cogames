from __future__ import annotations

from pydantic import Field

from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from mettagrid.config.action_config import (
    ActionsConfig,
    AttackActionConfig,
    ChangeVibeActionConfig,
    MoveActionConfig,
    NoopActionConfig,
)
from mettagrid.config.filter import actorHasTag, isNot
from mettagrid.config.game_value import stat
from mettagrid.config.handler_config import Handler
from mettagrid.config.mettagrid_config import (
    AgentConfig,
    GameConfig,
    GridObjectConfig,
    InventoryConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.config.mutation import EntityTarget, addTag, logActorAgentStat, logStatToGame
from mettagrid.config.obs_config import GlobalObsConfig, ObsConfig
from mettagrid.config.render_config import RenderConfig
from mettagrid.config.reward_config import AgentReward, reward

MINIMAL_ACTIONS = ActionsConfig(
    noop=NoopActionConfig(),
    move=MoveActionConfig(),
    attack=AttackActionConfig(enabled=False),
    change_vibe=ChangeVibeActionConfig(enabled=False, vibes=[]),
)


def success_handler(*filters) -> Handler:
    return Handler(
        filters=[isNot(actorHasTag("state:solved")), *filters],
        mutations=[
            addTag("state:solved", target=EntityTarget.ACTOR),
            logActorAgentStat("goal.reached"),
            logActorAgentStat("goal.steps_to_goal", source=stat("action.move.success")),
            logStatToGame("goal.reached"),
            logStatToGame("goal.steps_to_goal", source=stat("action.move.success")),
        ],
    )


class CognitiveSubstrateMission(CvCMission):
    default_variant: str | None = None
    num_agents: int = 1
    num_cogs: int | None = 1
    min_cogs: int = 1
    max_cogs: int = 1

    object_configs: dict[str, GridObjectConfig] = Field(default_factory=dict)
    game_tags: list[str] = Field(default_factory=list)
    resource_names: list[str] = Field(default_factory=list)
    agent_inventory: InventoryConfig = Field(default_factory=InventoryConfig)
    agent_rewards: dict[str, AgentReward] = Field(
        default_factory=lambda: {"goal": reward(stat("goal.reached"), weight=1.0)}
    )
    render_symbols: dict[str, str] = Field(default_factory=dict)
    obs_width: int = 7
    obs_height: int = 7
    obs_num_tokens: int = 64

    def make_base_env(self) -> MettaGridConfig:
        return MettaGridConfig(
            game=GameConfig(
                map_builder=self.map_builder,
                max_steps=self.max_steps,
                num_agents=self.num_agents,
                resource_names=list(self.resource_names),
                obs=ObsConfig(
                    width=self.obs_width,
                    height=self.obs_height,
                    num_tokens=self.obs_num_tokens,
                    global_obs=GlobalObsConfig(
                        episode_completion_pct=False,
                        last_action=False,
                        last_reward=False,
                        last_action_move=True,
                    ),
                    aoe_mask=False,
                ),
                actions=MINIMAL_ACTIONS.model_copy(deep=True),
                agent=AgentConfig(
                    inventory=self.agent_inventory.model_copy(deep=True),
                    rewards={name: cfg.model_copy(deep=True) for name, cfg in self.agent_rewards.items()},
                ),
                objects={
                    "wall": WallConfig(name="wall", map_name="wall"),
                    **{name: cfg.model_copy(deep=True) for name, cfg in self.object_configs.items()},
                },
                tags=list(self.game_tags),
                protocol_details_obs=False,
                render=RenderConfig(symbols={"wall": "#", **self.render_symbols}),
                events={},
            )
        )


def rotate_clockwise(map_rows: list[str]) -> list[str]:
    height = len(map_rows)
    width = len(map_rows[0])
    return ["".join(map_rows[height - row - 1][col] for row in range(height)) for col in range(width)]


def rotated_options(map_rows: list[str], rotations: int = 4) -> list[list[list[str]]]:
    options: list[list[list[str]]] = []
    current = list(map_rows)
    for _ in range(rotations):
        options.append([list(row) for row in current])
        current = rotate_clockwise(current)
    return options
