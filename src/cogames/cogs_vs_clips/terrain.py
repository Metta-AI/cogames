from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING, Any, Literal, override

import numpy as np

from cogames.core import CoGameMissionVariant

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import CvCMission
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mapgen.area import AreaWhere
from mettagrid.mapgen.mapgen import MapGen, MapGenConfig
from mettagrid.mapgen.random.int import IntConstantDistribution
from mettagrid.mapgen.scene import (
    AnySceneConfig,
    ChildrenAction,
    GridTransform,
    Scene,
    SceneConfig,
)
from mettagrid.mapgen.scenes.asteroid_mask import AsteroidMaskConfig
from mettagrid.mapgen.scenes.base_hub import BaseHub, BaseHubConfig
from mettagrid.mapgen.scenes.biome_caves import BiomeCavesConfig
from mettagrid.mapgen.scenes.biome_city import BiomeCityConfig
from mettagrid.mapgen.scenes.biome_desert import BiomeDesertConfig
from mettagrid.mapgen.scenes.biome_forest import BiomeForestConfig
from mettagrid.mapgen.scenes.biome_plains import BiomePlainsConfig
from mettagrid.mapgen.scenes.bounded_layout import BoundedLayout
from mettagrid.mapgen.scenes.bsp import BSPConfig, BSPLayout
from mettagrid.mapgen.scenes.building_distributions import (
    DistributionConfig,
    UniformExtractorParams,
)
from mettagrid.mapgen.scenes.make_connected import MakeConnected
from mettagrid.mapgen.scenes.maze import MazeConfig
from mettagrid.mapgen.scenes.radial_maze import RadialMaze
from mettagrid.mapgen.scenes.random_scene import RandomScene, RandomSceneCandidate, RandomSceneConfig

HubBundle = Literal["extractors", "none", "custom"]


class MapCornerPlacementsConfig(SceneConfig):
    """Place objects at map corners. Corner indices: 0=TL, 1=TR, 2=BL, 3=BR."""

    placements: list[tuple[str, int]] = []


class MapCornerPlacements(Scene[MapCornerPlacementsConfig]):
    def render(self) -> None:
        cfg = self.config
        h, w = self.height, self.width
        offset = 2
        corners = [
            (offset, offset),
            (offset, w - 1 - offset),
            (h - 1 - offset, offset),
            (h - 1 - offset, w - 1 - offset),
        ]
        for obj_name, corner_idx in cfg.placements:
            if 0 <= corner_idx < 4 and obj_name:
                r, c = corners[corner_idx]
                if 0 <= r < h and 0 <= c < w:
                    self.grid[r, c] = obj_name


class PerimeterPlacementsConfig(SceneConfig):
    """Place objects randomly on the map perimeter at a fixed offset from edges."""

    placements: list[tuple[str, int]] = []
    offset: int = 2


class PerimeterPlacements(Scene[PerimeterPlacementsConfig]):
    """Place configured objects on unique random cells along an inset perimeter ring."""

    def render(self) -> None:
        cfg = self.config
        h, w = self.height, self.width
        offset = max(0, int(cfg.offset))

        perimeter_positions = (
            [(offset, c) for c in range(offset, w - offset)]
            + [(h - 1 - offset, c) for c in range(offset, w - offset)]
            + [(r, offset) for r in range(offset + 1, h - 1 - offset)]
            + [(r, w - 1 - offset) for r in range(offset + 1, h - 1 - offset)]
        )
        available_positions = list(dict.fromkeys(perimeter_positions))
        if not available_positions:
            return

        for obj_name, count in cfg.placements:
            spawn_count = min(max(0, int(count)), len(available_positions))
            if not obj_name or spawn_count <= 0:
                continue
            chosen = self.rng.choice(len(available_positions), size=spawn_count, replace=False)
            for idx in sorted((int(i) for i in np.atleast_1d(chosen)), reverse=True):
                r, c = available_positions.pop(idx)
                self.grid[r, c] = obj_name


class EnsureHubReachableJunctionConfig(SceneConfig):
    """Ensure each hub has at least one nearby neutral junction."""

    hub_suffix: str = ":hub"
    hub_name: str = "hub"
    junction_name: str = "junction"
    min_distance: int = 4
    max_distance: int = 15


class EnsureHubReachableJunction(Scene[EnsureHubReachableJunctionConfig]):
    def _is_hub_cell(self, value: object) -> bool:
        if not isinstance(value, str):
            return False
        return value.endswith(self.config.hub_suffix) or value == self.config.hub_name

    def _is_passable(self, value: object) -> bool:
        if not isinstance(value, str):
            return False
        if value == "empty":
            return True
        if value == self.config.junction_name:
            return True
        return self._is_hub_cell(value)

    def _reachable_cells(self, start_r: int, start_c: int) -> set[tuple[int, int]]:
        grid = self.grid
        h, w = self.height, self.width
        if not (0 <= start_r < h and 0 <= start_c < w):
            return set()
        if not self._is_passable(grid[start_r, start_c]):
            return set()

        q = deque([(start_r, start_c)])
        seen = {(start_r, start_c)}
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if (nr, nc) in seen:
                    continue
                if not self._is_passable(grid[nr, nc]):
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
        return seen

    def render(self) -> None:
        cfg = self.config
        grid = self.grid
        h, w = self.height, self.width

        hubs: list[tuple[int, int]] = []
        junctions: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                cell = grid[r, c]
                if self._is_hub_cell(cell):
                    hubs.append((r, c))
                elif cell == cfg.junction_name:
                    junctions.append((r, c))

        if not hubs:
            return

        min_r2 = cfg.min_distance * cfg.min_distance
        max_r2 = cfg.max_distance * cfg.max_distance

        for hr, hc in hubs:
            reachable = self._reachable_cells(hr, hc)
            has_reachable_nearby_junction = any(
                (jr - hr) * (jr - hr) + (jc - hc) * (jc - hc) <= max_r2 and (jr, jc) in reachable
                for jr, jc in junctions
            )
            if has_reachable_nearby_junction:
                continue

            candidates: list[tuple[int, int, int]] = []
            r0 = max(0, hr - cfg.max_distance)
            r1 = min(h, hr + cfg.max_distance + 1)
            c0 = max(0, hc - cfg.max_distance)
            c1 = min(w, hc + cfg.max_distance + 1)
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if grid[r, c] != "empty":
                        continue
                    if (r, c) not in reachable:
                        continue
                    d2 = (r - hr) * (r - hr) + (c - hc) * (c - hc)
                    if d2 < min_r2 or d2 > max_r2:
                        continue
                    candidates.append((d2, r, c))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[0])
            best_d2 = candidates[0][0]
            best = [(r, c) for d2, r, c in candidates if d2 == best_d2]
            idx = int(self.rng.integers(0, len(best)))
            rr, cc = best[idx]
            grid[rr, cc] = cfg.junction_name
            junctions.append((rr, cc))


class MachinaArenaConfig(SceneConfig):
    # Core composition
    spawn_count: int

    # Biome / dungeon structure
    base_biome: str = "plains"
    base_biome_config: dict[str, Any] = {}

    # Corner balancing: ensure roughly equal path distance from center to each corner.
    balance_corners: bool = False
    balance_tolerance: float = 3
    max_balance_shortcuts: int = 10

    #### Building placement ####

    # How much of the map is covered by buildings
    building_coverage: float = 0.05
    # Resource placement (building-based API)
    # Defines the set of buildings that can be placed on the map
    building_names: list[str] | None = None
    # What proportion of buildings are of a type, falls back to default if not set
    # If building_names is not set, this is used to determine the buildings
    building_weights: dict[str, float] | None = None

    # Hub config. `spawn_count` will be set based on `spawn_count` in this config.
    hub: BaseHubConfig = BaseHubConfig(
        hub_object="c:hub",
        corner_bundle="extractors",
        cross_bundle="none",
        cross_distance=7,
    )

    # Optional asteroid-shaped boundary mask.
    asteroid_mask: AsteroidMaskConfig | None = None

    # Objects to place at map corners (not BaseHub corners). List of (object_name, corner_index).
    # Corner indices: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right.
    map_corner_placements: list[tuple[str, int]] = []
    # Objects to place randomly on the perimeter (object_name, count).
    map_perimeter_placements: list[tuple[str, int]] = []

    #### Layers ####

    biome_weights: dict[str, float] | None = None
    dungeon_weights: dict[str, float] | None = None
    biome_count: int | None = None
    dungeon_count: int | None = None
    density_scale: float = 0.9
    max_biome_zone_fraction: float = 0.27
    max_dungeon_zone_fraction: float = 0.2

    #### Distributions ####

    # How buildings are distributed on the map
    distribution: DistributionConfig = DistributionConfig()

    # How buildings are distributed on the map per building type, falls back to global distribution if not set
    building_distributions: dict[str, DistributionConfig] | None = None


class MachinaArena(Scene[MachinaArenaConfig]):
    def render(self) -> None:
        # No direct drawing; composition is done via children actions
        return

    def get_children(self) -> list[ChildrenAction]:
        cfg = self.config

        # Base biome map
        biome_map: dict[str, type[SceneConfig]] = {
            "caves": BiomeCavesConfig,
            "forest": BiomeForestConfig,
            "desert": BiomeDesertConfig,
            "city": BiomeCityConfig,
            "plains": BiomePlainsConfig,
        }
        if cfg.base_biome not in biome_map:
            raise ValueError(f"Unknown base_biome '{cfg.base_biome}'. Valid: {sorted(biome_map.keys())}")
        BaseCfgModel: type[SceneConfig] = biome_map[cfg.base_biome]
        base_cfg: SceneConfig = BaseCfgModel.model_validate(cfg.base_biome_config or {})

        # Building weights
        default_building_weights = {
            "junction": 0.7,
            "germanium_extractor": 0.3,
            "silicon_extractor": 0.3,
            "oxygen_extractor": 0.3,
            "carbon_extractor": 0.3,
        }

        weights_dict: dict[str, float] = (
            {str(k): v for k, v in cfg.building_weights.items()} if cfg.building_weights is not None else {}
        )
        if not weights_dict:
            if cfg.building_names is not None:
                weights_dict = {name: default_building_weights.get(name, 1.0) for name in cfg.building_names}
            else:
                weights_dict = {k: v for k, v in default_building_weights.items()}

        building_names_final = list(dict.fromkeys(list((cfg.building_names or list(weights_dict.keys())))))
        building_weights_final = {
            name: weights_dict.get(name, default_building_weights.get(name, 1.0)) for name in building_names_final
        }

        # Autoscale counts
        def _autoscale_zone_counts(
            w: int, h: int, *, biome_density: float = 1.0, dungeon_density: float = 1.0
        ) -> tuple[int, int]:
            area = max(1, w * h)
            biome_divisor = max(800, int(1600 / max(0.1, biome_density)))
            dungeon_divisor = max(800, int(1500 / max(0.1, dungeon_density)))
            biomes = max(3, min(48, area // biome_divisor))
            dungeons = max(3, min(48, area // dungeon_divisor))
            return int(biomes), int(dungeons)

        biome_count = cfg.biome_count
        dungeon_count = cfg.dungeon_count
        if biome_count is None or dungeon_count is None:
            auto_biomes, auto_dungeons = _autoscale_zone_counts(
                self.width, self.height, biome_density=cfg.density_scale, dungeon_density=cfg.density_scale
            )
            biome_count = auto_biomes if biome_count is None else biome_count
            dungeon_count = auto_dungeons if dungeon_count is None else dungeon_count

        def _min_count_for_fraction(frac: float) -> int:
            if frac <= 0:
                return 1
            return int(np.ceil(1.0 / min(0.9, max(0.02, float(frac)))))

        biome_count = max(int(biome_count), _min_count_for_fraction(cfg.max_biome_zone_fraction))
        dungeon_count = max(int(dungeon_count), _min_count_for_fraction(cfg.max_dungeon_zone_fraction))

        # Candidates
        def _make_biome_candidates(weights: dict[str, float] | None) -> list[RandomSceneCandidate]:
            defaults = {"caves": 0.0, "forest": 1.0, "desert": 1.0, "city": 1.0, "plains": 1.0}
            w = {**defaults, **(weights or {})}
            cands: list[RandomSceneCandidate] = []
            if w.get("caves", 0) > 0:
                cands.append(RandomSceneCandidate(scene=BiomeCavesConfig(), weight=w["caves"]))
            if w.get("forest", 0) > 0:
                cands.append(RandomSceneCandidate(scene=BiomeForestConfig(), weight=w["forest"]))
            if w.get("desert", 0) > 0:
                cands.append(RandomSceneCandidate(scene=BiomeDesertConfig(), weight=w["desert"]))
            if w.get("city", 0) > 0:
                cands.append(RandomSceneCandidate(scene=BiomeCityConfig(), weight=w["city"]))
            if w.get("plains", 0) > 0:
                cands.append(RandomSceneCandidate(scene=BiomePlainsConfig(), weight=w["plains"]))
            return cands

        def _make_dungeon_candidates(weights: dict[str, float] | None) -> list[RandomSceneCandidate]:
            defaults = {"bsp": 0.0, "maze": 1.0, "radial": 1.0}
            w = {**defaults, **(weights or {})}
            cands: list[RandomSceneCandidate] = []
            if w.get("bsp", 0) > 0:
                cands.append(
                    RandomSceneCandidate(
                        scene=BSPConfig(
                            rooms=4,
                            min_room_size=6,
                            min_room_size_ratio=0.35,
                            max_room_size_ratio=0.75,
                        ),
                        weight=w["bsp"],
                    )
                )
            if w.get("maze", 0) > 0:
                maze_weight = w["maze"]
                cands.append(
                    RandomSceneCandidate(
                        scene=MazeConfig(
                            algorithm="dfs",
                            room_size=IntConstantDistribution(value=3),
                            wall_size=IntConstantDistribution(value=1),
                        ),
                        weight=maze_weight * 0.6,
                    )
                )
                cands.append(
                    RandomSceneCandidate(
                        scene=MazeConfig(
                            algorithm="kruskal",
                            room_size=IntConstantDistribution(value=3),
                            wall_size=IntConstantDistribution(value=1),
                        ),
                        weight=maze_weight * 0.4,
                    )
                )
            if w.get("radial", 0) > 0:
                cands.append(
                    RandomSceneCandidate(
                        scene=RadialMaze.Config(arms=8, arm_width=1, clear_background=False, outline_walls=False),
                        weight=w["radial"],
                    )
                )
            return cands

        biome_max_w = max(10, int(min(self.width * cfg.max_biome_zone_fraction, self.width // 2)))
        biome_max_h = max(10, int(min(self.height * cfg.max_biome_zone_fraction, self.height // 2)))
        dungeon_max_w = max(10, int(min(self.width * cfg.max_dungeon_zone_fraction, self.width // 2)))
        dungeon_max_h = max(10, int(min(self.height * cfg.max_dungeon_zone_fraction, self.height // 2)))

        def _wrap_in_layout(scene_cfg: SceneConfig, tag: str, max_w: int, max_h: int) -> SceneConfig:
            return BoundedLayout.Config(
                max_width=max_w,
                max_height=max_h,
                tag=tag,
                children=[
                    ChildrenAction(
                        scene=scene_cfg,
                        where=AreaWhere(tags=[tag]),
                        limit=1,
                        order_by="first",
                    )
                ],
            )

        biome_layer: ChildrenAction | None = None
        biome_cands = _make_biome_candidates(cfg.biome_weights)
        if biome_cands:
            biome_fill_count = max(1, int(biome_count * 0.6))
            biome_layer = ChildrenAction(
                scene=BSPLayout.Config(
                    area_count=biome_count,
                    children=[
                        ChildrenAction(
                            scene=_wrap_in_layout(
                                RandomScene.Config(candidates=biome_cands),
                                tag="biome.zone",
                                max_w=biome_max_w,
                                max_h=biome_max_h,
                            ),
                            where=AreaWhere(tags=["zone"]),
                            order_by="random",
                            limit=biome_fill_count,
                        )
                    ],
                ),
                where="full",
            )

        dungeon_layer: ChildrenAction | None = None
        dungeon_cands = _make_dungeon_candidates(cfg.dungeon_weights)
        if dungeon_cands:
            dungeon_fill_count = max(1, int(dungeon_count * 0.5))
            dungeon_layer = ChildrenAction(
                scene=BSPLayout.Config(
                    area_count=dungeon_count,
                    children=[
                        ChildrenAction(
                            scene=_wrap_in_layout(
                                RandomSceneConfig(candidates=dungeon_cands),
                                tag="dungeon.zone",
                                max_w=dungeon_max_w,
                                max_h=dungeon_max_h,
                            ),
                            where=AreaWhere(tags=["zone"]),
                            order_by="random",
                            limit=dungeon_fill_count,
                        )
                    ],
                ),
                where="full",
                order_by="first",
                limit=1,
            )

        children: list[ChildrenAction] = []

        # Base shell first
        children.append(ChildrenAction(scene=base_cfg, where="full"))

        if biome_layer is not None:
            children.append(biome_layer)
        if dungeon_layer is not None:
            children.append(dungeon_layer)

        asteroid_mask = cfg.asteroid_mask
        if asteroid_mask is None and min(self.width, self.height) >= 80:
            asteroid_mask = AsteroidMaskConfig()
        if asteroid_mask is not None:
            children.append(ChildrenAction(scene=asteroid_mask, where="full"))

        # Resources
        children.append(
            ChildrenAction(
                scene=UniformExtractorParams(
                    target_coverage=cfg.building_coverage,
                    building_names=building_names_final,
                    building_weights=building_weights_final,
                    clear_existing=False,
                    distribution=cfg.distribution,
                    building_distributions=cfg.building_distributions,
                ),
                where="full",
            )
        )

        # Connectivity + hub
        children.append(
            ChildrenAction(
                scene=cfg.hub.model_copy(deep=True, update={"spawn_count": cfg.spawn_count}),
                where="full",
            )
        )

        children.append(
            ChildrenAction(
                scene=MakeConnected.Config(
                    balance_corners=cfg.balance_corners,
                    balance_tolerance=cfg.balance_tolerance,
                    max_balance_shortcuts=cfg.max_balance_shortcuts,
                ),
                where="full",
            )
        )

        return children


class SequentialMachinaArenaConfig(MachinaArenaConfig):
    _scene_cls = None


class SequentialMachinaArena(Scene[SequentialMachinaArenaConfig]):
    def render(self) -> None:
        pass

    def get_children(self) -> list[ChildrenAction]:
        cfg = self.config
        biome_map: dict[str, type[SceneConfig]] = {
            "caves": BiomeCavesConfig,
            "forest": BiomeForestConfig,
            "desert": BiomeDesertConfig,
            "city": BiomeCityConfig,
            "plains": BiomePlainsConfig,
        }
        BaseCfgModel = biome_map.get(cfg.base_biome)
        if BaseCfgModel is None:
            raise ValueError(f"Unknown base_biome '{cfg.base_biome}'. Valid: {sorted(biome_map.keys())}")
        base_cfg: SceneConfig = BaseCfgModel.model_validate(cfg.base_biome_config or {})
        default_building_weights = {
            "junction": 0.6,
            "germanium_extractor": 0.2,
            "silicon_extractor": 0.2,
            "oxygen_extractor": 0.2,
            "carbon_extractor": 0.2,
        }
        weights_dict: dict[str, float] = {str(k): v for k, v in (cfg.building_weights or {}).items()}
        if not weights_dict:
            names = cfg.building_names or list(default_building_weights.keys())
            weights_dict = {name: default_building_weights.get(name, 1.0) for name in names}
        building_names_final = list(dict.fromkeys(cfg.building_names or list(weights_dict)))
        building_weights_final = {
            name: weights_dict.get(name, default_building_weights.get(name, 1.0)) for name in building_names_final
        }

        def _make_biomes(weights: dict[str, float] | None) -> list[SceneConfig]:
            if weights is not None and "none" in weights:
                return []
            defaults = {"caves": 0.0, "forest": 1.0, "desert": 1.0, "city": 1.0, "plains": 1.0}
            w = {**defaults, **(weights or {})}
            biome_defs = [
                ("caves", BiomeCavesConfig()),
                ("forest", BiomeForestConfig()),
                ("desert", BiomeDesertConfig()),
                ("city", BiomeCityConfig()),
                ("plains", BiomePlainsConfig()),
            ]
            return [cfg for key, cfg in biome_defs if float(w.get(key, 0.0)) > 0]

        def _make_dungeons(weights: dict[str, float] | None) -> list[SceneConfig]:
            if weights is not None and "none" in weights:
                return []
            defaults = {"maze": 1.0, "radial": 1.0}
            w = {**defaults, **(weights or {})}
            dungeons: list[SceneConfig] = []
            if float(w.get("maze", 0.0)) > 0:
                dungeons.append(
                    RandomScene.Config(
                        candidates=[
                            RandomSceneCandidate(
                                scene=MazeConfig(
                                    algorithm="dfs",
                                    room_size=IntConstantDistribution(value=3),
                                    wall_size=IntConstantDistribution(value=1),
                                ),
                                weight=0.6,
                            ),
                            RandomSceneCandidate(
                                scene=MazeConfig(
                                    algorithm="kruskal",
                                    room_size=IntConstantDistribution(value=3),
                                    wall_size=IntConstantDistribution(value=1),
                                ),
                                weight=0.4,
                            ),
                        ]
                    )
                )
            if float(w.get("radial", 0.0)) > 0:
                dungeons.append(RadialMaze.Config(arms=8, arm_width=1, clear_background=False, outline_walls=False))
            return dungeons

        biome_max_w = max(10, int(min(self.width * cfg.max_biome_zone_fraction, self.width // 2)))
        biome_max_h = max(10, int(min(self.height * cfg.max_biome_zone_fraction, self.height // 2)))
        dungeon_max_w = max(10, int(min(self.width * cfg.max_dungeon_zone_fraction, self.width // 2)))
        dungeon_max_h = max(10, int(min(self.height * cfg.max_dungeon_zone_fraction, self.height // 2)))

        def _wrap_in_layout(scene_cfg: SceneConfig, tag: str, max_w: int, max_h: int) -> SceneConfig:
            return BoundedLayout.Config(
                max_width=max_w,
                max_height=max_h,
                tag=tag,
                children=[
                    ChildrenAction(
                        scene=scene_cfg,
                        where=AreaWhere(tags=[tag]),
                        limit=1,
                        order_by="first",
                    )
                ],
            )

        def _make_layer(
            configs: list[SceneConfig],
            tag: str,
            max_w: int,
            max_h: int,
        ) -> ChildrenAction | None:
            if not configs:
                return None
            children = [
                ChildrenAction(
                    scene=_wrap_in_layout(scene_cfg, tag=tag, max_w=max_w, max_h=max_h),
                    where=AreaWhere(tags=["zone"]),
                    order_by="random",
                    limit=1,
                    lock=tag,
                )
                for scene_cfg in configs
            ]
            return ChildrenAction(
                scene=BSPLayout.Config(
                    area_count=len(configs),
                    children=children,
                ),
                where="full",
            )

        biomes = _make_biomes(cfg.biome_weights)
        dungeons = _make_dungeons(cfg.dungeon_weights)
        biome_layer = _make_layer(biomes, "biome.zone", biome_max_w, biome_max_h) if biomes else None
        dungeon_layer = _make_layer(dungeons, "dungeon.zone", dungeon_max_w, dungeon_max_h) if dungeons else None
        children: list[ChildrenAction] = []
        children.append(ChildrenAction(scene=base_cfg, where="full"))
        if biome_layer is not None:
            children.append(biome_layer)
        if dungeon_layer is not None:
            children.append(dungeon_layer)
        asteroid_mask = cfg.asteroid_mask
        if asteroid_mask is None and min(self.width, self.height) >= 80:
            asteroid_mask = AsteroidMaskConfig()
        if asteroid_mask is not None:
            children.append(ChildrenAction(scene=asteroid_mask, where="full"))
        children.append(
            ChildrenAction(
                scene=UniformExtractorParams(
                    target_coverage=cfg.building_coverage,
                    building_names=building_names_final,
                    building_weights=building_weights_final,
                    clear_existing=False,
                    distribution=cfg.distribution,
                    building_distributions=cfg.building_distributions,
                ),
                where="full",
            )
        )
        children.append(
            ChildrenAction(
                scene=cfg.hub.model_copy(deep=True, update={"spawn_count": cfg.spawn_count}),
                where="full",
            )
        )
        children.append(
            ChildrenAction(
                scene=MakeConnected.Config(
                    balance_corners=cfg.balance_corners,
                    balance_tolerance=cfg.balance_tolerance,
                    max_balance_shortcuts=cfg.max_balance_shortcuts,
                ),
                where="full",
            )
        )
        if cfg.map_perimeter_placements:
            children.append(
                ChildrenAction(
                    scene=PerimeterPlacements.Config(placements=cfg.map_perimeter_placements, offset=2),
                    where="full",
                )
            )
        if cfg.map_corner_placements:
            children.append(
                ChildrenAction(
                    scene=MapCornerPlacements.Config(placements=cfg.map_corner_placements),
                    where="full",
                )
            )
        children.append(
            ChildrenAction(
                scene=EnsureHubReachableJunctionConfig(max_distance=15),
                where="full",
            )
        )
        return children


class RandomTransformConfig(SceneConfig):
    scene: AnySceneConfig


class RandomTransform(Scene[RandomTransformConfig]):
    def render(self) -> None:
        return

    def get_children(self) -> list[ChildrenAction]:
        return [
            ChildrenAction(
                scene=self.config.scene.model_copy(
                    update={"transform": GridTransform(self.rng.choice(list(GridTransform)))}
                ),
                where="full",
            )
        ]


class EnvNodeVariant[T](CoGameMissionVariant, ABC):
    @abstractmethod
    def extract_node(self, env: MettaGridConfig) -> T: ...

    @abstractmethod
    def modify_node(self, node: T): ...

    @override
    def modify_env(self, mission, env) -> None:
        node = self.extract_node(env)
        self.modify_node(node)


class MapGenVariant(EnvNodeVariant[MapGenConfig]):
    @classmethod
    def extract_node(cls, env: MettaGridConfig) -> MapGenConfig:
        map_builder = env.game.map_builder
        if not isinstance(map_builder, MapGen.Config):
            raise TypeError("MapGenConfigVariant can only be applied to MapGen.Config builders")
        return map_builder


class MapSeedVariant(MapGenVariant):
    """Variant that sets the MapGen seed for deterministic map generation.

    This is primarily meant for programmatic control from experiments / pipelines:

        mission = base_mission.with_variants([MapSeedVariant(seed=1234)])
        env_cfg = mission.make_env()

    """

    name: str = "map_seed"
    description: str = "Set MapGen seed for deterministic map generation."
    seed: int

    @override
    def modify_node(self, node: MapGenConfig) -> None:
        node.seed = int(self.seed)


class BaseHubVariant(EnvNodeVariant[BaseHubConfig]):
    @override
    def compat(self, mission: CvCMission) -> bool:
        env = mission.make_env()
        if not isinstance(env.game.map_builder, MapGen.Config):
            return False
        instance = env.game.map_builder.instance
        if not isinstance(instance, BaseHub.Config):
            return False
        if isinstance(instance, RandomTransform.Config) and isinstance(instance.scene, BaseHub.Config):
            return True
        if isinstance(instance, MachinaArena.Config):
            return True
        return False

    @classmethod
    def extract_node(cls, env: MettaGridConfig) -> BaseHubConfig:
        assert isinstance(env.game.map_builder, MapGen.Config)
        instance = env.game.map_builder.instance

        if isinstance(instance, RandomTransform.Config) and isinstance(instance.scene, BaseHub.Config):
            return instance.scene

        elif isinstance(instance, MachinaArena.Config):
            return instance.hub

        raise TypeError("BaseHubVariant can only be applied RandomTransform/BaseHub or MachinaArena scenes")


class MachinaArenaVariant(EnvNodeVariant[MachinaArenaConfig]):
    def compat(self, mission: CvCMission) -> bool:
        env = mission.make_env()
        return isinstance(env.game.map_builder, MapGen.Config) and isinstance(
            env.game.map_builder.instance, MachinaArena.Config
        )

    @classmethod
    def extract_node(cls, env: MettaGridConfig) -> MachinaArenaConfig:
        assert isinstance(env.game.map_builder, MapGen.Config)
        assert isinstance(env.game.map_builder.instance, MachinaArena.Config)
        return env.game.map_builder.instance


class SequentialMachinaArenaVariant(EnvNodeVariant[SequentialMachinaArenaConfig]):
    def compat(self, mission: CvCMission) -> bool:
        env = mission.make_env()
        return isinstance(env.game.map_builder, MapGen.Config) and isinstance(
            env.game.map_builder.instance, SequentialMachinaArena.Config
        )

    @classmethod
    def extract_node(cls, env: MettaGridConfig) -> SequentialMachinaArenaConfig:
        assert isinstance(env.game.map_builder, MapGen.Config)
        assert isinstance(env.game.map_builder.instance, SequentialMachinaArena.Config)
        return env.game.map_builder.instance
