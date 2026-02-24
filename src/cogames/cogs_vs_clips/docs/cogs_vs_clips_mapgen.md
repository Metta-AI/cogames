## Cogs vs Clips: Procedural Map Generation, Variants, and Missions

This guide explains how the procedural map system is wired together and how to extend it safely.

---

### Core Modules

- `cogs_vs_clips/terrain.py`
  - `MachinaArena` / `MachinaArenaConfig` scene
  - `SequentialMachinaArena` / `SequentialMachinaArenaConfig` scene
  - Variant helpers (`MapSeedVariant`, `BaseHubVariant`, `MachinaArenaVariant`)
- `mettagrid/mapgen/scenes/building_distributions.py`
  - `UniformExtractorScene` and `DistributionConfig` for building placement
- `cogames/core.py`
  - Base `CoGameSite`, `CoGameMission`, `CoGameMissionVariant` types
- `cogs_vs_clips/mission.py`
  - `CvCMission` (the concrete mission class for Cogs vs Clips)
- `cogs_vs_clips/sites.py`
  - Catalog of sites
- `cogs_vs_clips/variants.py`
  - Catalog of variants
- `cogames/cli/mission.py`
  - CLI glue (`cogames play`, `cogames missions`, `cogames train`) and variant composition

Everything ultimately produces a `MapBuilderConfig` that feeds into a `MettaGridConfig`. Missions and variants
coordinate map building, agent setup, and post-processing such as hub rewrites.

---

### Procedural Composition

#### `MachinaArena` (Scene)

Asteroid arena built as a Scene graph: base-biome shell, optional biome/dungeon overlays, resource placement,
connectivity, and a central hub.

Config fields (all keyword-only on `MachinaArenaConfig`):

| Category                | Parameters                                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| Size (MapGen)           | `width`, `height`                                                                               |
| Randomness (MapGen)     | `seed`                                                                                          |
| Base biome              | `base_biome` (`"caves"`, `"forest"`, `"desert"`, `"city"`, `"plains"`), `base_biome_config`     |
| Biome overlays          | `biome_weights`, `biome_count`, `density_scale`, `max_biome_zone_fraction`                      |
| Dungeon overlays        | `dungeon_weights`, `dungeon_count`, `max_dungeon_zone_fraction`                                 |
| Buildings               | `building_names`, `building_weights`, `building_coverage`                                       |
| Placement distributions | `distribution` (global `DistributionConfig`), `building_distributions` (per-building overrides) |
| Hub layout              | `hub` (`BaseHubConfig` with `corner_bundle`, `cross_bundle`, `cross_distance`, etc.)            |

Important details:

- Building parameters are expressed in "buildings" (stations). Legacy extractor fields are not accepted.
- The top-level `MapGen.Config` (not the scene) carries `seed`. Scenes inherit RNGs spawned from the root, so the same
  seed reproduces terrain and placement exactly.
- Hub defaults place extractors in the corners, but variants commonly override bundles and cross spacing via
  `BaseHubVariant`.

#### Example: site-level builder

```python
from cogames.cogs_vs_clips.terrain import MachinaArenaConfig
from cogames.core import CoGameSite
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import BaseHubConfig

MACHINA_PROCEDURAL_200 = CoGameSite(
    name="machina_procedural_200",
    description="Large procedural arena",
    map_builder=MapGen.Config(
        width=200,
        height=200,
        seed=12345,
        instance=MachinaArenaConfig(
            spawn_count=4,
            base_biome="caves",
            hub=BaseHubConfig(
                corner_bundle="extractors",
                cross_bundle="extractors",
                cross_distance=7,
            ),
            building_names=[
                "chest",
                "junction",
                "carbon_extractor",
                "oxygen_extractor",
                "germanium_extractor",
                "silicon_extractor",
            ],
            building_weights={
                "chest": 0.2,
                "junction": 0.6,
                "carbon_extractor": 0.3,
                "oxygen_extractor": 0.3,
                "germanium_extractor": 0.3,
                "silicon_extractor": 0.3,
            },
            building_coverage=0.01,
            distribution={"type": "bimodal", "cluster_std": 0.15},
            building_distributions={
                "chest": {"type": "exponential", "decay_rate": 5.0, "origin_x": 0.0, "origin_y": 0.0},
                "junction": {"type": "poisson"},
            },
        ),
    ),
    min_cogs=1,
    max_cogs=20,
)
```

---

### Placement Distributions (`building_distributions.py`)

`UniformExtractorScene` (configured via `UniformExtractorParams`) handles actual placement of buildings. It works in two
modes:

1. **Coverage-driven**: If `target_coverage` is set, it samples enough center points to hit the requested coverage using
   the supplied distributions.
2. **Grid-driven**: Without coverage, it places objects on a jittered grid defined by `rows`, `cols`, and `padding`.

Every random sample uses `self.rng`, which comes from the parent scene. Because `SceneConfig.seed` defaults to the
parent's RNG when unset, the top-level `MapGen.Config(seed=...)` ensures deterministic results.

`DistributionConfig` supports:

- `type`: `"uniform"`, `"normal"`, `"exponential"`, `"poisson"`, `"bimodal"`
- Additional parameters per distribution (means, standard deviations, decay rates, cluster centers, etc.)

Per-building overrides live in `building_distributions` and accept the same schema. Omitted buildings fall back to the
global `distribution`.

---

### Missions and Variants

#### Sites (`sites.py`)

Sites describe reusable environments. They point to either a static map (`get_map("name.map")`) or a procedural builder.
Examples:

- `TRAINING_FACILITY`: hub-only builder, 13x13
- `HELLO_WORLD`: 100x100 procedural arena
- `MACHINA_1`: 88x88 procedural arena
- `COGSGUARD_MACHINA_1`: 50x50 CogsGuard variant
- `COGSGUARD_ARENA`: 50x50 compact training map

Each site defines `min_cogs`/`max_cogs`. CLI calls (`--cogs`) override the default during mission instantiation.

#### Mission lifecycle (`mission.py`)

1. `mission.with_variants(variants_list)` (optional) clones the mission and attaches variants to it
   - `variant.modify_mission()` is applied immediately
   - `variant.modify_env()` is applied when `make_env` is called
2. `mission.make_env()` finalizes the `MettaGridConfig` and applies variants to the environment

#### Variants

Variants inherit from `CoGameMissionVariant` and override either `modify_mission(self, mission)`,
`modify_env(self, mission, env)`, or both.

Common patterns:

- Modify terrain config via typed variant helpers:

  ```python
  from cogames.cogs_vs_clips.terrain import MachinaArenaVariant, MachinaArenaConfig

  class CityVariant(MachinaArenaVariant):
      name: str = "city"
      description: str = "Ancient city ruins provide structured pathways."

      def modify_node(self, cfg: MachinaArenaConfig) -> None:
          cfg.biome_weights = {"city": 1.0, "caves": 0.0, "desert": 0.0, "forest": 0.0}
          cfg.base_biome = "city"
          cfg.density_scale = 1.0
          cfg.biome_count = 1
          cfg.max_biome_zone_fraction = 0.95
  ```

- Adjust hub bundles via `BaseHubVariant`:

  ```python
  from cogames.cogs_vs_clips.terrain import BaseHubVariant
  from mettagrid.mapgen.scenes.base_hub import BaseHubConfig

  class ExtractorCrossVariant(BaseHubVariant):
      name: str = "extractor_cross"
      description: str = "Extractors on cross arms."

      def modify_node(self, cfg: BaseHubConfig) -> None:
          cfg.cross_bundle = "extractors"
          cfg.cross_distance = 7
  ```

- Adjust env properties directly:

  ```python
  from cogames.core import CoGameMissionVariant
  from cogames.cogs_vs_clips.mission import CvCMission
  from mettagrid.config.mettagrid_config import MettaGridConfig

  class MyVariant(CoGameMissionVariant):
      name: str = "my"

      def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
          # Modify env config before simulation starts
          ...
  ```

CLI variants are composed in order, so `cogames play -m cogsguard_machina_1.basic -v city -v extractor_cross` applies
`city`, then `extractor_cross`.

---

### Seeds and Reproducibility

- `MapGen.Config.seed` (`env.game.map_builder.seed`) controls **map layout**.
- If the mission sets a MapGen seed, all commands use it unless you pass `--map-seed`.
- `--map-seed` overrides the MapGen seed for procedural maps.
- `--seed` sets the simulator/policy RNG for the run (sim dynamics, policy sampling, assignment shuffles).
- When a MapGen seed is set (`--map-seed` or mission seed), the map layout follows a deterministic per-episode seed
  sequence. If `maps_cache_size` is set, the sequence repeats every `maps_cache_size` episodes; otherwise it increases
  monotonically.
- When no MapGen seed is set, the map layout is random. With caching enabled, you see up to `maps_cache_size` distinct
  layouts; with caching disabled (`maps_cache_size=None`), you get a fresh layout each episode.
- For fully reproducible play/eval runs, set **both** `--seed` and `--map-seed`.

Example programmatic override using the `MapSeedVariant` helper:

```python
from cogames.cogs_vs_clips.terrain import MapSeedVariant

base_mission = my_mission
seeded_mission = base_mission.with_variants([MapSeedVariant(seed=1234)])
env_cfg = seeded_mission.make_env()
# env_cfg.game.map_builder is a MapGen.Config with seed=1234; calling builder.build()
# will now deterministically reproduce the same grid.
```

---

### Building New Missions

1. **Define or reuse a `CoGameSite`** with the desired map builder.
2. **Define the desired behavior in a variant** (inheriting from `CoGameMissionVariant` or a typed helper like
   `MachinaArenaVariant`).
3. **Create a `CvCMission` object**:

```python
from cogames.cogs_vs_clips.mission import CvCMission

mission = CvCMission(
    name="my_mission",
    description="My mission",
    site=my_site,
    variants=[Variant1(), Variant2()],
)
```

4. **Add the mission object to `MISSIONS`** so the CLI picks it up.

---

### CLI Reference

- List missions/variants:

  ```bash
  cogames missions
  ```

- Play with variants and overrides:

  ```bash
  cogames play --mission cogsguard_machina_1.basic \
               --variant city \
               --cogs 8 \
               --policy random
  ```

- Reproduce a procedural layout:
  ```bash
  cogames play -m cogsguard_machina_1.basic --variant city --map-seed 24601 --seed 24601
  ```
  (Use `--map-seed` for layout determinism; include `--seed` to reproduce simulator/policy RNG.)

---

### Recommended Workflow

1. Start with an existing Site + Mission pair (e.g., `COGSGUARD_ARENA`, `COGSGUARD_MACHINA_1`).
2. Copy the mission, adjust its properties, and add it to `MISSIONS`.
3. Define variants for reusable tweaks.
4. Use CLI commands (`missions`, `play`) to iterate quickly.
