# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CoGames: Getting Started
#
# Welcome to CoGames! This notebook will get you from zero to training your own agent.
#
# CoGames is built on **MettaGrid**, a grid-world simulation engine. The flagship game mode
# is **Cogs vs Clips** — a cooperative territory-control game where teams of specialized agents
# (Cogs) capture and defend junctions against automated expansion by Clips.
#
# This notebook covers:
# 1. Installing cogames and playing the interactive tutorials
# 2. How to create an environment and what parameters you can configure
# 3. What the observation and action spaces look like

# %% [markdown]
# ## 1. Install and Play
#
# Install the cogames package:
#
# ```bash
# pip install cogames
# ```
#
# Then run the two interactive tutorials to learn the game mechanics before writing any code.
#
# ### `cogames tutorial play`
#
# This launches a guided walkthrough in MettaScope where you control a single agent.
# It teaches you the basics step by step:
# - Camera controls (scroll/pinch to zoom, drag to pan)
# - Movement and energy (WASD/arrows, battery drains, recharge near Hub)
# - Gear stations (pick a role: Aligner, Scrambler, Miner, Scout)
# - Resources and hearts (extract resources at Extractors, craft Hearts at the Assembler)
# - Junction control (Aligners capture neutral junctions, Scramblers neutralize enemy ones)
#
# ```bash
# cogames tutorial play
# ```
#
# ### `cogames tutorial cogsguard`
#
# This is a deeper tutorial focused on the full Cogs vs Clips game loop on a smaller 35x35 arena.
# It walks through multi-phase strategy:
# - Gear up and craft hearts
# - Expand from the Hub by capturing nearby junctions
# - Handle Clips pressure as they scramble your junctions
# - Use territory (friendly junctions restore HP and energy)
# - Coordinate roles and maintain the resource-to-heart pipeline
#
# ```bash
# cogames tutorial cogsguard
# ```
#
# Once you understand the game, come back here to learn how to build environments
# and train agents programmatically.

# %% [markdown]
# ## 2. Creating an Environment
#
# A Cogs vs Clips environment is built from a **mission**, which bundles a map, agent count,
# game rules, and **variants** (modifiers that change difficulty or mechanics).
#
# The mission produces a `MettaGridConfig`, which is passed to a `Simulator` and wrapped
# in a `MettaGridPufferEnv` (a Gymnasium-compatible env).

# %%
# %pip install mettagrid cogames pufferlib-core --quiet

# %%
from cogames.cogs_vs_clips.clip_difficulty import EASY
from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import make_cogsguard_machina1_site
from cogames.cogs_vs_clips.variants import NoVibesVariant
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulator

NUM_AGENTS = 4
MAX_STEPS = 1000

mission = CvCMission(
    name="my_mission",
    description="A simple Cogs vs Clips mission.",
    site=make_cogsguard_machina1_site(NUM_AGENTS),
    num_cogs=NUM_AGENTS,
    max_steps=MAX_STEPS,
    teams={"cogs": CogTeam(name="cogs", num_agents=NUM_AGENTS)},
    variants=[EASY, NoVibesVariant()],
)

env_cfg = mission.make_env()
policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)

# Create and reset the environment
simulator = Simulator()
env = MettaGridPufferEnv(simulator, env_cfg, seed=42)
obs, infos = env.reset()

print(f"Agents: {env.num_agents}")
print(f"Max steps: {MAX_STEPS}")
print(f"Observation shape: {obs.shape}")
print(f"Action space: {policy_env_info.action_space}")

# %% [markdown]
# ### Key Mission Parameters
#
# | Parameter | What it controls |
# |-----------|------------------|
# | `site` | Map layout and size. `make_cogsguard_machina1_site(n)` gives an 88x88 arena. |
# | `num_cogs` | Number of agents on your team. |
# | `max_steps` | Episode length in ticks. |
# | `teams` | Team definitions (name, agent count, initial wealth). |
# | `variants` | List of modifiers that change game rules (see below). |
#
# ### Variants
#
# Variants modify the environment after the base config is built. Stack them to create
# custom training scenarios.
#
# **Role reward shaping** (dense rewards for learning a specific role):
#
# | Variant | Effect |
# |---------|--------|
# | `MinerRewardsVariant()` | Rewards gear pickup, resource extraction, and deposits |
# | `AlignerRewardsVariant()` | Rewards heart management and junction alignment |
# | `ScramblerRewardsVariant()` | Rewards gear pickup and junction scrambling |
# | `ScoutRewardsVariant()` | Rewards gear pickup and exploration/cell visitation |
#
# **Difficulty** (Clips pressure control):
#
# | Variant | Effect |
# |---------|--------|
# | `EASY` | Disables Clips events entirely |
# | `MEDIUM` | A few early Clips events, ending at step 300 |
# | `HARD` | Full Clips event system active |
#
# **Gameplay modifiers**:
#
# | Variant | Effect |
# |---------|--------|
# | `NoVibesVariant()` | Removes vibe actions (simplifies action space) |
# | `EnergizedVariant()` | Max energy + full regen, agents never run dry |
# | `BraveheartVariant()` | Hub starts with 255 hearts |
# | `ThickSkinnedVariant()` | No passive HP drain, only in enemy territory |
# | `DarkSideVariant()` | Zero solar energy regeneration |
# | `SuperChargedVariant()` | +2 to all energy regen |
# | `NoClipsVariant()` | Disable Clips entirely (same as EASY) |
#
# ### Under the Hood: `MettaGridConfig`
#
# `mission.make_env()` returns a `MettaGridConfig` with a `GameConfig` inside. The game
# config controls everything the simulator needs:

# %%
game = env_cfg.game

print("=== Game Config ===")
print(f"  num_agents:     {game.num_agents}")
print(f"  max_steps:      {game.max_steps}")
print(f"  resources:      {game.resource_names}")
print(f"  obs window:     {game.obs.height}x{game.obs.width}")
print(f"  obs tokens:     {game.obs.num_tokens} x {game.obs.token_dim}")
print(f"  map builder:    {type(game.map_builder).__name__}")
print(f"  objects:        {list(game.objects.keys())}")

actions = game.actions.actions()
print(f"  actions:        {[a.name for a in actions]}")

# %% [markdown]
# ## 3. Observation Space
#
# Each agent receives a **sparse token observation** every step: a fixed-size buffer of
# 3-element tokens `[coordinate, feature_id, value]`, dtype `uint8`.
#
# ```
# obs shape per agent: (num_tokens, 3)   # default: (200, 3)
# ```
#
# - **`coordinate`**: Packed `(row, col)` in a 13x13 egocentric window centered on the agent.
#   `0xFE` = global (non-spatial) token. `0xFF` = padding (end of valid tokens).
# - **`feature_id`**: Which feature this token describes (see list below).
# - **`value`**: The feature's value (0-255). Inventory amounts > 255 use multiple tokens
#   with `:p1`, `:p2` suffixes — reconstruct as `base + p1*256 + p2*65536`.
#
# The observation is sparse: valid tokens come first, padding fills the rest.

# %%
print(f"Observation shape per agent: {policy_env_info.observation_shape}")
print(f"Egocentric window: {policy_env_info.obs_height}x{policy_env_info.obs_width}")
print(f"\nObservation features ({len(policy_env_info.obs_features)} total):")
print(f"{'ID':>4}  {'Name':<30}  {'Normalization':>13}")
print("-" * 52)
for feat in policy_env_info.obs_features:
    print(f"{feat.id:>4}  {feat.name:<30}  {feat.normalization:>13.1f}")

# %% [markdown]
# Key features to know:
# - **`tag`** — object type at a cell (wall, junction, extractor, agent, etc.)
# - **`vibe`** — role/resource identity of the object
# - **`agent:group`** — team membership
# - **`aoe_mask`** — territory: 0=neutral, 1=friendly, 2=enemy
# - **`inv:*`** — agent inventory (energy, heart, resources, gear)
# - **`episode_completion_pct`**, **`last_action`**, **`last_reward`** — global agent state
#
# The `tag` feature maps to object types via integer indices:

# %%
print("Tag mapping (tag value -> object type):")
for tag_id, tag_name in policy_env_info.tag_id_to_name.items():
    print(f"  {tag_id:>3}: {tag_name}")

# %% [markdown]
# ### Converting to a Spatial Grid for CNNs
#
# The raw sparse tokens aren't directly CNN-friendly. `GridObsWrapper` wraps the env
# so that `reset()` and `step()` return dense `(num_agents, C, H, W)` float32 grids
# instead of sparse tokens — directly usable with CNNs.
#
# It handles coordinate decoding (nibble-packed `(y, x)`), padding filtering (`0xFF`),
# global token placement (`0xFE` → grid center), and per-feature normalization.

# %%
from mettagrid.envs.grid_obs_wrapper import GridObsWrapper

grid_env = GridObsWrapper(env)
grid_obs, _ = grid_env.reset()

print(f"Sparse tokens: {obs.shape}  ->  Dense grid: {grid_obs.shape}")
print(
    f"  {grid_obs.shape[0]} agents, {grid_obs.shape[1]} feature channels, "
    f"{grid_obs.shape[2]}x{grid_obs.shape[3]} spatial"
)

# %% [markdown]
# ## 4. Action Space
#
# The action space is `Discrete(N)` — each step, every agent picks one integer action.
#
# Actions are split into **primary actions** (movement + noop) and **vibe actions**
# (changing the agent's vibe state). The `NoVibesVariant` used above removes vibe actions
# entirely, leaving only movement.

# %%
print(f"Primary action space: Discrete({len(policy_env_info.action_names)})")
print("\nPrimary actions:")
for i, name in enumerate(policy_env_info.action_names):
    print(f"  {i}: {name}")

if policy_env_info.vibe_action_names:
    print(f"\nVibe actions ({len(policy_env_info.vibe_action_names)}):")
    for i, name in enumerate(policy_env_info.vibe_action_names):
        print(f"  {i}: {name}")
else:
    print("\nVibe actions: none (removed by NoVibesVariant)")

if policy_env_info.move_energy_cost is not None:
    print(f"\nMove energy cost: {policy_env_info.move_energy_cost} per step")

# %% [markdown]
# ### How Actions Work
#
# **Movement** (`move_north`, `move_south`, `move_west`, `move_east`): Moves the agent
# one cell in the given direction. Costs energy. If the agent moves onto a building
# (extractor, gear station, junction, etc.), the building's handler fires — this is how
# agents interact with the world. Moving into a wall or off the map does nothing.
#
# **Noop**: Do nothing this step. Useful when waiting for cooldowns or conserving energy.
#
# **Vibe changes** (`change_vibe_*`): Switch the agent's vibe. Vibes determine how objects
# react to the agent. In Cogs vs Clips, vibes represent resource types and roles — for example,
# an agent must be vibing `heart` to deposit hearts, or vibing `aligner` to capture a junction.
# When using `NoVibesVariant`, these are removed and vibes are handled automatically.
#
# ### Interaction Model
#
# There are no explicit "use" or "pick up" actions. All interactions happen through
# **movement**: walk onto an object to trigger it. This keeps the action space small and
# forces spatial reasoning — agents must navigate to the right objects.
#
# ### Stepping the Environment

# %%
import numpy as np

# Take 10 random steps
for _step in range(10):
    actions = np.array([grid_env.single_action_space.sample() for _ in range(grid_env.num_agents)])
    obs, rewards, terminals, truncations, infos = grid_env.step(actions)

print(f"obs shape:          {obs.shape}")
print(f"rewards shape:      {rewards.shape}")
print(f"terminals shape:    {terminals.shape}")
print(f"truncations shape:  {truncations.shape}")
print(f"\nRewards after 10 random steps: {rewards}")

grid_env.close()

# %% [markdown]
# ## Next Steps
#
# Now that you know how the environment works, pick a role and train a specialist:
#
# - `TRAIN_MINER.ipynb` — Train a Miner (resource extraction and deposits)
# - `TRAIN_ALIGNER.ipynb` — Train an Aligner (heart management and junction capture)
# - `TRAIN_SCRAMBLER.ipynb` — Train a Scrambler (junction scrambling)
# - `TRAIN_SCOUT.ipynb` — Train a Scout (exploration and cell visitation)
