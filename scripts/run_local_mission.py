#!/usr/bin/env python3
"""Run a CoGames mission with the local Nemotron Nano 9B V2 model.

This script sets up verbose logging so every LLM call (prompt + response +
latency) is visible in the terminal and written to a log file.

Usage:
    python scripts/run_local_mission.py

Prerequisites:
    1. Download the model:
           python scripts/download_nemotron.py
    2. Set the model path:
           export LOCAL_LLM_MODEL_PATH=~/.cache/cogames/models/nemotron-nano-9b-v2
    3. Install cogames (in the repo venv):
           pip install -e .

Environment variables:
    LOCAL_LLM_MODEL_PATH   Path to the downloaded model directory (required).
    COGAMES_MISSION        Mission to run (default: miner_tutorial.basic).
    COGAMES_MAX_STEPS      Max steps per episode (default: 300).
    COGAMES_LOG_FILE       Where to write the log file (default: local_llm_run.log).
    COGAMES_POLICY         Policy short name (default: llm_miner).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ── logging setup ──────────────────────────────────────────────────────────────
LOG_FILE = os.environ.get("COGAMES_LOG_FILE", "local_llm_run.log")

_fmt = "%(asctime)s %(levelname)-7s %(name)s │ %(message)s"
_datefmt = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_fmt,
    datefmt=_datefmt,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
# Show all LLM-related logs at DEBUG level so we see every prompt/response.
for _name in (
    "cogames.policy.llm_miner",
    "cogames.policy.machina_llm_roles",
    "cogames.policy.local_llm",
):
    logging.getLogger(_name).setLevel(logging.DEBUG)

log = logging.getLogger("run_local_mission")

# ── environment checks ─────────────────────────────────────────────────────────

MISSION_NAME = os.environ.get("COGAMES_MISSION", "miner_tutorial.basic")
MAX_STEPS = int(os.environ.get("COGAMES_MAX_STEPS", "300"))
POLICY_NAME = os.environ.get("COGAMES_POLICY", "llm_miner")
LOCAL_MODEL = os.environ.get("LOCAL_LLM_MODEL_PATH", "")

if not LOCAL_MODEL:
    log.error(
        "LOCAL_LLM_MODEL_PATH is not set.\n"
        "  1. Run:  python scripts/download_nemotron.py\n"
        "  2. Then: export LOCAL_LLM_MODEL_PATH=~/.cache/cogames/models/nemotron-nano-9b-v2"
    )
    sys.exit(1)

if not Path(LOCAL_MODEL).exists():
    log.error("Model directory does not exist: %s", LOCAL_MODEL)
    log.error("Run: python scripts/download_nemotron.py")
    sys.exit(1)

log.info("Model path : %s", LOCAL_MODEL)
log.info("Mission    : %s", MISSION_NAME)
log.info("Policy     : %s", POLICY_NAME)
log.info("Max steps  : %d", MAX_STEPS)
log.info("Log file   : %s", Path(LOG_FILE).resolve())

# ── cogames imports ────────────────────────────────────────────────────────────

try:
    from cogames.cogs_vs_clips.missions import get_core_missions
    from cogames.play import play
    from mettagrid.policy.policy import PolicySpec
    from rich.console import Console
except ImportError as exc:
    log.error("cogames import failed: %s", exc)
    log.error("Make sure you have activated the cogames virtual environment.")
    sys.exit(1)

# ── find mission ───────────────────────────────────────────────────────────────

_all = get_core_missions()
_missions = {m.name: m for m in _all}
_site, _, _name = MISSION_NAME.partition(".")
# Support both "miner_tutorial" and "miner_tutorial.basic" style names
_lookup = _name if _name else _site

if _lookup not in _missions:
    log.error("Unknown mission %r.  Available: %s", MISSION_NAME, ", ".join(sorted(_missions)))
    sys.exit(1)

mission = _missions[_lookup]
env_cfg = mission.make_env()
# Override max_steps if provided
if MAX_STEPS != mission.max_steps:
    env_cfg = env_cfg.model_copy(
        update={"game": env_cfg.game.model_copy(update={"max_steps": MAX_STEPS})}
    )

log.info("Mission loaded: %s agents=%d max_steps=%d", mission.description, env_cfg.game.num_agents, MAX_STEPS)

# ── build policy spec ──────────────────────────────────────────────────────────

# PolicySpec takes a class path.  The local model path is threaded through the
# class kwargs so the policy reads it at construction time.
policy_spec = PolicySpec(
    policy_cls_path=f"cogames.policy:{POLICY_NAME}",
    policy_kwargs={},  # LOCAL_LLM_MODEL_PATH is picked up from env automatically
)

log.info("Policy spec: %s", policy_spec)
log.info("=" * 70)
log.info("Starting episode …")

# ── run ────────────────────────────────────────────────────────────────────────

console = Console()
play(
    console=console,
    env_cfg=env_cfg,
    policy_spec=policy_spec,
    game_name=MISSION_NAME,
    seed=42,
    device="cpu",
    render_mode="none",
)

log.info("=" * 70)
log.info("Episode finished.  Full log saved to: %s", Path(LOG_FILE).resolve())
