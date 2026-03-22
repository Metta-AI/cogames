#!/usr/bin/env python3
"""End-to-end integration test: run a miner mission with a mocked local LLM.

This verifies the full pipeline — environment, policy, LLM planner,
skill dispatch — works correctly.  It uses a deterministic stub instead of
a real model, so it runs without GPU or a downloaded checkpoint.

Usage:
    python scripts/test_local_mission_mock.py

For a real run with the downloaded Nemotron model, set:
    export LOCAL_LLM_MODEL_PATH=/path/to/nemotron-nano-9b-v2
    python scripts/run_local_mission.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ── logging ────────────────────────────────────────────────────────────────────
LOG_FILE = "mock_local_llm_run.log"
_fmt = "%(asctime)s %(levelname)-7s %(name)s │ %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_fmt,
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
for _name in ("cogames.policy.llm_miner", "cogames.policy.machina_llm_roles", "cogames.policy.local_llm"):
    logging.getLogger(_name).setLevel(logging.DEBUG)

log = logging.getLogger("test_local_mission_mock")

# ── deterministic stub LLM ─────────────────────────────────────────────────────

_SKILL_CYCLE = [
    "explore",
    "gear_up",
    "mine_until_full",
    "deposit_to_hub",
    "mine_until_full",
    "deposit_to_hub",
]
_call_counter: dict[str, int] = {"n": 0}


def _mock_responder(prompt: str) -> str:
    """Cycle through skills in a fixed order to exercise all code paths."""
    idx = _call_counter["n"] % len(_SKILL_CYCLE)
    skill = _SKILL_CYCLE[idx]
    _call_counter["n"] += 1
    response = json.dumps({"skill": skill, "reason": f"mock call #{_call_counter['n']}"})
    log.info("[MOCK LLM] call=%d  skill=%s  (simulating Nemotron Nano 9B V2)", _call_counter["n"], skill)
    return response


# ── cogames imports ────────────────────────────────────────────────────────────

try:
    from cogames.cogs_vs_clips.missions import get_core_missions
    from cogames.play import play
    from mettagrid.policy.policy import PolicySpec
    from rich.console import Console
except ImportError as exc:
    log.error("Import failed: %s", exc)
    log.error("Activate the cogames venv before running this script.")
    sys.exit(1)

# ── find miner_tutorial mission ────────────────────────────────────────────────

_all = get_core_missions()
_mission = next((m for m in _all if m.name == "miner_tutorial"), None)
if _mission is None:
    log.error("miner_tutorial not found; available: %s", [m.name for m in _all])
    sys.exit(1)

MAX_STEPS = 150
env_cfg = _mission.make_env()
# Override max_steps for a quick smoke test
env_cfg = env_cfg.model_copy(update={"game": env_cfg.game.model_copy(update={"max_steps": MAX_STEPS})})

log.info("Mission : %s  agents=%d  max_steps=%d", _mission.description, env_cfg.game.num_agents, MAX_STEPS)
log.info("Policy  : llm_miner  (mock LLM responder — simulating Nemotron Nano 9B V2)")
log.info("=" * 70)

# ── patch responder at class level ────────────────────────────────────────────
# We set init_kwargs so the LLMMinerPolicy picks up our mock responder
# via the llm_responder parameter, completely bypassing OpenRouter.

policy_spec = PolicySpec(
    class_path="llm_miner",
    init_kwargs={"llm_model": None, "llm_responder": _mock_responder},
)

log.info("Starting episode with mock LLM …")
console = Console(no_color=True)

try:
    play(
        console=console,
        env_cfg=env_cfg,
        policy_spec=policy_spec,
        game_name="miner_tutorial",
        seed=42,
        device="cpu",
        render_mode="none",
    )
    log.info("=" * 70)
    log.info("Episode finished successfully!")
    log.info("Total mock LLM calls: %d", _call_counter["n"])
    log.info("Full log saved to: %s", Path(LOG_FILE).resolve())
except Exception as exc:
    log.error("Episode failed: %s", exc, exc_info=True)
    sys.exit(1)
