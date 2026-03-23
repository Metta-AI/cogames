#!/usr/bin/env python3
"""Test SharedMap + HP monitoring with mock LLM (no GPU needed)."""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

LOG_FILE = "shared_map_test.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, mode="w")],
)
for _name in ("cogames.policy.llm_miner", "cogames.policy.machina_llm_roles", "cogames.policy.local_llm", "cogames.policy.aligner_agent", "cogames.policy.scout_agent"):
    logging.getLogger(_name).setLevel(logging.DEBUG)
log = logging.getLogger("test_shared_map")

# ── mock LLM ──
_ALIGNER_SKILLS = ["gear_up", "get_heart", "align_neutral", "explore", "unstuck"]
_call_counter = {"n": 0}

def _mock_responder(prompt: str) -> str:
    _call_counter["n"] += 1
    # Simple heuristic: pick skill based on prompt state
    if "has_aligner: False" in prompt:
        skill = "gear_up"
    elif "has_heart: False" in prompt:
        skill = "get_heart"
    elif "known_alignable_junctions: 0" in prompt:
        skill = "explore"
    else:
        skill = "align_neutral"
    return json.dumps({"skill": skill, "reason": f"mock#{_call_counter['n']}"})

# ── cogames imports ──
from cogames.cli.mission import get_mission
from cogames.play import play
from mettagrid.policy.policy import PolicySpec
from rich.console import Console

# ── config ──
import os
MAX_STEPS = int(os.environ.get("TEST_STEPS", "200"))
NUM_COGS = int(os.environ.get("TEST_COGS", "4"))
NUM_ALIGNERS = int(os.environ.get("TEST_ALIGNERS", "3"))
NUM_SCOUTS = int(os.environ.get("TEST_SCOUTS", "1"))

name, env_cfg, _ = get_mission("cogsguard_machina_1")
env_cfg = env_cfg.model_copy(update={"game": env_cfg.game.model_copy(update={"num_agents": NUM_COGS, "max_steps": MAX_STEPS})})

log.info("Mission: %s  cogs=%d  aligners=%d  scouts=%d  steps=%d", name, NUM_COGS, NUM_ALIGNERS, NUM_SCOUTS, MAX_STEPS)

policy_spec = PolicySpec(
    class_path="machina_llm_roles",
    init_kwargs={
        "llm_model": None,
        "llm_responder": _mock_responder,
        "num_aligners": NUM_ALIGNERS,
        "num_scouts": NUM_SCOUTS,
    },
)

console = Console(no_color=True)
try:
    play(
        console=console,
        env_cfg=env_cfg,
        policy_spec=policy_spec,
        game_name="cogsguard_machina_1.basic",
        seed=42,
        device="cpu",
        render_mode="none",
    )
    log.info("=" * 70)
    log.info("Episode finished! LLM calls: %d", _call_counter["n"])
    log.info("Log: %s", Path(LOG_FILE).resolve())
except Exception as exc:
    log.error("Failed: %s", exc, exc_info=True)
    sys.exit(1)
