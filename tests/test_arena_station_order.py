from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def _arena_hub_stations_for_hash_seed(hash_seed: str) -> str:
    code = textwrap.dedent(
        """
        from cogsguard.missions.arena import make_basic_mission
        from cogsguard.missions.terrain import find_machina_arena

        env = make_basic_mission(max_steps=10).make_env()
        arena = find_machina_arena(env.game.map_builder)
        assert arena is not None
        print(",".join(arena.hub.stations))
        """
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def test_arena_hub_station_order_is_hash_seed_stable() -> None:
    assert _arena_hub_stations_for_hash_seed("1") == _arena_hub_stations_for_hash_seed("2")
