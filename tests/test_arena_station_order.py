from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def _arena_hub_stations_for_hash_seed(hash_seed: str) -> str:
    code = textwrap.dedent(
        """
        # Registers CvCMission's default "machina_1" variant.
        import cogsguard.missions.machina_1  # noqa: F401
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
    # Under Bazel's sandbox, the test process's sys.path is constructed from
    # runfiles; the spawned subprocess only inherits PYTHONPATH from env.
    # Forward it so `import cogsguard.*` resolves the same way the parent
    # process resolves it. Outside Bazel this is a no-op (sys.path is the
    # same in parent and child since they share the same site-packages).
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def test_arena_hub_station_order_is_hash_seed_stable() -> None:
    assert _arena_hub_stations_for_hash_seed("1") == _arena_hub_stations_for_hash_seed("2")
