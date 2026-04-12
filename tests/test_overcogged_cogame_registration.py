import os
import subprocess
import sys
from pathlib import Path

from cogames.game import get_game


def test_overcogged_cogame_registers_canonical_game_module() -> None:
    game = get_game("overcogged")
    full_variant = game.variant_registry.get("full")

    assert game.__class__.__name__ == "OvercookedCoGame"
    assert game.__class__.__module__ == "cogames.games.overcogged.game.game"
    assert game.missions
    assert [mission.name for mission in game.missions] == ["basic", "classic"]
    assert game.variant_registry.has("rush_hour")
    assert full_variant is not None
    assert full_variant.name == "full"


def test_importing_cogames_keeps_overcogged_lazy_until_requested(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            [
                str(repo_root / "packages/cogames/src"),
                str(repo_root / "packages/mettagrid/python/src"),
            ]
        )
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import cogames; "
                "from cogames.game import get_game; "
                "assert 'cogames.games.overcogged.game.game' not in sys.modules; "
                "assert get_game('cogs_vs_clips').name == 'cogs_vs_clips'; "
                "assert 'cogames.games.overcogged.game.game' not in sys.modules; "
                "game = get_game('overcogged'); "
                "assert game.name == 'overcogged'; "
                "assert [mission.name for mission in game.missions] == ['basic', 'classic']; "
                "assert game.variant_registry.get('full').name == 'full'; "
                "assert 'cogames.games.overcogged.game.game' in sys.modules"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
