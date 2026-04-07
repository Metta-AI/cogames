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
    assert game.missions[0].name == "basic"
    assert game.variant_registry.has("rush_hour")
    assert full_variant is not None
    assert full_variant.name == "full"


def test_importing_cogames_registers_overcogged_without_requiring_metta(tmp_path: Path) -> None:
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
                "import cogames; "
                "from cogames.game import get_game; "
                "game = get_game('overcogged'); "
                "assert game.name == 'overcogged'; "
                "assert game.missions[0].name == 'basic'; "
                "assert game.variant_registry.get('full').name == 'full'"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
