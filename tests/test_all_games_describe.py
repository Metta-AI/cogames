"""Test that the missions command describes every registered mission."""

import pytest
from typer.testing import CliRunner

from cogames.cli.mission import get_all_missions
from cogames.main import app

runner = CliRunner()


@pytest.mark.parametrize("mission_name", get_all_missions())
@pytest.mark.timeout(60)
def test_mission_describe(mission_name):
    """Test that 'cogames missions -m <mission_name>' works for all games."""

    result = runner.invoke(
        app,
        ["missions", "-m", mission_name],
    )

    assert result.exit_code == 0, f"Failed to describe mission {mission_name}: {result.stderr}"
    assert "Mission Configuration:" in result.output
    assert "Available Actions:" in result.output
