"""Tests for cogames CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from cogames.main import app

runner = CliRunner()


def test_help_command():
    """Test that 'cogames --help' shows help text."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, f"Command failed:\n{result.output}"

    output = result.output
    assert "bitworld" in output
    assert "play" in output
    assert "tutorial" in output


def test_play_help_does_not_reference_removed_missions_command() -> None:
    result = runner.invoke(app, ["play", "--help"])

    assert result.exit_code == 0, f"Command failed:\n{result.output}"
    output = " ".join(result.output.split())
    assert "cogames missions" not in output
    assert "path to a mission" in output
    assert "mission config" in output
    assert "file" in output
    assert "starter" in output
    assert "baseline" not in output


def test_docs_mission_command() -> None:
    """Test that `cogames docs mission` prints the packaged mission briefing."""
    package_root = Path(__file__).resolve().parents[1]
    result = runner.invoke(app, ["docs", "mission"])

    assert result.exit_code == 0, f"Command failed:\n{result.output}"
    mission_title = next(line for line in (package_root / "MISSION.md").read_text().splitlines() if line.strip())
    assert mission_title in result.output


def test_docs_readme_prints_raw_markdown() -> None:
    result = runner.invoke(app, ["docs", "readme"])

    assert result.exit_code == 0, f"Command failed:\n{result.output}"
    assert "[cogames](https://pypi.org/project/cogames/)" in result.output
