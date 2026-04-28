"""Tests for cogames CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from cogames.main import app

runner = CliRunner()


def test_missions_list_command():
    """Test that 'cogames missions' lists only top-level missions."""
    result = runner.invoke(app, ["missions"])

    assert result.exit_code == 0, f"Command failed:\n{result.output}"

    # Check that the output contains expected content (CvC is the default game)
    output = result.output
    assert "arena" in output
    assert "arena" in output
    assert "Cogs" in output
    assert "Map Size" in output
    assert "machina_1.clips" in output
    assert "machina_1.desert" not in output


def test_missions_describe_command():
    """Test that 'cogames missions <mission_name>' describes a specific mission."""
    result = runner.invoke(app, ["missions", "-m", "arena"])

    assert result.exit_code == 0, f"Command failed:\n{result.output}"

    # Check that the output contains expected game details
    output = result.output
    assert "arena" in output
    assert "Mission Configuration:" in output
    assert "Number of agents:" in output
    assert "Available Actions:" in output


def test_missions_list_with_filter():
    """Test that a positional filter argument filters missions by name."""
    result = runner.invoke(app, ["missions", "arena"])

    assert result.exit_code == 0, f"Command failed:\n{result.output}"

    output = result.output
    assert "arena" in output


def test_missions_nonexistent_mission():
    """Test that describing a nonexistent game returns an error."""
    result = runner.invoke(app, ["missions", "-m", "nonexistent_mission"])

    assert result.exit_code == 0, "Command should succeed but show error message for nonexistent mission"
    combined_output = result.output.lower()
    assert "could not find" in combined_output or "not found" in combined_output, (
        f"Expected 'not found' message, got:\n{result.output}"
    )


def test_missions_help_command():
    """Test that 'cogames --help' shows help text."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, f"Command failed:\n{result.output}"

    # Check that help text contains expected commands
    output = result.output
    assert "bitworld" in output
    assert "missions" in output
    assert "play" in output
    assert "tutorial" in output


def test_docs_mission_command() -> None:
    """Test that `cogames docs mission` prints the packaged mission briefing."""
    package_root = Path(__file__).resolve().parents[1]
    result = runner.invoke(app, ["docs", "mission"])

    assert result.exit_code == 0, f"Command failed:\n{result.output}"
    mission_title = next(line for line in (package_root / "MISSION.md").read_text().splitlines() if line.strip())
    assert mission_title in result.output


def test_make_mission_command(tmp_path: Path):
    """Test that 'cogames make-mission' creates a new mission configuration."""
    mission_path = tmp_path / "mission.json"

    # Note: Don't set width/height or agents since arena uses an AsciiMapBuilder
    # with fixed dimensions and spawn points.
    result = runner.invoke(app, ["make-mission", "-m", "arena", "--output", str(mission_path)])
    assert result.exit_code == 0, f"make-mission failed:\n{result.output}"

    result = runner.invoke(app, ["missions", "-m", str(mission_path)])
    assert result.exit_code == 0, f"missions failed:\n{result.output}"

    assert mission_path.exists()
