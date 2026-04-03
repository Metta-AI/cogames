"""Tests for cogames CLI commands."""

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def test_missions_list_command():
    """Test that 'cogames missions' lists only top-level missions."""
    result = subprocess.run(
        ["cogames", "missions"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check that the output contains expected content (CvC is the default game)
    output = result.stdout
    assert "arena" in output
    assert "arena" in output
    assert "Cogs" in output
    assert "Map Size" in output
    assert "machina_1.clips" in output
    assert "machina_1.desert" not in output


def test_missions_describe_command():
    """Test that 'cogames missions <mission_name>' describes a specific mission."""
    result = subprocess.run(
        ["cogames", "missions", "-m", "arena"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check that the output contains expected game details
    output = result.stdout
    assert "arena" in output
    assert "Mission Configuration:" in output
    assert "Number of agents:" in output
    assert "Available Actions:" in output


def test_missions_list_with_filter():
    """Test that a positional filter argument filters missions by name."""
    result = subprocess.run(
        ["cogames", "missions", "arena"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    output = result.stdout
    assert "arena" in output


def test_missions_nonexistent_mission():
    """Test that describing a nonexistent game returns an error."""
    result = subprocess.run(
        ["cogames", "missions", "-m", "nonexistent_mission"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, "Command should succeed but show error message for nonexistent mission"
    combined_output = (result.stdout + result.stderr).lower()
    assert "could not find" in combined_output or "not found" in combined_output, (
        f"Expected 'not found' message, got:\n{result.stdout}\n{result.stderr}"
    )


def test_missions_help_command():
    """Test that 'cogames --help' shows help text."""
    result = subprocess.run(
        ["cogames", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check that help text contains expected commands
    output = result.stdout
    assert "missions" in output
    assert "play" in output
    assert "tutorial" in output


def test_docs_mission_command() -> None:
    """Test that `cogames docs mission` prints the packaged mission briefing."""
    package_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["cogames", "docs", "mission"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    mission_title = next(line for line in (package_root / "MISSION.md").read_text().splitlines() if line.strip())
    assert mission_title in result.stdout


def test_cogames_wheel_includes_docs_resources(tmp_path: Path) -> None:
    """The wheel must package the same docs content that `cogames docs` reads from source."""
    repo_root = Path(__file__).resolve().parents[3]
    package_root = repo_root / "packages" / "cogames"
    shutil.rmtree(package_root / "build", ignore_errors=True)
    shutil.rmtree(package_root / "src" / "cogames.egg-info", ignore_errors=True)
    result = subprocess.run(
        [
            "uv",
            "build",
            "--package",
            "cogames",
            "--wheel",
            "-o",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=repo_root,
    )

    assert result.returncode == 0, f"uv build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    wheel_path = next(tmp_path.glob("cogames-*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        packaged_docs = {
            "cogames/docs/MISSION.md": package_root / "MISSION.md",
            "cogames/docs/TECHNICAL_MANUAL.md": package_root / "TECHNICAL_MANUAL.md",
            "cogames/docs/SCRIPTED_AGENT.md": package_root / "src" / "cogames" / "docs" / "SCRIPTED_AGENT.md",
            "cogames/games/cogs_vs_clips/docs/cogs_vs_clips_mapgen.md": (
                package_root / "src" / "cogames" / "games" / "cogs_vs_clips" / "docs" / "cogs_vs_clips_mapgen.md"
            ),
            "cogames/games/cogs_vs_clips/evals/README.md": (
                package_root / "src" / "cogames" / "games" / "cogs_vs_clips" / "evals" / "README.md"
            ),
        }

        for wheel_entry, source_path in packaged_docs.items():
            assert wheel.read(wheel_entry).decode() == source_path.read_text(), (
                f"{wheel_entry} did not match {source_path}"
            )


def test_make_mission_command():
    """Test that 'cogames make-mission' creates a new mission configuration."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        # Run make-game and write to temp file
        # Note: Don't set width/height or agents since training_facility uses an AsciiMapBuilder
        # with fixed dimensions and spawn points
        result = subprocess.run(
            [
                "cogames",
                "make-mission",
                "-m",
                "arena",
                "--output",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"make-mission failed: {result.stderr}"

        # Run games command with the generated file
        result = subprocess.run(
            ["cogames", "missions", "-m", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"missions failed: {result.stderr}"

        assert tmp_path.exists()
    finally:
        tmp_path.unlink(missing_ok=True)
