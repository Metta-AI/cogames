"""Tests for `cogames docsync check` subcommand."""

import tempfile
from pathlib import Path

import nbformat
from helpers import create_notebook
from nbformat.v4 import new_code_cell
from typer.testing import CliRunner

from cogames.cli.docsync._nb_md_sync import convert_nb_to_md
from cogames.cli.docsync._nb_py_sync import convert_nb_to_py
from cogames.cli.docsync.docsync import app

runner = CliRunner()


def _create_synced_notebook_set(tmp_path: Path) -> None:
    """Create a README.ipynb/.py/.md set that is in sync."""
    # Create .ipynb
    nb = create_notebook(cells=[new_code_cell("print('hello')")])
    nb_path = tmp_path / "README.ipynb"
    nbformat.write(nb, nb_path)

    # Create .py from .ipynb (so they match)
    py_path = tmp_path / "README.py"
    convert_nb_to_py(nb_path=nb_path, py_path=py_path)

    # Create .md from .ipynb (so they match)
    convert_nb_to_md(nb_path=nb_path, should_rerun=False, cogames_root=tmp_path)


def test_cli_check_passes_when_in_sync():
    """Test check command passes when all files exist and are in sync."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _create_synced_notebook_set(tmp_path)

        result = runner.invoke(app, ["check", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 0
        assert "out of sync" not in result.output.lower()


def test_cli_check_fails_missing_py():
    """Test check command fails when .py file is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _create_synced_notebook_set(tmp_path)

        # Remove .py file
        (tmp_path / "README.py").unlink()

        result = runner.invoke(app, ["check", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 1
        assert "README.py" in result.output
        assert "missing" in result.output.lower()


def test_cli_check_fails_missing_md():
    """Test check command fails when .md file is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _create_synced_notebook_set(tmp_path)

        # Remove .md file
        (tmp_path / "README.md").unlink()

        result = runner.invoke(app, ["check", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 1
        assert "README.md" in result.output
        assert "missing" in result.output.lower()


def test_cli_check_fails_py_out_of_sync():
    """Test check command fails when .py doesn't match .ipynb."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _create_synced_notebook_set(tmp_path)

        # Modify .py to be out of sync
        py_path = tmp_path / "README.py"
        py_path.write_text(py_path.read_text() + "\n# extra line that makes it different\n")

        result = runner.invoke(app, ["check", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 1
        assert "README.py" in result.output
        assert "doesn't match" in result.output.lower() or "out of sync" in result.output.lower()


def test_cli_check_fails_md_out_of_sync():
    """Test check command fails when .md doesn't match .ipynb."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _create_synced_notebook_set(tmp_path)

        # Modify .md to be out of sync
        md_path = tmp_path / "README.md"
        md_path.write_text(md_path.read_text() + "\n\n<!-- extra content -->\n")

        result = runner.invoke(app, ["check", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 1
        assert "README.md" in result.output
        assert "doesn't match" in result.output.lower() or "out of sync" in result.output.lower()
