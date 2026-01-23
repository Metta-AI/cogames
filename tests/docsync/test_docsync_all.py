"""Tests for `cogames docsync all` subcommand."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import nbformat
from helpers import create_notebook, create_py_content
from nbformat.v4 import new_code_cell
from typer.testing import CliRunner

from cogames.cli.docsync.docsync import app

runner = CliRunner()


def test_cli_all_syncs_ipynb_to_py_when_user_selects_ipynb():
    """Test all command syncs .ipynb to .py when user selects .ipynb as source."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create .ipynb
        nb = create_notebook(cells=[new_code_cell("notebook_code()")])
        nb_path = tmp_path / "README.ipynb"
        nbformat.write(nb, nb_path)

        # Create different .py
        py_path = tmp_path / "README.py"
        py_path.write_text(create_py_content())

        # Make .ipynb newer
        time.sleep(0.1)
        os.utime(nb_path, (time.time() + 100, time.time() + 100))

        # Mock questionary to select .ipynb as source (nb→py)
        with patch("cogames.cli.docsync._nb_py_sync.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = ".ipynb → .py (newer)"
            result = runner.invoke(app, ["all", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 0
        # .py should be updated with notebook content
        assert "notebook_code" in py_path.read_text()
        # .md should be created
        assert (tmp_path / "README.md").exists()


def test_cli_all_syncs_py_to_ipynb_when_user_selects_py():
    """Test all command syncs .py to .ipynb when user selects .py as source."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create .py
        py_content = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
py_file_code()
"""
        py_path = tmp_path / "README.py"
        py_path.write_text(py_content)

        # Create different .ipynb
        nb = create_notebook(cells=[new_code_cell("old_notebook_code()")])
        nb_path = tmp_path / "README.ipynb"
        nbformat.write(nb, nb_path)

        # Make .py newer
        time.sleep(0.1)
        os.utime(py_path, (time.time() + 100, time.time() + 100))

        # Mock questionary to select .py as source (py→nb)
        with patch("cogames.cli.docsync._nb_py_sync.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = ".py → .ipynb (newer)"
            result = runner.invoke(app, ["all", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 0
        # .ipynb should be updated with .py content
        nb_result = nbformat.read(nb_path, as_version=4)
        assert "py_file_code" in nb_result.cells[0].source


def test_cli_all_creates_missing_py():
    """Test all command creates .py when only .ipynb exists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create only .ipynb
        nb = create_notebook(cells=[new_code_cell("only_notebook()")])
        nb_path = tmp_path / "README.ipynb"
        nbformat.write(nb, nb_path)

        result = runner.invoke(app, ["all", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 0
        # .py should be created
        py_path = tmp_path / "README.py"
        assert py_path.exists()
        assert "only_notebook" in py_path.read_text()


def test_cli_all_noops_when_in_sync():
    """Test all command does nothing when .py and .ipynb are already in sync."""
    import jupytext

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create .ipynb first
        nb = create_notebook(cells=[new_code_cell("synced_code()")])
        nb_path = tmp_path / "README.ipynb"
        nbformat.write(nb, nb_path)

        # Create .py by converting from .ipynb (ensures they're truly in sync)
        py_path = tmp_path / "README.py"
        nb_for_py = jupytext.read(nb_path)
        jupytext.write(nb_for_py, py_path, fmt="py:percent")

        # Run - should succeed without prompting because files are in sync
        result = runner.invoke(app, ["all", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 0
        assert "already in sync" in result.output


def test_cli_all_skips_when_user_selects_skip():
    """Test all command skips file when user selects Skip option."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create .py
        py_content = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
py_code()
"""
        py_path = tmp_path / "README.py"
        py_path.write_text(py_content)

        # Create different .ipynb
        nb = create_notebook(cells=[new_code_cell("different_nb_code()")])
        nb_path = tmp_path / "README.ipynb"
        nbformat.write(nb, nb_path)

        # Make .py newer
        time.sleep(0.1)
        os.utime(py_path, (time.time() + 100, time.time() + 100))

        # Mock questionary to select Skip
        with patch("cogames.cli.docsync._nb_py_sync.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "Skip"
            result = runner.invoke(app, ["all", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 0
        # Should have skipped
        assert "Skipping README" in result.output
        # .ipynb should NOT be updated (still has original content)
        nb_result = nbformat.read(nb_path, as_version=4)
        assert "different_nb_code" in nb_result.cells[0].source


def test_cli_all_skips_when_user_cancels():
    """Test all command skips file when user cancels (Ctrl+C returns None)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create .py
        py_content = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
py_code()
"""
        py_path = tmp_path / "README.py"
        py_path.write_text(py_content)

        # Create different .ipynb
        nb = create_notebook(cells=[new_code_cell("different_nb_code()")])
        nb_path = tmp_path / "README.ipynb"
        nbformat.write(nb, nb_path)

        # Make .py newer
        time.sleep(0.1)
        os.utime(py_path, (time.time() + 100, time.time() + 100))

        # Mock questionary to return None (simulates Ctrl+C)
        with patch("cogames.cli.docsync._nb_py_sync.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = None
            result = runner.invoke(app, ["all", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 0
        # Should have skipped
        assert "Skipping README" in result.output
        # .ipynb should NOT be updated (still has original content)
        nb_result = nbformat.read(nb_path, as_version=4)
        assert "different_nb_code" in nb_result.cells[0].source


def test_cli_all_newer_label_on_py():
    """Test that (newer) label appears on .py option when .py is newer."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create .py
        py_content = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
py_code()
"""
        py_path = tmp_path / "README.py"
        py_path.write_text(py_content)

        # Create different .ipynb
        nb = create_notebook(cells=[new_code_cell("nb_code()")])
        nb_path = tmp_path / "README.ipynb"
        nbformat.write(nb, nb_path)

        # Make .py newer
        time.sleep(0.1)
        os.utime(py_path, (time.time() + 100, time.time() + 100))

        # Mock questionary and capture the choices passed to it
        with patch("cogames.cli.docsync._nb_py_sync.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "Skip"
            runner.invoke(app, ["all", "--skip-rerun", "--cogames-root", str(tmp_path)])

            # Check that the choices include (newer) on the py option
            call_kwargs = mock_select.call_args[1]
            choices = call_kwargs["choices"]
            # choices order: [nb_label, py_label, "Skip"]
            # .py option should have (newer), .ipynb option should not
            assert ".ipynb → .py" in choices[0] and "(newer)" not in choices[0]
            assert ".py → .ipynb" in choices[1] and "(newer)" in choices[1]


def test_cli_all_newer_label_on_ipynb():
    """Test that (newer) label appears on .ipynb option when .ipynb is newer."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create .py (older)
        py_content = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
py_code()
"""
        py_path = tmp_path / "README.py"
        py_path.write_text(py_content)

        # Create different .ipynb (newer)
        nb = create_notebook(cells=[new_code_cell("nb_code()")])
        nb_path = tmp_path / "README.ipynb"
        nbformat.write(nb, nb_path)

        # Make .ipynb newer
        time.sleep(0.1)
        os.utime(nb_path, (time.time() + 100, time.time() + 100))

        # Mock questionary and capture the choices passed to it
        with patch("cogames.cli.docsync._nb_py_sync.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "Skip"
            runner.invoke(app, ["all", "--skip-rerun", "--cogames-root", str(tmp_path)])

            # Check that the choices include (newer) on the ipynb option
            call_kwargs = mock_select.call_args[1]
            choices = call_kwargs["choices"]
            # choices order: [nb_label, py_label, "Skip"]
            # .ipynb option should have (newer), .py option should not
            assert ".ipynb → .py" in choices[0] and "(newer)" in choices[0]
            assert ".py → .ipynb" in choices[1] and "(newer)" not in choices[1]


def test_cli_all_touches_source_after_sync():
    """Test all command touches source file after sync so it stays 'newer'."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create .py
        py_content = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
source_code()
"""
        py_path = tmp_path / "README.py"
        py_path.write_text(py_content)

        # Create different .ipynb
        nb = create_notebook(cells=[new_code_cell("target_code()")])
        nb_path = tmp_path / "README.ipynb"
        nbformat.write(nb, nb_path)

        # Make .py older initially
        time.sleep(0.1)
        old_time = time.time() - 1000
        os.utime(py_path, (old_time, old_time))

        # Make .ipynb newer (so it's the source)
        os.utime(nb_path, (time.time(), time.time()))
        nb_mtime_before = nb_path.stat().st_mtime

        # Mock questionary to select .ipynb as source (nb→py)
        with patch("cogames.cli.docsync._nb_py_sync.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = ".ipynb → .py (newer)"
            result = runner.invoke(app, ["all", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 0
        # .ipynb (source) should have been touched (mtime updated)
        nb_mtime_after = nb_path.stat().st_mtime
        assert nb_mtime_after >= nb_mtime_before


def test_cli_all_user_can_override_mtime_suggestion():
    """Test that user can select the older file as source, overriding mtime suggestion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create .py (newer)
        py_content = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
newer_py_code()
"""
        py_path = tmp_path / "README.py"
        py_path.write_text(py_content)

        # Create different .ipynb (older)
        nb = create_notebook(cells=[new_code_cell("older_nb_code()")])
        nb_path = tmp_path / "README.ipynb"
        nbformat.write(nb, nb_path)

        # Make .py newer
        time.sleep(0.1)
        os.utime(py_path, (time.time() + 100, time.time() + 100))

        # Mock questionary to select .ipynb (the OLDER file) as source
        # This tests that user can override the mtime suggestion
        with patch("cogames.cli.docsync._nb_py_sync.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = ".ipynb → .py"  # Note: no (newer)
            result = runner.invoke(app, ["all", "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 0
        # .py should be updated with the OLDER notebook content (user's choice)
        assert "older_nb_code" in py_path.read_text()
        assert "newer_py_code" not in py_path.read_text()
