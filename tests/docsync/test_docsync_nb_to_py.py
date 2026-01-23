"""Tests for `cogames docsync nb-to-py` subcommand."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import nbformat
from helpers import create_notebook, create_py_content
from nbformat.v4 import new_code_cell, new_markdown_cell, new_output
from typer.testing import CliRunner

from cogames.cli.docsync._nb_py_sync import convert_nb_to_py
from cogames.cli.docsync._utils import lint_py_file, run_notebook
from cogames.cli.docsync.docsync import app

runner = CliRunner()


# region Function tests


def test_nb_to_py_basic():
    """Test basic notebook to .py conversion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(
            cells=[
                new_markdown_cell("# Hello World"),
                new_code_cell("print('hello')"),
            ]
        )

        nb_path = tmp_path / "test.ipynb"
        nbformat.write(nb, nb_path)

        py_path = nb_path.with_suffix(".py")
        convert_nb_to_py(nb_path=nb_path, py_path=py_path)

        assert py_path.exists()
        assert py_path.suffix == ".py"

        content = py_path.read_text()
        assert "# %% [markdown]" in content
        assert "# Hello World" in content
        assert "# %%" in content
        assert 'print("hello")' in content


def test_nb_to_py_percent_format():
    """Test that output is valid percent format with YAML header."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(cells=[new_code_cell("x = 1")])
        nb_path = tmp_path / "test.ipynb"
        nbformat.write(nb, nb_path)

        py_path = nb_path.with_suffix(".py")
        convert_nb_to_py(nb_path=nb_path, py_path=py_path)
        content = py_path.read_text()

        # Should have YAML header with jupytext metadata
        assert content.startswith("# ---")
        assert "jupytext:" in content
        assert "format_name: percent" in content


def test_nb_to_py_directives_preserved():
    """Test that # <<directive>> directives are preserved in .py output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(
            cells=[
                new_code_cell("# <<hide>>\nsecret_code()"),
                new_code_cell("# <<hide-input>>\ncompute()"),
                new_code_cell("# <<hide-output>>\nnoisy_code()"),
                new_code_cell("# <<collapse-input>>\nlong_code()"),
                new_code_cell("# <<collapse-output>>\nverbose_code()"),
            ]
        )

        nb_path = tmp_path / "test.ipynb"
        nbformat.write(nb, nb_path)

        py_path = nb_path.with_suffix(".py")
        convert_nb_to_py(nb_path=nb_path, py_path=py_path)
        content = py_path.read_text()

        assert "# <<hide>>" in content
        assert "# <<hide-input>>" in content
        assert "# <<hide-output>>" in content
        assert "# <<collapse-input>>" in content
        assert "# <<collapse-output>>" in content


def test_nb_to_py_outputs_stripped():
    """Test that outputs are not included in .py file (code only)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        cell = new_code_cell("print('hello')")
        cell.outputs = [new_output(output_type="stream", name="stdout", text="hello\n")]

        nb = create_notebook(cells=[cell])
        nb_path = tmp_path / "test.ipynb"
        nbformat.write(nb, nb_path)

        py_path = nb_path.with_suffix(".py")
        convert_nb_to_py(nb_path=nb_path, py_path=py_path)
        content = py_path.read_text()

        # Code should be present (ruff reformats to double quotes)
        assert 'print("hello")' in content
        # Output text should not be in the .py file (it's code only)
        # Note: "hello" appears in the code, so check for stream output markers instead
        assert "stdout" not in content.lower()


def test_nb_to_py_markdown_cells():
    """Test that markdown cells are converted to # %% [markdown] blocks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(
            cells=[
                new_markdown_cell("# Title\n\nSome text here."),
                new_code_cell("code()"),
                new_markdown_cell("## Section\n\nMore text."),
            ]
        )

        nb_path = tmp_path / "test.ipynb"
        nbformat.write(nb, nb_path)

        py_path = nb_path.with_suffix(".py")
        convert_nb_to_py(nb_path=nb_path, py_path=py_path)
        content = py_path.read_text()

        # Count markdown cell markers
        assert content.count("# %% [markdown]") == 2
        assert "# Title" in content
        assert "# Some text here." in content
        assert "# ## Section" in content


def test_nb_to_py_multiple_code_cells():
    """Test that multiple code cells get separate %% markers."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(
            cells=[
                new_code_cell("cell1()"),
                new_code_cell("cell2()"),
                new_code_cell("cell3()"),
            ]
        )

        nb_path = tmp_path / "test.ipynb"
        nbformat.write(nb, nb_path)

        py_path = nb_path.with_suffix(".py")
        convert_nb_to_py(nb_path=nb_path, py_path=py_path)
        content = py_path.read_text()

        # Each code cell should have its own %% marker
        # Count lines that are exactly "# %%" (code cell markers, not markdown)
        code_cell_markers = [line for line in content.split("\n") if line.strip() == "# %%"]
        assert len(code_cell_markers) == 3

        assert "cell1()" in content
        assert "cell2()" in content
        assert "cell3()" in content


def test_run_notebook_sets_cwd_to_notebook_parent():
    """Test that run_notebook executes jupyter with cwd set to notebook's parent directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        nb = create_notebook(cells=[new_code_cell("print('hello')")])
        nb_path = subdir / "test.ipynb"
        nbformat.write(nb, nb_path)

        with patch("cogames.cli.docsync._utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            run_notebook(nb_path=nb_path)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["cwd"] == subdir


def test_format_py_runs_ruff():
    """Test that format_py runs ruff format on .py files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create .py file with bad formatting (no spaces around =)
        py_path = tmp_path / "test.py"
        py_path.write_text("x=1\ny=2\n")

        lint_py_file(py_path)

        # Ruff should have fixed the formatting (spaces around =)
        content = py_path.read_text()
        assert "x = 1" in content
        assert "y = 2" in content


# endregion
# region CLI tests


def test_cli_nb_to_py_basic():
    """Test nb-to-py command creates .py file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(cells=[new_code_cell("print('hello')")])
        nb_path = tmp_path / "test.ipynb"
        nbformat.write(nb, nb_path)

        result = runner.invoke(app, ["nb-to-py", str(nb_path)])

        assert result.exit_code == 0
        assert "Created" in result.output

        py_path = tmp_path / "test.py"
        assert py_path.exists()


def test_cli_nb_to_py_rejects_non_ipynb():
    """Test nb-to-py command rejects non-.ipynb files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        py_path = tmp_path / "test.py"
        py_path.write_text("print('hello')")

        result = runner.invoke(app, ["nb-to-py", str(py_path)])

        assert result.exit_code == 1
        assert "expected .ipynb" in result.output


def test_cli_nb_to_py_rejects_missing_file():
    """Test nb-to-py command rejects missing files."""
    result = runner.invoke(app, ["nb-to-py", "/nonexistent/test.ipynb"])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_nb_to_py_overwrites_existing():
    """Test nb-to-py overwrites existing .py file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create notebook
        nb = create_notebook(cells=[new_code_cell("new_code()")])
        nb_path = tmp_path / "test.ipynb"
        nbformat.write(nb, nb_path)

        # Create existing .py file with different content
        py_path = tmp_path / "test.py"
        py_path.write_text(create_py_content())

        # Run nb-to-py
        result = runner.invoke(app, ["nb-to-py", str(nb_path)])

        assert result.exit_code == 0
        # File should be overwritten with new content
        assert "new_code" in py_path.read_text()


# endregion
