"""Tests for `cogames docsync py-to-nb` subcommand."""

import tempfile
from pathlib import Path

import nbformat
from helpers import create_notebook, create_py_content
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from typer.testing import CliRunner

from cogames.cli.docsync._nb_py_sync import convert_nb_to_py, convert_py_to_nb
from cogames.cli.docsync.docsync import app

runner = CliRunner()


# region Function tests


def test_py_to_nb_basic():
    """Test basic .py to notebook conversion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        py_content = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Hello World

# %%
print('hello')
"""
        py_path = tmp_path / "test.py"
        py_path.write_text(py_content)

        nb_path = py_path.with_suffix(".ipynb")
        convert_py_to_nb(py_path=py_path, nb_path=nb_path, should_rerun=False)

        assert nb_path.exists()
        assert nb_path.suffix == ".ipynb"

        nb = nbformat.read(nb_path, as_version=4)
        assert len(nb.cells) == 2
        assert nb.cells[0].cell_type == "markdown"
        assert "Hello World" in nb.cells[0].source
        assert nb.cells[1].cell_type == "code"
        assert "print('hello')" in nb.cells[1].source


def test_py_to_nb_directives_preserved():
    """Test that # <<directive>> directives are preserved in notebook cells."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        py_content = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %%
# <<hide>>
secret_code()

# %%
# <<hide-input>>
compute()
"""
        py_path = tmp_path / "test.py"
        py_path.write_text(py_content)

        nb_path = py_path.with_suffix(".ipynb")
        convert_py_to_nb(py_path=py_path, nb_path=nb_path, should_rerun=False)
        nb = nbformat.read(nb_path, as_version=4)

        assert "# <<hide>>" in nb.cells[0].source
        assert "# <<hide-input>>" in nb.cells[1].source


def test_py_to_nb_multiple_cells():
    """Test conversion with multiple code and markdown cells."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        py_content = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Title

# %%
cell1()

# %% [markdown]
# ## Section

# %%
cell2()

# %%
cell3()
"""
        py_path = tmp_path / "test.py"
        py_path.write_text(py_content)

        nb_path = py_path.with_suffix(".ipynb")
        convert_py_to_nb(py_path=py_path, nb_path=nb_path, should_rerun=False)
        nb = nbformat.read(nb_path, as_version=4)

        assert len(nb.cells) == 5
        assert nb.cells[0].cell_type == "markdown"
        assert nb.cells[1].cell_type == "code"
        assert nb.cells[2].cell_type == "markdown"
        assert nb.cells[3].cell_type == "code"
        assert nb.cells[4].cell_type == "code"


def test_py_to_nb_no_outputs_without_rerun():
    """Test that should_rerun=False doesn't execute the notebook."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        py_content = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %%
print('hello')
"""
        py_path = tmp_path / "test.py"
        py_path.write_text(py_content)

        nb_path = py_path.with_suffix(".ipynb")
        convert_py_to_nb(py_path=py_path, nb_path=nb_path, should_rerun=False)
        nb = nbformat.read(nb_path, as_version=4)

        # Cell should have no outputs since we didn't run it
        assert len(nb.cells[0].outputs) == 0


def test_py_to_nb_with_rerun_executes():
    """Test that should_rerun=True executes the notebook and produces output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

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
print('executed output')
"""
        py_path = tmp_path / "test.py"
        py_path.write_text(py_content)

        nb_path = py_path.with_suffix(".ipynb")
        convert_py_to_nb(py_path=py_path, nb_path=nb_path, should_rerun=True)
        nb = nbformat.read(nb_path, as_version=4)

        # Cell should have output since we ran it
        assert len(nb.cells[0].outputs) > 0
        # Find the stdout output
        stdout_output = None
        for output in nb.cells[0].outputs:
            if output.get("name") == "stdout":
                stdout_output = output
                break
        assert stdout_output is not None
        assert "executed output" in stdout_output.get("text", "")


def test_py_to_nb_roundtrip():
    """Test that nb -> py -> nb preserves content."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create original notebook
        nb_orig = new_notebook()
        nb_orig.cells = [
            new_markdown_cell("# Test Title"),
            new_code_cell("# <<hide-input>>\ncompute()"),
            new_code_cell("visible_code()"),
        ]
        nb_orig.metadata.kernelspec = {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }

        nb_path = tmp_path / "test.ipynb"
        nbformat.write(nb_orig, nb_path)

        # Convert to .py
        py_path = nb_path.with_suffix(".py")
        convert_nb_to_py(nb_path=nb_path, py_path=py_path)

        # Delete original notebook
        nb_path.unlink()

        # Convert back to notebook
        nb_path_new = py_path.with_suffix(".ipynb")
        convert_py_to_nb(py_path=py_path, nb_path=nb_path_new, should_rerun=False)

        # Read result
        nb_result = nbformat.read(nb_path_new, as_version=4)

        # Verify content preserved
        assert len(nb_result.cells) == 3
        assert nb_result.cells[0].cell_type == "markdown"
        assert "Test Title" in nb_result.cells[0].source
        assert "# <<hide-input>>" in nb_result.cells[1].source
        assert "visible_code" in nb_result.cells[2].source


# endregion
# region CLI tests


def test_cli_py_to_nb_basic():
    """Test py-to-nb command creates .ipynb file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        py_path = tmp_path / "test.py"
        py_path.write_text(create_py_content())

        result = runner.invoke(app, ["py-to-nb", str(py_path), "--skip-rerun"])

        assert result.exit_code == 0
        assert "Created" in result.output

        nb_path = tmp_path / "test.ipynb"
        assert nb_path.exists()


def test_cli_py_to_nb_rejects_non_py():
    """Test py-to-nb command rejects non-.py files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb_path = tmp_path / "test.ipynb"
        nb = create_notebook(cells=[])
        nbformat.write(nb, nb_path)

        result = runner.invoke(app, ["py-to-nb", str(nb_path)])

        assert result.exit_code == 1
        assert "expected .py" in result.output


def test_cli_py_to_nb_rejects_missing_file():
    """Test py-to-nb command rejects missing files."""
    result = runner.invoke(app, ["py-to-nb", "/nonexistent/test.py"])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_py_to_nb_overwrites_existing():
    """Test py-to-nb overwrites existing .ipynb file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create .py file
        py_path = tmp_path / "test.py"
        py_path.write_text(create_py_content())

        # Create existing .ipynb file with different content
        nb = create_notebook(cells=[new_code_cell("old_notebook_code()")])
        nb_path = tmp_path / "test.ipynb"
        nbformat.write(nb, nb_path)

        # Run py-to-nb
        result = runner.invoke(app, ["py-to-nb", str(py_path), "--skip-rerun"])

        assert result.exit_code == 0
        # File should be overwritten with .py content
        nb_result = nbformat.read(nb_path, as_version=4)
        # The default content from create_py_content() has print('hello')
        assert "print" in nb_result.cells[0].source
        assert "old_notebook_code" not in nb_result.cells[0].source


# endregion
