"""Tests for `cogames docsync nb-to-md` subcommand."""

import re
import tempfile
from pathlib import Path

import nbformat
from helpers import create_notebook
from nbformat.v4 import new_code_cell, new_markdown_cell, new_output
from typer.testing import CliRunner

from cogames.cli.docsync._nb_md_sync import convert_nb_to_md, convert_nb_to_md_in_memory
from cogames.cli.docsync._utils import clean_notebook_metadata
from cogames.cli.docsync.docsync import app

runner = CliRunner()


def _export_and_read(*, nb: nbformat.NotebookNode, tmp_path: Path) -> tuple[str, Path]:
    """Save notebook, export it, and return the markdown content and files directory."""
    notebook_path = tmp_path / "test.ipynb"
    nbformat.write(nb, notebook_path)
    convert_nb_to_md(nb_path=notebook_path, should_rerun=False, cogames_root=tmp_path)

    md_path = tmp_path / "test.md"
    md = md_path.read_text()
    files_dir = tmp_path / "test_files"
    return md, files_dir


# region Function tests


def test_export_basic():
    """Test basic notebook export with code and markdown cells."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(
            cells=[
                new_markdown_cell("# Hello World"),
                new_code_cell("print('hello')"),
            ]
        )

        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        assert "# Hello World" in md
        assert "```python" in md
        assert "print('hello')" in md


def test_export_hide_directive():
    """Test that # <<hide>> removes entire cell from output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        hidden_cell = new_code_cell("# <<hide>>\nsecret_code()")
        visible_cell = new_code_cell("visible_code()")

        nb = create_notebook(cells=[hidden_cell, visible_cell])
        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        assert "secret_code" not in md
        assert "visible_code" in md


def test_export_hide_input_directive():
    """Test that # <<hide-input>> removes code but keeps output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        cell = new_code_cell("# <<hide-input>>\nhidden_code()")
        # Add output to the cell
        cell.outputs = [new_output(output_type="stream", name="stdout", text="output text here")]

        nb = create_notebook(cells=[cell])
        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        # Code should be hidden
        assert "hidden_code" not in md
        # Output should be present
        assert "output text here" in md


def test_export_hide_output_directive():
    """Test that # <<hide-output>> removes output but keeps code."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        cell = new_code_cell("# <<hide-output>>\nshown_code()")
        # Add output that should be hidden
        cell.outputs = [new_output(output_type="stream", name="stdout", text="hidden output")]

        nb = create_notebook(cells=[cell])
        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        # Code should be shown (minus the directive line)
        assert "shown_code" in md
        # Output should be hidden
        assert "hidden output" not in md


def test_export_collapse_input_directive():
    """Test that # <<collapse-input>> wraps code in <details> tag."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        cell = new_code_cell("# <<collapse-input>>\ncollapsed_code()")

        nb = create_notebook(cells=[cell])
        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        # Code should be wrapped in details
        assert "<details>" in md
        assert "<summary>Code</summary>" in md
        assert "collapsed_code" in md
        assert "</details>" in md


def test_export_collapse_output_directive():
    """Test that # <<collapse-output>> wraps output in <details> tag."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        cell = new_code_cell("# <<collapse-output>>\nsome_code()")
        cell.outputs = [new_output(output_type="stream", name="stdout", text="collapsed output here")]

        nb = create_notebook(cells=[cell])
        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        # Code should be visible normally
        assert "some_code" in md
        # Output should be wrapped in details
        assert "<details>" in md
        assert "<summary>Output</summary>" in md
        assert "collapsed output here" in md


def test_export_empty_cells_removed():
    """Test that empty code cells are removed from output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(
            cells=[
                new_code_cell(""),  # Empty cell
                new_code_cell("actual_code()"),
            ]
        )

        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        # Should only have one code block
        assert md.count("```python") == 1
        assert "actual_code" in md


def test_export_image_output():
    """Test that image outputs are written to {stem}_files directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a minimal 1x1 red PNG image (base64 encoded)
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

        cell = new_code_cell("import matplotlib.pyplot as plt")
        # Add a display_data output with PNG image using new_output helper
        cell.outputs = [
            new_output(
                output_type="display_data",
                data={"image/png": png_b64},
                metadata={},
            )
        ]

        nb = create_notebook(cells=[cell])
        md, files_dir = _export_and_read(nb=nb, tmp_path=tmp_path)

        # Markdown should reference image in the {stem}_files subdirectory
        assert "![png](test_files/" in md
        assert ".png)" in md

        # Image file should exist on disk
        assert files_dir.exists()
        png_files = list(files_dir.glob("*.png"))
        assert len(png_files) == 1


def test_export_multiple_images():
    """Test that multiple images are all written correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

        cell1 = new_code_cell("plt.plot([1,2,3])")
        cell1.outputs = [new_output(output_type="display_data", data={"image/png": png_b64}, metadata={})]

        cell2 = new_code_cell("plt.plot([4,5,6])")
        cell2.outputs = [new_output(output_type="display_data", data={"image/png": png_b64}, metadata={})]

        nb = create_notebook(cells=[cell1, cell2])
        md, files_dir = _export_and_read(nb=nb, tmp_path=tmp_path)

        # Both images should be referenced in markdown
        assert md.count("![png]") == 2

        # Both image files should exist
        png_files = list(files_dir.glob("*.png"))
        assert len(png_files) == 2


def test_export_mixed_directives():
    """Test notebook with multiple cells using different directives."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        cells = [
            new_markdown_cell("# Introduction"),
            new_code_cell("# <<hide>>\nsetup_code()"),
            new_code_cell("visible_code()"),
            new_code_cell("# <<hide-input>>\ncompute()"),
            new_code_cell("# <<collapse-input>>\nlong_code()"),
        ]
        # Add output to hide-input cell
        cells[3].outputs = [new_output(output_type="stream", name="stdout", text="result: 42")]

        nb = create_notebook(cells=cells)
        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        assert "# Introduction" in md
        assert "setup_code" not in md  # hidden
        assert "visible_code" in md
        assert "compute" not in md  # hide-input
        assert "result: 42" in md  # but output shown
        assert "long_code" in md
        assert "<details>" in md  # collapse-input


def test_export_execute_result():
    """Test that execute_result outputs (like Out[1]: 42) are included."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        cell = new_code_cell("1 + 1")
        cell.outputs = [
            new_output(
                output_type="execute_result",
                data={"text/plain": "2"},
                metadata={},
                execution_count=1,
            )
        ]

        nb = create_notebook(cells=[cell])
        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        assert "1 + 1" in md
        assert "2" in md


def test_export_cleans_metadata():
    """Test that export strips metadata but keeps outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        cell = new_code_cell("1 + 1")
        cell.execution_count = 42
        cell.outputs = [
            new_output(
                output_type="execute_result",
                data={"text/plain": "2"},
                metadata={},
                execution_count=42,
            )
        ]

        nb = create_notebook(cells=[cell])
        notebook_path = tmp_path / "test.ipynb"
        nbformat.write(nb, notebook_path)

        convert_nb_to_md(nb_path=notebook_path, should_rerun=False, cogames_root=tmp_path)

        # Read back the notebook
        nb_after = nbformat.read(notebook_path, as_version=4)

        # Outputs should be preserved
        assert len(nb_after.cells[0].outputs) == 1
        assert nb_after.cells[0].outputs[0]["data"]["text/plain"] == "2"

        # Execution count should be stripped
        assert nb_after.cells[0].execution_count is None


def test_export_hide_input_markdown_display():
    """Test that # <<hide-input>> with display(Markdown()) doesn't leave empty code blocks.

    This tests the case where a cell uses display(Markdown(...)) to render markdown,
    and we want to hide the code but show the rendered markdown output.
    The exported markdown should NOT contain empty code blocks like ```\\n\\n```.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        cell = new_code_cell(
            "# <<hide-input>>\nfrom IPython.display import display, Markdown\ndisplay(Markdown('# Generated Heading'))"
        )
        # Simulate the output from display(Markdown(...))
        cell.outputs = [
            new_output(
                output_type="display_data",
                data={
                    "text/markdown": "# Generated Heading",
                    "text/plain": "<IPython.core.display.Markdown object>",
                },
                metadata={},
            )
        ]

        nb = create_notebook(cells=[cell])
        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        # Code should be hidden
        assert "hide-input" not in md
        assert "display(Markdown" not in md

        # Markdown output should be rendered directly (not in a code block)
        assert "# Generated Heading" in md

        # Should NOT have empty code blocks (with or without language identifier)
        # Match code blocks that are empty or contain only whitespace
        empty_code_block = re.search(r"```(?:python)?\s*```", md)
        assert not empty_code_block, f"Found empty code block in markdown:\n{md}"


def test_export_with_rerun_executes_notebook():
    """Test that should_rerun=True executes the notebook and produces output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a notebook with code but NO outputs
        cell = new_code_cell("print('executed output')")
        # Explicitly no outputs - the notebook hasn't been run yet
        cell.outputs = []

        nb = create_notebook(cells=[cell])
        notebook_path = tmp_path / "test.ipynb"
        nbformat.write(nb, notebook_path)

        # Export with should_rerun=True - this should execute the notebook
        convert_nb_to_md(nb_path=notebook_path, should_rerun=True, cogames_root=tmp_path)

        # Read the exported markdown
        md_path = tmp_path / "test.md"
        md = md_path.read_text()

        # The output should appear because the notebook was executed
        assert "executed output" in md


def test_export_without_rerun_preserves_existing_output():
    """Test that should_rerun=False doesn't execute, keeping existing outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a notebook with pre-existing output
        cell = new_code_cell("print('this would print something else if re-run')")
        cell.outputs = [new_output(output_type="stream", name="stdout", text="pre-existing output")]

        nb = create_notebook(cells=[cell])
        notebook_path = tmp_path / "test.ipynb"
        nbformat.write(nb, notebook_path)

        # Export with should_rerun=False - should NOT execute
        convert_nb_to_md(nb_path=notebook_path, should_rerun=False, cogames_root=tmp_path)

        # Read the exported markdown
        md_path = tmp_path / "test.md"
        md = md_path.read_text()

        # The pre-existing output should be preserved (not re-run)
        assert "pre-existing output" in md


def test_export_without_rerun_no_output_when_not_run():
    """Test that should_rerun=False with no outputs results in no output in markdown."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a notebook with code but NO outputs (never been run)
        # Use a unique marker that only appears in output, not in the code itself
        cell = new_code_cell("x = 123 + 456\nprint(x)")
        cell.outputs = []

        nb = create_notebook(cells=[cell])
        notebook_path = tmp_path / "test.ipynb"
        nbformat.write(nb, notebook_path)

        # Export with should_rerun=False - should NOT execute
        convert_nb_to_md(nb_path=notebook_path, should_rerun=False, cogames_root=tmp_path)

        # Read the exported markdown
        md_path = tmp_path / "test.md"
        md = md_path.read_text()

        # The code should be present
        assert "x = 123 + 456" in md
        assert "print(x)" in md
        # But "579" (the result of 123+456) should NOT appear since we didn't run it
        assert "579" not in md


def test_export_colab_link_placeholder():
    """Test that <<colab-link>> in markdown cells is replaced with actual Colab URL."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(
            cells=[
                new_markdown_cell(
                    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<<colab-link>>)"
                ),
                new_code_cell("print('hello')"),
            ]
        )

        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        # The placeholder should be replaced with actual URL
        assert "<<colab-link>>" not in md
        assert "https://colab.research.google.com/github/Metta-AI/cogames/blob/main/test.ipynb" in md


def test_export_multiple_colab_link_placeholders():
    """Test that multiple <<colab-link>> placeholders are all replaced."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(
            cells=[
                new_markdown_cell("Link 1: <<colab-link>>\n\nLink 2: <<colab-link>>"),
            ]
        )

        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        assert "<<colab-link>>" not in md
        expected_url = "https://colab.research.google.com/github/Metta-AI/cogames/blob/main/test.ipynb"
        assert md.count(expected_url) == 2


def test_export_colab_link_not_replaced_in_code_cells():
    """Test that <<colab-link>> in code cells is NOT replaced (only markdown cells)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(
            cells=[
                new_code_cell("url = '<<colab-link>>'"),
            ]
        )

        md, _ = _export_and_read(nb=nb, tmp_path=tmp_path)

        # The placeholder should still be there in the code cell
        assert "<<colab-link>>" in md


def test_export_colab_link_subdirectory():
    """Test that <<colab-link>> preserves subdirectory in URL."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a subdirectory structure
        subdir = tmp_path / "tutorials"
        subdir.mkdir()

        nb = create_notebook(
            cells=[
                new_markdown_cell(
                    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<<colab-link>>)"
                ),
            ]
        )

        notebook_path = subdir / "getting_started.ipynb"
        nbformat.write(nb, notebook_path)

        # Export with cogames_root = tmp_path (simulating cogames as root)
        md, _ = convert_nb_to_md_in_memory(
            nb_path=notebook_path,
            cogames_root=tmp_path,
        )

        # URL should include the subdirectory
        assert "Metta-AI/cogames/blob/main/tutorials/getting_started.ipynb" in md
        # Should NOT be just the stem
        assert "Metta-AI/cogames/blob/main/getting_started.ipynb" not in md


def test_export_colab_link_root_notebook():
    """Test that <<colab-link>> works for notebooks at cogames root."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(
            cells=[
                new_markdown_cell(
                    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<<colab-link>>)"
                ),
            ]
        )

        notebook_path = tmp_path / "README.ipynb"
        nbformat.write(nb, notebook_path)

        # Export with notebook at root level
        md, _ = convert_nb_to_md_in_memory(
            nb_path=notebook_path,
            cogames_root=tmp_path,
        )

        # URL should be at the root (no subdirectory)
        assert "Metta-AI/cogames/blob/main/README.ipynb" in md
        # Should NOT have extra path components
        assert "Metta-AI/cogames/blob/main/tutorials/README.ipynb" not in md


# endregion
# region clean_notebook_metadata tests


def test_clean_notebook_metadata_strips_execution_timestamps():
    """Test that clean_notebook_metadata strips execution timestamps from cells."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a notebook with execution metadata (as would be added by jupyter execute)
        nb = create_notebook(cells=[new_code_cell("print('hello')")])
        nb.cells[0].metadata["execution"] = {
            "iopub.execute_input": "2024-01-15T10:00:00.000000Z",
            "iopub.status.busy": "2024-01-15T10:00:00.000000Z",
            "iopub.status.idle": "2024-01-15T10:00:00.100000Z",
            "shell.execute_reply": "2024-01-15T10:00:00.100000Z",
        }
        nb_path = tmp_path / "test.ipynb"
        nbformat.write(nb, nb_path)

        # Verify execution metadata is present before cleaning
        nb_before = nbformat.read(nb_path, as_version=4)
        assert "execution" in nb_before.cells[0].metadata

        # Run clean_notebook_metadata
        clean_notebook_metadata(nb_path=nb_path)

        # Verify execution metadata is stripped
        nb_after = nbformat.read(nb_path, as_version=4)
        assert "execution" not in nb_after.cells[0].metadata, (
            "clean_notebook_metadata should strip cell.metadata.execution timestamps"
        )


def test_clean_notebook_metadata_sets_gpu_accelerator():
    """Test that clean_notebook_metadata sets accelerator to GPU for Colab."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a notebook without accelerator metadata
        nb = create_notebook(cells=[new_code_cell("print('hello')")])
        assert "accelerator" not in nb.metadata

        notebook_path = tmp_path / "test.ipynb"
        nbformat.write(nb, notebook_path)

        # Clean the notebook metadata
        clean_notebook_metadata(nb_path=notebook_path)

        # Read back and verify accelerator is set to GPU
        nb_after = nbformat.read(notebook_path, as_version=4)
        assert nb_after.metadata.get("accelerator") == "GPU"


# endregion
# region CLI tests


def test_cli_nb_to_md_basic():
    """Test nb-to-md command creates .md file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        nb = create_notebook(
            cells=[
                new_markdown_cell("# Hello"),
                new_code_cell("print('hello')"),
            ]
        )
        nb_path = tmp_path / "test.ipynb"
        nbformat.write(nb, nb_path)

        result = runner.invoke(app, ["nb-to-md", str(nb_path), "--skip-rerun", "--cogames-root", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "Exporting" in result.output

        md_path = tmp_path / "test.md"
        assert md_path.exists()
        assert "# Hello" in md_path.read_text()


def test_cli_nb_to_md_rejects_non_ipynb():
    """Test nb-to-md command rejects non-.ipynb files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        py_path = tmp_path / "test.py"
        py_path.write_text("print('hello')")

        result = runner.invoke(app, ["nb-to-md", str(py_path)])

        assert result.exit_code == 1
        assert "expected .ipynb" in result.output


# endregion
