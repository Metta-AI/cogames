import subprocess
import tempfile
from pathlib import Path

import jupytext
import nbformat
import typer
from nbstripout import strip_output


def get_cogames_root() -> Path:
    """Get cogames root path by finding pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find cogames root (no pyproject.toml found)")


def lint_py_file(path: Path, /) -> None:
    """Run ruff format on a Python file."""
    subprocess.run(["ruff", "format", str(path)], check=True, capture_output=True)


def run_notebook(*, nb_path: Path) -> None:
    """Execute notebook in place."""
    typer.echo(f"  Executing {nb_path.name}...")
    result = subprocess.run(
        ["jupyter", "execute", nb_path.name, "--inplace"],
        cwd=nb_path.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        typer.echo(f"  Error: notebook execution failed: {result.stderr}", err=True)
        raise typer.Exit(1)
    typer.echo(f"  Executed {nb_path.name}")


def clean_notebook_metadata(*, nb_path: Path) -> None:
    """Strip metadata (execution counts, cell IDs, execution timestamps) but keep outputs.

    Also ensures accelerator is set to GPU for Colab.
    """
    typer.echo(f"  Cleaning metadata from {nb_path.name}...")
    nb = nbformat.read(nb_path, as_version=4)
    nb = strip_output(
        nb,
        keep_output=True,
        keep_count=False,
        keep_id=False,
        extra_keys=["cell.metadata.execution"],
    )
    # Ensure GPU accelerator for Colab
    # See: https://github.com/mwouts/jupytext/pull/235#issuecomment-495010137
    nb.metadata["accelerator"] = "GPU"
    nbformat.write(nb, nb_path)
    typer.echo(f"  Cleaned metadata from {nb_path.name}")


def files_equal(path1: str | Path, path2: str | Path, /) -> bool:
    """Check if two files have identical content."""
    return Path(path1).read_bytes() == Path(path2).read_bytes()


def py_nb_content_equal(*, py_path: Path, nb_path: Path) -> bool:
    """Check if .py and .ipynb have equivalent content (ignoring outputs and linting)."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp_py = Path(tmp.name)
    try:
        # Convert current .ipynb to .py and compare with existing .py
        nb = jupytext.read(nb_path)
        jupytext.write(nb, tmp_py, fmt="py:percent")
        lint_py_file(tmp_py)  # Format generated .py for fair comparison
        return files_equal(py_path, tmp_py)
    finally:
        tmp_py.unlink(missing_ok=True)
