"""
Convert notebook to markdown using nbconvert.
"""

from pathlib import Path
from typing import TypedDict

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import TagRemovePreprocessor

from cogames.cli.docsync._nb_md_directive_processing import DirectivePreprocessor, directive_post_process
from cogames.cli.docsync._utils import clean_notebook_metadata, run_notebook


class NotebookResources(TypedDict):
    """Resources returned by nbconvert."""

    outputs: dict[str, bytes]
    output_files_dir: str


def convert_nb_to_md_in_memory(
    *, nb_path: Path, cogames_root: Path, notebook_relpath: Path | None = None
) -> tuple[str, NotebookResources]:
    """Convert .ipynb to .md and resource files using nbconvert.

    Args:
        nb_path: Path to the notebook file to read.
        cogames_root: Root path for cogames package (used to compute relative path if notebook_relpath not provided).
        notebook_relpath: Optional override for the relative path used in Colab URL generation.
            Useful when the notebook is in a temp directory but we need the original relative path.
    """

    # This tells nbconvert where we'll write output files later, so it can reference them now.
    output_files_dir = f"{nb_path.stem}_files"

    exporter = MarkdownExporter()
    exporter.register_preprocessor(DirectivePreprocessor, enabled=True)

    # Register TagRemovePreprocessor to properly hide input for # <<hide-input>> cells.
    # This removes the code block entirely rather than leaving an empty code block.
    tag_remove_preprocessor = TagRemovePreprocessor()
    tag_remove_preprocessor.remove_input_tags = {"remove_input"}
    exporter.register_preprocessor(tag_remove_preprocessor, enabled=True)

    nb = nbformat.read(nb_path, as_version=4)
    if notebook_relpath is None:
        notebook_relpath = nb_path.relative_to(cogames_root)
    md, resources = exporter.from_notebook_node(
        nb=nb,
        resources={
            "output_files_dir": output_files_dir,
            "notebook_relpath": notebook_relpath,
        },
    )

    # Run preprocessor again on nb to set collapse metadata (nbconvert modifies a copy internally).
    DirectivePreprocessor().preprocess(nb=nb, resources={"notebook_relpath": notebook_relpath})
    md = directive_post_process(md=md, nb=nb)

    return md, resources


def _write_notebook_output(*, to_dir: Path, name: str, md: str, resources: NotebookResources) -> None:
    """Write notebook output to the given directory."""

    # Write resources (images, etc.)
    # Note: filename keys already include the subdirectory (e.g., "test_files/output_0_0.png")
    output_files = resources["outputs"]
    output_files_dir = resources["output_files_dir"]
    if len(output_files) > 0:
        files_dir = to_dir / output_files_dir
        files_dir.mkdir(exist_ok=True)
        for filename, data in output_files.items():
            (to_dir / filename).write_bytes(data)

    # Write markdown.
    output_path = to_dir / f"{name}.md"
    output_path.write_text(md)


def convert_nb_to_md(*, nb_path: Path, should_rerun: bool, cogames_root: Path) -> None:
    """Export a notebook to markdown in the same directory.

    Args:
        nb_path: Path to the notebook file.
        should_rerun: Whether to re-run the notebook before exporting.
        cogames_root: Root path for cogames package, used to compute the relative
            path for Colab URL generation.
    """
    # Resolve to absolute paths for consistent handling (macOS symlinks /var -> /private/var)
    nb_path = nb_path.resolve()
    cogames_root = cogames_root.resolve()

    if should_rerun:
        run_notebook(nb_path=nb_path)

    md, resources = convert_nb_to_md_in_memory(nb_path=nb_path, cogames_root=cogames_root)

    _write_notebook_output(to_dir=nb_path.parent, name=nb_path.stem, md=md, resources=resources)

    clean_notebook_metadata(nb_path=nb_path)
