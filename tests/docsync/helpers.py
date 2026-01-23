"""Shared test helpers for `cogames docsync` tests."""

import nbformat
from nbformat.v4 import new_notebook


def create_notebook(*, cells: list[nbformat.NotebookNode]) -> nbformat.NotebookNode:
    """Create a notebook with the given cells and standard metadata."""
    nb = new_notebook()
    nb.cells = cells
    nb.metadata.kernelspec = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata.language_info = {
        "name": "python",
        "file_extension": ".py",
    }
    return nb


def create_py_content() -> str:
    """Create minimal valid percent format Python content."""
    return """\
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
