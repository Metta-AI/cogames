"""
Convert notebook to markdown using nbconvert.
"""

import nbformat
from nbconvert.preprocessors import Preprocessor

_COLAB_URL_BASE = "https://colab.research.google.com/github/Metta-AI/cogames/blob/main"


class DirectivePreprocessor(Preprocessor):
    """Process directives in code cells."""

    def preprocess_cell(self, cell, resources, index):
        del index  # unused

        # Handle <<colab-link>> placeholder in markdown cells
        if cell.cell_type == "markdown" and "<<colab-link>>" in cell.source:
            notebook_relpath = resources["notebook_relpath"]
            colab_url = f"{_COLAB_URL_BASE}/{notebook_relpath}"
            cell.source = cell.source.replace("<<colab-link>>", colab_url)
            return cell, resources

        if cell.cell_type != "code":
            return cell, resources

        source = cell.source
        if not source.strip():
            return cell, resources

        lines = source.split("\n")
        first_line = lines[0].strip()

        if not (first_line.startswith("# <<") and ">>" in first_line):
            return cell, resources

        # Extract directive from "# <<directive>>" format
        directive = first_line[4 : first_line.index(">>")].strip()
        rest_source = "\n".join(lines[1:])

        if directive == "hide":
            # Mark cell for removal
            cell.source = ""
            cell.outputs = []
            cell.metadata["remove_cell"] = True

        elif directive == "hide-input":
            # Keep outputs, hide code using TagRemovePreprocessor
            cell.source = rest_source
            cell.metadata.setdefault("tags", []).append("remove_input")

        elif directive == "hide-output":
            # Keep code, hide outputs
            cell.source = rest_source
            cell.outputs = []

        elif directive == "collapse-input":
            # Will be handled in post-processing
            cell.source = rest_source
            cell.metadata["collapse_input"] = True

        elif directive == "collapse-output":
            # Will be handled in post-processing
            cell.source = rest_source
            cell.metadata["collapse_output"] = True

        return cell, resources

    def preprocess(self, nb, resources):
        # First pass: process directives
        nb, resources = super().preprocess(nb=nb, resources=resources)

        # Second pass: remove cells marked for removal
        nb.cells = [cell for cell in nb.cells if not cell.get("metadata", {}).get("remove_cell", False)]

        # Also remove empty code cells (from hide-input with no output)
        nb.cells = [
            cell for cell in nb.cells if not (cell.cell_type == "code" and not cell.source.strip() and not cell.outputs)
        ]

        return nb, resources


def directive_post_process(*, md: str, nb: nbformat.NotebookNode) -> str:
    """Wrap collapsed cells in <details> tags."""

    # Build list of (collapse_input, collapse_output) for each code cell that has visible code.
    # Skip cells with "remove_input" tag since their code block is removed from markdown.
    collapse_info = []
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source.strip():
            meta = cell.get("metadata", {})
            tags = meta.get("tags", [])
            # Skip cells whose input is removed (no code block in markdown)
            if "remove_input" in tags:
                continue
            collapse_info.append(
                (
                    meta.get("collapse_input", False),
                    meta.get("collapse_output", False),
                )
            )

    if not any(collapse_input or collapse_output for collapse_input, collapse_output in collapse_info):
        return md

    # Process markdown line by line
    # nbconvert outputs: ```python ... ``` for code, then indented lines for output
    lines = md.split("\n")
    result = []
    code_cell_idx = 0
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect start of code block
        if line.startswith("```python"):
            code_block_lines = [line]
            i += 1

            # Collect code block
            while i < len(lines) and not (lines[i].startswith("```") and len(lines[i].strip()) == 3):
                code_block_lines.append(lines[i])
                i += 1
            if i < len(lines):
                code_block_lines.append(lines[i])  # closing ```
                i += 1

            # Check for collapse_input
            collapse_input = False
            collapse_output = False
            if code_cell_idx < len(collapse_info):
                collapse_input, collapse_output = collapse_info[code_cell_idx]

            # Output code block (possibly wrapped)
            if collapse_input:
                result.append("<details>")
                result.append("<summary>Code</summary>")
                result.append("")
                result.extend(code_block_lines)
                result.append("")
                result.append("</details>")
            else:
                result.extend(code_block_lines)

            # Collect output (indented lines or empty lines between code and next content)
            output_lines = []
            while i < len(lines):
                # Output lines are typically indented with 4 spaces
                if lines[i].startswith("    ") or lines[i] == "":
                    output_lines.append(lines[i])
                    i += 1
                else:
                    break

            # Strip trailing empty lines from output
            while output_lines and output_lines[-1] == "":
                output_lines.pop()

            # Output the output (possibly wrapped)
            if output_lines:
                if collapse_output:
                    result.append("")
                    result.append("<details>")
                    result.append("<summary>Output</summary>")
                    result.append("")
                    result.extend(output_lines)
                    result.append("")
                    result.append("</details>")
                else:
                    result.extend(output_lines)

            result.append("")  # blank line after cell
            code_cell_idx += 1
        else:
            result.append(line)
            i += 1

    return "\n".join(result)
