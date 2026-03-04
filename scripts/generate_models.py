#!/usr/bin/env -S uv run
"""Regenerate generated_models.py from the public OpenAPI spec.

Usage:
    uv run packages/cogames/scripts/generate_models.py [path/to/openapi.json]
    uv run packages/cogames/scripts/generate_models.py --output /tmp/fresh.py

If no positional path is given, generates a fresh public OpenAPI spec via
app_backend/scripts/export_openapi.py and uses that as input.

Called by `metta dev generate-api-types` alongside the TypeScript codegen.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EXPORT_OPENAPI_SCRIPT = _REPO_ROOT / "app_backend" / "scripts" / "export_openapi.py"
_DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "src" / "cogames" / "cli" / "generated_models.py"


def generate(spec: Path, output: Path) -> None:
    """Run datamodel-codegen and ruff format to produce generated_models.py."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "datamodel_code_generator",
            "--input",
            str(spec),
            "--input-file-type",
            "openapi",
            "--output",
            str(output),
            "--output-model-type",
            "pydantic_v2.BaseModel",
            "--target-python-version",
            "3.12",
            "--use-standard-collections",
            "--use-annotated",
            "--use-field-description",
            "--field-constraints",
            "--base-class",
            "cogames.cli._model_base.CLIModel",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"datamodel-codegen failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    fmt = subprocess.run(
        [sys.executable, "-m", "ruff", "format", str(output)],
        capture_output=True,
        text=True,
    )
    if fmt.returncode != 0:
        print(f"ruff format failed:\n{fmt.stderr}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = sys.argv[1:]

    output = _DEFAULT_OUTPUT
    if "--output" in args:
        idx = args.index("--output")
        output = Path(args[idx + 1])
        args = args[:idx] + args[idx + 2 :]

    if args:
        spec = Path(args[0])
        if not spec.exists():
            print(f"OpenAPI spec not found: {spec}", file=sys.stderr)
            print("Run: uv run app_backend/scripts/export_openapi.py", file=sys.stderr)
            sys.exit(1)
        generate(spec, output)
        print(f"Generated: {output}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result = subprocess.run(
            [sys.executable, str(_EXPORT_OPENAPI_SCRIPT), "--output", str(output_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Failed to export OpenAPI spec:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

        spec = output_dir / "openapi.json"
        if not spec.exists():
            print(f"OpenAPI spec not found after export: {spec}", file=sys.stderr)
            sys.exit(1)
        generate(spec, output)

    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
