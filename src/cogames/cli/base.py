from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console

console = Console()


def emit_json(payload: Any) -> None:
    """Write machine-parseable JSON directly to stdout, bypassing Rich console.

    Using sys.stdout.write instead of console.print ensures the output is clean
    JSON without Rich markup, terminal width wrapping, or ANSI escape codes that
    would corrupt the output when piped to other programs.
    """
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
