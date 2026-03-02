from __future__ import annotations

import json
from typing import Any

from rich.console import Console

console = Console()


def emit_json(payload: Any) -> None:
    """Write machine-parseable JSON without terminal width wrapping."""
    console.print(json.dumps(payload, indent=2), soft_wrap=True, highlight=False)
