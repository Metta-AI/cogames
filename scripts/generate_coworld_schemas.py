from __future__ import annotations

import json
from pathlib import Path

from cogames.coworld.types import coworld_episode_request_schema, coworld_manifest_schema

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "cogames" / "coworld"


def main() -> None:
    schemas = {
        "coworld_manifest_schema.json": coworld_manifest_schema(),
        "episode_request_schema.json": coworld_episode_request_schema(),
    }
    for filename, schema in schemas.items():
        (PACKAGE_ROOT / filename).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
