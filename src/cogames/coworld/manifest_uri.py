from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from cogames.coworld.schema_validation import JsonObject


class RemoteCoworldManifestResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    manifest: JsonObject = Field(min_length=1)


@contextmanager
def materialized_manifest_path(manifest_uri: str) -> Iterator[Path]:
    parsed = urlparse(manifest_uri)
    if parsed.scheme == "file":
        yield Path(unquote(parsed.path)).resolve()
        return
    if parsed.scheme in ("http", "https"):
        with tempfile.TemporaryDirectory(prefix="coworld-manifest-") as temp_dir:
            manifest_path = Path(temp_dir) / "coworld_manifest.json"
            manifest_path.write_text(json.dumps(_download_manifest(manifest_uri)), encoding="utf-8")
            yield manifest_path
        return

    yield Path(manifest_uri).resolve()


def _download_manifest(manifest_uri: str) -> JsonObject:
    response = httpx.get(manifest_uri, timeout=60.0)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object from {manifest_uri}")
    if "game" in value:
        return value
    return RemoteCoworldManifestResponse.model_validate(value).manifest
