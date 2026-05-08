from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urljoin, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from cogames.coworld.schema_validation import JsonObject


class RemoteCoworldManifestResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    manifest: JsonObject = Field(min_length=1)


@contextmanager
def materialized_manifest_path(manifest_uri: str, *, server: str | None = None) -> Iterator[Path]:
    resolved_uri = _resolve_manifest_uri(manifest_uri, server=server)
    parsed = urlparse(resolved_uri)
    if parsed.scheme == "file":
        yield Path(unquote(parsed.path)).resolve()
        return
    if parsed.scheme in ("http", "https"):
        with tempfile.TemporaryDirectory(prefix="coworld-manifest-") as temp_dir:
            manifest_path = Path(temp_dir) / "coworld_manifest.json"
            manifest_path.write_text(json.dumps(_download_manifest(resolved_uri)), encoding="utf-8")
            yield manifest_path
        return

    yield Path(resolved_uri).resolve()


def _resolve_manifest_uri(manifest_uri: str, *, server: str | None = None) -> str:
    parsed = urlparse(manifest_uri)
    if parsed.scheme:
        return manifest_uri
    if manifest_uri.startswith("/v2/coworlds/"):
        if server is None:
            raise ValueError(f"Backend Coworld manifest URI requires --server: {manifest_uri}")
        return urljoin(f"{server.rstrip('/')}/", manifest_uri.lstrip("/"))
    return manifest_uri


def _download_manifest(manifest_uri: str) -> JsonObject:
    response = httpx.get(manifest_uri, timeout=60.0)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object from {manifest_uri}")
    if "game" in value:
        return value
    return RemoteCoworldManifestResponse.model_validate(value).manifest
