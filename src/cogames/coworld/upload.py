from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import httpx
import typer
from pydantic import BaseModel

from cogames.cli.submit import DEFAULT_SUBMIT_SERVER
from cogames.coworld.certifier import certify_coworld, load_coworld_package, resolve_manifest_uri
from softmax.auth import DEFAULT_COGAMES_SERVER, load_current_cogames_token

RUNNABLE_LIST_SECTIONS = ("player", "grader", "reporter", "commissioner", "diagnoser", "optimizer")


class CoworldUploadResponse(BaseModel):
    upload_id: str
    name: str
    version: str
    manifest: dict[str, Any]
    size_bytes: int


class AwsCredentials(BaseModel):
    access_key_id: str
    secret_access_key: str
    session_token: str


class EcrPushInfo(BaseModel):
    kind: str = "ecr"
    region: str
    registry: str
    repository: str
    tag: str
    image_uri: str
    endpoint_url: str | None = None
    credentials: AwsCredentials


class ContainerImageResponse(BaseModel):
    id: str
    name: str
    version: int
    client_hash: str | None = None
    status: str
    image_uri: str | None = None
    image_digest: str | None = None
    public_image_uri: str | None = None


class ImageUploadResponse(BaseModel):
    image: ContainerImageResponse
    pre_signed_info: EcrPushInfo | None = None


@dataclass(frozen=True)
class CoworldUploadResult:
    name: str
    version: str
    manifest: dict[str, Any]
    size_bytes: int


class CoworldUploadClient:
    def __init__(self, server_url: str, token: str):
        self._http_client = httpx.Client(base_url=server_url, timeout=30.0)
        self._token = token

    @classmethod
    def from_login(cls, *, server_url: str, login_server: str) -> Self:
        token = load_current_cogames_token(login_server=login_server)
        if token is None:
            raise RuntimeError("Not authenticated. Run: cogames auth login")
        return cls(server_url=server_url, token=token)

    def close(self) -> None:
        self._http_client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return {"X-Auth-Token": self._token}

    def upload_manifest(self, manifest: dict[str, object]) -> CoworldUploadResponse:
        response = self._http_client.post(
            "/v2/coworlds/upload",
            headers=self._headers(),
            json={"manifest": manifest},
            timeout=120.0,
        )
        response.raise_for_status()
        return CoworldUploadResponse.model_validate(response.json())

    def request_image_upload(self, *, name: str, client_hash: str) -> ImageUploadResponse:
        response = self._http_client.post(
            "/v2/container_images/upload",
            headers=self._headers(),
            json={"name": name, "client_hash": client_hash},
            timeout=60.0,
        )
        response.raise_for_status()
        return ImageUploadResponse.model_validate(response.json())

    def complete_image_upload(self, image_id: str) -> ContainerImageResponse:
        response = self._http_client.post(
            "/v2/container_images/upload/complete",
            headers=self._headers(),
            json={"id": image_id},
            timeout=120.0,
        )
        response.raise_for_status()
        return ContainerImageResponse.model_validate(response.json())


def upload_coworld(
    manifest_path: Path,
    *,
    server: str = DEFAULT_SUBMIT_SERVER,
    login_server: str = DEFAULT_COGAMES_SERVER,
    timeout_seconds: float = 60.0,
) -> CoworldUploadResult:
    package = load_coworld_package(manifest_path)
    certify_coworld(package.manifest_path, timeout_seconds=timeout_seconds)

    with CoworldUploadClient.from_login(server_url=server, login_server=login_server) as client:
        upload_manifest = _manifest_with_softmax_images(client, package.manifest)
        upload_manifest = _manifest_with_inlined_docs(upload_manifest, package.manifest_path.parent)
        response = client.upload_manifest(upload_manifest)

    return CoworldUploadResult(
        name=response.name,
        version=response.version,
        manifest=response.manifest,
        size_bytes=response.size_bytes,
    )


def upload_coworld_cmd(
    manifest_path: Path,
    server: str = DEFAULT_SUBMIT_SERVER,
    login_server: str = DEFAULT_COGAMES_SERVER,
    timeout_seconds: float = 60.0,
) -> None:
    result = upload_coworld(
        manifest_path,
        server=server,
        login_server=login_server,
        timeout_seconds=timeout_seconds,
    )
    typer.echo(f"Upload complete: {result.name}:{result.version}")
    typer.echo(f"Size: {result.size_bytes} bytes")


def _manifest_with_softmax_images(client: CoworldUploadClient, manifest: dict[str, object]) -> dict[str, object]:
    upload_manifest = copy.deepcopy(manifest)
    replacements = {
        image: _upload_container_image(client, image).image_uri
        for image in sorted({runnable["image"] for runnable in _manifest_runnables(upload_manifest)})
    }
    for runnable in _manifest_runnables(upload_manifest):
        runnable["image"] = replacements[runnable["image"]]
    return upload_manifest


def _manifest_runnables(manifest: dict[str, object]) -> list[dict[str, Any]]:
    game = manifest["game"]
    if not isinstance(game, dict):
        raise TypeError("Coworld manifest game must be an object")
    runnable = game["runnable"]
    if not isinstance(runnable, dict):
        raise TypeError("Coworld manifest game.runnable must be an object")

    runnables = [runnable]
    for section in RUNNABLE_LIST_SECTIONS:
        if section not in manifest:
            continue
        items = manifest[section]
        if not isinstance(items, list):
            raise TypeError(f"Coworld manifest {section} must be a list")
        for item in items:
            if not isinstance(item, dict):
                raise TypeError(f"Coworld manifest {section} items must be objects")
            runnables.append(item)

    for item in runnables:
        if not isinstance(item["image"], str):
            raise TypeError("Coworld runnable image must be a string")
    return runnables


def _upload_container_image(client: CoworldUploadClient, image: str) -> ContainerImageResponse:
    client_hash = _local_image_client_hash(image)
    response = client.request_image_upload(name=_image_upload_name(image), client_hash=client_hash)

    if response.pre_signed_info is not None:
        _push_container_image(image, response.pre_signed_info)
        completed = client.complete_image_upload(response.image.id)
    else:
        completed = response.image

    if completed.image_uri is None:
        raise RuntimeError(f"Softmax image upload did not return an executable image URI for {image}")
    return completed


def _local_image_client_hash(image: str) -> str:
    with tempfile.TemporaryFile() as archive:
        subprocess.run(["docker", "image", "save", image], check=True, stdout=archive)
        archive.seek(0)
        return _docker_archive_client_hash(archive)


def _docker_archive_client_hash(archive: Any) -> str:
    with tarfile.open(fileobj=archive, mode="r:*") as tar:
        manifest_json = _read_tar_member(tar, "manifest.json")
        manifest = json.loads(manifest_json)
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise RuntimeError("Docker image archive must contain exactly one image manifest")
        image_manifest = manifest[0]
        if not isinstance(image_manifest, dict):
            raise RuntimeError("Docker image archive manifest entry must be an object")

        config_name = image_manifest["Config"]
        layers = image_manifest["Layers"]
        if not isinstance(config_name, str) or not isinstance(layers, list):
            raise RuntimeError("Docker image archive manifest has invalid config or layers")

        layer_hashes = []
        for layer in layers:
            if not isinstance(layer, str):
                raise RuntimeError("Docker image archive manifest layers must be strings")
            layer_hashes.append(_sha256_digest(_read_tar_member(tar, layer)))

        content = {
            "config": _sha256_digest(_read_tar_member(tar, config_name)),
            "layers": layer_hashes,
        }
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_digest(encoded)


def _read_tar_member(tar: tarfile.TarFile, name: str) -> bytes:
    member = tar.extractfile(name)
    if member is None:
        raise RuntimeError(f"Docker image archive is missing {name}")
    return member.read()


def _sha256_digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _push_container_image(source_image: str, push_info: EcrPushInfo) -> None:
    aws_env = os.environ | {
        "AWS_ACCESS_KEY_ID": push_info.credentials.access_key_id,
        "AWS_SECRET_ACCESS_KEY": push_info.credentials.secret_access_key,
        "AWS_SESSION_TOKEN": push_info.credentials.session_token,
    }
    aws_env.pop("AWS_PROFILE", None)
    login_command = ["aws", "ecr", "get-login-password", "--region", push_info.region]
    if push_info.endpoint_url is not None:
        login_command.extend(["--endpoint-url", push_info.endpoint_url])
    password = subprocess.run(
        login_command,
        env=aws_env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    with tempfile.TemporaryDirectory(prefix="coworld-docker-config-") as docker_config_dir:
        auth = b64encode(f"AWS:{password.strip()}".encode()).decode()
        (Path(docker_config_dir) / "config.json").write_text(
            json.dumps({"auths": {push_info.registry: {"auth": auth}}}) + "\n"
        )
        docker_env = os.environ | {"DOCKER_CONFIG": docker_config_dir}
        subprocess.run(["docker", "tag", source_image, push_info.image_uri], check=True)
        subprocess.run(["docker", "push", push_info.image_uri], env=docker_env, check=True)


def _image_upload_name(image: str) -> str:
    name = image.rsplit("/", 1)[-1].split("@", 1)[0].split(":", 1)[0]
    return name or "coworld-image"


# --- Documentation inlining ---


def _is_url(path: str) -> bool:
    return path.startswith("https://")


def _manifest_doc_paths(manifest: dict[str, object]) -> list[tuple[list[str | int], str]]:
    """Return (json_path, value) pairs for all local doc path fields in the manifest."""
    paths: list[tuple[list[str | int], str]] = []
    game = manifest.get("game")
    if not isinstance(game, dict):
        return paths

    protocols = game.get("protocols")
    if isinstance(protocols, dict):
        for key in ("player", "global"):
            val = protocols.get(key)
            if isinstance(val, str) and not _is_url(val):
                paths.append((["game", "protocols", key], val))

    docs = game.get("docs")
    if isinstance(docs, dict):
        readme = docs.get("readme")
        if isinstance(readme, str) and not _is_url(readme):
            paths.append((["game", "docs", "readme"], readme))
        pages = docs.get("pages")
        if isinstance(pages, list):
            for i, page in enumerate(pages):
                if isinstance(page, dict):
                    page_content = page.get("content")
                    if isinstance(page_content, str) and not _is_url(page_content):
                        paths.append((["game", "docs", "pages", i, "content"], page_content))

    return paths


def _manifest_with_inlined_docs(
    manifest: dict[str, object],
    manifest_dir: Path,
) -> dict[str, object]:
    """Read local markdown files referenced in the manifest and inline their content."""
    doc_paths = _manifest_doc_paths(manifest)
    if not doc_paths:
        return manifest

    upload_manifest = copy.deepcopy(manifest)

    for json_path, local_path in doc_paths:
        file_path = resolve_manifest_uri(manifest_dir, local_path)
        content = file_path.read_text(encoding="utf-8")

        target: Any = upload_manifest
        for key in json_path[:-1]:
            target = target[key]
        target[json_path[-1]] = content

    return upload_manifest
