"""Tests for run_evaluation policy device overrides."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_run_evaluation_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_evaluation.py"
    spec = importlib.util.spec_from_file_location("cogames_run_evaluation", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_evaluation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uri_policy_device_override_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    run_evaluation = _load_run_evaluation_module()

    def fake_policy_spec_from_uri(_uri: str, *, device: str = "cpu", remove_downloaded_copy_on_exit: bool = False):
        return run_evaluation.PolicySpec(class_path="policy", data_path=None, init_kwargs={})

    monkeypatch.setattr(run_evaluation, "policy_spec_from_uri", fake_policy_spec_from_uri)

    agent_config = run_evaluation.AgentConfig(
        key="test",
        label="test",
        policy_path="s3://bucket/run:v1",
    )

    spec = run_evaluation._resolve_policy_spec(agent_config, "cuda")

    assert spec.init_kwargs.get("device") == "cuda"
