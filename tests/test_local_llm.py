"""Tests for the local LLM inference integration.

These tests do NOT load an actual model — they verify that:
- LocalLLMInference raises a clear error when the model path is missing.
- LLMMinerPlannerClient routes to local inference when LOCAL_LLM_MODEL_PATH is set.
- LLMMinerPlannerClient falls back to the responder/OpenRouter when it is not set.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

from cogames.policy.local_llm import LocalLLMInference
from cogames.policy.llm_miner_policy import LLMMinerPlannerClient


# ── LocalLLMInference unit tests ───────────────────────────────────────────────


def test_local_llm_inference_raises_when_path_missing(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="LOCAL_LLM_MODEL_PATH"):
        LocalLLMInference()


def test_local_llm_inference_accepts_explicit_path():
    inf = LocalLLMInference("/some/path")
    assert inf._model_path == "/some/path"
    assert inf._pipeline is None  # not yet loaded


def test_local_llm_inference_accepts_env_var(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MODEL_PATH", "/env/path")
    inf = LocalLLMInference()
    assert inf._model_path == "/env/path"


# ── LLMMinerPlannerClient routing tests ────────────────────────────────────────


def test_planner_uses_local_inference_when_path_set(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MODEL_PATH", "/fake/model")

    fake_response = '{"skill": "explore", "reason": "testing"}'

    # Patch LocalLLMInference at its definition site; LLMMinerPlannerClient
    # imports it lazily inside __init__ via a local import.
    mock_inference = MagicMock()
    mock_inference.complete.return_value = fake_response

    with patch("cogames.policy.local_llm.LocalLLMInference", return_value=mock_inference):
        client = LLMMinerPlannerClient()

    assert client._local_inference is mock_inference
    result = client.complete("some prompt")
    mock_inference.complete.assert_called_once_with("some prompt")
    assert result == fake_response


def test_planner_uses_responder_over_local_inference(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MODEL_PATH", "/fake/model")

    responder_called = []

    def my_responder(prompt: str) -> str:
        responder_called.append(prompt)
        return '{"skill": "gear_up", "reason": "responder"}'

    mock_inference = MagicMock()
    with patch("cogames.policy.local_llm.LocalLLMInference", return_value=mock_inference):
        client = LLMMinerPlannerClient(responder=my_responder)

    result = client.complete("hello")
    # responder takes precedence over local inference
    assert result == '{"skill": "gear_up", "reason": "responder"}'
    mock_inference.complete.assert_not_called()


def test_planner_no_local_path_no_model_raises(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_MODEL_PATH", raising=False)
    client = LLMMinerPlannerClient(model=None, api_url=None)
    with pytest.raises(RuntimeError, match="not configured"):
        client.complete("test")
