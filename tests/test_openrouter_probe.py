import os

import httpx
from typer.testing import CliRunner

from cogames.main import app
from cogames.policy.llm_miner_policy import LLMMinerPlannerClient


def test_complete_strict_returns_responder_text() -> None:
    client = LLMMinerPlannerClient(responder=lambda _: '{"skill":"explore","reason":"ok"}')

    assert client.complete_strict("prompt") == '{"skill":"explore","reason":"ok"}'


def test_complete_preserves_fallback_behavior_when_openrouter_fails(monkeypatch) -> None:
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    client = LLMMinerPlannerClient(model="openai/gpt-4o-mini")

    def raise_connect_error(_prompt: str, _api_key: str | None) -> str:
        raise httpx.ConnectError("dns failed")

    monkeypatch.setattr(client, "_complete_openrouter", raise_connect_error)

    assert client.complete("prompt") == ""

    try:
        client.complete_strict("prompt")
    except httpx.ConnectError as exc:
        assert "dns failed" in str(exc)
    else:
        raise AssertionError("complete_strict should surface transport failures")


def test_openrouter_probe_command_prints_response(monkeypatch) -> None:
    runner = CliRunner()

    def fake_complete_strict(self, prompt: str) -> str:
        assert prompt == 'Return exactly {"skill":"explore","reason":"connectivity test"}'
        return '{"skill":"explore","reason":"ok"}'

    monkeypatch.setattr(LLMMinerPlannerClient, "complete_strict", fake_complete_strict)

    result = runner.invoke(app, ["openrouter-probe"])

    assert result.exit_code == 0, result.output
    assert '{"skill":"explore","reason":"ok"}' in result.output


def test_openrouter_probe_command_reports_failure(monkeypatch) -> None:
    runner = CliRunner()

    def fake_complete_strict(self, prompt: str) -> str:
        raise RuntimeError("Missing API key in environment variable OPENROUTER_API_KEY")

    monkeypatch.setattr(LLMMinerPlannerClient, "complete_strict", fake_complete_strict)

    result = runner.invoke(app, ["openrouter-probe"])

    assert result.exit_code == 1
    assert "OpenRouter probe failed" in result.output
    assert "Missing API key" in result.output
