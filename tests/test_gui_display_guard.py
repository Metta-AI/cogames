from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import cogames.cli.episode as episode_module
import cogames.display_detect as display_detect
import cogames.main as main_module

runner = CliRunner()


def test_tutorial_play_requires_display(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "has_display", lambda: False)

    result = runner.invoke(main_module.app, ["tutorial", "play"])

    assert result.exit_code == 1
    assert "This command requires a GUI display" in result.stdout


def test_tutorial_cvc_requires_display(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "has_display", lambda: False)

    result = runner.invoke(main_module.app, ["tutorial", "cvc"])

    assert result.exit_code == 1
    assert "This command requires a GUI display" in result.stdout


def test_macos_display_env_without_session_is_false(monkeypatch) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(display_detect, "_macos_session_has_display", lambda: False)
    monkeypatch.setattr(display_detect, "_macos_launchctl_has_display", lambda: False)

    assert display_detect._macos_has_display() is False


def test_macos_display_uses_session_check(monkeypatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(display_detect, "_macos_session_has_display", lambda: True)
    monkeypatch.setattr(display_detect, "_macos_launchctl_has_display", lambda: False)

    assert display_detect._macos_has_display() is True


def test_macos_display_uses_launchctl_fallback(monkeypatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(display_detect, "_macos_session_has_display", lambda: False)
    monkeypatch.setattr(display_detect.shutil, "which", lambda _: "/usr/bin/launchctl")
    monkeypatch.setattr(
        display_detect.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="gui/501 = {\n\tsession = Aqua\n}",
            stderr="",
        ),
    )

    assert display_detect._macos_has_display() is True


def test_play_gui_requires_display_before_loading_mission(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "has_display", lambda: False)
    monkeypatch.setattr(
        main_module,
        "get_mission_name_and_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve mission")),
    )

    result = runner.invoke(main_module.app, ["play", "-m", "arena", "--render", "gui"])

    assert result.exit_code == 1
    assert "This render mode requires a GUI display" in result.stdout


def test_play_auto_uses_unicode_without_display(monkeypatch) -> None:
    captured: dict[str, object] = {}
    env_cfg = SimpleNamespace(game=SimpleNamespace(max_steps=10))
    mission_cfg = object()

    monkeypatch.setattr(main_module, "has_display", lambda: False)
    monkeypatch.setattr(
        main_module,
        "get_mission_name_and_config",
        lambda *_args, **_kwargs: ("arena", env_cfg, mission_cfg),
    )
    monkeypatch.setattr(main_module, "resolve_training_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(main_module, "get_policy_spec", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        main_module.play_module,
        "play",
        lambda *_args, **kwargs: captured.update(kwargs),
    )

    result = runner.invoke(main_module.app, ["play", "-m", "arena"])

    assert result.exit_code == 0
    assert captured["render_mode"] == "unicode"
    assert "Render: unicode" in result.stdout


def test_play_auto_uses_gui_with_display(monkeypatch) -> None:
    captured: dict[str, object] = {}
    env_cfg = SimpleNamespace(game=SimpleNamespace(max_steps=10))
    mission_cfg = object()

    monkeypatch.setattr(main_module, "has_display", lambda: True)
    monkeypatch.setattr(
        main_module,
        "get_mission_name_and_config",
        lambda *_args, **_kwargs: ("arena", env_cfg, mission_cfg),
    )
    monkeypatch.setattr(main_module, "resolve_training_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(main_module, "get_policy_spec", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        main_module.play_module,
        "play",
        lambda *_args, **kwargs: captured.update(kwargs),
    )

    result = runner.invoke(main_module.app, ["play", "-m", "arena"])

    assert result.exit_code == 0
    assert captured["render_mode"] == "gui"
    assert "Render: gui" in result.stdout


def test_replay_requires_display_before_viewer_launch(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "test.replay"
    replay_path.write_text("replay")
    monkeypatch.setattr(main_module, "has_display", lambda: False)
    monkeypatch.setattr(
        main_module,
        "launch_replay_path",
        lambda _request: (_ for _ in ()).throw(AssertionError("should not launch replay viewer")),
    )

    result = runner.invoke(main_module.app, ["replay", str(replay_path)])

    assert result.exit_code == 1
    assert "This command requires a GUI display" in result.stdout


def test_replay_delegates_to_replay_viewer_contract(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "test.bitreplay"
    replay_path.write_bytes(b"BITWORLD\x02\x00")
    launched: list[object] = []
    monkeypatch.setattr(main_module, "has_display", lambda: True)
    monkeypatch.setattr(
        main_module,
        "launch_replay_path",
        lambda request: launched.append(request.replay_path) or launched.append(request.duration) or 0,
    )

    result = runner.invoke(main_module.app, ["replay", str(replay_path), "--duration", "1"])

    assert result.exit_code == 0
    assert launched == [replay_path, 1.0]


class _FakeClient:
    def __init__(self, ep: SimpleNamespace):
        self._ep = ep

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def get_episode(self, _ep_uuid: uuid.UUID) -> SimpleNamespace:
        return self._ep


class _FakeHttpResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_episode_replay_requires_display_before_launch(monkeypatch) -> None:
    episode_id = uuid.uuid4()
    monkeypatch.setattr(episode_module, "has_display", lambda: False)
    monkeypatch.setattr(
        episode_module,
        "_get_anon_client",
        lambda _server: _FakeClient(SimpleNamespace(id=episode_id, replay_url="https://example.com/replay")),
    )
    monkeypatch.setattr(
        episode_module.httpx,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not download replay")),
    )
    monkeypatch.setattr(
        episode_module,
        "launch_replay_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not launch replay viewer")),
    )

    result = runner.invoke(main_module.app, ["episode", "replay", str(episode_id)])

    assert result.exit_code == 1
    assert "This command requires a GUI display" in result.stdout


def test_episode_replay_delegates_to_replay_viewer_contract(monkeypatch) -> None:
    episode_id = uuid.uuid4()
    launched: list[object] = []
    monkeypatch.setattr(episode_module, "has_display", lambda: True)
    monkeypatch.setattr(
        episode_module,
        "_get_anon_client",
        lambda _server: _FakeClient(SimpleNamespace(id=episode_id, replay_url="https://example.com/replay")),
    )
    monkeypatch.setattr(episode_module.httpx, "get", lambda *_args, **_kwargs: _FakeHttpResponse(b"BITWORLD\x02\x00"))
    monkeypatch.setattr(
        episode_module,
        "launch_replay_bytes",
        lambda replay_bytes, *, prefix: launched.append(replay_bytes) or launched.append(prefix) or 0,
    )

    result = runner.invoke(main_module.app, ["episode", "replay", str(episode_id)])

    assert result.exit_code == 0
    assert launched == [b"BITWORLD\x02\x00", f"episode-{str(episode_id)[:8]}-"]
