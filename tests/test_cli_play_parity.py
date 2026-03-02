import random
from types import SimpleNamespace

import numpy as np
import torch
from typer.testing import CliRunner

import cogames.main as main_module

runner = CliRunner()


def _stub_mission(*_args, **kwargs):  # type: ignore[no-untyped-def]
    env_cfg = SimpleNamespace(game=SimpleNamespace(max_steps=10000, num_agents=1, map_builder=None))
    return "dummy.mission", env_cfg, None


def _patch_play_dependencies(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(main_module, "get_mission_name_and_config", _stub_mission)
    monkeypatch.setattr(main_module, "get_policy_spec", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(main_module, "resolve_training_device", lambda *_args, **_kwargs: torch.device("cpu"))


def test_play_forwards_steps_only_when_explicit(monkeypatch) -> None:
    captured: dict[str, int | None] = {}

    def _capture_steps(*_args, **kwargs):  # type: ignore[no-untyped-def]
        captured["steps"] = kwargs.get("steps")
        return _stub_mission()

    monkeypatch.setattr(main_module, "get_mission_name_and_config", _capture_steps)
    monkeypatch.setattr(main_module, "get_policy_spec", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(main_module, "resolve_training_device", lambda *_args, **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(main_module.play_module, "play", lambda *_args, **_kwargs: None)

    explicit = runner.invoke(main_module.app, ["play", "-m", "dummy", "-s", "1000", "--render", "none"])
    assert explicit.exit_code == 0, explicit.output
    assert captured["steps"] == 1000

    default = runner.invoke(main_module.app, ["play", "-m", "dummy", "--render", "none"])
    assert default.exit_code == 0, default.output
    assert captured["steps"] is None


def test_play_seeds_global_rng_with_seed_flag(monkeypatch) -> None:
    samples: list[tuple[float, float, float]] = []
    _patch_play_dependencies(monkeypatch)

    def _capture_rng(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        samples.append((random.random(), float(np.random.rand()), float(torch.rand(1).item())))

    monkeypatch.setattr(main_module.play_module, "play", _capture_rng)

    first = runner.invoke(main_module.app, ["play", "-m", "dummy", "--render", "none", "--seed", "42"])
    second = runner.invoke(main_module.app, ["play", "-m", "dummy", "--render", "none", "--seed", "42"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert len(samples) == 2
    assert samples[0] == samples[1]


def test_pickup_forwards_steps_without_posthoc_override(monkeypatch) -> None:
    captured_steps: dict[str, int | None] = {}
    env_cfg = SimpleNamespace(game=SimpleNamespace(max_steps=777))

    def _capture_mission(*_args, **kwargs):  # type: ignore[no-untyped-def]
        captured_steps["steps"] = kwargs.get("steps")
        return "dummy.mission", env_cfg, None

    class _DummyParsedPolicy:
        def to_policy_spec(self):  # type: ignore[no-untyped-def]
            return object()

    monkeypatch.setattr(main_module, "get_mission_name_and_config", _capture_mission)
    monkeypatch.setattr(main_module, "resolve_training_device", lambda *_args, **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(main_module, "get_policy_spec", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(main_module, "parse_policy_spec", lambda *_args, **_kwargs: _DummyParsedPolicy())
    monkeypatch.setattr(main_module.pickup_module, "pickup", lambda *_args, **_kwargs: None)

    result = runner.invoke(
        main_module.app,
        [
            "pickup",
            "-m",
            "dummy",
            "-p",
            "candidate",
            "--pool",
            "pool",
            "--steps",
            "123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_steps["steps"] == 123
    # pickup should rely on mission resolution for steps handling, not mutate max_steps afterward.
    assert env_cfg.game.max_steps == 777
