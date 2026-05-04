import random
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

import cogames.main as main_module

runner = CliRunner()
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _normalize_cli_text(text: str) -> str:
    return " ".join(_ANSI_ESCAPE_RE.sub("", text).split())


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


def test_play_forwards_action_timeout_ms(monkeypatch) -> None:
    captured: dict[str, int] = {}
    _patch_play_dependencies(monkeypatch)

    def _capture_play(*_args, **kwargs):  # type: ignore[no-untyped-def]
        captured["action_timeout_ms"] = kwargs["action_timeout_ms"]

    monkeypatch.setattr(main_module.play_module, "play", _capture_play)

    result = runner.invoke(
        main_module.app,
        ["play", "-m", "dummy", "--render", "none", "--action-timeout-ms", "1234"],
    )

    assert result.exit_code == 0, result.output
    assert captured["action_timeout_ms"] == 1234


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


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["play", "--help"], "cogames play -m arena"),
        (["tutorial", "train", "--help"], "cogames tutorial train -m arena"),
        (["run", "--help"], "cogames run -m arena -p lstm"),
        (["scrimmage", "--help"], "cogames scrimmage -m arena -p lstm"),
    ],
)
def test_help_examples_use_valid_arena_mission(args: list[str], expected: str) -> None:
    result = runner.invoke(main_module.app, args)
    normalized_stdout = _normalize_cli_text(result.stdout)

    assert result.exit_code == 0, result.output
    assert expected in normalized_stdout
    assert "machina_1.basic" not in normalized_stdout
    assert "arena.battle" not in normalized_stdout


def test_make_policy_examples_use_valid_arena_mission(tmp_path) -> None:
    scripted_path = tmp_path / "my_scripted_policy.py"
    scripted = runner.invoke(
        main_module.app,
        ["tutorial", "make-policy", "--scripted", "-o", str(scripted_path)],
    )

    assert scripted.exit_code == 0, scripted.output
    assert "Play with: cogames play -m arena -p class=my_scripted_policy.StarterPolicy" in scripted.stdout
    assert "machina_1.basic" not in scripted.stdout

    trainable_path = tmp_path / "my_trainable_policy.py"
    trainable = runner.invoke(
        main_module.app,
        ["tutorial", "make-policy", "--trainable", "-o", str(trainable_path)],
    )

    assert trainable.exit_code == 0, trainable.output
    assert "Train with: cogames tutorial train -m arena -p" in trainable.stdout
    assert "class=my_trainable_policy.MyTrainablePolicy" in trainable.stdout
    assert "machina_1.basic" not in trainable.stdout

    amongthem_path = tmp_path / "amongthem_policy.py"
    amongthem = runner.invoke(
        main_module.app,
        ["tutorial", "make-policy", "--amongthem", "-o", str(amongthem_path)],
    )

    assert amongthem.exit_code == 0, amongthem.output
    assert "Dry-run validation: cogames upload -p class=amongthem_policy.AmongThemPolicy" in amongthem.stdout
    assert "Ship: cogames ship -p class=amongthem_policy.AmongThemPolicy" in amongthem.stdout
    assert "Score: cogames leaderboard <season> --policy $USER-amongthem-practice" in amongthem.stdout
    assert "Walkthrough: cogames docs amongthem_policy" in amongthem.stdout


def test_make_policy_rejects_unimportable_output_stem(tmp_path) -> None:
    output_path = tmp_path / "my-scripted-policy.py"
    result = runner.invoke(
        main_module.app,
        ["tutorial", "make-policy", "--scripted", "-o", str(output_path)],
    )

    assert result.exit_code == 1, result.output
    assert "is not importable as a Python module" in result.stdout
    assert not output_path.exists()


def test_amongthem_policy_walkthrough_is_packaged() -> None:
    result = runner.invoke(main_module.app, ["docs", "amongthem_policy"])

    assert result.exit_code == 0, result.output
    assert "AmongThem Policy Practice" in result.stdout
    assert "cogames tutorial make-policy --amongthem" in result.stdout
    assert "cogames leaderboard <season>" in result.stdout


def test_pickup_uses_arena_as_default_mission(monkeypatch) -> None:
    captured: dict[str, object] = {}
    env_cfg = SimpleNamespace(game=SimpleNamespace(max_steps=777))

    def _capture_mission(_ctx, mission, **kwargs):  # type: ignore[no-untyped-def]
        captured["mission"] = mission
        captured["steps"] = kwargs.get("steps")
        return "arena", env_cfg, None

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
            "-p",
            "candidate",
            "--pool",
            "pool",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["mission"] == "arena"
