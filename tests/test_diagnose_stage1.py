import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cogames.diagnose as diagnose_module
import cogames.main as main_module
from mettagrid.simulator.multi_episode.summary import MultiEpisodeRolloutPolicySummary, MultiEpisodeRolloutSummary

runner = CliRunner()

_STAGE1_PROBE_MISSIONS = (
    "eval_balanced_spread",
    "eval_collect_resources",
    "eval_divide_and_conquer",
)


def _stage1_case_names(*, cogs: int = 8, mission_set: str = "cogsguard_evals") -> list[str]:
    return [f"{mission_set}.{mission} (cogs={cogs})" for mission in _STAGE1_PROBE_MISSIONS]


def _diagnose_case(name: str) -> diagnose_module.DiagnoseCase:
    return diagnose_module.DiagnoseCase(name=name, env_cfg=None)  # type: ignore[arg-type]


def _stage1_pack_cases() -> list[diagnose_module.DiagnoseCase]:
    return [_diagnose_case(case_name) for case_name in _stage1_case_names(cogs=1)]


def _run_diagnose(*, output_dir: Path, mission_set: str, extra_args: list[str] | None = None):
    args = [
        "diagnose",
        "class=random",
        "--mission-set",
        mission_set,
        "--output-dir",
        str(output_dir),
        "--episodes",
        "1",
        "--steps",
        "10",
    ]
    if extra_args:
        args.extend(extra_args)
    return runner.invoke(main_module.app, args)


def _read_state(output_dir: Path) -> diagnose_module.DiagnoseRunState:
    return diagnose_module.DiagnoseRunState.model_validate_json((output_dir / "diagnose_state.json").read_text())


def _patch_evaluate_to_fail(monkeypatch: pytest.MonkeyPatch, *, message: str) -> None:
    def _should_not_run_evaluate(*_args, **_kwargs):
        raise AssertionError(message)

    monkeypatch.setattr(diagnose_module.evaluate_module, "evaluate", _should_not_run_evaluate)


def _patch_diagnose_runtime(monkeypatch: pytest.MonkeyPatch, *, evaluate_fn) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(diagnose_module.evaluate_module, "evaluate", evaluate_fn)
    monkeypatch.setattr(diagnose_module, "resolve_training_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(diagnose_module, "get_policy_spec", lambda *_args, **_kwargs: object())


def _make_summary(*, episodes: int, num_policies: int) -> MultiEpisodeRolloutSummary:
    policy_summaries = [
        MultiEpisodeRolloutPolicySummary(
            agent_count=1,
            avg_agent_metrics={
                "action.move.success": 0.9,
                "action.failed": 0.05,
                "status.max_steps_without_motion": 2.0,
            },
            action_timeouts=0,
        )
        for _ in range(num_policies)
    ]
    rewards: list[float | None] = [1.0, *([0.8] * max(0, num_policies - 1))]
    return MultiEpisodeRolloutSummary(
        episodes=episodes,
        policy_summaries=policy_summaries,
        avg_game_stats={"aligned.junction.held": 1.0},
        per_episode_per_policy_avg_rewards=dict.fromkeys(range(episodes), rewards),
    )


def _fake_evaluate_without_replays(*_args, **kwargs):  # type: ignore[no-untyped-def]
    missions = kwargs["missions"]
    policy_specs = kwargs["policy_specs"]
    episodes = kwargs["episodes"]
    return [_make_summary(episodes=episodes, num_policies=len(policy_specs)) for _mission_name, _cfg in missions]


def _fake_evaluate_with_replays(*_args, **kwargs):  # type: ignore[no-untyped-def]
    missions = kwargs["missions"]
    policy_specs = kwargs["policy_specs"]
    episodes = kwargs["episodes"]
    save_replay = kwargs.get("save_replay")
    if save_replay is not None:
        replay_dir = Path(save_replay)
        replay_dir.mkdir(parents=True, exist_ok=True)
        for mission_name, _mission_cfg in missions:
            replay_name = mission_name.replace("/", "_")
            for episode_idx in range(episodes):
                (replay_dir / f"{replay_name}_ep{episode_idx}.json.z").write_text("{}", encoding="utf-8")
    return [_make_summary(episodes=episodes, num_policies=len(policy_specs)) for _mission_name, _cfg in missions]


def test_evaluate_stage1_gate_marks_missing_axes() -> None:
    case_names = [_stage1_case_names()[0]]
    gate = diagnose_module.evaluate_stage1_gate(
        case_names=case_names,
        pack=diagnose_module.COGSGUARD_STAGE1_PACK_V1,
    )

    assert not gate.satisfied
    results_by_axis = {result.axis: result for result in gate.results}
    assert results_by_axis[diagnose_module.DiagnoseAxis.STABILITY].satisfied
    assert not results_by_axis[diagnose_module.DiagnoseAxis.EFFICIENCY].satisfied
    assert not results_by_axis[diagnose_module.DiagnoseAxis.CONTROL].satisfied


def test_assess_stage1_signals_confirms_with_metrics_and_replays() -> None:
    gate = diagnose_module.evaluate_stage1_gate(
        case_names=_stage1_case_names(),
        pack=diagnose_module.COGSGUARD_STAGE1_PACK_V1,
    )
    metrics_by_mission = {
        mission: [
            diagnose_module.Stage1MissionMetrics(
                mission_name=mission,
                reward_variance=0.0,
                non_zero_episode_pct=0.0,
                timeout_rate=0.0,
                mean_move_success=1.0,
                mean_action_failed=0.0,
                mean_stuck_steps=2.0,
            )
        ]
        for mission in _STAGE1_PROBE_MISSIONS
    }

    signals = diagnose_module.assess_stage1_signals(
        requirement_results=gate.results,
        metrics_by_mission=metrics_by_mission,
        replay_refs=["replays/example.json.z"],
    )

    assert all(signal.confirmed for signal in signals)
    assert all(signal.metric_refs for signal in signals)
    assert all(signal.replay_refs for signal in signals)


def test_write_replay_bundle_includes_replays_or_writes_readme(tmp_path: Path) -> None:
    bundle_path = diagnose_module.write_replay_bundle(tmp_path)
    with zipfile.ZipFile(bundle_path, "r") as bundle:
        assert bundle.namelist() == ["README.txt"]

    (tmp_path / "replays").mkdir(parents=True, exist_ok=True)
    (tmp_path / "replays" / "episode_0.json.z").write_text("{}", encoding="utf-8")
    bundle_path = diagnose_module.write_replay_bundle(tmp_path)
    with zipfile.ZipFile(bundle_path, "r") as bundle:
        assert "replays/episode_0.json.z" in bundle.namelist()


def test_diagnose_cli_marks_incomplete_when_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [_diagnose_case("diagnostic_evals.diagnostic_chest_navigation1 (cogs=1)")]
    monkeypatch.setattr(diagnose_module, "_build_diagnose_cases", lambda **_kwargs: cases)
    _patch_evaluate_to_fail(monkeypatch, message="evaluate() should not run when Stage 1 gate fails")

    result = _run_diagnose(output_dir=tmp_path, mission_set="diagnostic_evals")

    assert result.exit_code == 0, result.output
    state = _read_state(tmp_path)
    assert state.stage_status == diagnose_module.DiagnoseStageStatus.STAGE1_INCOMPLETE
    assert state.run_status == diagnose_module.DiagnoseRunStatus.INCOMPLETE
    assert state.missing_requirements

    for relative_path in ("diagnose_state.json", "doctor_note.json", "diagnose_report.html", "manifest.json"):
        assert (tmp_path / relative_path).exists()


def test_diagnose_cli_reused_output_dir_does_not_count_stale_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    stale_replay_dir = output_dir / "replays"
    stale_replay_dir.mkdir(parents=True, exist_ok=True)
    for episode_idx in range(3):
        (stale_replay_dir / f"stale_{episode_idx}.json.z").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(diagnose_module, "_build_diagnose_cases", lambda **_kwargs: _stage1_pack_cases())
    _patch_diagnose_runtime(monkeypatch, evaluate_fn=_fake_evaluate_without_replays)

    result = _run_diagnose(output_dir=output_dir, mission_set="cogsguard_evals")

    assert result.exit_code == 0, result.output
    assert not list(stale_replay_dir.glob("*.json.z"))
    state = _read_state(output_dir)
    assert state.stage_status == diagnose_module.DiagnoseStageStatus.STAGE1_INCOMPLETE
    assert state.run_status == diagnose_module.DiagnoseRunStatus.INCOMPLETE


def test_diagnose_cli_writes_portable_replay_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    monkeypatch.setattr(diagnose_module, "_build_diagnose_cases", lambda **_kwargs: _stage1_pack_cases())
    _patch_diagnose_runtime(monkeypatch, evaluate_fn=_fake_evaluate_with_replays)

    result = _run_diagnose(output_dir=output_dir, mission_set="cogsguard_evals")

    assert result.exit_code == 0, result.output
    doctor_note = diagnose_module.load_doctor_note(output_dir / "doctor_note.json")
    replay_refs = doctor_note.evidence_index["replay_refs"]
    assert replay_refs
    assert all(str(output_dir) not in replay_ref for replay_ref in replay_refs)
    assert all(not Path(replay_ref).is_absolute() for replay_ref in replay_refs)


def test_diagnose_cli_completes_stage2_with_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    monkeypatch.setattr(diagnose_module, "_build_diagnose_cases", lambda **_kwargs: _stage1_pack_cases())
    _patch_diagnose_runtime(monkeypatch, evaluate_fn=_fake_evaluate_with_replays)

    result = _run_diagnose(output_dir=output_dir, mission_set="cogsguard_evals")

    assert result.exit_code == 0, result.output
    state = _read_state(output_dir)
    assert state.stage_status == diagnose_module.DiagnoseStageStatus.STAGE2_COMPLETED
    assert state.run_status == diagnose_module.DiagnoseRunStatus.COMPLETE

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["stage_status"] == "stage2_completed"
    assert manifest["run_status"] == "complete"
