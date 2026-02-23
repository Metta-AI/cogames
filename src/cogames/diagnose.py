from __future__ import annotations

import importlib
import json
import math
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from html import escape
from pathlib import Path
from typing import Callable, Literal, Optional

import typer
from pydantic import BaseModel

from cogames import evaluate as evaluate_module
from cogames.cli.base import console
from cogames.cli.policy import get_policy_spec, policy_arg_example
from cogames.cogs_vs_clips.mission import CvCMission, NumCogsVariant
from cogames.device import resolve_training_device
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.simulator.multi_episode.summary import MultiEpisodeRolloutSummary


class DiagnoseAxis(str, Enum):
    STABILITY = "stability"
    EFFICIENCY = "efficiency"
    CONTROL = "control"
    SOCIAL_COORDINATION = "social_coordination"


class DiagnoseStageStatus(str, Enum):
    STAGE1_RUNNING = "stage1_running"
    STAGE1_PASSED_FOR_SOCIAL = "stage1_passed_for_social"
    STAGE1_INCOMPLETE = "stage1_incomplete"
    STAGE2_RUNNING = "stage2_running"
    STAGE2_COMPLETED = "stage2_completed"
    STAGE2_INCOMPLETE = "stage2_incomplete"


class DiagnoseRunStatus(str, Enum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


class DiagnosisLifecycleStatus(str, Enum):
    DIAGNOSIS_COMPLETE = "diagnosis_complete"
    DIAGNOSIS_INCOMPLETE = "diagnosis_incomplete"


class DoctorNoteStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class DiagnoseDominantIssue(str, Enum):
    STABILITY = "stability"
    SPEED = "speed"
    STRATEGY = "strategy"
    SOCIAL_COORDINATION = "social_coordination"
    MIXED = "mixed"


class DiagnosePackRequirement(BaseModel):
    axis: DiagnoseAxis
    probe_missions: list[str]
    min_required: int = 1


class DiagnosePack(BaseModel):
    pack_id: str
    pack_version: str
    mission_set: str
    requirements: list[DiagnosePackRequirement]


class Stage1RequirementResult(BaseModel):
    axis: DiagnoseAxis
    satisfied: bool
    matched_missions: list[str]
    required_count: int
    accepted_probe_missions: list[str]


class DiagnoseRunConfig(BaseModel):
    run_id: str
    created_at: datetime
    output_dir: Path
    policy: str
    mission_set: str
    experiments: list[str]
    cogs: list[int]
    steps: int
    episodes: int
    pack_id: str
    pack_version: str
    scripted_baseline_policy: Optional[str] = None
    known_strong_policy: Optional[str] = None


class DiagnoseRunState(BaseModel):
    run_id: str
    created_at: datetime
    stage_status: DiagnoseStageStatus
    run_status: DiagnoseRunStatus
    diagnosis_status: Optional[DiagnosisLifecycleStatus] = None
    pack_id: str
    pack_version: str
    missing_requirements: list[Stage1RequirementResult]
    requirement_results: list[Stage1RequirementResult]
    expected_replay_count: int
    replay_count: int
    output_dir: str
    notes: list[str]
    stage1_signals: list["Stage1AxisSignal"]


class Stage1GateEvaluation(BaseModel):
    satisfied: bool
    results: list[Stage1RequirementResult]


class Stage1MissionMetrics(BaseModel):
    mission_name: str
    reward_variance: float
    non_zero_episode_pct: float
    timeout_rate: float
    mean_move_success: float
    mean_action_failed: float
    mean_stuck_steps: float


class Stage1AxisSignal(BaseModel):
    axis: DiagnoseAxis
    confirmed: bool
    metric_refs: list[str]
    replay_refs: list[str]
    summary: str


class Stage1ProbeDefinition(BaseModel):
    probe_id: str
    axis: DiagnoseAxis
    mission: str
    scenario: str
    question: str
    validation_metric: str
    pass_fail_threshold: str
    threshold_rules: list["Stage1ProbeThresholdRule"] = []


class Stage1ProbeThresholdRule(BaseModel):
    metric: str
    operator: str
    value: float


class Stage1ProbeEvaluation(BaseModel):
    probe_id: str
    axis: DiagnoseAxis
    mission: str
    passed: bool
    summary: str
    evidence_refs: list[str]


class Stage1ProbeThresholdProfile(BaseModel):
    profile_id: str
    probes: list[Stage1ProbeDefinition]


class Stage1MetricCorrelation(BaseModel):
    metric: str
    correlation: Optional[float] = None
    sample_count: int
    direction: str
    interpretation: str


class Stage1ProbeThresholdCalibration(BaseModel):
    probe_id: str
    mission: str
    metric: str
    operator: str
    current_threshold: float
    recommended_threshold: Optional[float] = None
    delta: Optional[float] = None
    sample_count: int
    notes: list[str] = []


class Stage1MetricCorrelationReport(BaseModel):
    objective_metric: str
    sample_count: int
    correlations: list[Stage1MetricCorrelation]
    probe_calibrations: list[Stage1ProbeThresholdCalibration]
    tuned: bool
    notes: list[str]


class Stage2Mode(str, Enum):
    ABSOLUTE = "absolute"
    MIRROR = "mirror"


class Stage2MissionMetrics(BaseModel):
    mission_name: str
    policy_reward_mean: float
    reward_variance: float
    non_zero_episode_pct: float
    timeout_rate: float
    mean_move_success: float
    mean_action_failed: float
    mean_stuck_steps: float


class Stage2ModeSummary(BaseModel):
    mode: Stage2Mode
    seed: int
    case_count: int
    mission_metrics: dict[str, list[Stage2MissionMetrics]]


class Stage2SocialSignal(BaseModel):
    confirmed: bool
    severity: float
    confidence: float
    summary: str
    evidence_refs: list[str]


class Stage2SocialProbeDefinition(BaseModel):
    probe_id: str
    mode: Stage2Mode
    scenario: str
    question: str
    validation_metric: str
    pass_fail_threshold: str


class Stage1DerivedMetrics(BaseModel):
    reward_variance: float
    non_zero_episode_pct: float
    timeout_rate: float


class TournamentObjectiveContext(BaseModel):
    aligned_junction_held_stage1: Optional[float] = None
    aligned_junction_held_stage2_absolute: Optional[float] = None
    aligned_junction_held_stage2_mirror: Optional[float] = None


class Stage1AxisDerivedMetrics(BaseModel):
    reward_variance: float
    non_zero_episode_pct: float
    timeout_rate: float
    mean_move_success: float
    mean_action_failed: float
    mean_stuck_steps: float


class Stage1AxisScore(BaseModel):
    axis: DiagnoseAxis
    normalized_score: float
    raw_score: float
    confirmed: bool
    derived_metrics: Stage1AxisDerivedMetrics
    metric_refs: list[str]
    replay_refs: list[str]
    baseline_refs: list[str] = []
    baseline_coverage_count: int = 0
    normalization_mode: str = "heuristic_thresholds"


class Stage1BaselinePolicySummary(BaseModel):
    role: str
    policy: str
    axis_scores: list[Stage1AxisScore]


class Stage1BaselineContext(BaseModel):
    normalization_mode: str
    normalization_notes: list[str]
    baselines: list[Stage1BaselinePolicySummary]


class ReplayExemplarRefs(BaseModel):
    best: Optional[str] = None
    worst: Optional[str] = None
    most_diagnostic: Optional[str] = None


class DiagnoseSymptomEvidenceRefs(BaseModel):
    metric_refs: list[str]
    replay_refs: list[str]


class DiagnoseSymptom(BaseModel):
    symptom_id: str
    axis: DiagnoseAxis
    severity: float
    confidence: float
    evidence_refs: DiagnoseSymptomEvidenceRefs
    likely_cause: str
    action: str
    expected_effect: str


class DiagnosePrescription(BaseModel):
    symptom_id: str
    action: str
    owner: str
    validation_metric: str
    pass_fail_threshold: str


class Stage2DiagnosisDelta(BaseModel):
    stage1_dominant_issue: DiagnoseDominantIssue
    final_dominant_issue: DiagnoseDominantIssue
    changed: bool
    summary: str
    evidence_refs: list[str]


class DiagnoseDoctorNote(BaseModel):
    schema_version: str
    run_id: str
    status: DoctorNoteStatus
    stage_status: DiagnoseStageStatus
    diagnosis_status: Optional[DiagnosisLifecycleStatus] = None
    dominant_issue: DiagnoseDominantIssue
    axes: list[Stage1AxisScore]
    stage1_probe_threshold_profile_id: str = "cogsguard_stage1_probe_thresholds_v1"
    stage1_probe_catalog: list[Stage1ProbeDefinition] = []
    stage1_probe_evaluations: list[Stage1ProbeEvaluation] = []
    tournament_objective_context: TournamentObjectiveContext = TournamentObjectiveContext()
    baseline_context: Stage1BaselineContext
    derived_metrics: Stage1DerivedMetrics
    social_review: Optional[Stage2SocialSignal] = None
    social_probe_catalog: list[Stage2SocialProbeDefinition] = []
    stage2_diagnosis_delta: Optional[Stage2DiagnosisDelta] = None
    replay_exemplars: ReplayExemplarRefs = ReplayExemplarRefs()
    symptoms: list[DiagnoseSymptom]
    prescriptions: list[DiagnosePrescription]
    evidence_index: dict[str, list[str]]
    missing_requirements: list[Stage1RequirementResult]
    notes: list[str]


class InterpretationSnapshot(BaseModel):
    label: str
    run_id: str
    dominant_issue: DiagnoseDominantIssue
    top_symptom_ids: list[str]


class InterpretationStabilityReport(BaseModel):
    snapshot_count: int
    compared_run_ids: list[str]
    dominant_issue_stable: bool
    top_symptom_stable: bool
    stable: bool
    notes: list[str]
    snapshots: list[InterpretationSnapshot]


class DiagnoseValidityCheck(BaseModel):
    check_id: str
    passed: bool
    details: str


class DiagnoseValidityReport(BaseModel):
    valid: bool
    failed_check_ids: list[str]
    checks: list[DiagnoseValidityCheck]


class DiagnosePackContractCheck(BaseModel):
    check_id: str
    passed: bool
    details: str


class DiagnosePackContractReport(BaseModel):
    valid: bool
    checks: list[DiagnosePackContractCheck]


class DiagnoseManifest(BaseModel):
    schema_version: str
    run_id: str
    created_at: datetime
    command: str
    git_sha: Optional[str]
    pack_id: str
    pack_version: str
    mission_set: str
    policy: str
    scripted_baseline_policy: Optional[str]
    known_strong_policy: Optional[str]
    stage_status: DiagnoseStageStatus
    run_status: DiagnoseRunStatus
    seeds: dict[str, int]
    artifact_files: list[str]
    interpretation_stability: InterpretationStabilityReport
    diagnose_validity: DiagnoseValidityReport


COGSGUARD_STAGE1_PACK_V1 = DiagnosePack(
    pack_id="cogsguard_stage1",
    pack_version="v1",
    mission_set="cogsguard_evals",
    requirements=[
        DiagnosePackRequirement(
            axis=DiagnoseAxis.STABILITY,
            probe_missions=["eval_balanced_spread", "eval_single_use_world"],
            min_required=1,
        ),
        DiagnosePackRequirement(
            axis=DiagnoseAxis.EFFICIENCY,
            probe_missions=["eval_collect_resources", "eval_collect_resources_medium"],
            min_required=1,
        ),
        DiagnosePackRequirement(
            axis=DiagnoseAxis.CONTROL,
            probe_missions=["eval_divide_and_conquer", "eval_clip_oxygen"],
            min_required=1,
        ),
    ],
)


COGSGUARD_STAGE1_FIXED_COGS = [1, 2, 4]
COGSGUARD_STAGE1_FIXED_STEPS = 1000
COGSGUARD_STAGE1_FIXED_EPISODES = 3


STAGE1_PROBE_THRESHOLD_PROFILE_ID = "cogsguard_stage1_probe_thresholds_v1"


STAGE1_PROBE_CATALOG_V1: list[Stage1ProbeDefinition] = [
    Stage1ProbeDefinition(
        probe_id="stage1.food_under_pressure",
        axis=DiagnoseAxis.STABILITY,
        mission="eval_balanced_spread",
        scenario="High-value resource appears under hostile pressure with opportunity cost.",
        question="Does the policy stay stable and opportunistic without collapsing under pressure?",
        validation_metric="reward_variance + timeout_rate",
        pass_fail_threshold="reward_variance <= 1.0 and timeout_rate <= 0.05",
        threshold_rules=[
            Stage1ProbeThresholdRule(metric="reward_variance", operator="<=", value=1.0),
            Stage1ProbeThresholdRule(metric="timeout_rate", operator="<=", value=0.05),
        ],
    ),
    Stage1ProbeDefinition(
        probe_id="stage1.heart_lever_diversion",
        axis=DiagnoseAxis.EFFICIENCY,
        mission="eval_collect_resources",
        scenario="Resource sits behind an indirect unlock path; policy must choose lever route vs camping.",
        question="Does the policy discover and execute indirect paths efficiently under competing priorities?",
        validation_metric="action.move.success + status.max_steps_without_motion",
        pass_fail_threshold="action.move.success >= 0.70 and stuck_steps <= 30",
        threshold_rules=[
            Stage1ProbeThresholdRule(metric="mean_move_success", operator=">=", value=0.7),
            Stage1ProbeThresholdRule(metric="mean_stuck_steps", operator="<=", value=30.0),
        ],
    ),
    Stage1ProbeDefinition(
        probe_id="stage1.junction_light_shift",
        axis=DiagnoseAxis.CONTROL,
        mission="eval_divide_and_conquer",
        scenario="Priority target shifts mid-episode requiring fast control-policy retargeting.",
        question="Does the policy adapt quickly without wrong-target persistence?",
        validation_metric="action.failed + timeout_rate",
        pass_fail_threshold="action.failed <= 0.20 and timeout_rate <= 0.05",
        threshold_rules=[
            Stage1ProbeThresholdRule(metric="mean_action_failed", operator="<=", value=0.2),
            Stage1ProbeThresholdRule(metric="timeout_rate", operator="<=", value=0.05),
        ],
    ),
]


STAGE2_SOCIAL_PROBE_CATALOG_V1: list[Stage2SocialProbeDefinition] = [
    Stage2SocialProbeDefinition(
        probe_id="stage2.timed_bridge_window",
        mode=Stage2Mode.ABSOLUTE,
        scenario="Teammate opens a bridge window for 5 seconds; agent must cross and complete follow-up objective.",
        question="Does behavior remain coordinated and timely when dependent on teammate action windows?",
        validation_metric="timed_bridge.success_rate",
        pass_fail_threshold=">= 0.70 success across fixed-seed episodes",
    ),
    Stage2SocialProbeDefinition(
        probe_id="stage2.corridor_interference",
        mode=Stage2Mode.MIRROR,
        scenario="Two allied agents share a choke corridor; evaluate yielding behavior vs deadlock time.",
        question="Does policy avoid blocking teammates and maintain throughput under shared-path contention?",
        validation_metric="corridor.deadlock_time",
        pass_fail_threshold="<= 5% of episode steps spent in deadlock",
    ),
]


def validate_pack_definition(pack: DiagnosePack) -> None:
    seen_axes: set[DiagnoseAxis] = set()
    for requirement in pack.requirements:
        if requirement.axis in seen_axes:
            raise ValueError(f"Duplicate axis in diagnose pack: {requirement.axis}")
        seen_axes.add(requirement.axis)
        if requirement.min_required <= 0:
            raise ValueError(f"min_required must be > 0 for axis {requirement.axis}")
        if not requirement.probe_missions:
            raise ValueError(f"No probe missions configured for axis {requirement.axis}")


def evaluate_stage1_pack_contract(
    *,
    mission_set: str,
    cogs: list[int],
    steps: int,
    episodes: int,
    pack: DiagnosePack,
) -> DiagnosePackContractReport:
    sorted_cogs = sorted(cogs)
    checks = [
        DiagnosePackContractCheck(
            check_id="pack.mission_set",
            passed=mission_set == pack.mission_set,
            details=f"mission_set={mission_set}; expected={pack.mission_set}",
        ),
        DiagnosePackContractCheck(
            check_id="pack.cogs",
            passed=sorted_cogs == COGSGUARD_STAGE1_FIXED_COGS,
            details=f"cogs={sorted_cogs}; expected={COGSGUARD_STAGE1_FIXED_COGS}",
        ),
        DiagnosePackContractCheck(
            check_id="pack.steps",
            passed=steps == COGSGUARD_STAGE1_FIXED_STEPS,
            details=f"steps={steps}; expected={COGSGUARD_STAGE1_FIXED_STEPS}",
        ),
        DiagnosePackContractCheck(
            check_id="pack.episodes",
            passed=episodes == COGSGUARD_STAGE1_FIXED_EPISODES,
            details=f"episodes={episodes}; expected={COGSGUARD_STAGE1_FIXED_EPISODES}",
        ),
    ]
    return DiagnosePackContractReport(
        valid=all(check.passed for check in checks),
        checks=checks,
    )


def make_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def resolve_output_dir(output_dir: Optional[Path], run_id: str) -> Path:
    if output_dir is not None:
        return output_dir
    return Path("outputs") / "cogames-diagnose" / run_id


def replay_artifact_paths() -> tuple[str, ...]:
    return ("replays", "replays_stage2", "stability_reruns")


def write_json(path: Path, payload: BaseModel | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_html(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def write_replay_bundle(output_dir: Path) -> Path:
    bundle_path = output_dir / "replay_bundle.zip"
    replay_paths: list[Path] = []
    for replay_path in output_dir.rglob("*.json.z"):
        rel_path = replay_path.relative_to(output_dir)
        rel_path_str = rel_path.as_posix()
        if (
            rel_path_str.startswith("replays/")
            or rel_path_str.startswith("replays_stage2/")
            or rel_path_str.startswith("stability_reruns/")
        ):
            replay_paths.append(replay_path)
    replay_paths.sort(key=lambda path: str(path.relative_to(output_dir)))

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        if replay_paths:
            for replay_path in replay_paths:
                bundle.write(replay_path, arcname=replay_path.relative_to(output_dir).as_posix())
        else:
            bundle.writestr("README.txt", "No replay files were captured for this diagnose run.\n")
    return bundle_path


def write_bundle_guide(
    *,
    output_dir: Path,
    state: DiagnoseRunState,
    interpretation_stability: InterpretationStabilityReport,
    diagnose_validity: DiagnoseValidityReport,
) -> Path:
    guide_path = output_dir / "bundle_guide.md"
    failed_checks = ", ".join(diagnose_validity.failed_check_ids) if diagnose_validity.failed_check_ids else "none"
    assert state.diagnosis_status is not None, "DiagnoseRunState.diagnosis_status must be set"
    diagnosis_status = state.diagnosis_status
    guide = "\n".join(
        [
            "# Cogames Diagnose Bundle Guide",
            "",
            f"- Stage status: `{state.stage_status.value}`",
            f"- Run status: `{state.run_status.value}`",
            f"- Diagnosis status: `{diagnosis_status.value}`",
            f"- Visit completeness checks pass: `{str(diagnose_validity.valid).lower()}`",
            f"- Stability check: `{str(interpretation_stability.stable).lower()}`",
            f"- Failed completeness checks: `{failed_checks}`",
            "",
            "## 5-Minute Workflow",
            "1. Open `diagnose_report.html` and classify the dominant issue.",
            "2. Confirm `diagnose_validity.json` indicates a complete doctor-visit signal set.",
            "3. Confirm `interpretation_stability.json` is stable across snapshots.",
            (
                "4. Open `stage1_metric_correlation.json` and validate threshold tuning "
                "direction against objective context."
            ),
            "5. Open `doctor_note.json` and run the top prescription using its validation threshold.",
            "6. Share `diagnose_report.html`, `doctor_note.json`, `manifest.json`, and `replay_bundle.zip`.",
            "",
            "## Tournament Anchor",
            "- Keep `aligned.junction.held` visible when interpreting results.",
            "- Treat non-aligned diagnostics as supporting signals.",
        ]
    )
    guide_path.write_text(guide + "\n")
    return guide_path


def load_doctor_note(path: Path) -> DiagnoseDoctorNote:
    return DiagnoseDoctorNote.model_validate_json(path.read_text())


def stage1_probe_catalog() -> list[Stage1ProbeDefinition]:
    return [probe.model_copy(deep=True) for probe in STAGE1_PROBE_CATALOG_V1]


def stage1_probe_threshold_profile() -> Stage1ProbeThresholdProfile:
    return Stage1ProbeThresholdProfile(
        profile_id=STAGE1_PROBE_THRESHOLD_PROFILE_ID,
        probes=stage1_probe_catalog(),
    )


def stage2_social_probe_catalog() -> list[Stage2SocialProbeDefinition]:
    return [probe.model_copy(deep=True) for probe in STAGE2_SOCIAL_PROBE_CATALOG_V1]


def mission_from_case_name(case_name: str) -> str:
    prefix = case_name.split(" (cogs=", 1)[0]
    if "." in prefix:
        return prefix.split(".", 1)[1]
    return prefix


def evaluate_stage1_gate(*, case_names: list[str], pack: DiagnosePack) -> Stage1GateEvaluation:
    validate_pack_definition(pack)
    selected_missions = {mission_from_case_name(case_name) for case_name in case_names}

    results: list[Stage1RequirementResult] = []
    for requirement in pack.requirements:
        matched = sorted(selected_missions.intersection(requirement.probe_missions))
        results.append(
            Stage1RequirementResult(
                axis=requirement.axis,
                satisfied=len(matched) >= requirement.min_required,
                matched_missions=matched,
                required_count=requirement.min_required,
                accepted_probe_missions=requirement.probe_missions,
            )
        )

    return Stage1GateEvaluation(satisfied=all(result.satisfied for result in results), results=results)


def build_run_config(
    *,
    output_dir: Optional[Path],
    policy: str,
    mission_set: str,
    experiments: Optional[list[str]],
    cogs: Optional[list[int]],
    steps: int,
    episodes: int,
    pack: DiagnosePack,
    scripted_baseline_policy: Optional[str] = None,
    known_strong_policy: Optional[str] = None,
) -> DiagnoseRunConfig:
    run_id = make_run_id()
    return DiagnoseRunConfig(
        run_id=run_id,
        created_at=datetime.now(UTC),
        output_dir=resolve_output_dir(output_dir, run_id),
        policy=policy,
        mission_set=mission_set,
        experiments=experiments or [],
        cogs=cogs or [],
        steps=steps,
        episodes=episodes,
        pack_id=pack.pack_id,
        pack_version=pack.pack_version,
        scripted_baseline_policy=scripted_baseline_policy,
        known_strong_policy=known_strong_policy,
    )


def build_incomplete_state(
    *,
    config: DiagnoseRunConfig,
    stage_status: DiagnoseStageStatus,
    requirement_results: list[Stage1RequirementResult],
    expected_replay_count: int,
    replay_count: int,
    notes: list[str],
    stage1_signals: Optional[list[Stage1AxisSignal]] = None,
) -> DiagnoseRunState:
    missing_requirements = [result for result in requirement_results if not result.satisfied]
    return DiagnoseRunState(
        run_id=config.run_id,
        created_at=config.created_at,
        stage_status=stage_status,
        run_status=DiagnoseRunStatus.INCOMPLETE,
        diagnosis_status=DiagnosisLifecycleStatus.DIAGNOSIS_INCOMPLETE,
        pack_id=config.pack_id,
        pack_version=config.pack_version,
        missing_requirements=missing_requirements,
        requirement_results=requirement_results,
        expected_replay_count=expected_replay_count,
        replay_count=replay_count,
        output_dir=str(config.output_dir),
        notes=notes,
        stage1_signals=stage1_signals or [],
    )


def build_stage1_passed_state(
    *,
    config: DiagnoseRunConfig,
    requirement_results: list[Stage1RequirementResult],
    expected_replay_count: int,
    replay_count: int,
    stage1_signals: list[Stage1AxisSignal],
) -> DiagnoseRunState:
    return DiagnoseRunState(
        run_id=config.run_id,
        created_at=config.created_at,
        stage_status=DiagnoseStageStatus.STAGE1_PASSED_FOR_SOCIAL,
        run_status=DiagnoseRunStatus.INCOMPLETE,
        diagnosis_status=DiagnosisLifecycleStatus.DIAGNOSIS_INCOMPLETE,
        pack_id=config.pack_id,
        pack_version=config.pack_version,
        missing_requirements=[],
        requirement_results=requirement_results,
        expected_replay_count=expected_replay_count,
        replay_count=replay_count,
        output_dir=str(config.output_dir),
        notes=["Stage 1 passed. Stage 2 social review is required before final diagnosis."],
        stage1_signals=stage1_signals,
    )


def build_stage2_completed_state(
    *,
    config: DiagnoseRunConfig,
    requirement_results: list[Stage1RequirementResult],
    expected_replay_count: int,
    replay_count: int,
    stage1_signals: list[Stage1AxisSignal],
    notes: Optional[list[str]] = None,
) -> DiagnoseRunState:
    return DiagnoseRunState(
        run_id=config.run_id,
        created_at=config.created_at,
        stage_status=DiagnoseStageStatus.STAGE2_COMPLETED,
        run_status=DiagnoseRunStatus.COMPLETE,
        diagnosis_status=DiagnosisLifecycleStatus.DIAGNOSIS_COMPLETE,
        pack_id=config.pack_id,
        pack_version=config.pack_version,
        missing_requirements=[],
        requirement_results=requirement_results,
        expected_replay_count=expected_replay_count,
        replay_count=replay_count,
        output_dir=str(config.output_dir),
        notes=notes or ["Stage 1 and Stage 2 completed."],
        stage1_signals=stage1_signals,
    )


def count_replays(replay_dir: Path) -> int:
    if not replay_dir.exists():
        return 0
    return sum(1 for _ in replay_dir.glob("*.json.z"))


def expected_replays(case_count: int, episodes: int) -> int:
    return case_count * episodes


def stage1_summary_payload(*, mission_names: list[str], episode_count: int, case_names: list[str]) -> dict:
    return {
        "stage": "stage1",
        "mission_count": len(mission_names),
        "episodes_per_case": episode_count,
        "cases": case_names,
    }


def use_stage1_pack(mission_set: str) -> bool:
    return mission_set == COGSGUARD_STAGE1_PACK_V1.mission_set


def _policy_episode_rewards(summary: MultiEpisodeRolloutSummary, policy_index: int = 0) -> list[float]:
    rewards: list[float] = []
    for episode_rewards in summary.per_episode_per_policy_avg_rewards.values():
        if policy_index >= len(episode_rewards):
            continue
        reward = episode_rewards[policy_index]
        if reward is not None:
            rewards.append(float(reward))
    return rewards


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def mission_metrics_from_summary(
    *,
    mission_name: str,
    summary: MultiEpisodeRolloutSummary,
    policy_index: int = 0,
) -> Stage1MissionMetrics:
    policy_summary = summary.policy_summaries[policy_index]
    avg_agent_metrics = policy_summary.avg_agent_metrics
    rewards = _policy_episode_rewards(summary, policy_index)

    non_zero_episode_count = sum(1 for reward in rewards if reward != 0.0)
    non_zero_episode_pct = (non_zero_episode_count / len(rewards) * 100.0) if rewards else 0.0

    denom = max(summary.episodes * max(policy_summary.agent_count, 1), 1)
    timeout_rate = policy_summary.action_timeouts / denom

    return Stage1MissionMetrics(
        mission_name=mission_name,
        reward_variance=_variance(rewards),
        non_zero_episode_pct=non_zero_episode_pct,
        timeout_rate=timeout_rate,
        mean_move_success=float(avg_agent_metrics.get("action.move.success", 0.0)),
        mean_action_failed=float(avg_agent_metrics.get("action.failed", 0.0)),
        mean_stuck_steps=float(avg_agent_metrics.get("status.max_steps_without_motion", 0.0)),
    )


def stage1_metrics_by_mission(
    *,
    case_names: list[str],
    summaries: list[MultiEpisodeRolloutSummary],
) -> dict[str, list[Stage1MissionMetrics]]:
    metrics_by_mission: dict[str, list[Stage1MissionMetrics]] = {}
    for case_name, summary in zip(case_names, summaries, strict=True):
        mission_name = mission_from_case_name(case_name)
        mission_metrics = mission_metrics_from_summary(mission_name=mission_name, summary=summary)
        metrics_by_mission.setdefault(mission_name, []).append(mission_metrics)
    return metrics_by_mission


def compute_tournament_objective_value(
    *,
    summaries: list[MultiEpisodeRolloutSummary],
    metric_key: str = "aligned.junction.held",
) -> Optional[float]:
    values: list[float] = []
    for summary in summaries:
        metric = summary.avg_game_stats.get(metric_key)
        if metric is not None:
            values.append(float(metric))
    if not values:
        return None
    return round(_safe_mean(values), 4)


def stage1_objective_values_by_mission(
    *,
    case_names: list[str],
    summaries: list[MultiEpisodeRolloutSummary],
    metric_key: str = "aligned.junction.held",
) -> dict[str, list[float]]:
    objective_by_mission: dict[str, list[float]] = {}
    for case_name, summary in zip(case_names, summaries, strict=True):
        objective = summary.avg_game_stats.get(metric_key)
        if objective is None:
            continue
        mission_name = mission_from_case_name(case_name)
        objective_by_mission.setdefault(mission_name, []).append(float(objective))
    return objective_by_mission


def _pearson_correlation(values_x: list[float], values_y: list[float]) -> Optional[float]:
    if len(values_x) < 2 or len(values_y) < 2 or len(values_x) != len(values_y):
        return None
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    centered_x = [value - mean_x for value in values_x]
    centered_y = [value - mean_y for value in values_y]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x <= 1e-9 or denom_y <= 1e-9:
        return None
    numerator = sum(x * y for x, y in zip(centered_x, centered_y, strict=True))
    return max(min(numerator / (denom_x * denom_y), 1.0), -1.0)


def evaluate_stage1_metric_correlation(
    *,
    metrics_by_mission: dict[str, list[Stage1MissionMetrics]],
    objective_by_mission: dict[str, list[float]],
    objective_metric: str = "aligned.junction.held",
    threshold_profile: Optional[Stage1ProbeThresholdProfile] = None,
) -> Stage1MetricCorrelationReport:
    metric_extractors: list[tuple[str, Callable[[Stage1MissionMetrics], float]]] = [
        ("reward_variance", lambda metric: metric.reward_variance),
        ("non_zero_episode_pct", lambda metric: metric.non_zero_episode_pct),
        ("timeout_rate", lambda metric: metric.timeout_rate),
        ("mean_move_success", lambda metric: metric.mean_move_success),
        ("mean_action_failed", lambda metric: metric.mean_action_failed),
        ("mean_stuck_steps", lambda metric: metric.mean_stuck_steps),
    ]
    paired_values: dict[str, list[float]] = {metric_name: [] for metric_name, _extract in metric_extractors}
    paired_objectives: dict[str, list[float]] = {metric_name: [] for metric_name, _extract in metric_extractors}
    total_pairs = 0

    for mission_name, mission_metrics in metrics_by_mission.items():
        objective_values = objective_by_mission.get(mission_name, [])
        pair_count = min(len(mission_metrics), len(objective_values))
        if pair_count == 0:
            continue
        total_pairs += pair_count
        for metric in mission_metrics[:pair_count]:
            for metric_name, extractor in metric_extractors:
                paired_values[metric_name].append(float(extractor(metric)))
        for objective in objective_values[:pair_count]:
            for metric_name, _extractor in metric_extractors:
                paired_objectives[metric_name].append(float(objective))

    correlations: list[Stage1MetricCorrelation] = []
    for metric_name, _extractor in metric_extractors:
        values = paired_values[metric_name]
        objective_values = paired_objectives[metric_name]
        correlation = _pearson_correlation(values, objective_values)
        if correlation is None:
            direction = "insufficient_data"
            interpretation = "Need at least 2 non-constant paired samples."
        else:
            direction = "positive" if correlation >= 0 else "negative"
            strength = abs(correlation)
            if strength >= 0.6:
                interpretation = "Strong relationship to tournament objective."
            elif strength >= 0.3:
                interpretation = "Moderate relationship to tournament objective."
            else:
                interpretation = "Weak relationship to tournament objective."
        correlations.append(
            Stage1MetricCorrelation(
                metric=metric_name,
                correlation=None if correlation is None else round(correlation, 4),
                sample_count=len(values),
                direction=direction,
                interpretation=interpretation,
            )
        )

    calibrations: list[Stage1ProbeThresholdCalibration] = []
    profile = threshold_profile or stage1_probe_threshold_profile()
    for probe in profile.probes:
        mission_metrics = metrics_by_mission.get(probe.mission, [])
        objective_values = objective_by_mission.get(probe.mission, [])
        pair_count = min(len(mission_metrics), len(objective_values))
        top_indices: list[int] = []
        if pair_count > 0:
            paired_objective_values = objective_values[:pair_count]
            sorted_objectives = sorted(paired_objective_values)
            median = sorted_objectives[pair_count // 2]
            top_indices = [idx for idx, value in enumerate(paired_objective_values) if value >= median]
        for rule in probe.threshold_rules:
            notes: list[str] = []
            recommended_threshold: Optional[float] = None
            delta: Optional[float] = None
            if pair_count == 0:
                notes.append("No paired mission metrics + objective samples for probe calibration.")
            elif not top_indices:
                notes.append("No top-objective samples found for probe calibration.")
            else:
                metric_samples = [mission_metrics[idx].model_dump(mode="json").get(rule.metric) for idx in top_indices]
                numeric_samples = [float(sample) for sample in metric_samples if sample is not None]
                if not numeric_samples:
                    notes.append("Probe metric was missing from mission samples.")
                elif rule.operator == "<=":
                    recommended_threshold = round(max(numeric_samples), 4)
                    delta = round(recommended_threshold - rule.value, 4)
                else:
                    recommended_threshold = round(min(numeric_samples), 4)
                    delta = round(recommended_threshold - rule.value, 4)
            calibrations.append(
                Stage1ProbeThresholdCalibration(
                    probe_id=probe.probe_id,
                    mission=probe.mission,
                    metric=rule.metric,
                    operator=rule.operator,
                    current_threshold=rule.value,
                    recommended_threshold=recommended_threshold,
                    delta=delta,
                    sample_count=pair_count,
                    notes=notes,
                )
            )

    tuned = any(
        calibration.recommended_threshold is not None
        and calibration.delta is not None
        and abs(calibration.delta) > 1e-6
        for calibration in calibrations
    )
    notes: list[str] = []
    if total_pairs < 2:
        notes.append("Insufficient paired samples for stable correlation estimates.")
    if not tuned:
        notes.append("No threshold deltas detected from current top-objective samples.")
    return Stage1MetricCorrelationReport(
        objective_metric=objective_metric,
        sample_count=total_pairs,
        correlations=correlations,
        probe_calibrations=calibrations,
        tuned=tuned,
        notes=notes,
    )


def stage2_mission_metrics_from_summary(
    *,
    mission_name: str,
    summary: MultiEpisodeRolloutSummary,
    policy_index: int,
) -> Stage2MissionMetrics:
    policy_summary = summary.policy_summaries[policy_index]
    avg_agent_metrics = policy_summary.avg_agent_metrics
    rewards = _policy_episode_rewards(summary, policy_index)
    non_zero_episode_count = sum(1 for reward in rewards if reward != 0.0)
    non_zero_episode_pct = (non_zero_episode_count / len(rewards) * 100.0) if rewards else 0.0
    denom = max(summary.episodes * max(policy_summary.agent_count, 1), 1)
    timeout_rate = policy_summary.action_timeouts / denom
    policy_reward_mean = _safe_mean(rewards)
    return Stage2MissionMetrics(
        mission_name=mission_name,
        policy_reward_mean=policy_reward_mean,
        reward_variance=_variance(rewards),
        non_zero_episode_pct=non_zero_episode_pct,
        timeout_rate=timeout_rate,
        mean_move_success=float(avg_agent_metrics.get("action.move.success", 0.0)),
        mean_action_failed=float(avg_agent_metrics.get("action.failed", 0.0)),
        mean_stuck_steps=float(avg_agent_metrics.get("status.max_steps_without_motion", 0.0)),
    )


def stage2_metrics_by_mission(
    *,
    case_names: list[str],
    summaries: list[MultiEpisodeRolloutSummary],
) -> dict[str, list[Stage2MissionMetrics]]:
    metrics_by_mission: dict[str, list[Stage2MissionMetrics]] = {}
    for case_name, summary in zip(case_names, summaries, strict=True):
        mission_name = mission_from_case_name(case_name)
        metrics_by_mission.setdefault(mission_name, [])
        for policy_index in range(len(summary.policy_summaries)):
            metrics_by_mission[mission_name].append(
                stage2_mission_metrics_from_summary(
                    mission_name=mission_name,
                    summary=summary,
                    policy_index=policy_index,
                )
            )
    return metrics_by_mission


def build_stage2_mode_summary(
    *,
    mode: Stage2Mode,
    seed: int,
    case_names: list[str],
    summaries: list[MultiEpisodeRolloutSummary],
) -> Stage2ModeSummary:
    return Stage2ModeSummary(
        mode=mode,
        seed=seed,
        case_count=len(case_names),
        mission_metrics=stage2_metrics_by_mission(case_names=case_names, summaries=summaries),
    )


def _aggregate_stage2_metrics(metrics: dict[str, list[Stage2MissionMetrics]]) -> Stage2MissionMetrics:
    all_metrics = [metric for mission_metrics in metrics.values() for metric in mission_metrics]
    return Stage2MissionMetrics(
        mission_name="aggregate",
        policy_reward_mean=_safe_mean([metric.policy_reward_mean for metric in all_metrics]),
        reward_variance=_safe_mean([metric.reward_variance for metric in all_metrics]),
        non_zero_episode_pct=_safe_mean([metric.non_zero_episode_pct for metric in all_metrics]),
        timeout_rate=_safe_mean([metric.timeout_rate for metric in all_metrics]),
        mean_move_success=_safe_mean([metric.mean_move_success for metric in all_metrics]),
        mean_action_failed=_safe_mean([metric.mean_action_failed for metric in all_metrics]),
        mean_stuck_steps=_safe_mean([metric.mean_stuck_steps for metric in all_metrics]),
    )


def _mirror_policy_reward_gap(mirror_metrics: dict[str, list[Stage2MissionMetrics]]) -> float:
    per_mission_gaps: list[float] = []
    for mission_metrics in mirror_metrics.values():
        if len(mission_metrics) < 2:
            continue
        first = mission_metrics[0].policy_reward_mean
        second = mission_metrics[1].policy_reward_mean
        per_mission_gaps.append(abs(first - second))
    return _safe_mean(per_mission_gaps)


def assess_stage2_social_signal(
    *,
    absolute_summary: Stage2ModeSummary,
    mirror_summary: Stage2ModeSummary,
    replay_refs: list[str],
) -> Stage2SocialSignal:
    absolute_agg = _aggregate_stage2_metrics(absolute_summary.mission_metrics)
    mirror_agg = _aggregate_stage2_metrics(mirror_summary.mission_metrics)
    mirror_reward_gap = _mirror_policy_reward_gap(mirror_summary.mission_metrics)

    reward_drop = max(0.0, absolute_agg.policy_reward_mean - mirror_agg.policy_reward_mean)
    timeout_increase = max(0.0, mirror_agg.timeout_rate - absolute_agg.timeout_rate)
    stuck_increase = max(0.0, mirror_agg.mean_stuck_steps - absolute_agg.mean_stuck_steps)

    severity_components = [
        _clamp01(reward_drop / 1.0),
        _clamp01(mirror_reward_gap / 0.5),
        _clamp01(timeout_increase / 0.03),
        _clamp01(stuck_increase / 8.0),
    ]
    severity = round(_safe_mean(severity_components), 2)

    evidence_refs = [
        f"absolute:policy_reward_mean={absolute_agg.policy_reward_mean:.4f}",
        f"mirror:policy_reward_mean={mirror_agg.policy_reward_mean:.4f}",
        f"mirror:policy_reward_gap={mirror_reward_gap:.4f}",
        f"absolute:timeout_rate={absolute_agg.timeout_rate:.4f}",
        f"mirror:timeout_rate={mirror_agg.timeout_rate:.4f}",
        f"absolute:mean_stuck_steps={absolute_agg.mean_stuck_steps:.4f}",
        f"mirror:mean_stuck_steps={mirror_agg.mean_stuck_steps:.4f}",
    ]

    has_metrics = absolute_summary.case_count > 0 and mirror_summary.case_count > 0
    has_replays = bool(replay_refs)
    confirmed = has_metrics and has_replays

    confidence = 0.4
    if has_metrics:
        confidence += 0.25
    if has_replays:
        confidence += 0.2
    confidence += 0.1 if severity >= 0.4 else 0.0
    confidence = round(_clamp01(confidence), 2)

    if not confirmed:
        summary = "Stage 2 social review evidence is incomplete."
    elif severity >= 0.55:
        summary = "Social scrimmage indicates coordination degradation under mirror conditions."
    elif severity >= 0.3:
        summary = "Social scrimmage shows moderate coordination risk."
    else:
        summary = "Social scrimmage confirms Stage 1 diagnosis without major coordination regressions."

    return Stage2SocialSignal(
        confirmed=confirmed,
        severity=severity,
        confidence=confidence,
        summary=summary,
        evidence_refs=evidence_refs,
    )


def _metric_refs_for_axis(axis: DiagnoseAxis, mission_metrics: Stage1MissionMetrics) -> list[str]:
    if axis == DiagnoseAxis.STABILITY:
        return [
            f"{mission_metrics.mission_name}:reward_variance={mission_metrics.reward_variance:.4f}",
            f"{mission_metrics.mission_name}:timeout_rate={mission_metrics.timeout_rate:.4f}",
            f"{mission_metrics.mission_name}:non_zero_episode_pct={mission_metrics.non_zero_episode_pct:.2f}",
        ]
    if axis == DiagnoseAxis.EFFICIENCY:
        return [
            f"{mission_metrics.mission_name}:action.move.success={mission_metrics.mean_move_success:.4f}",
            f"{mission_metrics.mission_name}:status.max_steps_without_motion={mission_metrics.mean_stuck_steps:.4f}",
        ]
    if axis == DiagnoseAxis.CONTROL:
        return [
            f"{mission_metrics.mission_name}:action.failed={mission_metrics.mean_action_failed:.4f}",
            f"{mission_metrics.mission_name}:action.move.success={mission_metrics.mean_move_success:.4f}",
        ]
    raise ValueError(f"Unsupported axis: {axis}")


def assess_stage1_signals(
    *,
    requirement_results: list[Stage1RequirementResult],
    metrics_by_mission: dict[str, list[Stage1MissionMetrics]],
    replay_refs: list[str],
    require_replay_evidence: bool = True,
) -> list[Stage1AxisSignal]:
    signals: list[Stage1AxisSignal] = []
    for requirement_result in requirement_results:
        metric_refs: list[str] = []
        for mission_name in requirement_result.matched_missions:
            for mission_metrics in metrics_by_mission.get(mission_name, []):
                metric_refs.extend(_metric_refs_for_axis(requirement_result.axis, mission_metrics))

        has_replay_evidence = bool(replay_refs) or not require_replay_evidence
        confirmed = requirement_result.satisfied and bool(metric_refs) and has_replay_evidence
        if confirmed:
            summary = f"Confirmed {requirement_result.axis} signal with metrics and replay evidence."
        elif not requirement_result.satisfied:
            summary = f"Missing required probe coverage for axis {requirement_result.axis}."
        elif not metric_refs:
            summary = f"No metrics captured for axis {requirement_result.axis}."
        elif require_replay_evidence:
            summary = f"No replay evidence captured for axis {requirement_result.axis}."
        else:
            summary = f"Confirmed {requirement_result.axis} signal with metric-only baseline evidence."

        signals.append(
            Stage1AxisSignal(
                axis=requirement_result.axis,
                confirmed=confirmed,
                metric_refs=metric_refs,
                replay_refs=replay_refs if confirmed and require_replay_evidence else [],
                summary=summary,
            )
        )
    return signals


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _inverse_threshold_score(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clamp01(1.0 - (value / threshold))


def _aggregate_axis_metrics(metrics: list[Stage1MissionMetrics]) -> Stage1AxisDerivedMetrics:
    return Stage1AxisDerivedMetrics(
        reward_variance=_safe_mean([metric.reward_variance for metric in metrics]),
        non_zero_episode_pct=_safe_mean([metric.non_zero_episode_pct for metric in metrics]),
        timeout_rate=_safe_mean([metric.timeout_rate for metric in metrics]),
        mean_move_success=_safe_mean([metric.mean_move_success for metric in metrics]),
        mean_action_failed=_safe_mean([metric.mean_action_failed for metric in metrics]),
        mean_stuck_steps=_safe_mean([metric.mean_stuck_steps for metric in metrics]),
    )


def evaluate_stage1_probe_catalog(
    *,
    metrics_by_mission: dict[str, list[Stage1MissionMetrics]],
) -> list[Stage1ProbeEvaluation]:
    evaluations: list[Stage1ProbeEvaluation] = []
    for probe in stage1_probe_catalog():
        mission_metrics = metrics_by_mission.get(probe.mission, [])
        if not mission_metrics:
            evaluations.append(
                Stage1ProbeEvaluation(
                    probe_id=probe.probe_id,
                    axis=probe.axis,
                    mission=probe.mission,
                    passed=False,
                    summary="Missing mission metrics for probe mission.",
                    evidence_refs=[],
                )
            )
            continue

        aggregated = _aggregate_axis_metrics(mission_metrics)
        metric_values = aggregated.model_dump(mode="json")
        evidence_refs: list[str] = []
        passed = bool(probe.threshold_rules)
        for rule in probe.threshold_rules:
            metric_value = metric_values.get(rule.metric)
            if metric_value is None:
                passed = False
                evidence_refs.append(f"{probe.mission}:{rule.metric}=missing")
                continue
            metric_value = float(metric_value)
            if rule.operator == "<=":
                rule_passed = metric_value <= rule.value
            elif rule.operator == ">=":
                rule_passed = metric_value >= rule.value
            else:
                rule_passed = False
            passed = passed and rule_passed
            evidence_refs.append(f"{probe.mission}:{rule.metric}={metric_value:.4f} ({rule.operator} {rule.value:.4f})")

        evaluations.append(
            Stage1ProbeEvaluation(
                probe_id=probe.probe_id,
                axis=probe.axis,
                mission=probe.mission,
                passed=passed,
                summary="Probe threshold passed." if passed else "Probe threshold failed.",
                evidence_refs=evidence_refs,
            )
        )
    return evaluations


def _normalized_axis_score(axis: DiagnoseAxis, metrics: Stage1AxisDerivedMetrics) -> float:
    if axis == DiagnoseAxis.STABILITY:
        components = [
            _inverse_threshold_score(metrics.reward_variance, threshold=1.0),
            _inverse_threshold_score(metrics.timeout_rate, threshold=0.05),
            _clamp01(metrics.non_zero_episode_pct / 100.0),
        ]
    elif axis == DiagnoseAxis.EFFICIENCY:
        components = [
            _clamp01(metrics.mean_move_success),
            _inverse_threshold_score(metrics.mean_stuck_steps, threshold=30.0),
            _clamp01(metrics.non_zero_episode_pct / 100.0),
        ]
    elif axis == DiagnoseAxis.CONTROL:
        components = [
            _inverse_threshold_score(metrics.mean_action_failed, threshold=0.2),
            _clamp01(metrics.mean_move_success),
            _inverse_threshold_score(metrics.timeout_rate, threshold=0.05),
        ]
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    return round(_safe_mean(components) * 100.0, 2)


def compute_stage1_derived_metrics(metrics_by_mission: dict[str, list[Stage1MissionMetrics]]) -> Stage1DerivedMetrics:
    all_metrics = [metric for metrics in metrics_by_mission.values() for metric in metrics]
    return Stage1DerivedMetrics(
        reward_variance=_safe_mean([metric.reward_variance for metric in all_metrics]),
        non_zero_episode_pct=_safe_mean([metric.non_zero_episode_pct for metric in all_metrics]),
        timeout_rate=_safe_mean([metric.timeout_rate for metric in all_metrics]),
    )


def compute_stage1_axis_scores(
    *,
    requirement_results: list[Stage1RequirementResult],
    metrics_by_mission: dict[str, list[Stage1MissionMetrics]],
    stage1_signals: list[Stage1AxisSignal],
) -> list[Stage1AxisScore]:
    signals_by_axis = {signal.axis: signal for signal in stage1_signals}
    axis_scores: list[Stage1AxisScore] = []

    for requirement_result in requirement_results:
        axis_metrics = [
            mission_metrics
            for mission_name in requirement_result.matched_missions
            for mission_metrics in metrics_by_mission.get(mission_name, [])
        ]
        aggregated_metrics = _aggregate_axis_metrics(axis_metrics)
        normalized_score = _normalized_axis_score(requirement_result.axis, aggregated_metrics)

        signal = signals_by_axis.get(requirement_result.axis)
        confirmed = signal.confirmed if signal is not None else False
        if not requirement_result.satisfied:
            normalized_score = min(normalized_score, 40.0)
        elif not confirmed:
            normalized_score = min(normalized_score, 60.0)

        axis_scores.append(
            Stage1AxisScore(
                axis=requirement_result.axis,
                normalized_score=normalized_score,
                raw_score=normalized_score,
                confirmed=confirmed,
                derived_metrics=aggregated_metrics,
                metric_refs=signal.metric_refs if signal is not None else [],
                replay_refs=signal.replay_refs if signal is not None else [],
                baseline_refs=[],
                baseline_coverage_count=0,
                normalization_mode="heuristic_thresholds",
            )
        )

    return axis_scores


def _axis_score_map(axis_scores: list[Stage1AxisScore]) -> dict[DiagnoseAxis, Stage1AxisScore]:
    return {axis_score.axis: axis_score for axis_score in axis_scores}


def normalize_stage1_axis_scores(
    *,
    axis_scores: list[Stage1AxisScore],
    scripted_baseline_axis_scores: Optional[list[Stage1AxisScore]],
    known_strong_axis_scores: Optional[list[Stage1AxisScore]],
    scripted_policy: Optional[str],
    known_strong_policy: Optional[str],
) -> tuple[list[Stage1AxisScore], Stage1BaselineContext]:
    scripted_by_axis = _axis_score_map(scripted_baseline_axis_scores or [])
    strong_by_axis = _axis_score_map(known_strong_axis_scores or [])

    context_notes: list[str] = []
    normalized_axis_scores: list[Stage1AxisScore] = []
    uses_dual_baselines = bool(scripted_by_axis) and bool(strong_by_axis)
    uses_single_baseline = bool(scripted_by_axis) != bool(strong_by_axis)

    for axis_score in axis_scores:
        scripted_score = scripted_by_axis.get(axis_score.axis)
        strong_score = strong_by_axis.get(axis_score.axis)
        raw_score = axis_score.normalized_score

        normalized_score = raw_score
        normalization_mode = "heuristic_thresholds"
        baseline_refs: list[str] = []
        baseline_coverage_count = 0

        if scripted_score is not None:
            baseline_coverage_count += 1
            baseline_refs.append(f"scripted:{axis_score.axis.value}={scripted_score.normalized_score:.2f}")
        if strong_score is not None:
            baseline_coverage_count += 1
            baseline_refs.append(f"known_strong:{axis_score.axis.value}={strong_score.normalized_score:.2f}")

        if scripted_score is not None and strong_score is not None:
            baseline_span = strong_score.normalized_score - scripted_score.normalized_score
            if abs(baseline_span) >= 1e-6:
                normalized_score = _clamp01((raw_score - scripted_score.normalized_score) / baseline_span) * 100.0
                normalization_mode = "scripted_to_known_strong"
            else:
                context_notes.append(
                    f"Axis {axis_score.axis.value}: scripted and known-strong baseline scores are identical."
                )
        elif strong_score is not None and strong_score.normalized_score > 0:
            normalized_score = _clamp01(raw_score / strong_score.normalized_score) * 100.0
            normalization_mode = "known_strong_only"
        elif scripted_score is not None and scripted_score.normalized_score < 100.0:
            upper_span = 100.0 - scripted_score.normalized_score
            normalized_score = _clamp01((raw_score - scripted_score.normalized_score) / max(upper_span, 1e-6)) * 100.0
            normalization_mode = "scripted_only"

        normalized_axis_scores.append(
            Stage1AxisScore(
                axis=axis_score.axis,
                normalized_score=round(normalized_score, 2),
                raw_score=round(raw_score, 2),
                confirmed=axis_score.confirmed,
                derived_metrics=axis_score.derived_metrics,
                metric_refs=axis_score.metric_refs,
                replay_refs=axis_score.replay_refs,
                baseline_refs=baseline_refs,
                baseline_coverage_count=baseline_coverage_count,
                normalization_mode=normalization_mode,
            )
        )

    if uses_dual_baselines:
        normalization_mode = "scripted_to_known_strong"
    elif uses_single_baseline:
        normalization_mode = "single_baseline"
    else:
        normalization_mode = "heuristic_only"
        context_notes.append("No reference baselines provided; normalized scores rely on heuristic thresholds only.")

    baseline_summaries: list[Stage1BaselinePolicySummary] = []
    if scripted_policy is not None and scripted_baseline_axis_scores:
        baseline_summaries.append(
            Stage1BaselinePolicySummary(
                role="scripted_baseline",
                policy=scripted_policy,
                axis_scores=scripted_baseline_axis_scores,
            )
        )
    if known_strong_policy is not None and known_strong_axis_scores:
        baseline_summaries.append(
            Stage1BaselinePolicySummary(
                role="known_strong",
                policy=known_strong_policy,
                axis_scores=known_strong_axis_scores,
            )
        )

    return normalized_axis_scores, Stage1BaselineContext(
        normalization_mode=normalization_mode,
        normalization_notes=context_notes,
        baselines=baseline_summaries,
    )


def _axis_symptom_profile(axis: DiagnoseAxis) -> tuple[str, str, str, str]:
    if axis == DiagnoseAxis.STABILITY:
        return (
            "Behavior varies across episodes or degrades under sustained pressure.",
            "Increase fixed-seed episode count and tune policy for lower variance and fewer timeouts.",
            "Lower reward variance and timeout rate with more reliable outcomes.",
            "stability.normalized_score >= 80 and timeout_rate <= 0.03",
        )
    if axis == DiagnoseAxis.EFFICIENCY:
        return (
            "Movement or objective progression is slower than expected.",
            "Tune pathing and objective prioritization to reduce stuck steps and improve move success.",
            "Higher throughput and faster completion in resource and traversal probes.",
            "efficiency.normalized_score >= 80 and mean_stuck_steps <= 8",
        )
    if axis == DiagnoseAxis.CONTROL:
        return (
            "Action selection is error-prone under pressure.",
            "Reduce invalid actions and improve decision timing on high-risk interactions.",
            "Fewer failed actions and better execution accuracy in control probes.",
            "control.normalized_score >= 80 and action.failed <= 0.08",
        )
    if axis == DiagnoseAxis.SOCIAL_COORDINATION:
        return (
            "Behavior degrades when teammates/opponents are present.",
            "Run social scrimmage probes and tune coordination rules to reduce interference and missed timing windows.",
            "Improved team throughput and lower deadlock/missed-window rates in social probes.",
            "stage2.social_coordination.severity <= 0.30",
        )
    raise ValueError(f"Unsupported axis: {axis}")


def rank_stage1_symptoms(axis_scores: list[Stage1AxisScore]) -> list[DiagnoseSymptom]:
    symptoms: list[DiagnoseSymptom] = []

    for axis_score in axis_scores:
        score_gap = _clamp01((100.0 - axis_score.normalized_score) / 100.0)
        severity = score_gap if axis_score.confirmed else max(score_gap, 0.6)
        severity = round(_clamp01(severity), 2)

        if severity < 0.2 and axis_score.confirmed:
            continue

        evidence_count = min(len(axis_score.metric_refs), 6)
        confidence = 0.35 + (0.05 * evidence_count)
        if axis_score.replay_refs:
            confidence += 0.2
        confidence += 0.2 if axis_score.confirmed else 0.05
        confidence += 0.08 * min(axis_score.baseline_coverage_count, 2)
        if axis_score.normalization_mode == "heuristic_thresholds":
            confidence -= 0.08
        confidence = round(_clamp01(confidence), 2)

        likely_cause, action, expected_effect, _threshold = _axis_symptom_profile(axis_score.axis)
        symptoms.append(
            DiagnoseSymptom(
                symptom_id=f"stage1.{axis_score.axis.value}.risk",
                axis=axis_score.axis,
                severity=severity,
                confidence=confidence,
                evidence_refs=DiagnoseSymptomEvidenceRefs(
                    metric_refs=axis_score.metric_refs,
                    replay_refs=axis_score.replay_refs,
                ),
                likely_cause=likely_cause,
                action=action,
                expected_effect=expected_effect,
            )
        )

    symptoms.sort(key=lambda symptom: (symptom.severity, symptom.confidence), reverse=True)
    return symptoms


def build_stage2_social_symptom(
    *,
    social_signal: Optional[Stage2SocialSignal],
    replay_refs: list[str],
) -> Optional[DiagnoseSymptom]:
    if social_signal is None or not social_signal.confirmed:
        return None
    if social_signal.severity < 0.3:
        return None

    likely_cause, action, expected_effect, _threshold = _axis_symptom_profile(DiagnoseAxis.SOCIAL_COORDINATION)
    return DiagnoseSymptom(
        symptom_id="stage2.social_coordination.risk",
        axis=DiagnoseAxis.SOCIAL_COORDINATION,
        severity=round(_clamp01(social_signal.severity), 2),
        confidence=round(_clamp01(social_signal.confidence), 2),
        evidence_refs=DiagnoseSymptomEvidenceRefs(
            metric_refs=social_signal.evidence_refs,
            replay_refs=replay_refs,
        ),
        likely_cause=likely_cause,
        action=action,
        expected_effect=expected_effect,
    )


def build_stage2_diagnosis_delta(
    *,
    axis_scores: list[Stage1AxisScore],
    social_signal: Optional[Stage2SocialSignal],
) -> Optional[Stage2DiagnosisDelta]:
    if social_signal is None or not social_signal.confirmed:
        return None

    stage1_issue = _dominant_issue(axis_scores, social_signal=None)
    final_issue = _dominant_issue(axis_scores, social_signal=social_signal)
    changed = stage1_issue != final_issue
    if changed:
        summary = f"Stage 2 social review changed diagnosis from {stage1_issue.value} to {final_issue.value}."
    else:
        summary = f"Stage 2 social review confirmed {final_issue.value} as the dominant issue."
    return Stage2DiagnosisDelta(
        stage1_dominant_issue=stage1_issue,
        final_dominant_issue=final_issue,
        changed=changed,
        summary=summary,
        evidence_refs=social_signal.evidence_refs,
    )


def _slugify_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _mission_replay_ref(*, replay_refs: list[str], mission_name: str) -> Optional[str]:
    mission = mission_name.split(" (cogs=")[0]
    mission_tokens = {
        mission.lower(),
        mission.replace(".", "/").lower(),
        mission.replace(".", "_").lower(),
        _slugify_for_match(mission),
        mission.split(".")[-1].lower(),
    }
    for replay_ref in replay_refs:
        lowered = replay_ref.lower()
        replay_token = _slugify_for_match(Path(replay_ref).stem)
        if any(token and (token in lowered or token == replay_token) for token in mission_tokens):
            return replay_ref
    return None


def _mission_behavior_score(metrics: list[Stage1MissionMetrics]) -> float:
    if not metrics:
        return 0.0
    move_success = _safe_mean([metric.mean_move_success for metric in metrics])
    non_zero = _clamp01(_safe_mean([metric.non_zero_episode_pct for metric in metrics]) / 100.0)
    timeout = _safe_mean([metric.timeout_rate for metric in metrics])
    action_failed = _safe_mean([metric.mean_action_failed for metric in metrics])
    stuck_penalty = min(_safe_mean([metric.mean_stuck_steps for metric in metrics]) / 50.0, 1.0)
    variance_penalty = min(_safe_mean([metric.reward_variance for metric in metrics]) / 2.0, 1.0)
    return (
        (0.5 * move_success)
        + (0.3 * non_zero)
        - (0.3 * timeout)
        - (0.2 * action_failed)
        - (0.2 * stuck_penalty)
        - (0.1 * variance_penalty)
    )


def select_replay_exemplars(
    *,
    metrics_by_mission: dict[str, list[Stage1MissionMetrics]],
    replay_refs: list[str],
    symptoms: list[DiagnoseSymptom],
    mission_replay_refs: Optional[dict[str, list[str]]] = None,
) -> ReplayExemplarRefs:
    if not replay_refs:
        return ReplayExemplarRefs()

    sorted_refs = sorted(replay_refs)
    mission_replay_refs = mission_replay_refs or {}
    mission_scores = [
        (mission_name, _mission_behavior_score(mission_metrics))
        for mission_name, mission_metrics in metrics_by_mission.items()
        if mission_metrics
    ]
    best_ref = sorted_refs[0]
    worst_ref = sorted_refs[-1]
    if mission_scores:
        mission_scores.sort(key=lambda item: item[1])
        worst_mission, _ = mission_scores[0]
        best_mission, _ = mission_scores[-1]
        matched_best = next(iter(sorted(mission_replay_refs.get(best_mission, []))), None) or _mission_replay_ref(
            replay_refs=sorted_refs,
            mission_name=best_mission,
        )
        matched_worst = next(iter(sorted(mission_replay_refs.get(worst_mission, []))), None) or _mission_replay_ref(
            replay_refs=sorted_refs,
            mission_name=worst_mission,
        )
        if matched_best is not None:
            best_ref = matched_best
        if matched_worst is not None:
            worst_ref = matched_worst

    most_diagnostic_ref: Optional[str] = None
    if symptoms:
        top_symptom = symptoms[0]
        if top_symptom.axis == DiagnoseAxis.SOCIAL_COORDINATION:
            most_diagnostic_ref = next(
                (ref for ref in sorted_refs if "/replays_stage2/" in ref or "\\replays_stage2\\" in ref),
                None,
            )
        else:
            probe_mission = next(
                (probe.mission for probe in stage1_probe_catalog() if probe.axis == top_symptom.axis),
                None,
            )
            if probe_mission is not None:
                most_diagnostic_ref = next(iter(sorted(mission_replay_refs.get(probe_mission, []))), None) or (
                    _mission_replay_ref(replay_refs=sorted_refs, mission_name=probe_mission)
                )
    if most_diagnostic_ref is None:
        most_diagnostic_ref = worst_ref

    return ReplayExemplarRefs(
        best=best_ref,
        worst=worst_ref,
        most_diagnostic=most_diagnostic_ref,
    )


def build_stage1_prescriptions(symptoms: list[DiagnoseSymptom]) -> list[DiagnosePrescription]:
    prescriptions: list[DiagnosePrescription] = []
    for symptom in symptoms:
        _likely_cause, action, _expected_effect, threshold = _axis_symptom_profile(symptom.axis)
        prescriptions.append(
            DiagnosePrescription(
                symptom_id=symptom.symptom_id,
                action=action,
                owner="policy_researcher",
                validation_metric=f"{symptom.axis.value}.normalized_score",
                pass_fail_threshold=threshold,
            )
        )
    return prescriptions


def _dominant_issue(
    axis_scores: list[Stage1AxisScore],
    social_signal: Optional[Stage2SocialSignal] = None,
) -> DiagnoseDominantIssue:
    if social_signal is not None and social_signal.confirmed and social_signal.severity >= 0.55:
        return DiagnoseDominantIssue.SOCIAL_COORDINATION
    if not axis_scores:
        return DiagnoseDominantIssue.MIXED

    sorted_scores = sorted(axis_scores, key=lambda score: score.normalized_score)
    if len(sorted_scores) > 1 and (sorted_scores[1].normalized_score - sorted_scores[0].normalized_score) <= 5.0:
        return DiagnoseDominantIssue.MIXED

    weakest_axis = sorted_scores[0].axis
    if weakest_axis == DiagnoseAxis.STABILITY:
        return DiagnoseDominantIssue.STABILITY
    if weakest_axis == DiagnoseAxis.EFFICIENCY:
        return DiagnoseDominantIssue.SPEED
    if weakest_axis == DiagnoseAxis.CONTROL:
        return DiagnoseDominantIssue.STRATEGY
    return DiagnoseDominantIssue.MIXED


def build_stage1_doctor_note(
    *,
    config: DiagnoseRunConfig,
    stage_status: DiagnoseStageStatus,
    run_status: DiagnoseRunStatus,
    requirement_results: list[Stage1RequirementResult],
    metrics_by_mission: dict[str, list[Stage1MissionMetrics]],
    stage1_signals: list[Stage1AxisSignal],
    replay_refs: list[str],
    mission_replay_refs: Optional[dict[str, list[str]]] = None,
    notes: list[str],
    scripted_baseline_axis_scores: Optional[list[Stage1AxisScore]] = None,
    known_strong_axis_scores: Optional[list[Stage1AxisScore]] = None,
    social_signal: Optional[Stage2SocialSignal] = None,
    tournament_objective_context: Optional[TournamentObjectiveContext] = None,
) -> DiagnoseDoctorNote:
    raw_axis_scores = compute_stage1_axis_scores(
        requirement_results=requirement_results,
        metrics_by_mission=metrics_by_mission,
        stage1_signals=stage1_signals,
    )
    axis_scores, baseline_context = normalize_stage1_axis_scores(
        axis_scores=raw_axis_scores,
        scripted_baseline_axis_scores=scripted_baseline_axis_scores,
        known_strong_axis_scores=known_strong_axis_scores,
        scripted_policy=config.scripted_baseline_policy,
        known_strong_policy=config.known_strong_policy,
    )
    stage1_probe_evaluations = evaluate_stage1_probe_catalog(metrics_by_mission=metrics_by_mission)
    symptoms = rank_stage1_symptoms(axis_scores)
    social_symptom = build_stage2_social_symptom(social_signal=social_signal, replay_refs=replay_refs)
    if social_symptom is not None:
        symptoms.append(social_symptom)
        symptoms.sort(key=lambda symptom: (symptom.severity, symptom.confidence), reverse=True)
    stage2_diagnosis_delta = build_stage2_diagnosis_delta(axis_scores=axis_scores, social_signal=social_signal)
    replay_exemplars = select_replay_exemplars(
        metrics_by_mission=metrics_by_mission,
        replay_refs=replay_refs,
        symptoms=symptoms,
        mission_replay_refs=mission_replay_refs,
    )
    prescriptions = build_stage1_prescriptions(symptoms) if run_status == DiagnoseRunStatus.COMPLETE else []
    metric_refs = sorted({ref for axis_score in axis_scores for ref in axis_score.metric_refs})
    replay_ref_index = sorted({ref for axis_score in axis_scores for ref in axis_score.replay_refs}.union(replay_refs))
    baseline_refs = sorted({ref for axis_score in axis_scores for ref in axis_score.baseline_refs})
    note_items = list(notes)
    note_items.extend(baseline_context.normalization_notes)
    if config.scripted_baseline_policy is None or config.known_strong_policy is None:
        note_items.append(
            "Reference baselines are optional. Provide both --scripted-baseline-policy and "
            "--known-strong-policy for cross-policy normalization; otherwise standalone "
            "heuristic normalization is used."
        )
    if run_status == DiagnoseRunStatus.INCOMPLETE:
        note_items.append("Prescriptions withheld until diagnosis reaches complete status.")

    status = DoctorNoteStatus.COMPLETE if run_status != DiagnoseRunStatus.INCOMPLETE else DoctorNoteStatus.INCOMPLETE
    return DiagnoseDoctorNote(
        schema_version="v1",
        run_id=config.run_id,
        status=status,
        stage_status=stage_status,
        diagnosis_status=(
            DiagnosisLifecycleStatus.DIAGNOSIS_COMPLETE
            if run_status == DiagnoseRunStatus.COMPLETE
            else DiagnosisLifecycleStatus.DIAGNOSIS_INCOMPLETE
        ),
        dominant_issue=_dominant_issue(axis_scores, social_signal=social_signal),
        axes=axis_scores,
        stage1_probe_threshold_profile_id=STAGE1_PROBE_THRESHOLD_PROFILE_ID,
        stage1_probe_catalog=stage1_probe_catalog(),
        stage1_probe_evaluations=stage1_probe_evaluations,
        tournament_objective_context=tournament_objective_context or TournamentObjectiveContext(),
        baseline_context=baseline_context,
        derived_metrics=compute_stage1_derived_metrics(metrics_by_mission),
        social_review=social_signal,
        social_probe_catalog=stage2_social_probe_catalog(),
        stage2_diagnosis_delta=stage2_diagnosis_delta,
        replay_exemplars=replay_exemplars,
        symptoms=symptoms,
        prescriptions=prescriptions,
        evidence_index={
            "metric_refs": metric_refs,
            "replay_refs": replay_ref_index,
            "baseline_refs": baseline_refs,
        },
        missing_requirements=[result for result in requirement_results if not result.satisfied],
        notes=note_items,
    )


def _format_score(value: float) -> str:
    return f"{value:.1f}"


def _format_optional_score(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return _format_score(value)


def _replay_label(replay_ref: Optional[str]) -> str:
    if replay_ref is None:
        return "none"
    return Path(replay_ref).name


def _radar_point(center: float, radius: float, angle_deg: float, fraction: float) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    return (
        center + (radius * fraction * math.cos(radians)),
        center + (radius * fraction * math.sin(radians)),
    )


def _render_radar_chart(axis_scores: list[Stage1AxisScore]) -> str:
    if not axis_scores:
        return "<p>No axis scores available yet.</p>"

    score_by_axis = {axis_score.axis: axis_score.normalized_score for axis_score in axis_scores}
    axis_order = [
        (DiagnoseAxis.STABILITY, -90.0),
        (DiagnoseAxis.EFFICIENCY, 30.0),
        (DiagnoseAxis.CONTROL, 150.0),
    ]
    center = 160.0
    radius = 110.0

    grid_levels = [0.25, 0.5, 0.75, 1.0]
    grid_polygons = []
    for level in grid_levels:
        points = [
            _radar_point(center=center, radius=radius, angle_deg=angle, fraction=level) for _axis, angle in axis_order
        ]
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        grid_polygons.append(f'<polygon points="{point_text}" class="grid-level" />')

    axis_lines = []
    axis_labels = []
    for axis, angle in axis_order:
        end_x, end_y = _radar_point(center=center, radius=radius, angle_deg=angle, fraction=1.0)
        label_x, label_y = _radar_point(center=center, radius=radius, angle_deg=angle, fraction=1.18)
        axis_lines.append(
            f'<line x1="{center:.1f}" y1="{center:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" class="axis-line" />'
        )
        axis_labels.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" class="axis-label">{escape(axis.value.title())}</text>'
        )

    data_points = []
    for axis, angle in axis_order:
        score = score_by_axis.get(axis, 0.0)
        point = _radar_point(center=center, radius=radius, angle_deg=angle, fraction=_clamp01(score / 100.0))
        data_points.append(point)
    data_polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in data_points)

    return f"""
<svg class="radar-chart" viewBox="0 0 320 320" role="img" aria-label="Stage 1 radar chart">
  <g class="grid">
    {"".join(grid_polygons)}
  </g>
  <g class="axes">
    {"".join(axis_lines)}
    {"".join(axis_labels)}
  </g>
  <polygon points="{data_polygon}" class="data-polygon" />
</svg>
""".strip()


def render_diagnose_report_html(doctor_note: DiagnoseDoctorNote) -> str:
    assert doctor_note.diagnosis_status is not None, "DiagnoseDoctorNote.diagnosis_status must be set"
    diagnosis_status = doctor_note.diagnosis_status
    status_line = (
        f"run_id: {escape(doctor_note.run_id)} | "
        f"stage: {escape(doctor_note.stage_status.value)} | "
        f"status: {escape(doctor_note.status.value)} | "
        f"diagnosis: {escape(diagnosis_status.value)}"
    )
    reward_variance = _format_score(doctor_note.derived_metrics.reward_variance)
    non_zero_episode_pct = _format_score(doctor_note.derived_metrics.non_zero_episode_pct)
    timeout_rate = _format_score(doctor_note.derived_metrics.timeout_rate)
    top_symptom_id = doctor_note.symptoms[0].symptom_id if doctor_note.symptoms else "none"
    aligned_stage1 = _format_optional_score(doctor_note.tournament_objective_context.aligned_junction_held_stage1)
    aligned_stage2_absolute = _format_optional_score(
        doctor_note.tournament_objective_context.aligned_junction_held_stage2_absolute
    )
    aligned_stage2_mirror = _format_optional_score(
        doctor_note.tournament_objective_context.aligned_junction_held_stage2_mirror
    )

    axis_rows = "\n".join(
        f"<tr><td>{escape(axis_score.axis.value)}</td>"
        f"<td>{_format_score(axis_score.raw_score)}</td>"
        f"<td>{_format_score(axis_score.normalized_score)}</td>"
        f"<td>{escape(axis_score.normalization_mode)}</td>"
        f"<td>{escape(str(axis_score.confirmed).lower())}</td></tr>"
        for axis_score in doctor_note.axes
    )
    stage1_probe_rows = "\n".join(
        "<tr>"
        f"<td>{escape(probe.probe_id)}</td>"
        f"<td>{escape(probe.axis.value)}</td>"
        f"<td>{escape(probe.mission)}</td>"
        f"<td>{escape(probe.validation_metric)}</td>"
        f"<td>{escape(probe.pass_fail_threshold)}</td>"
        "</tr>"
        for probe in doctor_note.stage1_probe_catalog
    )
    if not stage1_probe_rows:
        stage1_probe_rows = "<tr><td colspan='5'>No Stage 1 probes configured.</td></tr>"
    stage1_probe_eval_rows = "\n".join(
        "<tr>"
        f"<td>{escape(evaluation.probe_id)}</td>"
        f"<td>{escape(str(evaluation.passed).lower())}</td>"
        f"<td>{escape(evaluation.summary)}</td>"
        f"<td>{escape('; '.join(evaluation.evidence_refs))}</td>"
        "</tr>"
        for evaluation in doctor_note.stage1_probe_evaluations
    )
    if not stage1_probe_eval_rows:
        stage1_probe_eval_rows = "<tr><td colspan='4'>No Stage 1 probe evaluations available.</td></tr>"

    symptom_rows = "\n".join(
        "<tr>"
        f"<td>{escape(symptom.symptom_id)}</td>"
        f"<td>{escape(symptom.axis.value)}</td>"
        f"<td>{_format_score(symptom.severity * 100.0)}</td>"
        f"<td>{_format_score(symptom.confidence * 100.0)}</td>"
        f"<td>{escape(symptom.likely_cause)}</td>"
        "</tr>"
        for symptom in doctor_note.symptoms
    )
    if not symptom_rows:
        symptom_rows = "<tr><td colspan='5'>No Stage 1 red-flag symptoms.</td></tr>"

    prescription_rows = "\n".join(
        "<tr>"
        f"<td>{escape(prescription.symptom_id)}</td>"
        f"<td>{escape(prescription.action)}</td>"
        f"<td>{escape(prescription.owner)}</td>"
        f"<td>{escape(prescription.validation_metric)}</td>"
        f"<td>{escape(prescription.pass_fail_threshold)}</td>"
        "</tr>"
        for prescription in doctor_note.prescriptions
    )
    if not prescription_rows:
        prescription_rows = "<tr><td colspan='5'>No prescriptions generated.</td></tr>"

    replay_refs = doctor_note.evidence_index["replay_refs"]
    replay_links = "\n".join(
        f"<li><a href='{escape(replay_ref)}'>{escape(Path(replay_ref).name)}</a></li>" for replay_ref in replay_refs
    )
    if not replay_links:
        replay_links = "<li>No replay evidence linked.</li>"
    replay_exemplar_lines = "\n".join(
        [
            f"<li><strong>best:</strong> {escape(_replay_label(doctor_note.replay_exemplars.best))}</li>",
            f"<li><strong>worst:</strong> {escape(_replay_label(doctor_note.replay_exemplars.worst))}</li>",
            (
                "<li><strong>most diagnostic:</strong> "
                f"{escape(_replay_label(doctor_note.replay_exemplars.most_diagnostic))}</li>"
            ),
        ]
    )

    note_items = "\n".join(f"<li>{escape(note)}</li>" for note in doctor_note.notes)
    if not note_items:
        note_items = "<li>No notes.</li>"

    baseline_rows = "\n".join(
        "<tr>"
        f"<td>{escape(baseline.role)}</td>"
        f"<td>{escape(baseline.policy)}</td>"
        f"<td>{_format_score(_safe_mean([axis.normalized_score for axis in baseline.axis_scores]))}</td>"
        "</tr>"
        for baseline in doctor_note.baseline_context.baselines
    )
    if not baseline_rows:
        baseline_rows = "<tr><td colspan='3'>No baseline policies evaluated.</td></tr>"

    social_delta_line = ""
    if doctor_note.stage2_diagnosis_delta is not None:
        social_delta_line = (
            f"<p><strong>stage2 diagnosis delta:</strong> {escape(doctor_note.stage2_diagnosis_delta.summary)}</p>"
        )

    social_section = ""
    if doctor_note.social_review is not None:
        social_section = (
            "<section class='card'>"
            "<h2>Stage 2 Social Review</h2>"
            f"<p><strong>confirmed:</strong> {escape(str(doctor_note.social_review.confirmed).lower())}</p>"
            f"<p><strong>severity:</strong> {_format_score(doctor_note.social_review.severity * 100.0)}</p>"
            f"<p><strong>confidence:</strong> {_format_score(doctor_note.social_review.confidence * 100.0)}</p>"
            f"<p>{escape(doctor_note.social_review.summary)}</p>"
            f"{social_delta_line}"
            "</section>"
        )

    social_probe_rows = "\n".join(
        "<tr>"
        f"<td>{escape(probe.probe_id)}</td>"
        f"<td>{escape(probe.mode.value)}</td>"
        f"<td>{escape(probe.validation_metric)}</td>"
        f"<td>{escape(probe.pass_fail_threshold)}</td>"
        "</tr>"
        for probe in doctor_note.social_probe_catalog
    )
    if not social_probe_rows:
        social_probe_rows = "<tr><td colspan='4'>No social probes configured.</td></tr>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cogames Diagnose Report</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --card: #fffdf8;
      --ink: #171412;
      --muted: #5b5148;
      --line: #d5c8b7;
      --accent: #0d6f65;
      --alert: #a63f2f;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
      background: radial-gradient(circle at top, #f9f3ea 0%, var(--bg) 60%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1080px;
      margin: 24px auto;
      padding: 0 16px 24px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 14px;
    }}
    h1, h2 {{
      margin: 0 0 10px;
      letter-spacing: 0.02em;
    }}
    .status {{
      color: var(--muted);
    }}
    .dominant {{
      color: var(--accent);
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f7efe4;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(160px, 1fr));
      gap: 10px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #fff9f1;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .metric .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .radar-wrap {{
      display: flex;
      justify-content: center;
    }}
    .radar-chart {{
      width: 360px;
      height: 360px;
    }}
    .grid-level {{
      fill: none;
      stroke: #dbcdbb;
      stroke-width: 1;
    }}
    .axis-line {{
      stroke: #c7b7a3;
      stroke-width: 1;
    }}
    .axis-label {{
      font-size: 11px;
      text-anchor: middle;
      fill: var(--muted);
    }}
    .data-polygon {{
      fill: rgba(13, 111, 101, 0.22);
      stroke: var(--accent);
      stroke-width: 2;
    }}
    .severity {{
      color: var(--alert);
      font-weight: 700;
    }}
    .triage-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(220px, 1fr));
      gap: 10px;
      margin-bottom: 10px;
    }}
    .triage-card {{
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fef8ef;
      padding: 10px;
    }}
    .triage-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .triage-value {{
      font-size: 20px;
      font-weight: 700;
    }}
    .objective-anchor {{
      border: 1px solid #c6b6a1;
      border-radius: 10px;
      background: #f7efe4;
      padding: 10px;
      margin-top: 10px;
    }}
    @media (max-width: 820px) {{
      .metrics {{
        grid-template-columns: 1fr;
      }}
      .triage-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Cogames Diagnose Doctor Note</h1>
      <div class="status">{status_line}</div>
      <p class="dominant">Dominant issue: {escape(doctor_note.dominant_issue.value)}</p>
      <ul>{note_items}</ul>
    </section>

    <section class="card">
      <h2>5-Minute Triage</h2>
      <div class="triage-grid">
        <div class="triage-card">
          <div class="triage-label">Primary Classification</div>
          <div class="triage-value">{escape(doctor_note.dominant_issue.value)}</div>
        </div>
        <div class="triage-card">
          <div class="triage-label">Top Red Flag</div>
          <div class="triage-value">{escape(top_symptom_id)}</div>
        </div>
      </div>
      <ol>
        <li>Confirm visit completeness: diagnosis should be diagnosis_complete and stage2_completed.</li>
        <li>Classify dominant issue from this card: stability, speed, strategy, or social_coordination.</li>
        <li>Use ranked symptoms + replay evidence to choose the first prescription to run.</li>
        <li>Validate pass/fail thresholds and rerun with fixed seeds to confirm interpretation stability.</li>
      </ol>
      <p><a href="bundle_guide.md">Open reproducible bundle guide (Markdown)</a></p>
      <p><a href="diagnose_validity.json">Open run validity checks (JSON)</a></p>
      <p><a href="stage1_metric_correlation.json">Open Stage 1 objective correlation + threshold tuning (JSON)</a></p>
      <p><a href="metrics.json">Open aggregated metrics (JSON)</a></p>
      <div class="objective-anchor">
        Tournament objective anchor: keep <code>aligned.junction.held</code> visible in analysis and treat non-aligned
        diagnostics as supporting evidence.
      </div>
    </section>

    <section class="card">
      <h2>Tournament Objective Context</h2>
      <table>
        <thead><tr><th>objective</th><th>stage1</th><th>stage2_absolute</th><th>stage2_mirror</th></tr></thead>
        <tbody>
          <tr><td>aligned.junction.held</td><td>{aligned_stage1}</td><td>{aligned_stage2_absolute}</td><td>{aligned_stage2_mirror}</td></tr>
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>Stage 1 Profile</h2>
      <div class="metrics">
        <div class="metric"><div class="label">Reward Variance</div><div class="value">{reward_variance}</div></div>
        <div class="metric">
          <div class="label">Non-zero Episode %</div><div class="value">{non_zero_episode_pct}</div>
        </div>
        <div class="metric"><div class="label">Timeout Rate</div><div class="value">{timeout_rate}</div></div>
      </div>
      <div class="radar-wrap">{_render_radar_chart(doctor_note.axes)}</div>
      <p>Normalization mode: {escape(doctor_note.baseline_context.normalization_mode)}</p>
      <table>
        <thead>
          <tr><th>Axis</th><th>Raw Score</th><th>Normalized Score</th><th>Normalization</th><th>Confirmed</th></tr>
        </thead>
        <tbody>{axis_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Stage 1 Probe Catalog</h2>
      <p>Threshold profile: <code>{escape(doctor_note.stage1_probe_threshold_profile_id)}</code></p>
      <table>
        <thead>
          <tr><th>probe_id</th><th>axis</th><th>mission</th><th>validation metric</th><th>pass/fail threshold</th></tr>
        </thead>
        <tbody>{stage1_probe_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Stage 1 Probe Results</h2>
      <table>
        <thead>
          <tr><th>probe_id</th><th>passed</th><th>summary</th><th>evidence</th></tr>
        </thead>
        <tbody>{stage1_probe_eval_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Baseline Context</h2>
      <table>
        <thead><tr><th>role</th><th>policy</th><th>avg axis score</th></tr></thead>
        <tbody>{baseline_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Ranked Symptoms</h2>
      <table>
        <thead>
          <tr><th>symptom_id</th><th>axis</th><th>severity</th><th>confidence</th><th>likely cause</th></tr>
        </thead>
        <tbody>{symptom_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Prescriptions</h2>
      <table>
        <thead>
          <tr>
            <th>symptom_id</th><th>action</th><th>owner</th><th>validation metric</th><th>pass/fail threshold</th>
          </tr>
        </thead>
        <tbody>{prescription_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Replay Evidence</h2>
      <p><a href="replay_bundle.zip">Download replay bundle (zip)</a></p>
      <p><a href="replay_exemplars.json">Open replay exemplars (JSON)</a></p>
      <ul>{replay_exemplar_lines}</ul>
      <ul>{replay_links}</ul>
    </section>
    <section class="card">
      <h2>Social Probe Catalog</h2>
      <table>
        <thead><tr><th>probe_id</th><th>mode</th><th>validation metric</th><th>pass/fail threshold</th></tr></thead>
        <tbody>{social_probe_rows}</tbody>
      </table>
    </section>
    {social_section}
  </div>
</body>
</html>
"""


def write_doctor_note_bundle(output_dir: Path, doctor_note: DiagnoseDoctorNote) -> None:
    write_json(output_dir / "doctor_note.json", doctor_note)
    write_html(output_dir / "diagnose_report.html", render_diagnose_report_html(doctor_note))


def interpretation_snapshot_from_doctor_note(doctor_note: DiagnoseDoctorNote, *, label: str) -> InterpretationSnapshot:
    top_symptom_ids = [symptom.symptom_id for symptom in doctor_note.symptoms[:3]]
    return InterpretationSnapshot(
        label=label,
        run_id=doctor_note.run_id,
        dominant_issue=doctor_note.dominant_issue,
        top_symptom_ids=top_symptom_ids,
    )


def evaluate_interpretation_stability(snapshots: list[InterpretationSnapshot]) -> InterpretationStabilityReport:
    if not snapshots:
        return InterpretationStabilityReport(
            snapshot_count=0,
            compared_run_ids=[],
            dominant_issue_stable=False,
            top_symptom_stable=False,
            stable=False,
            notes=["No interpretation snapshots were provided."],
            snapshots=[],
        )

    dominant_issues = {snapshot.dominant_issue for snapshot in snapshots}
    top_symptom_sets = {tuple(snapshot.top_symptom_ids) for snapshot in snapshots}
    dominant_issue_stable = len(dominant_issues) == 1
    top_symptom_stable = len(top_symptom_sets) == 1
    stable = dominant_issue_stable and top_symptom_stable and len(snapshots) > 1

    notes: list[str] = []
    if len(snapshots) <= 1:
        notes.append(
            "Only one interpretation snapshot available; rerun with fixed seeds and compare run dirs for stability."
        )
    if not dominant_issue_stable:
        notes.append("Dominant issue changed across compared runs.")
    if not top_symptom_stable:
        notes.append("Top symptom ranking changed across compared runs.")
    if dominant_issue_stable and top_symptom_stable and len(snapshots) > 1:
        notes.append("Interpretation is stable across compared runs.")

    return InterpretationStabilityReport(
        snapshot_count=len(snapshots),
        compared_run_ids=[snapshot.run_id for snapshot in snapshots],
        dominant_issue_stable=dominant_issue_stable,
        top_symptom_stable=top_symptom_stable,
        stable=stable,
        notes=notes,
        snapshots=snapshots,
    )


def evaluate_diagnose_validity(
    *,
    stage_status: DiagnoseStageStatus,
    requirement_results: list[Stage1RequirementResult],
    stage1_signals: list[Stage1AxisSignal],
    stage1_replay_count: int,
    expected_stage1_replay_count: int,
    stage2_absolute_summary: Optional[Stage2ModeSummary],
    stage2_mirror_summary: Optional[Stage2ModeSummary],
    stage2_replay_count: int,
    expected_stage2_replay_count: int,
    social_signal: Optional[Stage2SocialSignal],
) -> DiagnoseValidityReport:
    checks = [
        DiagnoseValidityCheck(
            check_id="stage1.required_axes",
            passed=bool(requirement_results) and all(result.satisfied for result in requirement_results),
            details="Stage 1 pack includes at least one probe for stability, efficiency, and control.",
        ),
        DiagnoseValidityCheck(
            check_id="stage1.replay_evidence",
            passed=stage1_replay_count >= expected_stage1_replay_count and expected_stage1_replay_count > 0,
            details=f"Stage 1 replay evidence {stage1_replay_count}/{expected_stage1_replay_count}.",
        ),
        DiagnoseValidityCheck(
            check_id="stage1.signal_confirmation",
            passed=bool(stage1_signals) and all(signal.confirmed for signal in stage1_signals),
            details="Stage 1 confirmed stability, efficiency, and control signals with replay evidence.",
        ),
        DiagnoseValidityCheck(
            check_id="stage2.absolute_mode",
            passed=stage2_absolute_summary is not None and stage2_absolute_summary.case_count > 0,
            details="Stage 2 absolute scrimmage mode was executed.",
        ),
        DiagnoseValidityCheck(
            check_id="stage2.mirror_mode",
            passed=stage2_mirror_summary is not None and stage2_mirror_summary.case_count > 0,
            details="Stage 2 mirror scrimmage mode was executed.",
        ),
        DiagnoseValidityCheck(
            check_id="stage2.replay_evidence",
            passed=stage2_replay_count >= expected_stage2_replay_count and expected_stage2_replay_count > 0,
            details=f"Stage 2 replay evidence {stage2_replay_count}/{expected_stage2_replay_count}.",
        ),
        DiagnoseValidityCheck(
            check_id="stage2.social_signal",
            passed=social_signal is not None and social_signal.confirmed,
            details="Stage 2 social signal was confirmed with absolute and mirror evidence.",
        ),
        DiagnoseValidityCheck(
            check_id="stage2.final_stage_status",
            passed=stage_status == DiagnoseStageStatus.STAGE2_COMPLETED,
            details=f"Final stage status is {stage_status.value}.",
        ),
    ]
    failed_check_ids = [check.check_id for check in checks if not check.passed]
    return DiagnoseValidityReport(
        valid=not failed_check_ids,
        failed_check_ids=failed_check_ids,
        checks=checks,
    )


def build_manifest(
    *,
    config: DiagnoseRunConfig,
    state: DiagnoseRunState,
    command: str,
    git_sha: Optional[str],
    seeds: dict[str, int],
    artifact_files: list[str],
    interpretation_stability: InterpretationStabilityReport,
    diagnose_validity: DiagnoseValidityReport,
) -> DiagnoseManifest:
    return DiagnoseManifest(
        schema_version="v1",
        run_id=config.run_id,
        created_at=config.created_at,
        command=command,
        git_sha=git_sha,
        pack_id=config.pack_id,
        pack_version=config.pack_version,
        mission_set=config.mission_set,
        policy=config.policy,
        scripted_baseline_policy=config.scripted_baseline_policy,
        known_strong_policy=config.known_strong_policy,
        stage_status=state.stage_status,
        run_status=state.run_status,
        seeds=seeds,
        artifact_files=artifact_files,
        interpretation_stability=interpretation_stability,
        diagnose_validity=diagnose_validity,
    )


@dataclass(frozen=True)
class DiagnoseCase:
    name: str
    env_cfg: MettaGridConfig


def _load_eval_missions(module_path: str) -> list[CvCMission]:
    module = importlib.import_module(module_path)
    missions = getattr(module, "EVAL_MISSIONS", None)
    if missions is None:
        raise AttributeError(f"Module '{module_path}' does not define EVAL_MISSIONS")
    return list(missions)


def _load_diagnose_missions(mission_set: str) -> list[CvCMission]:
    if mission_set == "all":
        from cogames.cogs_vs_clips.evals.cogsguard_evals import COGSGUARD_EVAL_MISSIONS  # noqa: PLC0415
        from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS  # noqa: PLC0415
        from cogames.cogs_vs_clips.missions import MISSIONS as ALL_MISSIONS  # noqa: PLC0415

        missions_list: list[CvCMission] = []
        missions_list.extend(COGSGUARD_EVAL_MISSIONS)
        missions_list.extend(_load_eval_missions("cogames.cogs_vs_clips.evals.integrated_evals"))
        missions_list.extend(_load_eval_missions("cogames.cogs_vs_clips.evals.spanning_evals"))
        missions_list.extend([mission_cls() for mission_cls in DIAGNOSTIC_EVALS])  # type: ignore[call-arg]
        eval_mission_names = {mission.name for mission in missions_list}
        for mission in ALL_MISSIONS:
            if mission.name not in eval_mission_names:
                missions_list.append(mission)
        return missions_list

    if mission_set == "cogsguard_evals":
        from cogames.cogs_vs_clips.evals.cogsguard_evals import COGSGUARD_EVAL_MISSIONS  # noqa: PLC0415

        return list(COGSGUARD_EVAL_MISSIONS)

    if mission_set == "diagnostic_evals":
        from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS  # noqa: PLC0415

        return [mission_cls() for mission_cls in DIAGNOSTIC_EVALS]  # type: ignore[call-arg]

    if mission_set == "tournament":
        from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS  # noqa: PLC0415

        missions_list = []
        missions_list.extend(_load_eval_missions("cogames.cogs_vs_clips.evals.integrated_evals"))
        missions_list.extend([mission_cls() for mission_cls in DIAGNOSTIC_EVALS])  # type: ignore[call-arg]
        return missions_list

    if mission_set == "integrated_evals":
        return _load_eval_missions("cogames.cogs_vs_clips.evals.integrated_evals")

    if mission_set == "spanning_evals":
        return _load_eval_missions("cogames.cogs_vs_clips.evals.spanning_evals")

    raise ValueError(f"Unknown mission set: {mission_set}")


def _matches_experiment(mission_name: str, experiment_filters: set[str]) -> bool:
    if not experiment_filters:
        return True
    if mission_name in experiment_filters:
        return True
    suffix = f".{mission_name}"
    return any(name.endswith(suffix) for name in experiment_filters)


def _cogs_for_mission(mission: CvCMission, cogs_list: list[int], respect_cogs_list: bool) -> list[int]:
    fixed_cogs = getattr(mission, "num_cogs", None)
    if fixed_cogs is not None:
        if respect_cogs_list and fixed_cogs not in cogs_list:
            return []
        return [fixed_cogs]
    site = getattr(mission, "site", None)
    if site is None:
        return list(cogs_list)
    min_cogs = getattr(site, "min_cogs", None)
    max_cogs = getattr(site, "max_cogs", None)
    return [
        num_cogs
        for num_cogs in cogs_list
        if (min_cogs is None or num_cogs >= min_cogs) and (max_cogs is None or num_cogs <= max_cogs)
    ]


def _build_diagnose_case(mission: CvCMission, num_cogs: int, steps: int) -> DiagnoseCase:
    mission_with_cogs = mission.with_variants([NumCogsVariant(num_cogs=num_cogs)])
    env_cfg = mission_with_cogs.make_env()
    env_cfg.game.max_steps = steps
    name = f"{mission.full_name()} (cogs={num_cogs})"
    return DiagnoseCase(name=name, env_cfg=env_cfg)


def _build_diagnose_cases(
    *,
    mission_set: str,
    experiments: Optional[list[str]],
    cogs: Optional[list[int]],
    steps: int,
) -> list[DiagnoseCase]:
    experiment_filters = set(experiments or [])
    cogs_list = cogs if cogs else [1, 2, 4]
    respect_cogs_list = cogs is not None
    cases: list[DiagnoseCase] = []

    missions = _load_diagnose_missions(mission_set)
    for mission in missions:
        if not _matches_experiment(mission.name, experiment_filters):
            continue
        for num_cogs in _cogs_for_mission(mission, cogs_list, respect_cogs_list):
            cases.append(_build_diagnose_case(mission, num_cogs, steps))

    return cases


def _reset_diagnose_replay_artifacts(output_dir: Path) -> None:
    def _remove_path(path: Path) -> None:
        # Never follow symlinks into recursive deletes.
        if path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    # Clear replay directories and stability rerun outputs.
    for relative_path in replay_artifact_paths():
        _remove_path(output_dir / relative_path)

    # When reusing an output dir, stale top-level artifacts from prior runs can
    # survive (e.g. stage2_* files) and get swept into the new manifest.
    for pattern in ("*.json", "*.html", "*.md", "*.zip"):
        for artifact_path in output_dir.glob(pattern):
            _remove_path(artifact_path)


def _help_callback(ctx: typer.Context, value: bool) -> None:
    if value:
        console.print(ctx.get_help())
        raise typer.Exit()


def diagnose_cmd(
    ctx: typer.Context,
    policy: str = typer.Argument(
        ...,
        metavar="POLICY",
        help=f"Policy specification: {policy_arg_example}",
    ),
    # --- Evaluation ---
    mission_set: Literal[
        "cogsguard_evals",
        "diagnostic_evals",
        "integrated_evals",
        "spanning_evals",
        "tournament",
        "all",
    ] = typer.Option(
        "cogsguard_evals",
        "--mission-set",
        "-S",
        metavar="SET",
        help="Eval suite to run (full Stage 2 diagnosis currently requires cogsguard_evals).",
        rich_help_panel="Evaluation",
    ),
    experiments: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--experiments",
        metavar="NAME",
        help="Specific experiments (subset of mission set).",
        rich_help_panel="Evaluation",
    ),
    cogs: Optional[list[int]] = typer.Option(  # noqa: B008
        None,
        "--cogs",
        "-c",
        metavar="N",
        help="Agent counts to test (repeatable).",
        rich_help_panel="Evaluation",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        metavar="DEVICE",
        help="Policy device (auto, cpu, cuda, cuda:0, etc.).",
        rich_help_panel="Evaluation",
    ),
    # --- Simulation ---
    steps: int = typer.Option(
        1000,
        "--steps",
        "-s",
        metavar="N",
        help="Max steps per episode.",
        rich_help_panel="Simulation",
    ),
    episodes: int = typer.Option(
        3,
        "--episodes",
        "-e",
        metavar="N",
        help="Episodes per case.",
        rich_help_panel="Simulation",
    ),
    output_dir: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        metavar="DIR",
        help="Directory for structured diagnose artifacts.",
        rich_help_panel="Output",
    ),
    scripted_baseline_policy: Optional[str] = typer.Option(  # noqa: B008
        None,
        "--scripted-baseline-policy",
        metavar="POLICY",
        help=f"Reference scripted baseline policy ({policy_arg_example})",
        rich_help_panel="Evaluation",
    ),
    known_strong_policy: Optional[str] = typer.Option(  # noqa: B008
        None,
        "--known-strong-policy",
        metavar="POLICY",
        help=f"Reference known-strong policy ({policy_arg_example})",
        rich_help_panel="Evaluation",
    ),
    compare_run_dir: Optional[list[Path]] = typer.Option(  # noqa: B008
        None,
        "--compare-run-dir",
        metavar="DIR",
        help="Previous diagnose run directory containing doctor_note.json for interpretation stability comparison.",
        rich_help_panel="Output",
    ),
    stability_reruns: int = typer.Option(
        0,
        "--stability-reruns",
        metavar="N",
        help="Number of additional fixed-seed reruns to compute interpretation stability snapshots.",
        min=0,
        rich_help_panel="Output",
    ),
    require_stable_interpretation: bool = typer.Option(
        False,
        "--require-stable-interpretation",
        help="Mark diagnosis incomplete when interpretation stability check fails.",
        rich_help_panel="Output",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    pack = COGSGUARD_STAGE1_PACK_V1
    resolved_cogs = cogs if cogs else list(COGSGUARD_STAGE1_FIXED_COGS)
    stage1_seed = 42
    stage2_absolute_seed = 43
    stage2_mirror_seed = 44
    config = build_run_config(
        output_dir=output_dir,
        policy=policy,
        mission_set=mission_set,
        experiments=experiments,
        cogs=resolved_cogs,
        steps=steps,
        episodes=episodes,
        pack=pack,
        scripted_baseline_policy=scripted_baseline_policy,
        known_strong_policy=known_strong_policy,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _reset_diagnose_replay_artifacts(config.output_dir)
    write_json(config.output_dir / "run_config.json", config)
    write_json(
        config.output_dir / "stage1_probe_catalog.json",
        {"probes": [probe.model_dump(mode="json") for probe in stage1_probe_catalog()]},
    )
    write_json(
        config.output_dir / "stage1_probe_threshold_profile.json",
        stage1_probe_threshold_profile(),
    )
    write_json(
        config.output_dir / "stage2_probe_catalog.json",
        {"probes": [probe.model_dump(mode="json") for probe in stage2_social_probe_catalog()]},
    )
    scripted_baseline_axis_scores: Optional[list[Stage1AxisScore]] = None
    known_strong_axis_scores: Optional[list[Stage1AxisScore]] = None
    baseline_notes: list[str] = []
    command = " ".join(sys.argv)
    comparison_snapshots: list[InterpretationSnapshot] = []
    stability_rerun_snapshots: list[InterpretationSnapshot] = []
    expected_stage1_replay_count = 0
    stage1_replay_count = 0
    stage2_expected_replay_count = 0
    stage2_replay_count = 0
    stage2_absolute_summary: Optional[Stage2ModeSummary] = None
    stage2_mirror_summary: Optional[Stage2ModeSummary] = None
    stage2_social_signal: Optional[Stage2SocialSignal] = None
    stage1_objective_by_mission: dict[str, list[float]] = {}
    stage1_mission_replay_refs: dict[str, list[str]] = {}
    pack_contract_report = evaluate_stage1_pack_contract(
        mission_set=mission_set,
        cogs=resolved_cogs,
        steps=steps,
        episodes=episodes,
        pack=pack,
    )
    write_json(config.output_dir / "stage1_pack_contract.json", pack_contract_report)
    tournament_objective_context = TournamentObjectiveContext()

    def _git_sha() -> Optional[str]:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _replay_ref(path: Path) -> str:
        # Keep refs relative to config.output_dir so bundles remain portable and
        # HTML report links don't accidentally double-prefix output_dir.
        base_dir = config.output_dir.resolve() if path.is_absolute() else config.output_dir
        try:
            return path.relative_to(base_dir).as_posix()
        except ValueError:
            return str(path)

    def _write_stage1_doctor_note(
        *,
        stage_status: DiagnoseStageStatus,
        run_status: DiagnoseRunStatus,
        requirement_results: list[Stage1RequirementResult],
        metrics_by_mission: dict[str, list[Stage1MissionMetrics]],
        stage1_signals: list[Stage1AxisSignal],
        replay_refs: list[str],
        mission_replay_refs: Optional[dict[str, list[str]]] = None,
        notes: list[str],
        social_signal: Optional[Stage2SocialSignal] = None,
    ) -> DiagnoseDoctorNote:
        doctor_note = build_stage1_doctor_note(
            config=config,
            stage_status=stage_status,
            run_status=run_status,
            requirement_results=requirement_results,
            metrics_by_mission=metrics_by_mission,
            stage1_signals=stage1_signals,
            replay_refs=replay_refs,
            mission_replay_refs=mission_replay_refs,
            notes=[*notes, *baseline_notes],
            scripted_baseline_axis_scores=scripted_baseline_axis_scores,
            known_strong_axis_scores=known_strong_axis_scores,
            social_signal=social_signal,
            tournament_objective_context=tournament_objective_context,
        )
        write_doctor_note_bundle(config.output_dir, doctor_note)
        write_json(
            config.output_dir / "stage1_probe_evaluations.json",
            {"probes": [probe.model_dump(mode="json") for probe in doctor_note.stage1_probe_evaluations]},
        )
        write_json(config.output_dir / "replay_exemplars.json", doctor_note.replay_exemplars)
        write_json(
            config.output_dir / "metrics.json",
            {
                "missions": {
                    mission_name: [metric.model_dump(mode="json") for metric in mission_metrics_list]
                    for mission_name, mission_metrics_list in metrics_by_mission.items()
                },
                "derived_metrics": doctor_note.derived_metrics.model_dump(mode="json"),
                "axis_scores_normalized": [axis.model_dump(mode="json") for axis in doctor_note.axes],
                "baseline_context": doctor_note.baseline_context.model_dump(mode="json"),
                "objective_by_mission": stage1_objective_by_mission,
            },
        )
        write_json(
            config.output_dir / "stage1_metric_correlation.json",
            evaluate_stage1_metric_correlation(
                metrics_by_mission=metrics_by_mission,
                objective_by_mission=stage1_objective_by_mission,
            ),
        )
        if doctor_note.stage2_diagnosis_delta is not None:
            write_json(
                config.output_dir / "stage2_diagnosis_delta.json",
                doctor_note.stage2_diagnosis_delta,
            )
        return doctor_note

    def _write_repro_artifacts(
        *,
        state: DiagnoseRunState,
        doctor_note: DiagnoseDoctorNote,
    ) -> InterpretationStabilityReport:
        write_replay_bundle(config.output_dir)
        snapshots = [interpretation_snapshot_from_doctor_note(doctor_note, label="current")]
        snapshots.extend(comparison_snapshots)
        snapshots.extend(stability_rerun_snapshots)
        stability = evaluate_interpretation_stability(snapshots)
        write_json(config.output_dir / "interpretation_stability.json", stability)
        validity = evaluate_diagnose_validity(
            stage_status=state.stage_status,
            requirement_results=state.requirement_results,
            stage1_signals=state.stage1_signals,
            stage1_replay_count=stage1_replay_count,
            expected_stage1_replay_count=expected_stage1_replay_count,
            stage2_absolute_summary=stage2_absolute_summary,
            stage2_mirror_summary=stage2_mirror_summary,
            stage2_replay_count=stage2_replay_count,
            expected_stage2_replay_count=stage2_expected_replay_count,
            social_signal=stage2_social_signal,
        )
        write_json(config.output_dir / "diagnose_validity.json", validity)
        write_bundle_guide(
            output_dir=config.output_dir,
            state=state,
            interpretation_stability=stability,
            diagnose_validity=validity,
        )

        artifact_files = sorted(
            str(path.relative_to(config.output_dir)) for path in config.output_dir.rglob("*") if path.is_file()
        )
        artifact_files = sorted({*artifact_files, "manifest.json"})
        manifest = build_manifest(
            config=config,
            state=state,
            command=command,
            git_sha=_git_sha(),
            seeds={
                "stage1": stage1_seed,
                "stage2_absolute": stage2_absolute_seed,
                "stage2_mirror": stage2_mirror_seed,
            },
            artifact_files=artifact_files,
            interpretation_stability=stability,
            diagnose_validity=validity,
        )
        write_json(config.output_dir / "manifest.json", manifest)
        return stability

    def _persist_incomplete_state(
        *,
        stage_status: DiagnoseStageStatus,
        requirement_results: list[Stage1RequirementResult],
        expected_replay_count: int,
        replay_count: int,
        notes: list[str],
        stage1_signals: list[Stage1AxisSignal],
        metrics_by_mission: dict[str, list[Stage1MissionMetrics]],
        replay_refs: list[str],
        mission_replay_refs: Optional[dict[str, list[str]]] = None,
        social_signal: Optional[Stage2SocialSignal] = None,
    ) -> None:
        state = build_incomplete_state(
            config=config,
            stage_status=stage_status,
            requirement_results=requirement_results,
            expected_replay_count=expected_replay_count,
            replay_count=replay_count,
            notes=notes,
            stage1_signals=stage1_signals,
        )
        write_json(config.output_dir / "diagnose_state.json", state)
        doctor_note = _write_stage1_doctor_note(
            stage_status=state.stage_status,
            run_status=state.run_status,
            requirement_results=requirement_results,
            metrics_by_mission=metrics_by_mission,
            stage1_signals=stage1_signals,
            replay_refs=replay_refs,
            mission_replay_refs=mission_replay_refs,
            notes=state.notes,
            social_signal=social_signal,
        )
        _write_repro_artifacts(state=state, doctor_note=doctor_note)

    for idx, run_dir in enumerate(compare_run_dir or [], start=1):
        doctor_note_path = run_dir / "doctor_note.json"
        if not doctor_note_path.exists():
            baseline_notes.append(f"Comparison run missing doctor_note.json: {doctor_note_path}")
            continue
        try:
            comparison_note = load_doctor_note(doctor_note_path)
        except (OSError, ValueError) as error:
            baseline_notes.append(f"Failed to parse comparison doctor note at {doctor_note_path}: {error}")
            continue
        comparison_snapshots.append(
            interpretation_snapshot_from_doctor_note(
                comparison_note,
                label=f"compare_{idx}",
            )
        )

    if mission_set == pack.mission_set and not pack_contract_report.valid:
        failed_details = [check.details for check in pack_contract_report.checks if not check.passed]
        baseline_notes.append("Standalone diagnose mode: fixed-pack contract checks did not pass.")
        baseline_notes.extend(f"Fixed-pack check: {detail}" for detail in failed_details)
        console.print(
            "[yellow]Continuing diagnose run despite fixed-pack contract mismatches (standalone mode).[/yellow]"
        )
        for detail in failed_details:
            console.print(f"[yellow]- {detail}[/yellow]")

    cases = _build_diagnose_cases(
        mission_set=mission_set,
        experiments=experiments,
        cogs=cogs,
        steps=steps,
    )
    if not cases:
        _persist_incomplete_state(
            stage_status=DiagnoseStageStatus.STAGE1_INCOMPLETE,
            requirement_results=[],
            expected_replay_count=0,
            replay_count=0,
            notes=["No evaluation cases matched the filters."],
            stage1_signals=[],
            metrics_by_mission={},
            replay_refs=[],
        )
        console.print("[red]No evaluation cases matched your filters.[/red]")
        console.print(f"[yellow]Diagnosis marked incomplete. Artifacts: {config.output_dir}[/yellow]")
        raise typer.Exit(1)

    case_names = [case.name for case in cases]
    gate = evaluate_stage1_gate(case_names=case_names, pack=pack)
    write_json(config.output_dir / "stage1_gate.json", gate)
    write_json(
        config.output_dir / "stage1_summary.json",
        stage1_summary_payload(
            mission_names=[mission_from_case_name(case.name) for case in cases],
            episode_count=episodes,
            case_names=case_names,
        ),
    )

    uses_stage1_pack = use_stage1_pack(mission_set)
    if not uses_stage1_pack or not gate.satisfied:
        notes = []
        if not uses_stage1_pack:
            notes.append(f"Selected mission_set does not match required Stage 1 pack mission_set ({pack.mission_set}).")
        if not gate.satisfied:
            notes.append("Stage 1 required signals were not all satisfied. Stage 2 is blocked.")
        _persist_incomplete_state(
            stage_status=DiagnoseStageStatus.STAGE1_INCOMPLETE,
            requirement_results=gate.results,
            expected_replay_count=0,
            replay_count=0,
            notes=notes,
            stage1_signals=[],
            metrics_by_mission={},
            replay_refs=[],
        )
        console.print("[yellow]Diagnosis incomplete: Stage 1 gate not satisfied.[/yellow]")
        console.print(f"[yellow]Artifacts written to: {config.output_dir}[/yellow]")
        return

    resolved_device = resolve_training_device(console, device)
    policy_spec = get_policy_spec(ctx, policy, device=str(resolved_device))
    replay_dir = config.output_dir / "replays"

    def _evaluate_stage1_cases_with_replay_refs(
        *,
        cases_to_run: list[DiagnoseCase],
    ) -> tuple[list[MultiEpisodeRolloutSummary], dict[str, list[str]]]:
        replay_dir.mkdir(parents=True, exist_ok=True)
        known_replay_paths = {path.resolve() for path in replay_dir.glob("*.json.z")}
        stage1_summaries: list[MultiEpisodeRolloutSummary] = []
        replay_refs_by_mission: dict[str, list[str]] = {}

        for case in cases_to_run:
            case_summaries = evaluate_module.evaluate(
                console,
                missions=[(case.name, case.env_cfg)],
                policy_specs=[policy_spec],
                proportions=[1.0],
                action_timeout_ms=10000,
                episodes=episodes,
                seed=stage1_seed,
                device=str(resolved_device),
                save_replay=str(replay_dir),
            )
            stage1_summaries.extend(case_summaries)

            current_replay_paths = {path.resolve() for path in replay_dir.glob("*.json.z")}
            new_replay_paths = sorted(_replay_ref(path) for path in (current_replay_paths - known_replay_paths))
            known_replay_paths = current_replay_paths
            mission_name = mission_from_case_name(case.name)
            replay_refs_by_mission.setdefault(mission_name, []).extend(new_replay_paths)

        return stage1_summaries, replay_refs_by_mission

    def _evaluate_baseline_axis_scores(
        *,
        baseline_label: str,
        baseline_policy_arg: Optional[str],
    ) -> tuple[Optional[list[Stage1AxisScore]], Optional[dict]]:
        if baseline_policy_arg is None:
            return None, None

        baseline_spec = get_policy_spec(ctx, baseline_policy_arg, device=str(resolved_device))
        console.print(f"[cyan]Running {baseline_label} baseline evaluation...[/cyan]")
        try:
            baseline_summaries = evaluate_module.evaluate(
                console,
                missions=[(case.name, case.env_cfg) for case in cases],
                policy_specs=[baseline_spec],
                proportions=[1.0],
                action_timeout_ms=10000,
                episodes=episodes,
                seed=stage1_seed,
                device=str(resolved_device),
                save_replay=None,
            )
        except Exception as error:  # pragma: no cover - depends on policy/runtime behavior
            note = f"{baseline_label} baseline evaluation failed for '{baseline_policy_arg}': {error}"
            baseline_notes.append(note)
            console.print(f"[yellow]{note}[/yellow]")
            return None, {"policy": baseline_policy_arg, "error": str(error)}
        baseline_metrics = stage1_metrics_by_mission(
            case_names=case_names,
            summaries=baseline_summaries,
        )
        baseline_signals = assess_stage1_signals(
            requirement_results=gate.results,
            metrics_by_mission=baseline_metrics,
            replay_refs=[],
            require_replay_evidence=False,
        )
        baseline_axis_scores = compute_stage1_axis_scores(
            requirement_results=gate.results,
            metrics_by_mission=baseline_metrics,
            stage1_signals=baseline_signals,
        )
        baseline_payload = {
            "policy": baseline_policy_arg,
            "missions": {
                mission_name: [metric.model_dump(mode="json") for metric in mission_metrics_list]
                for mission_name, mission_metrics_list in baseline_metrics.items()
            },
            "axis_scores": [axis_score.model_dump(mode="json") for axis_score in baseline_axis_scores],
        }
        return baseline_axis_scores, baseline_payload

    def _run_fixed_seed_stability_rerun(
        *,
        rerun_index: int,
    ) -> Optional[InterpretationSnapshot]:
        rerun_label = f"rerun_{rerun_index}"
        rerun_root = config.output_dir / "stability_reruns" / rerun_label
        rerun_stage1_replay_dir = rerun_root / "replays_stage1"
        rerun_stage2_absolute_replay_dir = rerun_root / "replays_stage2" / "absolute"
        rerun_stage2_mirror_replay_dir = rerun_root / "replays_stage2" / "mirror"

        try:
            rerun_summaries = evaluate_module.evaluate(
                console,
                missions=[(case.name, case.env_cfg) for case in cases],
                policy_specs=[policy_spec],
                proportions=[1.0],
                action_timeout_ms=10000,
                episodes=episodes,
                seed=stage1_seed,
                device=str(resolved_device),
                save_replay=str(rerun_stage1_replay_dir),
            )
        except Exception as error:  # pragma: no cover - runtime dependent
            baseline_notes.append(f"Stability {rerun_label} failed during Stage 1 rollout: {error}")
            return None

        rerun_stage1_metrics = stage1_metrics_by_mission(
            case_names=case_names,
            summaries=rerun_summaries,
        )
        rerun_stage1_replay_refs = sorted(_replay_ref(path) for path in rerun_stage1_replay_dir.glob("*.json.z"))
        rerun_stage1_signals = assess_stage1_signals(
            requirement_results=gate.results,
            metrics_by_mission=rerun_stage1_metrics,
            replay_refs=rerun_stage1_replay_refs,
        )

        if not all(signal.confirmed for signal in rerun_stage1_signals):
            note = build_stage1_doctor_note(
                config=config,
                stage_status=DiagnoseStageStatus.STAGE1_INCOMPLETE,
                run_status=DiagnoseRunStatus.INCOMPLETE,
                requirement_results=gate.results,
                metrics_by_mission=rerun_stage1_metrics,
                stage1_signals=rerun_stage1_signals,
                replay_refs=rerun_stage1_replay_refs,
                notes=[f"Stability {rerun_label}: Stage 1 not fully confirmed."],
                scripted_baseline_axis_scores=scripted_baseline_axis_scores,
                known_strong_axis_scores=known_strong_axis_scores,
            )
            return interpretation_snapshot_from_doctor_note(note, label=rerun_label)

        try:
            rerun_absolute_summaries = evaluate_module.evaluate(
                console,
                missions=[(case.name, case.env_cfg) for case in cases],
                policy_specs=[policy_spec],
                proportions=[1.0],
                action_timeout_ms=10000,
                episodes=episodes,
                seed=stage2_absolute_seed,
                device=str(resolved_device),
                save_replay=str(rerun_stage2_absolute_replay_dir),
            )
            rerun_mirror_summaries = evaluate_module.evaluate(
                console,
                missions=[(case.name, case.env_cfg) for case in cases],
                policy_specs=[policy_spec, policy_spec],
                proportions=[0.5, 0.5],
                action_timeout_ms=10000,
                episodes=episodes,
                seed=stage2_mirror_seed,
                device=str(resolved_device),
                save_replay=str(rerun_stage2_mirror_replay_dir),
            )
        except Exception as error:  # pragma: no cover - runtime dependent
            baseline_notes.append(f"Stability {rerun_label} failed during Stage 2 rollout: {error}")
            note = build_stage1_doctor_note(
                config=config,
                stage_status=DiagnoseStageStatus.STAGE2_INCOMPLETE,
                run_status=DiagnoseRunStatus.INCOMPLETE,
                requirement_results=gate.results,
                metrics_by_mission=rerun_stage1_metrics,
                stage1_signals=rerun_stage1_signals,
                replay_refs=rerun_stage1_replay_refs,
                notes=[f"Stability {rerun_label}: Stage 2 failed before completion."],
                scripted_baseline_axis_scores=scripted_baseline_axis_scores,
                known_strong_axis_scores=known_strong_axis_scores,
            )
            return interpretation_snapshot_from_doctor_note(note, label=rerun_label)

        rerun_absolute_summary = build_stage2_mode_summary(
            mode=Stage2Mode.ABSOLUTE,
            seed=stage2_absolute_seed,
            case_names=case_names,
            summaries=rerun_absolute_summaries,
        )
        rerun_mirror_summary = build_stage2_mode_summary(
            mode=Stage2Mode.MIRROR,
            seed=stage2_mirror_seed,
            case_names=case_names,
            summaries=rerun_mirror_summaries,
        )
        rerun_stage2_replay_refs = sorted(
            _replay_ref(path) for path in rerun_stage2_absolute_replay_dir.glob("*.json.z")
        )
        rerun_stage2_replay_refs.extend(
            sorted(_replay_ref(path) for path in rerun_stage2_mirror_replay_dir.glob("*.json.z"))
        )
        rerun_social_signal = assess_stage2_social_signal(
            absolute_summary=rerun_absolute_summary,
            mirror_summary=rerun_mirror_summary,
            replay_refs=rerun_stage2_replay_refs,
        )
        rerun_note = build_stage1_doctor_note(
            config=config,
            stage_status=(
                DiagnoseStageStatus.STAGE2_COMPLETED
                if rerun_social_signal.confirmed
                else DiagnoseStageStatus.STAGE2_INCOMPLETE
            ),
            run_status=(DiagnoseRunStatus.COMPLETE if rerun_social_signal.confirmed else DiagnoseRunStatus.INCOMPLETE),
            requirement_results=gate.results,
            metrics_by_mission=rerun_stage1_metrics,
            stage1_signals=rerun_stage1_signals,
            replay_refs=[*rerun_stage1_replay_refs, *rerun_stage2_replay_refs],
            notes=[f"Stability {rerun_label}: fixed-seed rerun snapshot."],
            scripted_baseline_axis_scores=scripted_baseline_axis_scores,
            known_strong_axis_scores=known_strong_axis_scores,
            social_signal=rerun_social_signal,
        )
        return interpretation_snapshot_from_doctor_note(rerun_note, label=rerun_label)

    running_state = DiagnoseRunState(
        run_id=config.run_id,
        created_at=config.created_at,
        stage_status=DiagnoseStageStatus.STAGE1_RUNNING,
        run_status=DiagnoseRunStatus.INCOMPLETE,
        diagnosis_status=DiagnosisLifecycleStatus.DIAGNOSIS_INCOMPLETE,
        pack_id=config.pack_id,
        pack_version=config.pack_version,
        missing_requirements=[],
        requirement_results=gate.results,
        expected_replay_count=expected_replays(len(cases), episodes),
        replay_count=0,
        output_dir=str(config.output_dir),
        notes=["Stage 1 is running."],
        stage1_signals=[],
    )
    write_json(config.output_dir / "diagnose_state.json", running_state)

    console.print(f"[cyan]Running diagnostic evaluation ({len(cases)} cases)...[/cyan]")
    summaries, stage1_mission_replay_refs = _evaluate_stage1_cases_with_replay_refs(cases_to_run=cases)
    tournament_objective_context = tournament_objective_context.model_copy(
        update={
            "aligned_junction_held_stage1": compute_tournament_objective_value(summaries=summaries),
        }
    )

    mission_payload = {
        "missions": [
            {
                "mission_name": case.name,
                "mission_summary": summary.model_dump(mode="json"),
            }
            for case, summary in zip(cases, summaries, strict=True)
        ]
    }
    write_json(config.output_dir / "stage1_rollout_summary.json", mission_payload)
    mission_metrics = stage1_metrics_by_mission(case_names=case_names, summaries=summaries)
    stage1_objective_by_mission = stage1_objective_values_by_mission(
        case_names=case_names,
        summaries=summaries,
    )
    stage1_metric_correlation = evaluate_stage1_metric_correlation(
        metrics_by_mission=mission_metrics,
        objective_by_mission=stage1_objective_by_mission,
    )
    expected_replay_count = expected_replays(len(cases), episodes)
    replay_count = count_replays(replay_dir)
    expected_stage1_replay_count = expected_replay_count
    stage1_replay_count = replay_count
    replay_refs = sorted(_replay_ref(path) for path in replay_dir.glob("*.json.z"))
    stage1_signals = assess_stage1_signals(
        requirement_results=gate.results,
        metrics_by_mission=mission_metrics,
        replay_refs=replay_refs,
    )
    scripted_baseline_payload = None
    if scripted_baseline_policy is not None:
        scripted_baseline_axis_scores, scripted_baseline_payload = _evaluate_baseline_axis_scores(
            baseline_label="scripted",
            baseline_policy_arg=scripted_baseline_policy,
        )

    known_strong_payload = None
    if known_strong_policy is not None:
        known_strong_axis_scores, known_strong_payload = _evaluate_baseline_axis_scores(
            baseline_label="known-strong",
            baseline_policy_arg=known_strong_policy,
        )

    axis_scores_raw = compute_stage1_axis_scores(
        requirement_results=gate.results,
        metrics_by_mission=mission_metrics,
        stage1_signals=stage1_signals,
    )
    axis_scores_normalized, baseline_context = normalize_stage1_axis_scores(
        axis_scores=axis_scores_raw,
        scripted_baseline_axis_scores=scripted_baseline_axis_scores,
        known_strong_axis_scores=known_strong_axis_scores,
        scripted_policy=scripted_baseline_policy,
        known_strong_policy=known_strong_policy,
    )
    baselines_payload = {
        "normalization_context": baseline_context.model_dump(mode="json"),
        "scripted_baseline": scripted_baseline_payload,
        "known_strong": known_strong_payload,
    }
    write_json(config.output_dir / "stage1_baselines.json", baselines_payload)
    metrics_payload = {
        "missions": {
            mission_name: [metric.model_dump(mode="json") for metric in mission_metrics_list]
            for mission_name, mission_metrics_list in mission_metrics.items()
        },
        "derived_metrics": compute_stage1_derived_metrics(mission_metrics).model_dump(mode="json"),
        "axis_scores_raw": [axis_score.model_dump(mode="json") for axis_score in axis_scores_raw],
        "axis_scores_normalized": [axis_score.model_dump(mode="json") for axis_score in axis_scores_normalized],
        "baseline_context": baseline_context.model_dump(mode="json"),
        "objective_by_mission": stage1_objective_by_mission,
        "stage1_metric_correlation": stage1_metric_correlation.model_dump(mode="json"),
    }
    write_json(config.output_dir / "stage1_metrics.json", metrics_payload)
    write_json(
        config.output_dir / "stage1_signals.json",
        {"signals": [signal.model_dump(mode="json") for signal in stage1_signals]},
    )

    if replay_count < expected_replay_count:
        _persist_incomplete_state(
            stage_status=DiagnoseStageStatus.STAGE1_INCOMPLETE,
            requirement_results=gate.results,
            expected_replay_count=expected_replay_count,
            replay_count=replay_count,
            notes=[
                "Replay evidence is insufficient for Stage 1.",
                "Stage 2 is blocked until Stage 1 replay evidence is complete.",
            ],
            stage1_signals=stage1_signals,
            metrics_by_mission=mission_metrics,
            replay_refs=replay_refs,
            mission_replay_refs=stage1_mission_replay_refs,
        )
        console.print("[yellow]Diagnosis incomplete: missing Stage 1 replay evidence.[/yellow]")
        console.print(f"[yellow]Artifacts written to: {config.output_dir}[/yellow]")
        return

    if not all(signal.confirmed for signal in stage1_signals):
        _persist_incomplete_state(
            stage_status=DiagnoseStageStatus.STAGE1_INCOMPLETE,
            requirement_results=gate.results,
            expected_replay_count=expected_replay_count,
            replay_count=replay_count,
            notes=[
                "Stage 1 signal confirmation is incomplete.",
                "Stage 2 is blocked until stability, efficiency, and control signals are confirmed.",
            ],
            stage1_signals=stage1_signals,
            metrics_by_mission=mission_metrics,
            replay_refs=replay_refs,
            mission_replay_refs=stage1_mission_replay_refs,
        )
        console.print("[yellow]Diagnosis incomplete: Stage 1 signals are not fully confirmed.[/yellow]")
        console.print(f"[yellow]Artifacts written to: {config.output_dir}[/yellow]")
        return

    stage1_passed_state = build_stage1_passed_state(
        config=config,
        requirement_results=gate.results,
        expected_replay_count=expected_replay_count,
        replay_count=replay_count,
        stage1_signals=stage1_signals,
    )
    write_json(config.output_dir / "diagnose_state.json", stage1_passed_state)
    console.print("[green]Stage 1 passed for social review.[/green]")

    stage2_expected_replay_count = expected_replays(len(cases), episodes) * 2
    stage2_running_state = DiagnoseRunState(
        run_id=config.run_id,
        created_at=config.created_at,
        stage_status=DiagnoseStageStatus.STAGE2_RUNNING,
        run_status=DiagnoseRunStatus.INCOMPLETE,
        diagnosis_status=DiagnosisLifecycleStatus.DIAGNOSIS_INCOMPLETE,
        pack_id=config.pack_id,
        pack_version=config.pack_version,
        missing_requirements=[],
        requirement_results=gate.results,
        expected_replay_count=expected_replay_count + stage2_expected_replay_count,
        replay_count=replay_count,
        output_dir=str(config.output_dir),
        notes=["Stage 2 social review is running."],
        stage1_signals=stage1_signals,
    )
    write_json(config.output_dir / "diagnose_state.json", stage2_running_state)

    stage2_absolute_replay_dir = config.output_dir / "replays_stage2" / "absolute"
    stage2_mirror_replay_dir = config.output_dir / "replays_stage2" / "mirror"

    console.print(f"[cyan]Running Stage 2 absolute social review ({len(cases)} cases)...[/cyan]")
    absolute_summaries = evaluate_module.evaluate(
        console,
        missions=[(case.name, case.env_cfg) for case in cases],
        policy_specs=[policy_spec],
        proportions=[1.0],
        action_timeout_ms=10000,
        episodes=episodes,
        seed=stage2_absolute_seed,
        device=str(resolved_device),
        save_replay=str(stage2_absolute_replay_dir),
    )
    absolute_stage2 = build_stage2_mode_summary(
        mode=Stage2Mode.ABSOLUTE,
        seed=stage2_absolute_seed,
        case_names=case_names,
        summaries=absolute_summaries,
    )
    tournament_objective_context = tournament_objective_context.model_copy(
        update={
            "aligned_junction_held_stage2_absolute": compute_tournament_objective_value(summaries=absolute_summaries),
        }
    )
    stage2_absolute_summary = absolute_stage2
    write_json(config.output_dir / "stage2_absolute_summary.json", absolute_stage2)

    console.print(f"[cyan]Running Stage 2 mirror social review ({len(cases)} cases)...[/cyan]")
    mirror_summaries = evaluate_module.evaluate(
        console,
        missions=[(case.name, case.env_cfg) for case in cases],
        policy_specs=[policy_spec, policy_spec],
        proportions=[0.5, 0.5],
        action_timeout_ms=10000,
        episodes=episodes,
        seed=stage2_mirror_seed,
        device=str(resolved_device),
        save_replay=str(stage2_mirror_replay_dir),
    )
    mirror_stage2 = build_stage2_mode_summary(
        mode=Stage2Mode.MIRROR,
        seed=stage2_mirror_seed,
        case_names=case_names,
        summaries=mirror_summaries,
    )
    tournament_objective_context = tournament_objective_context.model_copy(
        update={
            "aligned_junction_held_stage2_mirror": compute_tournament_objective_value(summaries=mirror_summaries),
        }
    )
    stage2_mirror_summary = mirror_stage2
    write_json(config.output_dir / "stage2_mirror_summary.json", mirror_stage2)

    stage2_absolute_replay_refs = sorted(_replay_ref(path) for path in stage2_absolute_replay_dir.glob("*.json.z"))
    stage2_mirror_replay_refs = sorted(_replay_ref(path) for path in stage2_mirror_replay_dir.glob("*.json.z"))
    stage2_replay_refs = [*stage2_absolute_replay_refs, *stage2_mirror_replay_refs]
    stage2_replay_count = len(stage2_replay_refs)
    total_replay_count = replay_count + stage2_replay_count
    expected_total_replays = expected_replay_count + stage2_expected_replay_count

    social_signal = assess_stage2_social_signal(
        absolute_summary=absolute_stage2,
        mirror_summary=mirror_stage2,
        replay_refs=stage2_replay_refs,
    )
    stage2_social_signal = social_signal
    write_json(config.output_dir / "stage2_social_signal.json", social_signal)

    if stage2_replay_count < stage2_expected_replay_count:
        _persist_incomplete_state(
            stage_status=DiagnoseStageStatus.STAGE2_INCOMPLETE,
            requirement_results=gate.results,
            expected_replay_count=expected_total_replays,
            replay_count=total_replay_count,
            notes=[
                "Stage 2 replay evidence is insufficient.",
                "Diagnosis remains incomplete until absolute and mirror social review replays are complete.",
            ],
            stage1_signals=stage1_signals,
            metrics_by_mission=mission_metrics,
            replay_refs=[*replay_refs, *stage2_replay_refs],
            mission_replay_refs=stage1_mission_replay_refs,
            social_signal=social_signal,
        )
        console.print("[yellow]Diagnosis incomplete: missing Stage 2 replay evidence.[/yellow]")
        console.print(f"[yellow]Artifacts written to: {config.output_dir}[/yellow]")
        return

    if not social_signal.confirmed:
        _persist_incomplete_state(
            stage_status=DiagnoseStageStatus.STAGE2_INCOMPLETE,
            requirement_results=gate.results,
            expected_replay_count=expected_total_replays,
            replay_count=total_replay_count,
            notes=[
                "Stage 2 social signal confirmation is incomplete.",
                "Diagnosis requires both absolute and mirror social evidence.",
            ],
            stage1_signals=stage1_signals,
            metrics_by_mission=mission_metrics,
            replay_refs=[*replay_refs, *stage2_replay_refs],
            mission_replay_refs=stage1_mission_replay_refs,
            social_signal=social_signal,
        )
        console.print("[yellow]Diagnosis incomplete: Stage 2 social signal not confirmed.[/yellow]")
        console.print(f"[yellow]Artifacts written to: {config.output_dir}[/yellow]")
        return

    complete_state = build_stage2_completed_state(
        config=config,
        requirement_results=gate.results,
        expected_replay_count=expected_total_replays,
        replay_count=total_replay_count,
        stage1_signals=stage1_signals,
        notes=[
            "Stage 1 and Stage 2 completed.",
            social_signal.summary,
        ],
    )
    write_json(config.output_dir / "diagnose_state.json", complete_state)
    doctor_note = _write_stage1_doctor_note(
        stage_status=complete_state.stage_status,
        run_status=complete_state.run_status,
        requirement_results=gate.results,
        metrics_by_mission=mission_metrics,
        stage1_signals=stage1_signals,
        replay_refs=[*replay_refs, *stage2_replay_refs],
        mission_replay_refs=stage1_mission_replay_refs,
        notes=complete_state.notes,
        social_signal=social_signal,
    )
    if stability_reruns > 0:
        console.print(f"[cyan]Running {stability_reruns} fixed-seed stability rerun(s)...[/cyan]")
        for rerun_index in range(1, stability_reruns + 1):
            snapshot = _run_fixed_seed_stability_rerun(rerun_index=rerun_index)
            if snapshot is not None:
                stability_rerun_snapshots.append(snapshot)
        write_json(
            config.output_dir / "stability_rerun_snapshots.json",
            {"snapshots": [snapshot.model_dump(mode="json") for snapshot in stability_rerun_snapshots]},
        )

    stability = _write_repro_artifacts(state=complete_state, doctor_note=doctor_note)
    if require_stable_interpretation and not stability.stable:
        _persist_incomplete_state(
            stage_status=DiagnoseStageStatus.STAGE2_INCOMPLETE,
            requirement_results=gate.results,
            expected_replay_count=expected_total_replays,
            replay_count=total_replay_count,
            notes=[
                "Interpretation stability check failed under required reproducibility gate.",
                "Diagnosis requires stable dominant issue and top symptoms across fixed-seed comparisons.",
                *stability.notes,
            ],
            stage1_signals=stage1_signals,
            metrics_by_mission=mission_metrics,
            replay_refs=[*replay_refs, *stage2_replay_refs],
            social_signal=social_signal,
        )
        console.print("[yellow]Diagnosis incomplete: interpretation stability requirement failed.[/yellow]")
        console.print(f"[yellow]Artifacts written to: {config.output_dir}[/yellow]")
        return

    console.print("[green]Stage 2 social review completed.[/green]")
    console.print(f"[green]Diagnosis complete. Artifacts written to: {config.output_dir}[/green]")
