from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import log
from pathlib import Path

from ...trajectory_generator import GeneratedTrajectoryDataset, TrajectoryArtifact, _class_by_name, _make_manual_trajectory, _tier_by_name
from ...utils.io import _write_json, _write_text, write_csv
from ...utils.math import _clamp
from ...analysis.feature_analysis import _one_dimensional_feature_context_from_trajectory
from ..gym_types import CorpusGymAction
from ..gym_utils import _class_validity_score, _leakage_penalty, _physical_invalidity_penalty


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    objective_id: str
    family: str
    level: str
    version: int
    status: str
    scope: dict[str, object]
    target: dict[str, object]
    score: dict[str, object]
    optimizer_support: dict[str, object]


@dataclass(frozen=True, slots=True)
class ObjectiveScore:
    objective_id: str
    candidate_id: str
    trajectory_id: str
    level: str
    score: float
    passed_constraints: bool
    primary_terms: dict[str, float]
    penalties: dict[str, float]
    metadata: dict[str, object] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        return {
            "objective_id": self.objective_id,
            "candidate_id": self.candidate_id,
            "trajectory_id": self.trajectory_id,
            "level": self.level,
            "score": self.score,
            "passed_constraints": self.passed_constraints,
            **self.primary_terms,
            **self.penalties,
            **self.metadata,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveProofArtifacts:
    run_dir: Path
    objective_spec_schema_path: Path
    objective_score_schema_path: Path
    objective_family_catalog_path: Path
    proof_ladder_path: Path
    posterior_target_spec_path: Path
    posterior_target_witness_scores_path: Path
    report_path: Path


def posterior_target_objective_spec(
    *,
    objective_id: str = "posterior_target__cv_ca_50_50",
    class_a: str = "constant_velocity",
    class_b: str = "constant_acceleration",
    weight_a: float = 0.5,
    weight_b: float = 0.5,
) -> ObjectiveSpec:
    return ObjectiveSpec(
        objective_id=objective_id,
        family="posterior_target_distribution",
        level="trajectory",
        version=1,
        status="experimental",
        scope={
            "backend_ids": ["controlled_1d", "parameter_only_1d", "sequential_control_1d"],
            "class_set_id": "common_1d_motion_classes",
            "feature_set_id": "common_1d_kinematic_features",
            "evidence_provider_id": "class_similarity_proxy_v1",
            "prior_regime_id": "uniform",
        },
        target={
            "posterior_distribution": {
                class_a: weight_a,
                class_b: weight_b,
            },
            "target_time": "final",
        },
        score={
            "primary_metric": "posterior_total_variation",
            "transform": "one_minus_tv",
            "min_validity": 0.35,
            "penalties": {
                "leakage": 1.0,
                "degeneracy": 1.0,
                "class_invalidity": 1.0,
            },
        },
        optimizer_support={
            "random_search": True,
            "cem": True,
            "ppo": "terminal_reward",
            "qd_archive": False,
        },
    )


def posterior_target_spec_from_payload(
    *,
    objective_id: str,
    target_distribution: dict[str, float],
    evidence_provider_id: str = "class_similarity_proxy_v1",
) -> ObjectiveSpec:
    normalised = _normalise_distribution(target_distribution)
    return ObjectiveSpec(
        objective_id=objective_id,
        family="posterior_target_distribution",
        level="trajectory",
        version=1,
        status="experimental",
        scope={
            "backend_ids": ["controlled_1d", "parameter_only_1d", "sequential_control_1d"],
            "class_set_id": "common_1d_motion_classes",
            "feature_set_id": "common_1d_kinematic_features",
            "evidence_provider_id": evidence_provider_id,
            "prior_regime_id": "uniform",
        },
        target={"posterior_distribution": normalised, "target_time": "final"},
        score={
            "primary_metric": "posterior_total_variation",
            "transform": "one_minus_tv",
            "min_validity": 0.35,
            "penalties": {"leakage": 1.0, "degeneracy": 1.0, "class_invalidity": 1.0},
        },
        optimizer_support={"random_search": True, "cem": True, "ppo": "terminal_reward", "qd_archive": False},
    )


def objective_spec_schema() -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ObjectiveSpec",
        "type": "object",
        "required": ["objective_id", "family", "level", "version", "status", "scope", "target", "score", "optimizer_support"],
        "properties": {
            "objective_id": {"type": "string"},
            "family": {"type": "string"},
            "level": {"enum": ["trajectory", "corpus", "study"]},
            "version": {"type": "integer"},
            "status": {"enum": ["experimental", "validated", "deprecated"]},
            "scope": {"type": "object"},
            "target": {"type": "object"},
            "score": {"type": "object"},
            "optimizer_support": {"type": "object"},
        },
    }


def objective_score_schema() -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ObjectiveScore",
        "type": "object",
        "required": ["objective_id", "candidate_id", "trajectory_id", "level", "score", "passed_constraints"],
        "properties": {
            "objective_id": {"type": "string"},
            "candidate_id": {"type": "string"},
            "trajectory_id": {"type": "string"},
            "level": {"enum": ["trajectory", "corpus", "study"]},
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "passed_constraints": {"type": "boolean"},
        },
    }


def objective_family_catalog_rows() -> tuple[dict[str, object], ...]:
    return (
        {"family": "class_composition", "level": "corpus", "purpose": "Build balanced or intentionally imbalanced corpora."},
        {"family": "posterior_target_distribution", "level": "trajectory", "purpose": "Generate trajectories with desired posterior mass."},
        {"family": "boundary_closeness", "level": "trajectory", "purpose": "Generate low-margin classifier-boundary cases as a proxy."},
        {"family": "feature_region", "level": "trajectory", "purpose": "Hit specific feature-space cells, rows, or zones."},
        {"family": "feature_excitation", "level": "corpus", "purpose": "Ensure intended features are exercised."},
        {"family": "novelty_archive_coverage", "level": "trajectory/corpus", "purpose": "Fill useful feature/class/backend archive cells."},
        {"family": "class_validity", "level": "trajectory/corpus", "purpose": "Ensure labels are meaningful under declared class definitions."},
        {"family": "leakage_degeneracy_penalty", "level": "trajectory/corpus", "purpose": "Prevent shortcut examples from winning."},
        {"family": "stress_difficulty", "level": "trajectory/corpus", "purpose": "Generate hard but valid examples."},
        {"family": "prior_sensitivity", "level": "trajectory/study", "purpose": "Find cases where conclusions depend on priors."},
        {"family": "oracle_gap", "level": "study", "purpose": "Find cases where richer evidence succeeds and a candidate rung fails."},
        {"family": "rung_specific_witness", "level": "study/corpus", "purpose": "Generate escalation witnesses for the method ladder."},
        {"family": "environment_provenance", "level": "corpus", "purpose": "Ensure examples carry enough context and backend metadata."},
    )


def proof_ladder_rows() -> tuple[dict[str, object], ...]:
    return (
        {"level": 0, "name": "contract_proof", "meaning": "Schema validates; score is finite, bounded, deterministic, and artifacted."},
        {"level": 1, "name": "unit_witness_proof", "meaning": "Canonical positive and negative examples score in the expected order."},
        {"level": 2, "name": "optimizer_proof", "meaning": "Random/CEM/PPO can improve the objective over baseline."},
        {"level": 3, "name": "cross_metric_proof", "meaning": "Optimizing the objective changes the downstream metric it claims to affect."},
        {"level": 4, "name": "decision_proof", "meaning": "The objective supports a promote/revise/reject/defer decision."},
        {"level": 5, "name": "mathematical_proof", "meaning": "Closed-form or assumption-bound proof for a simplified family."},
    )


def _normalise_distribution(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in values.values())
    if total <= 0.0:
        uniform = 1.0 / max(len(values), 1)
        return {key: uniform for key in values}
    return {key: max(0.0, float(value)) / total for key, value in values.items()}


def _trajectory_dataset(trajectory: TrajectoryArtifact, tier_name: str) -> GeneratedTrajectoryDataset:
    return GeneratedTrajectoryDataset(
        tier=tier_name,
        seed=trajectory.seed,
        class_definitions=(_class_by_name(trajectory.true_class),),
        tier_definition=_tier_by_name(tier_name),
        trajectories=(trajectory,),
    )


def class_similarity_posterior(trajectory: TrajectoryArtifact, *, class_names: tuple[str, ...], tier_name: str) -> dict[str, float]:
    context = _one_dimensional_feature_context_from_trajectory(_trajectory_dataset(trajectory, tier_name), trajectory)
    velocities = tuple(float(value) for value in trajectory.true_velocity or ())
    accelerations = tuple(float(value) for value in trajectory.true_acceleration or ())
    if not velocities or not accelerations:
        return _normalise_distribution({class_name: 1.0 for class_name in class_names})
    accel_abs_mean = sum(abs(value) for value in accelerations) / len(accelerations)
    accel_mean = sum(accelerations) / len(accelerations)
    velocity_delta = velocities[-1] - velocities[0]
    scores = {
        "constant_velocity": max(0.0, 1.0 - min(accel_abs_mean / 0.20, 1.0)) * max(0.0, 1.0 - min(context.acceleration_variance / 0.03, 1.0)),
        "constant_acceleration": min(accel_abs_mean / 0.45, 1.0) * max(0.0, 1.0 - min(context.acceleration_variance / 0.03, 1.0)),
        "braking": min(abs(min(accel_mean, 0.0)) / 0.55, 1.0) * min(max(-velocity_delta, 0.0) / 0.7, 1.0),
        "maneuver": min(context.acceleration_range / 0.50, 1.0) * max(context.acceleration_sign_changes / 1.0, 0.35 if context.acceleration_range > 0.25 else 0.0),
    }
    selected = {class_name: scores.get(class_name, 0.0) + 1e-6 for class_name in class_names}
    return _normalise_distribution(selected)


def _total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)


def _kl_divergence(target: dict[str, float], achieved: dict[str, float]) -> float:
    value = 0.0
    for key, target_mass in target.items():
        q = max(float(target_mass), 1e-12)
        p = max(float(achieved.get(key, 0.0)), 1e-12)
        value += q * log(q / p)
    return value


def _entropy(distribution: dict[str, float]) -> float:
    return -sum(float(value) * log(max(float(value), 1e-12)) for value in distribution.values())


def score_posterior_target_distribution(
    spec: ObjectiveSpec,
    trajectory: TrajectoryArtifact,
    *,
    action: CorpusGymAction | None = None,
    candidate_id: str = "candidate",
    backend_id: str = "unknown",
    tier_name: str = "boundary_v1",
) -> ObjectiveScore:
    target_distribution = _normalise_distribution(dict(spec.target["posterior_distribution"]))
    class_names = tuple(target_distribution)
    achieved = class_similarity_posterior(trajectory, class_names=class_names, tier_name=tier_name)
    tv_error = _total_variation(target_distribution, achieved)
    l1_error = 2.0 * tv_error
    kl_error = _kl_divergence(target_distribution, achieved)
    entropy = _entropy(achieved)
    base_score = _clamp(1.0 - tv_error, 0.0, 1.0)
    validity = _class_validity_score(trajectory, tier_name)
    leakage = _leakage_penalty(action, trajectory, tier_name) if action is not None else 0.0
    invalidity = _physical_invalidity_penalty(trajectory)
    min_validity = float(spec.score.get("min_validity", 0.0))
    passed = validity >= min_validity and leakage <= 0.50 and invalidity <= 0.50
    gate = 1.0 if passed else 0.0
    score = _clamp(base_score * gate, 0.0, 1.0)
    metadata: dict[str, object] = {
        "backend_id": backend_id,
        "evidence_provider_id": spec.scope.get("evidence_provider_id", ""),
        "prior_regime_id": spec.scope.get("prior_regime_id", ""),
        "target_distribution": str(target_distribution),
        "achieved_distribution": str(achieved),
    }
    for class_name, mass in target_distribution.items():
        metadata[f"target_posterior_{class_name}"] = mass
    for class_name, mass in achieved.items():
        metadata[f"achieved_posterior_{class_name}"] = mass
    return ObjectiveScore(
        objective_id=spec.objective_id,
        candidate_id=candidate_id,
        trajectory_id=trajectory.trajectory_id,
        level=spec.level,
        score=score,
        passed_constraints=passed,
        primary_terms={
            "posterior_tv_error": tv_error,
            "posterior_l1_error": l1_error,
            "posterior_kl_error": kl_error,
            "posterior_entropy": entropy,
            "posterior_target_score": base_score,
        },
        penalties={
            "leakage_penalty": leakage,
            "degeneracy_penalty": invalidity,
            "class_invalidity_penalty": 1.0 - validity,
        },
        metadata=metadata,
    )


def objective_proof_report() -> str:
    return "\n".join(
        [
            "# Objective Function Proof Status",
            "",
            "The objective suite claims contract-tested and empirically instrumented objectives, not universal mathematical optimality.",
            "",
            "## Current Proof Level",
            "",
            "- Existing feature, boundary, novelty, and posterior-target objectives target levels 0-3 of the proof ladder.",
            "- `boundary_closeness` remains a proxy for ambiguity.",
            "- `posterior_target_distribution` is the direct ambiguity objective for targets such as 50/50 class posterior mass.",
            "",
            "## Global Scoring Rules",
            "",
            "- Scores are normalized to `[0, 1]` unless documented otherwise.",
            "- Higher is always better.",
            "- Same candidate, seed, and objective spec should produce the same score.",
            "- Invalid, degenerate, or leaking candidates cannot receive high scores.",
            "- Composite scores emit component terms and provenance.",
            "- Scores are comparable within a family; cross-family comparison requires explicit utility weights.",
        ]
    )


def _constant_acceleration_witness(acceleration: float, *, trajectory_id: str) -> TrajectoryArtifact:
    times = tuple(index * 0.25 for index in range(9))
    velocities = tuple(acceleration * time for time in times)
    positions = tuple(0.5 * acceleration * time * time for time in times)
    accelerations = tuple(acceleration for _ in times)
    return _make_manual_trajectory(
        trajectory_id=trajectory_id,
        true_class="constant_acceleration",
        tier="boundary_v1",
        scenario_family="posterior_target_objective_proof",
        measurements=positions,
        times=times,
        true_position=positions,
        true_velocity=velocities,
        true_acceleration=accelerations,
        measurement_std=0.01,
        outlier_indices=[],
        seed=7,
        generator_parameters={"witness_acceleration": acceleration},
    )


def posterior_target_witness_score_rows() -> tuple[dict[str, object], ...]:
    spec = posterior_target_objective_spec()
    witnesses = (
        ("ambiguous_cv_ca_probe", _constant_acceleration_witness(0.15, trajectory_id="ambiguous_cv_ca_probe")),
        ("clear_cv_probe", _constant_acceleration_witness(0.0, trajectory_id="clear_cv_probe")),
        ("clear_ca_probe", _constant_acceleration_witness(0.35, trajectory_id="clear_ca_probe")),
    )
    return tuple(
        score_posterior_target_distribution(
            spec,
            trajectory,
            candidate_id=candidate_id,
            backend_id="unit_witness",
        ).as_row()
        for candidate_id, trajectory in witnesses
    )


def write_objective_proof_artifacts(output_dir: str | Path) -> ObjectiveProofArtifacts:
    run_dir = Path(output_dir) / "trajectory_objective_proofs"
    run_dir.mkdir(parents=True, exist_ok=True)
    objective_spec_schema_path = run_dir / "objective_spec_schema.json"
    objective_score_schema_path = run_dir / "objective_score_schema.json"
    objective_family_catalog_path = run_dir / "objective_family_catalog.csv"
    proof_ladder_path = run_dir / "objective_proof_ladder.csv"
    posterior_target_spec_path = run_dir / "posterior_target_50_50_spec.json"
    posterior_target_witness_scores_path = run_dir / "posterior_target_witness_scores.csv"
    report_path = run_dir / "objective_proof_report.md"

    spec = posterior_target_objective_spec()
    _write_json(objective_spec_schema_path, objective_spec_schema())
    _write_json(objective_score_schema_path, objective_score_schema())
    _write_json(posterior_target_spec_path, asdict(spec))
    write_csv(objective_family_catalog_path, list(objective_family_catalog_rows()), ["family", "level", "purpose"])
    write_csv(proof_ladder_path, list(proof_ladder_rows()), ["level", "name", "meaning"])
    _write_text(report_path, objective_proof_report())
    witness_rows = list(posterior_target_witness_score_rows())
    write_csv(posterior_target_witness_scores_path, witness_rows, list(witness_rows[0].keys()))
    return ObjectiveProofArtifacts(
        run_dir=run_dir,
        objective_spec_schema_path=objective_spec_schema_path,
        objective_score_schema_path=objective_score_schema_path,
        objective_family_catalog_path=objective_family_catalog_path,
        proof_ladder_path=proof_ladder_path,
        posterior_target_spec_path=posterior_target_spec_path,
        posterior_target_witness_scores_path=posterior_target_witness_scores_path,
        report_path=report_path,
    )
