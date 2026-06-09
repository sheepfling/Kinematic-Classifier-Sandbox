from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from math import exp, log, sqrt
from pathlib import Path

from .contracts import TrajectoryArtifact, validate_trajectory_artifact


@dataclass(frozen=True, slots=True)
class DimensionalLiftAuditResult:
    module_rows: tuple[dict[str, object], ...]
    scalar_assumption_rows: tuple[dict[str, object], ...]
    required_adapter_markdown: str
    audit_markdown: str
    vector_predictions_rows: tuple[dict[str, object], ...]
    vector_posterior_rows: tuple[dict[str, object], ...]
    vector_feature_rows: tuple[dict[str, object], ...]
    validation_results: dict[str, object]


@dataclass(frozen=True, slots=True)
class DimensionalLiftAuditArtifacts:
    run_dir: Path
    audit_report_path: Path
    module_status_path: Path
    scalar_assumption_inventory_path: Path
    required_adapters_path: Path
    vector_predictions_path: Path
    vector_posterior_history_path: Path
    vector_feature_matrix_path: Path
    validation_results_path: Path


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _module_rows() -> tuple[dict[str, object], ...]:
    return (
        {
            "module": "contracts.py",
            "layer": "artifact_contracts",
            "dimensional_status": "dimension_agnostic",
            "reason": "Trajectory and classifier artifacts already carry measurement/state dimension, axes, and frame metadata.",
            "required_3d_action": "none",
        },
        {
            "module": "shared_evaluation.py",
            "layer": "evaluator",
            "dimensional_status": "dimension_agnostic",
            "reason": "Shared classifier runs already propagate measurement_dim and coordinate_frame.",
            "required_3d_action": "none",
        },
        {
            "module": "common_dataset_comparison.py",
            "layer": "shared_corpus",
            "dimensional_status": "adapter_compatible",
            "reason": "Run summaries and adapters are dimension-aware, but the current shared corpus generator is still scalar dynamics only.",
            "required_3d_action": "add vector corpus generator and vector classifier adapters",
        },
        {
            "module": "common_experiment_harness.py",
            "layer": "experiment_harness",
            "dimensional_status": "adapter_compatible",
            "reason": "Output surfaces carry dimension metadata, but feature extraction and synthetic pair logic remain 1D-specific.",
            "required_3d_action": "route vector corpus and vector feature adapters through the same orchestration path",
        },
        {
            "module": "feature_analysis.py",
            "layer": "feature_extraction",
            "dimensional_status": "rewrite_required",
            "reason": "Current feature contexts and extractors assume scalar position, scalar velocity, and scalar acceleration sequences.",
            "required_3d_action": "add vector feature context and multi-axis/norm policies",
        },
        {
            "module": "kalman_filter_bank.py",
            "layer": "filter_backend",
            "dimensional_status": "rewrite_required",
            "reason": "Current state transition and measurement updates are scalar-position models with 1D latent state layouts.",
            "required_3d_action": "add vector state layout, vector measurement layout, and frame-aware process/measurement models",
        },
        {
            "module": "generic_inference_contract.py",
            "layer": "methodology_proof",
            "dimensional_status": "adapter_compatible",
            "reason": "Contract schemas are generic, but the current proof checks only scalar measurement runs.",
            "required_3d_action": "add vector-backed proof rows",
        },
        {
            "module": "generic_feature_taxonomy.py",
            "layer": "methodology_proof",
            "dimensional_status": "adapter_compatible",
            "reason": "Taxonomy carries dimensional_transfer metadata, but the current feature set inventory is still 1D-derived.",
            "required_3d_action": "register vector feature families and frame assumptions",
        },
        {
            "module": "generic_classification_evidence_proof.py",
            "layer": "methodology_proof",
            "dimensional_status": "dimension_agnostic",
            "reason": "Posterior updating only consumes class-conditioned evidence streams and is not tied to scalar observations.",
            "required_3d_action": "none",
        },
        {
            "module": "generic_filtering_contract.py",
            "layer": "methodology_proof",
            "dimensional_status": "adapter_compatible",
            "reason": "The filtering contract is generic, but current validation only covers scalar Kalman-style backends.",
            "required_3d_action": "add vector filter backend validation rows",
        },
    )


def _scalar_assumption_rows() -> tuple[dict[str, object], ...]:
    return (
        {
            "module": "feature_analysis.py",
            "assumption_id": "scalar_velocity_and_acceleration_sequences",
            "severity": "high",
            "blocking_for_3d": True,
            "current_assumption": "velocity_sign_changes, acceleration_range, and monotonicity are derived from scalar finite differences.",
            "3d_requirement": "define per-axis, projected, or norm-based semantics for velocity and acceleration features",
        },
        {
            "module": "feature_analysis.py",
            "assumption_id": "scalar_fit_residuals",
            "severity": "high",
            "blocking_for_3d": True,
            "current_assumption": "linear and quadratic residuals are computed on one scalar trajectory coordinate.",
            "3d_requirement": "define multi-axis fit residual policy or vector residual norm",
        },
        {
            "module": "kalman_filter_bank.py",
            "assumption_id": "scalar_position_measurement_model",
            "severity": "high",
            "blocking_for_3d": True,
            "current_assumption": "measurement update uses scalar position observations and 1D latent states.",
            "3d_requirement": "define vector state layout and measurement matrix",
        },
        {
            "module": "common_dataset_comparison.py",
            "assumption_id": "scalar_shared_corpus_dynamics",
            "severity": "medium",
            "blocking_for_3d": True,
            "current_assumption": "shared synthetic corpus uses scalar position-only kinematics.",
            "3d_requirement": "add vector-valued corpus generator with explicit frame and axes",
        },
        {
            "module": "common_experiment_harness.py",
            "assumption_id": "scalar_pairwise_feature_logic",
            "severity": "medium",
            "blocking_for_3d": True,
            "current_assumption": "pair scoring and feature extraction assume scalar features from scalar trajectories.",
            "3d_requirement": "move to adapter-provided vector-compatible feature/evidence surfaces",
        },
        {
            "module": "contracts.py",
            "assumption_id": "none_contract_ready",
            "severity": "low",
            "blocking_for_3d": False,
            "current_assumption": "contract layer already admits dimension, axes, and coordinate-frame metadata.",
            "3d_requirement": "none",
        },
    )


def _required_adapters_markdown() -> str:
    return "\n".join(
        [
            "# Required 3D Adapters",
            "",
            "1. Vector corpus adapter",
            "Expose `measurement_dim`, `measurement_axes`, `state_dim`, `state_axes`, and `coordinate_frame` on every trajectory, with vector measurement tuples serialized consistently.",
            "",
            "2. Vector feature adapter",
            "Define trivial vector-compatible features first: path_length, displacement_norm, mean_dt, sampling_irregularity, and axis-independent duration coverage.",
            "",
            "3. Vector evidence adapter",
            "Allow evidence providers to consume vector features or vector residual summaries while keeping posterior updating unchanged.",
            "",
            "4. Vector filter adapter",
            "Future vector filters must declare state layout, measurement layout, and frame assumptions while emitting the same posterior/evidence/diagnostic contract.",
            "",
            "5. Visualization adapter",
            "2D/3D plotting can remain deferred, but artifact writers must still emit standard CSV/JSON/markdown outputs from vector studies.",
        ]
    )


def _normalize_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
    pivot = max(log_scores.values())
    norm = pivot + log(sum(exp(value - pivot) for value in log_scores.values()))
    return {name: exp(value - norm) for name, value in log_scores.items()}


def _vector_norm(values: tuple[float, ...]) -> float:
    return sqrt(sum(value * value for value in values))


def _fake_vector_corpus() -> tuple[TrajectoryArtifact, ...]:
    return (
        TrajectoryArtifact(
            trajectory_id="vector_traj_a",
            true_class="slow_linear",
            scenario_id="vector_nominal",
            seed=101,
            times=(0.0, 1.0, 2.0, 3.0),
            measurements=((0.0, 0.0, 0.0), (0.8, 0.2, 0.0), (1.6, 0.4, 0.0), (2.4, 0.6, 0.0)),
            measurement_dim=3,
            measurement_axes=("x", "y", "z"),
            coordinate_frame="enu",
            state_dim=6,
            state_axes=("x", "y", "z", "vx", "vy", "vz"),
            truth_series={
                "x": (0.0, 0.8, 1.6, 2.4),
                "y": (0.0, 0.2, 0.4, 0.6),
                "z": (0.0, 0.0, 0.0, 0.0),
            },
        ),
        TrajectoryArtifact(
            trajectory_id="vector_traj_b",
            true_class="fast_linear",
            scenario_id="vector_nominal",
            seed=102,
            times=(0.0, 1.0, 2.0, 3.0),
            measurements=((0.0, 0.0, 0.0), (1.5, 0.4, 0.0), (3.0, 0.8, 0.0), (4.5, 1.2, 0.0)),
            measurement_dim=3,
            measurement_axes=("x", "y", "z"),
            coordinate_frame="enu",
            state_dim=6,
            state_axes=("x", "y", "z", "vx", "vy", "vz"),
            truth_series={
                "x": (0.0, 1.5, 3.0, 4.5),
                "y": (0.0, 0.4, 0.8, 1.2),
                "z": (0.0, 0.0, 0.0, 0.0),
            },
        ),
    )


def _vector_feature_rows(corpus: tuple[TrajectoryArtifact, ...]) -> tuple[dict[str, object], ...]:
    rows = []
    for trajectory in corpus:
        displacements = [
            tuple(
                trajectory.measurements[index][axis] - trajectory.measurements[index - 1][axis]
                for axis in range(trajectory.measurement_dim)
            )
            for index in range(1, len(trajectory.measurements))
        ]
        path_length = sum(_vector_norm(displacement) for displacement in displacements)
        displacement_norm = _vector_norm(
            tuple(
                trajectory.measurements[-1][axis] - trajectory.measurements[0][axis]
                for axis in range(trajectory.measurement_dim)
            )
        )
        dt_values = [trajectory.times[index] - trajectory.times[index - 1] for index in range(1, len(trajectory.times))]
        mean_dt = sum(dt_values) / max(len(dt_values), 1)
        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "scenario_id": trajectory.scenario_id,
                "true_class": trajectory.true_class,
                "measurement_dim": trajectory.measurement_dim,
                "coordinate_frame": trajectory.coordinate_frame,
                "duration": trajectory.times[-1] - trajectory.times[0],
                "path_length": path_length,
                "displacement_norm": displacement_norm,
                "mean_dt": mean_dt,
            }
        )
    return tuple(rows)


def _vector_predictions_and_posteriors(
    feature_rows: tuple[dict[str, object], ...],
    *,
    threshold: float = 3.4,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    predictions = []
    posteriors = []
    for row in feature_rows:
        score_fast = -(abs(float(row["path_length"]) - 4.66))
        score_slow = -(abs(float(row["path_length"]) - 2.47))
        weights = _normalize_log_scores({"slow_linear": score_slow, "fast_linear": score_fast})
        predicted_class = max(weights, key=weights.get)
        run_id = f"vector_baseline:{row['trajectory_id']}"
        predictions.append(
            {
                "run_id": run_id,
                "classifier_id": "vector_path_length_baseline",
                "sensor_regime_id": "position_only",
                "trajectory_id": row["trajectory_id"],
                "scenario_id": row["scenario_id"],
                "time": 3.0,
                "true_class": row["true_class"],
                "predicted_class": predicted_class,
                "confidence": weights[predicted_class],
                "posterior_slow_linear": weights["slow_linear"],
                "posterior_fast_linear": weights["fast_linear"],
                "measurement_dim": row["measurement_dim"],
                "coordinate_frame": row["coordinate_frame"],
            }
        )
        posteriors.append(
            {
                "run_id": run_id,
                "classifier_id": "vector_path_length_baseline",
                "sensor_regime_id": "position_only",
                "trajectory_id": row["trajectory_id"],
                "scenario_id": row["scenario_id"],
                "time": 3.0,
                "true_class": row["true_class"],
                "posterior_slow_linear": weights["slow_linear"],
                "posterior_fast_linear": weights["fast_linear"],
                "measurement_dim": row["measurement_dim"],
                "coordinate_frame": row["coordinate_frame"],
            }
        )
    return tuple(predictions), tuple(posteriors)


def analyze_dimensional_lift_audit() -> DimensionalLiftAuditResult:
    module_rows = _module_rows()
    scalar_assumption_rows = _scalar_assumption_rows()
    required_adapter_markdown = _required_adapters_markdown()
    corpus = _fake_vector_corpus()
    trajectory_validation = {
        trajectory.trajectory_id: validate_trajectory_artifact(trajectory)
        for trajectory in corpus
    }
    feature_rows = _vector_feature_rows(corpus)
    prediction_rows, posterior_rows = _vector_predictions_and_posteriors(feature_rows)
    validation_results = {
        "all_modules_labeled": all(bool(row["dimensional_status"]) for row in module_rows),
        "scalar_assumptions_listed": len(scalar_assumption_rows) > 0,
        "vector_corpus_loaded": all(not errors for errors in trajectory_validation.values()),
        "vector_features_emitted": len(feature_rows) == len(corpus),
        "vector_predictions_emitted": len(prediction_rows) == len(corpus),
        "vector_posteriors_emitted": len(posterior_rows) == len(corpus),
        "trajectory_validation": trajectory_validation,
        "overall_status": "pass"
        if all(not errors for errors in trajectory_validation.values())
        and len(feature_rows) == len(corpus)
        and len(prediction_rows) == len(corpus)
        and len(posterior_rows) == len(corpus)
        else "fail",
    }
    audit_markdown = "\n".join(
        [
            "# Dimensional Lift Audit",
            "",
            "This artifact audits which current modules are dimension-agnostic, adapter-compatible, or rewrite-required, and proves that a fake vector-valued corpus can still reach the standard methodology artifact surface.",
            "",
            "## Validation Summary",
            "",
            f"- Overall status: `{validation_results['overall_status']}`",
            f"- All modules labeled: `{validation_results['all_modules_labeled']}`",
            f"- Scalar assumptions listed: `{validation_results['scalar_assumptions_listed']}`",
            f"- Vector corpus loaded: `{validation_results['vector_corpus_loaded']}`",
            f"- Vector features emitted: `{validation_results['vector_features_emitted']}`",
            f"- Vector predictions emitted: `{validation_results['vector_predictions_emitted']}`",
            f"- Vector posterior rows emitted: `{validation_results['vector_posteriors_emitted']}`",
            "",
            "## Module Status",
            "",
            "| module | layer | dimensional_status | required_3d_action |",
            "| --- | --- | --- | --- |",
            *[
                f"| {row['module']} | {row['layer']} | {row['dimensional_status']} | {row['required_3d_action']} |"
                for row in module_rows
            ],
            "",
            "## Scalar Assumption Inventory",
            "",
            "| module | assumption_id | severity | blocking_for_3d |",
            "| --- | --- | --- | --- |",
            *[
                f"| {row['module']} | {row['assumption_id']} | {row['severity']} | {row['blocking_for_3d']} |"
                for row in scalar_assumption_rows
            ],
            "",
            "## Fake Vector Proof",
            "",
            "- The fake vector corpus uses `measurement_dim=3`, `measurement_axes=(x,y,z)`, and `coordinate_frame=enu`.",
            "- It emits a standard feature table, prediction table, and posterior table without relying on full 3D dynamics or a full 3D Kalman bank.",
            "",
            "## Required Adapters",
            "",
            required_adapter_markdown,
        ]
    )
    return DimensionalLiftAuditResult(
        module_rows=module_rows,
        scalar_assumption_rows=scalar_assumption_rows,
        required_adapter_markdown=required_adapter_markdown,
        audit_markdown=audit_markdown,
        vector_predictions_rows=prediction_rows,
        vector_posterior_rows=posterior_rows,
        vector_feature_rows=feature_rows,
        validation_results=validation_results,
    )


def write_dimensional_lift_audit_artifacts(
    output_dir: str | Path,
    *,
    result: DimensionalLiftAuditResult | None = None,
) -> DimensionalLiftAuditArtifacts:
    audit = result or analyze_dimensional_lift_audit()
    run_dir = Path(output_dir) / "dimensional_lift_audit"
    run_dir.mkdir(parents=True, exist_ok=True)

    audit_report_path = run_dir / "dimensional_lift_audit.md"
    module_status_path = run_dir / "module_dimension_status.csv"
    scalar_assumption_inventory_path = run_dir / "scalar_assumption_inventory.csv"
    required_adapters_path = run_dir / "required_3d_adapters.md"
    vector_predictions_path = run_dir / "vector_proof_predictions.csv"
    vector_posterior_history_path = run_dir / "vector_proof_posterior_history.csv"
    vector_feature_matrix_path = run_dir / "vector_proof_feature_matrix.csv"
    validation_results_path = run_dir / "validation_results.json"

    audit_report_path.write_text(audit.audit_markdown, encoding="utf-8")
    required_adapters_path.write_text(audit.required_adapter_markdown, encoding="utf-8")
    validation_results_path.write_text(json.dumps(audit.validation_results, indent=2), encoding="utf-8")
    _write_csv(
        module_status_path,
        list(audit.module_rows),
        ["module", "layer", "dimensional_status", "reason", "required_3d_action"],
    )
    _write_csv(
        scalar_assumption_inventory_path,
        list(audit.scalar_assumption_rows),
        ["module", "assumption_id", "severity", "blocking_for_3d", "current_assumption", "3d_requirement"],
    )
    _write_csv(
        vector_predictions_path,
        list(audit.vector_predictions_rows),
        [
            "run_id",
            "classifier_id",
            "sensor_regime_id",
            "trajectory_id",
            "scenario_id",
            "time",
            "true_class",
            "predicted_class",
            "confidence",
            "posterior_slow_linear",
            "posterior_fast_linear",
            "measurement_dim",
            "coordinate_frame",
        ],
    )
    _write_csv(
        vector_posterior_history_path,
        list(audit.vector_posterior_rows),
        [
            "run_id",
            "classifier_id",
            "sensor_regime_id",
            "trajectory_id",
            "scenario_id",
            "time",
            "true_class",
            "posterior_slow_linear",
            "posterior_fast_linear",
            "measurement_dim",
            "coordinate_frame",
        ],
    )
    _write_csv(
        vector_feature_matrix_path,
        list(audit.vector_feature_rows),
        [
            "trajectory_id",
            "scenario_id",
            "true_class",
            "measurement_dim",
            "coordinate_frame",
            "duration",
            "path_length",
            "displacement_norm",
            "mean_dt",
        ],
    )

    return DimensionalLiftAuditArtifacts(
        run_dir=run_dir,
        audit_report_path=audit_report_path,
        module_status_path=module_status_path,
        scalar_assumption_inventory_path=scalar_assumption_inventory_path,
        required_adapters_path=required_adapters_path,
        vector_predictions_path=vector_predictions_path,
        vector_posterior_history_path=vector_posterior_history_path,
        vector_feature_matrix_path=vector_feature_matrix_path,
        validation_results_path=validation_results_path,
    )
