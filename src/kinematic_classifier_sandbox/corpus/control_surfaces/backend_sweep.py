from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ...utils.io import _write_json, _write_text, union_fieldnames, write_csv
from ...utils.plotting import prepare_matplotlib, write_plot
from ..observation_surfaces.catalog import observation_surface_rows
from ..trajectory_exploration.objective_scoring import (
    ObjectiveSpec,
    posterior_target_objective_spec,
    score_posterior_target_distribution,
)
from .backends import default_control_surface_backends
from .contracts import ControlSurfaceBackend, ControlSurfaceMetadata, TrajectoryCandidate


@dataclass(frozen=True, slots=True)
class ControlSurfaceBackendSweepConfig:
    objective_id: str = "posterior_target__cv_ca_50_50"
    random_candidates_per_backend: int = 24
    cem_iterations: int = 4
    cem_population: int = 16
    elite_fraction: float = 0.25
    seed: int = 7


@dataclass(frozen=True, slots=True)
class ControlSurfaceBackendSweepArtifacts:
    run_dir: Path
    config_path: Path
    control_surface_manifest_path: Path
    backend_capability_matrix_path: Path
    backend_objective_achievability_path: Path
    posterior_target_backend_sweep_path: Path
    target_vs_achieved_posterior_path: Path
    generator_identification_probe_path: Path
    backend_identification_probe_path: Path
    backend_identification_confusion_path: Path
    observation_surface_manifest_path: Path
    achievability_plot_path: Path
    posterior_plot_path: Path
    backend_probe_plot_path: Path
    report_path: Path


def _metadata_row(metadata: ControlSurfaceMetadata) -> dict[str, object]:
    return {
        "backend_id": metadata.backend_id,
        "display_name": metadata.display_name,
        "control_surface_type": metadata.control_surface_type,
        "state_variables": ",".join(metadata.state_variables),
        "control_variables": ",".join(metadata.control_variables),
        "classifier_allowed_fields": ",".join(metadata.classifier_allowed_fields),
        "hidden_fields": ",".join(metadata.hidden_fields),
        "best_use": metadata.best_use,
        "lift_to_3d": metadata.lift_to_3d,
    }


def _capability_rows(backends: tuple[ControlSurfaceBackend, ...]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for backend in backends:
        metadata = backend.metadata()
        row: dict[str, object] = {"backend_id": metadata.backend_id, "control_surface_type": metadata.control_surface_type}
        row.update({f"supports_{key}": "implemented" if value else "not_applicable" for key, value in metadata.supports.items()})
        rows.append(row)
    return tuple(rows)


def _visible_feature_row(candidate: TrajectoryCandidate) -> dict[str, object]:
    trajectory = candidate.trajectory
    times = tuple(float(value) for value in trajectory.times)
    measurements = tuple(float(value) for value in trajectory.measurements)
    velocities = tuple(float(value) for value in trajectory.true_velocity or ())
    accelerations = tuple(float(value) for value in trajectory.true_acceleration or ())
    duration = times[-1] - times[0] if len(times) > 1 else 0.0
    displacement = measurements[-1] - measurements[0] if len(measurements) > 1 else 0.0
    measurement_range = max(measurements) - min(measurements) if measurements else 0.0
    mean_abs_velocity = sum(abs(value) for value in velocities) / max(len(velocities), 1)
    mean_abs_acceleration = sum(abs(value) for value in accelerations) / max(len(accelerations), 1)
    acceleration_range = max(accelerations) - min(accelerations) if accelerations else 0.0
    acceleration_mean = sum(accelerations) / max(len(accelerations), 1)
    acceleration_variance = (
        sum((value - acceleration_mean) ** 2 for value in accelerations) / len(accelerations)
        if accelerations
        else 0.0
    )
    return {
        "visible_duration": duration,
        "visible_displacement": displacement,
        "visible_measurement_range": measurement_range,
        "visible_mean_abs_velocity": mean_abs_velocity,
        "visible_mean_abs_acceleration": mean_abs_acceleration,
        "visible_acceleration_range": acceleration_range,
        "visible_acceleration_variance": acceleration_variance,
    }


def _score_candidate(
    *,
    spec: ObjectiveSpec,
    candidate: TrajectoryCandidate,
    optimizer_id: str,
    iteration: int,
) -> dict[str, object]:
    score = score_posterior_target_distribution(
        spec,
        candidate.trajectory,
        candidate_id=candidate.candidate_id,
        backend_id=candidate.backend_id,
        tier_name="boundary_v1",
    )
    row = score.as_row()
    row.update(
        {
            "optimizer_id": optimizer_id,
            "iteration": iteration,
            "accel_magnitude": candidate.params.get("accel_magnitude", ""),
            "shape": candidate.params.get("shape", ""),
            "dt": candidate.params.get("dt", ""),
            "steps": candidate.params.get("steps", ""),
            "measurement_std": candidate.params.get("measurement_std", ""),
            "control_trace_keys": ",".join(candidate.control_trace.keys()),
        }
    )
    row.update(_visible_feature_row(candidate))
    return row


def _random_search_rows(
    *,
    backend: ControlSurfaceBackend,
    spec: ObjectiveSpec,
    config: ControlSurfaceBackendSweepConfig,
    seed_offset: int,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for index in range(config.random_candidates_per_backend):
        seed = config.seed + seed_offset + index
        params = backend.sample_params(seed)
        candidate = backend.rollout(params, seed=seed, candidate_id=f"{backend.backend_id}__random__{index}")
        rows.append(_score_candidate(spec=spec, candidate=candidate, optimizer_id="random_search", iteration=0))
    return tuple(rows)


def _cem_rows(
    *,
    backend: ControlSurfaceBackend,
    spec: ObjectiveSpec,
    config: ControlSurfaceBackendSweepConfig,
    seed_offset: int,
) -> tuple[dict[str, object], ...]:
    rng = random.Random(config.seed + seed_offset + 50_000)
    mean = 0.25
    std = 0.16
    rows: list[dict[str, object]] = []
    for iteration in range(config.cem_iterations):
        iteration_rows: list[dict[str, object]] = []
        for index in range(config.cem_population):
            accel_magnitude = max(0.0, min(0.45, rng.gauss(mean, std)))
            seed = config.seed + seed_offset + iteration * 1000 + index + 200_000
            params = backend.sample_params(seed, accel_magnitude=accel_magnitude)
            candidate = backend.rollout(params, seed=seed, candidate_id=f"{backend.backend_id}__cem__{iteration}_{index}")
            iteration_rows.append(_score_candidate(spec=spec, candidate=candidate, optimizer_id="cem", iteration=iteration))
        rows.extend(iteration_rows)
        ranked = sorted(iteration_rows, key=lambda row: float(row["score"]), reverse=True)
        elite_count = max(1, int(len(ranked) * config.elite_fraction))
        elite_accels = [float(row["accel_magnitude"]) for row in ranked[:elite_count]]
        mean = sum(elite_accels) / len(elite_accels)
        std = max(0.025, sum(abs(value - mean) for value in elite_accels) / len(elite_accels))
    return tuple(rows)


def _best_rows(evaluation_rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in evaluation_rows:
        key = (str(row["backend_id"]), str(row["optimizer_id"]))
        if key not in grouped or float(row["score"]) > float(grouped[key]["score"]):
            grouped[key] = row
    return tuple(grouped[key] for key in sorted(grouped))


def _achievability_rows(evaluation_rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    best_by_backend: dict[str, dict[str, object]] = {}
    for row in evaluation_rows:
        backend_id = str(row["backend_id"])
        if backend_id not in best_by_backend or float(row["score"]) > float(best_by_backend[backend_id]["score"]):
            best_by_backend[backend_id] = row
    for backend_id, row in sorted(best_by_backend.items()):
        tv_error = float(row["posterior_tv_error"])
        passed_constraints = str(row["passed_constraints"]) == "True" or row["passed_constraints"] is True
        rows.append(
            {
                "backend_id": backend_id,
                "objective_id": row["objective_id"],
                "best_score": row["score"],
                "posterior_tv_error": tv_error,
                "posterior_l1_error": row["posterior_l1_error"],
                "achieved_posterior_constant_velocity": row.get("achieved_posterior_constant_velocity", ""),
                "achieved_posterior_constant_acceleration": row.get("achieved_posterior_constant_acceleration", ""),
                "passed_constraints": row["passed_constraints"],
                "best_optimizer_id": row["optimizer_id"],
                "status": "achieved" if passed_constraints and tv_error <= 0.12 else "partial",
            }
        )
    return tuple(rows)


def _posterior_rows(best_rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "backend_id": row["backend_id"],
            "optimizer_id": row["optimizer_id"],
            "target_cv": row.get("target_posterior_constant_velocity", ""),
            "target_ca": row.get("target_posterior_constant_acceleration", ""),
            "achieved_cv": row.get("achieved_posterior_constant_velocity", ""),
            "achieved_ca": row.get("achieved_posterior_constant_acceleration", ""),
            "posterior_tv_error": row["posterior_tv_error"],
            "score": row["score"],
        }
        for row in best_rows
    )


def _generator_probe_rows(evaluation_rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in evaluation_rows:
        grouped.setdefault(str(row["backend_id"]), []).append(row)
    global_scores = [float(row["score"]) for row in evaluation_rows]
    global_mean = sum(global_scores) / max(len(global_scores), 1)
    rows: list[dict[str, object]] = []
    for backend_id, rows_for_backend in sorted(grouped.items()):
        scores = [float(row["score"]) for row in rows_for_backend]
        mean_score = sum(scores) / len(scores)
        mean_accel = sum(float(row["accel_magnitude"]) for row in rows_for_backend) / len(rows_for_backend)
        risk = min(1.0, abs(mean_score - global_mean) * 2.0 + abs(mean_accel - 0.225))
        rows.append(
            {
                "backend_id": backend_id,
                "probe_type": "observable_distribution_proxy",
                "backend_signature_risk": risk,
                "mean_score": mean_score,
                "global_mean_score": global_mean,
                "mean_accel_magnitude": mean_accel,
                "interpretation": "audit_required" if risk > 0.35 else "no_strong_proxy_signal",
            }
        )
    return tuple(rows)


_PROBE_FEATURES = (
    "visible_duration",
    "visible_displacement",
    "visible_measurement_range",
    "visible_mean_abs_velocity",
    "visible_mean_abs_acceleration",
    "visible_acceleration_range",
    "visible_acceleration_variance",
)


def _is_probe_test_row(row: dict[str, object]) -> bool:
    candidate_id = str(row["candidate_id"])
    return sum(ord(character) for character in candidate_id) % 3 == 0


def _feature_vector(row: dict[str, object]) -> tuple[float, ...]:
    return tuple(float(row.get(feature, 0.0) or 0.0) for feature in _PROBE_FEATURES)


def _normalization_stats(rows: tuple[dict[str, object], ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    vectors = [_feature_vector(row) for row in rows]
    if not vectors:
        return tuple(0.0 for _ in _PROBE_FEATURES), tuple(1.0 for _ in _PROBE_FEATURES)
    means = tuple(sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(_PROBE_FEATURES)))
    scales = []
    for index, mean in enumerate(means):
        variance = sum((vector[index] - mean) ** 2 for vector in vectors) / len(vectors)
        scales.append(max(variance**0.5, 1e-9))
    return means, tuple(scales)


def _normalize(vector: tuple[float, ...], means: tuple[float, ...], scales: tuple[float, ...]) -> tuple[float, ...]:
    return tuple((value - means[index]) / scales[index] for index, value in enumerate(vector))


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum((left[index] - right[index]) ** 2 for index in range(len(left))) ** 0.5


def _backend_identification_probe_rows(evaluation_rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    train_rows = tuple(row for row in evaluation_rows if not _is_probe_test_row(row))
    test_rows = tuple(row for row in evaluation_rows if _is_probe_test_row(row))
    if not train_rows or not test_rows:
        return ()
    means, scales = _normalization_stats(train_rows)
    grouped: dict[str, list[tuple[float, ...]]] = {}
    for row in train_rows:
        grouped.setdefault(str(row["backend_id"]), []).append(_normalize(_feature_vector(row), means, scales))
    centroids = {
        backend_id: tuple(sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(_PROBE_FEATURES)))
        for backend_id, vectors in grouped.items()
    }
    rows: list[dict[str, object]] = []
    for row in test_rows:
        vector = _normalize(_feature_vector(row), means, scales)
        predicted_backend = min(centroids, key=lambda backend_id: _distance(vector, centroids[backend_id]))
        actual_backend = str(row["backend_id"])
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "actual_backend_id": actual_backend,
                "predicted_backend_id": predicted_backend,
                "correct": actual_backend == predicted_backend,
                "probe_model": "nearest_centroid_visible_features",
                "feature_columns": ",".join(_PROBE_FEATURES),
                "optimizer_id": row["optimizer_id"],
                "score": row["score"],
            }
        )
    return tuple(rows)


def _backend_identification_confusion_rows(probe_rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    backend_ids = sorted({str(row["actual_backend_id"]) for row in probe_rows} | {str(row["predicted_backend_id"]) for row in probe_rows})
    rows: list[dict[str, object]] = []
    for actual in backend_ids:
        actual_rows = [row for row in probe_rows if row["actual_backend_id"] == actual]
        total = max(len(actual_rows), 1)
        for predicted in backend_ids:
            count = sum(1 for row in actual_rows if row["predicted_backend_id"] == predicted)
            rows.append(
                {
                    "actual_backend_id": actual,
                    "predicted_backend_id": predicted,
                    "count": count,
                    "rate": count / total,
                }
            )
    return tuple(rows)


def _probe_accuracy(probe_rows: tuple[dict[str, object], ...]) -> float:
    if not probe_rows:
        return 0.0
    return sum(1 for row in probe_rows if row["correct"] is True) / len(probe_rows)


def _write_achievability_plot(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    plt = prepare_matplotlib()
    labels = [str(row["backend_id"]) for row in rows]
    scores = [float(row["best_score"]) for row in rows]
    colors = ["#2f855a" if row["status"] == "achieved" else "#c05621" for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(labels, scores, color=colors)
    ax.axhline(0.88, color="#4a5568", linestyle="--", linewidth=1.0, label="TV error <= 0.12")
    ax.set_title("Posterior-Target Achievability by Control Surface", loc="left", fontweight="bold")
    ax.set_ylabel("best score")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="lower right")
    write_plot(fig, path)


def _write_posterior_plot(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    plt = prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    for row in rows:
        ax.scatter(float(row["achieved_cv"]), float(row["achieved_ca"]), s=65, label=f"{row['backend_id']}:{row['optimizer_id']}")
    ax.scatter(0.5, 0.5, s=120, marker="x", color="#1a202c", label="target")
    ax.set_title("Target vs Achieved CV/CA Posterior", loc="left", fontweight="bold")
    ax.set_xlabel("P(constant_velocity)")
    ax.set_ylabel("P(constant_acceleration)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    write_plot(fig, path)


def _write_backend_probe_plot(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    plt = prepare_matplotlib()
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        grouped.setdefault(str(row["actual_backend_id"]), []).append(bool(row["correct"]))
    labels = sorted(grouped)
    accuracies = [sum(grouped[label]) / len(grouped[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(labels, accuracies, color="#2b6cb0")
    ax.set_title("Backend-ID Probe Accuracy from Visible Features", loc="left", fontweight="bold")
    ax.set_ylabel("probe accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    write_plot(fig, path)


def _report(
    *,
    config: ControlSurfaceBackendSweepConfig,
    achievability_rows: tuple[dict[str, object], ...],
    probe_rows: tuple[dict[str, object], ...],
    backend_id_probe_rows: tuple[dict[str, object], ...],
) -> str:
    achieved = sum(1 for row in achievability_rows if row["status"] == "achieved")
    risk_count = sum(1 for row in probe_rows if row["interpretation"] == "audit_required")
    backend_probe_accuracy = _probe_accuracy(backend_id_probe_rows)
    return "\n".join(
        [
            "# Control Surface Backend Sweep",
            "",
            "This artifact lane tests whether one posterior-target objective can run across multiple 1D generation surfaces.",
            "",
            "## Objective",
            "",
            f"- objective: `{config.objective_id}`",
            "- target posterior: `constant_velocity=0.5`, `constant_acceleration=0.5`",
            "- backends: direct parameters, acceleration sequence, jerk sequence, spline knots, hybrid schedule, stochastic process",
            "- optimizers: random search and a small CEM loop over acceleration magnitude",
            "",
            "## Result",
            "",
            f"- backend/objective rows generated: `{len(achievability_rows)}`",
            f"- backends reaching posterior TV error <= 0.12: `{achieved}`",
            f"- generator-probe rows requiring audit: `{risk_count}`",
            f"- held-out backend-ID probe accuracy from visible summaries: `{backend_probe_accuracy:.3f}`",
            "",
            "## Interpretation",
            "",
            "This is not a final generator-leakage proof. It is the first backend-agnostic objective proof: the same score contract and posterior target are applied to different truth-generation surfaces while hidden generation fields are documented separately from classifier-visible fields.",
            "",
            "The nearest-centroid backend-ID probe uses visible trajectory summaries only. High probe accuracy means backend signatures may be observable and should be audited before claiming generator-independent classifier behavior.",
        ]
    )


def run_control_surface_backend_sweep(
    config: ControlSurfaceBackendSweepConfig | None = None,
    *,
    backends: tuple[ControlSurfaceBackend, ...] | None = None,
) -> dict[str, tuple[dict[str, object], ...]]:
    resolved = config or ControlSurfaceBackendSweepConfig()
    selected_backends = backends or default_control_surface_backends()
    spec = posterior_target_objective_spec(objective_id=resolved.objective_id)
    evaluation_rows: list[dict[str, object]] = []
    for backend_index, backend in enumerate(selected_backends):
        seed_offset = backend_index * 10_000
        evaluation_rows.extend(_random_search_rows(backend=backend, spec=spec, config=resolved, seed_offset=seed_offset))
        evaluation_rows.extend(_cem_rows(backend=backend, spec=spec, config=resolved, seed_offset=seed_offset))
    best = _best_rows(tuple(evaluation_rows))
    backend_id_probe_rows = _backend_identification_probe_rows(tuple(evaluation_rows))
    return {
        "manifest_rows": tuple(_metadata_row(backend.metadata()) for backend in selected_backends),
        "capability_rows": _capability_rows(selected_backends),
        "observation_surface_rows": observation_surface_rows(),
        "evaluation_rows": tuple(evaluation_rows),
        "achievability_rows": _achievability_rows(tuple(evaluation_rows)),
        "target_vs_achieved_rows": _posterior_rows(best),
        "generator_probe_rows": _generator_probe_rows(tuple(evaluation_rows)),
        "backend_identification_probe_rows": backend_id_probe_rows,
        "backend_identification_confusion_rows": _backend_identification_confusion_rows(backend_id_probe_rows),
    }


def write_control_surface_backend_sweep_artifacts(
    output_dir: str | Path,
    config: ControlSurfaceBackendSweepConfig | None = None,
) -> ControlSurfaceBackendSweepArtifacts:
    resolved = config or ControlSurfaceBackendSweepConfig()
    run_dir = Path(output_dir) / "control_surfaces"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = run_control_surface_backend_sweep(resolved)
    config_path = run_dir / "posterior_target_backend_sweep_config.json"
    manifest_path = run_dir / "control_surface_manifest.csv"
    capability_path = run_dir / "backend_capability_matrix.csv"
    achievability_path = run_dir / "backend_objective_achievability.csv"
    sweep_path = run_dir / "posterior_target_backend_sweep.csv"
    posterior_path = run_dir / "target_vs_achieved_posterior_by_backend.csv"
    probe_path = run_dir / "generator_identification_probe.csv"
    backend_id_probe_path = run_dir / "backend_identification_probe.csv"
    backend_id_confusion_path = run_dir / "backend_identification_confusion.csv"
    observation_manifest_path = run_dir / "observation_surface_manifest.csv"
    achievability_plot_path = run_dir / "backend_objective_achievability.png"
    posterior_plot_path = run_dir / "target_vs_achieved_posterior_by_backend.png"
    backend_probe_plot_path = run_dir / "backend_identification_probe.png"
    report_path = run_dir / "control_surface_report.md"

    _write_json(config_path, asdict(resolved))
    write_csv(manifest_path, list(rows["manifest_rows"]), union_fieldnames(rows["manifest_rows"]))
    write_csv(capability_path, list(rows["capability_rows"]), union_fieldnames(rows["capability_rows"]))
    write_csv(achievability_path, list(rows["achievability_rows"]), union_fieldnames(rows["achievability_rows"]))
    write_csv(sweep_path, list(rows["evaluation_rows"]), union_fieldnames(rows["evaluation_rows"]))
    write_csv(posterior_path, list(rows["target_vs_achieved_rows"]), union_fieldnames(rows["target_vs_achieved_rows"]))
    write_csv(probe_path, list(rows["generator_probe_rows"]), union_fieldnames(rows["generator_probe_rows"]))
    write_csv(observation_manifest_path, list(rows["observation_surface_rows"]), union_fieldnames(rows["observation_surface_rows"]))
    write_csv(backend_id_probe_path, list(rows["backend_identification_probe_rows"]), union_fieldnames(rows["backend_identification_probe_rows"]))
    write_csv(backend_id_confusion_path, list(rows["backend_identification_confusion_rows"]), union_fieldnames(rows["backend_identification_confusion_rows"]))
    _write_achievability_plot(achievability_plot_path, rows["achievability_rows"])
    _write_posterior_plot(posterior_plot_path, rows["target_vs_achieved_rows"])
    _write_backend_probe_plot(backend_probe_plot_path, rows["backend_identification_probe_rows"])
    _write_text(
        report_path,
        _report(
            config=resolved,
            achievability_rows=rows["achievability_rows"],
            probe_rows=rows["generator_probe_rows"],
            backend_id_probe_rows=rows["backend_identification_probe_rows"],
        ),
    )
    return ControlSurfaceBackendSweepArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        control_surface_manifest_path=manifest_path,
        backend_capability_matrix_path=capability_path,
        backend_objective_achievability_path=achievability_path,
        posterior_target_backend_sweep_path=sweep_path,
        target_vs_achieved_posterior_path=posterior_path,
        generator_identification_probe_path=probe_path,
        backend_identification_probe_path=backend_id_probe_path,
        backend_identification_confusion_path=backend_id_confusion_path,
        observation_surface_manifest_path=observation_manifest_path,
        achievability_plot_path=achievability_plot_path,
        posterior_plot_path=posterior_plot_path,
        backend_probe_plot_path=backend_probe_plot_path,
        report_path=report_path,
    )
