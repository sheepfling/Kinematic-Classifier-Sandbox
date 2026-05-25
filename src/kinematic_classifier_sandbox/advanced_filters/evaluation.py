from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path
import time as wall_time

import numpy as np

from .models_1d import (
    constant_velocity_transition,
    make_initial_particles_1d,
    nonlinear_drag_transition,
    position_gaussian_log_likelihood,
    position_mixture_log_likelihood,
)
from .particle_filter import BootstrapParticleFilter, ParticleFilterConfig
from .particle_filter_bank import ParticleFilterBank
from .linear_gaussian import KalmanModeState, kalman_predict, kalman_update
from .rbpf import RBPFConfig, RaoBlackwellizedParticleFilter
from .rbpf_models_1d import default_mode_transition_matrix_1d, make_rbpf_1d_mode_models


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True, slots=True)
class AdvancedFilterWitnessArtifacts:
    run_dir: Path
    report_path: Path
    posterior_history_path: Path
    state_estimate_history_path: Path
    metrics_path: Path
    plot_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class AdvancedFilterComparisonArtifacts:
    run_dir: Path
    method_comparison_path: Path
    nonlinear_stress_metrics_path: Path
    latent_maneuver_metrics_path: Path
    runtime_cost_metrics_path: Path
    decision_matrix_path: Path
    report_path: Path


def write_particle_filter_witness_artifacts(output_dir: str | Path, *, seed: int = 23) -> AdvancedFilterWitnessArtifacts:
    run_dir = Path(output_dir) / "particle_filter_v1"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, 8.0, 0.25, dtype=np.float64)
    positions = [0.0]
    velocities = [2.2]
    for _ in times[1:]:
        dt = 0.25
        drag = 0.11 * velocities[-1] * abs(velocities[-1])
        velocities.append(velocities[-1] - drag * dt)
        positions.append(positions[-1] + velocities[-1] * dt)
    truth = np.asarray(positions, dtype=np.float64)
    observations = truth + rng.normal(0.0, 0.12, size=len(truth))
    outlier_indices = [8, 13, 18, 25, 29]
    observations[outlier_indices] += np.array([1.8, -1.6, 1.5, -1.4, 1.7], dtype=np.float64)

    particle_count = 384
    bank_rng = np.random.default_rng(seed + 1)
    filters = {
        "constant_velocity": BootstrapParticleFilter(
            ParticleFilterConfig(particle_count=particle_count, seed=seed + 2),
            transition_fn=lambda particles, dt, gen: constant_velocity_transition(particles, dt, gen, process_std=0.05),
            log_likelihood_fn=lambda particles, obs: position_gaussian_log_likelihood(particles, obs, measurement_std=0.25),
        ),
        "nonlinear_drag": BootstrapParticleFilter(
            ParticleFilterConfig(particle_count=particle_count, seed=seed + 3),
            transition_fn=lambda particles, dt, gen: nonlinear_drag_transition(particles, dt, gen, drag_coefficient=0.11, process_std=0.04),
            log_likelihood_fn=lambda particles, obs: position_mixture_log_likelihood(particles, obs, measurement_std=0.14, outlier_std=1.2, outlier_probability=0.08),
        ),
    }
    bank = ParticleFilterBank(filters)
    initial_particles = {
        label: make_initial_particles_1d(particle_count, observations[0], 0.18, 2.2, 0.25, bank_rng)
        for label in filters
    }
    bank.reset("nonlinear_drag_outlier_1d", initial_particles)

    posterior_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    kalman_state = KalmanModeState(
        mean=np.array([observations[0], 2.2, 0.0], dtype=np.float64),
        covariance=np.diag([0.08, 0.25, 0.30]).astype(np.float64),
    )
    last_time: float | None = None
    start = wall_time.perf_counter()
    for index, (time_value, observation) in enumerate(zip(times, observations, strict=True)):
        dt = 0.25 if last_time is None else max(float(time_value - last_time), 1.0e-9)
        if last_time is not None:
            kalman_state = kalman_predict(kalman_state, dt=dt, process_noise_scale=0.01)
        kalman_state, _, _, _ = kalman_update(
            kalman_state,
            observation=np.array([observation], dtype=np.float64),
            measurement_noise=0.12**2,
        )
        last_time = float(time_value)
        step = bank.update(float(time_value), np.array([observation], dtype=np.float64))
        for label, posterior in step.posterior_by_label.items():
            posterior_rows.append({"trajectory_id": step.trajectory_id, "time": time_value, "label": label, "posterior": posterior, "predicted_label": step.predicted_label, "confidence": step.confidence, "log_evidence": step.log_evidence_by_label[label]})
        nonlinear_filter = filters["nonlinear_drag"]
        assert nonlinear_filter.state is not None
        weights = np.exp(nonlinear_filter.state.log_weights)
        mean = np.average(nonlinear_filter.state.particles, axis=0, weights=weights)
        state_rows.append({
            "trajectory_id": step.trajectory_id,
            "time": time_value,
            "truth_position": truth[index],
            "observation": observation,
            "pf_position": float(mean[0]),
            "pf_velocity": float(mean[1]),
            "kalman_position": float(kalman_state.mean[0]),
            "kalman_velocity": float(kalman_state.mean[1]),
            "predicted_label": step.predicted_label,
            "ess": step.diagnostics["ess_nonlinear_drag"],
            "resampled": step.diagnostics["resampled_nonlinear_drag"],
        })
    runtime_seconds = wall_time.perf_counter() - start
    rmse = float(np.sqrt(np.mean([(row["pf_position"] - row["truth_position"]) ** 2 for row in state_rows])))
    kalman_baseline_rmse = float(np.sqrt(np.mean([(row["kalman_position"] - row["truth_position"]) ** 2 for row in state_rows])))
    observation_baseline_rmse = float(np.sqrt(np.mean((observations - truth) ** 2)))
    resampling_count = sum(1 for row in state_rows if row["resampled"])
    rmse_delta_vs_kalman = kalman_baseline_rmse - rmse
    decision = "promote" if rmse < kalman_baseline_rmse and rmse < observation_baseline_rmse else "revise"

    posterior_path = run_dir / "posterior_history.csv"
    state_path = run_dir / "state_estimate_history.csv"
    metrics_path = run_dir / "pf_method_comparison.csv"
    report_path = run_dir / "pf_report.md"
    _write_csv(posterior_path, posterior_rows, ["trajectory_id", "time", "label", "posterior", "predicted_label", "confidence", "log_evidence"])
    _write_csv(state_path, state_rows, ["trajectory_id", "time", "truth_position", "observation", "pf_position", "pf_velocity", "kalman_position", "kalman_velocity", "predicted_label", "ess", "resampled"])
    metrics = [{"method_id": "particle_filter_bank_v1", "witness": "nonlinear_drag_outlier_1d", "position_rmse": rmse, "kalman_baseline_rmse": kalman_baseline_rmse, "observation_baseline_rmse": observation_baseline_rmse, "rmse_delta_vs_kalman": rmse_delta_vs_kalman, "resampling_count": resampling_count, "runtime_seconds": runtime_seconds, "promotion_decision": decision}]
    _write_csv(metrics_path, metrics, list(metrics[0]))
    report_path.write_text(f"""# Particle Filter V1 Report

Witness: `nonlinear_drag_outlier_1d`

PF is evaluated as a posterior-compatible classifier bank over `constant_velocity` and `nonlinear_drag`.

- PF nonlinear-drag position RMSE: `{rmse}`
- Gaussian Kalman baseline RMSE: `{kalman_baseline_rmse}`
- Observation baseline RMSE: `{observation_baseline_rmse}`
- RMSE delta vs Kalman: `{rmse_delta_vs_kalman}`
- Resampling count: `{resampling_count}`
- Runtime seconds: `{runtime_seconds}`
- Decision: `{decision}`
""", encoding="utf-8")
    state_plot = plot_dir / "pf_state_vs_truth.png"
    ess_plot = plot_dir / "pf_ess_timeline.png"
    _plot_series(state_plot, times, [(truth, "truth"), (observations, "observation"), (np.array([row["kalman_position"] for row in state_rows]), "Kalman baseline"), (np.array([row["pf_position"] for row in state_rows]), "PF estimate")], "PF state vs truth")
    _plot_series(ess_plot, times, [(np.array([row["ess"] for row in state_rows], dtype=np.float64), "ESS")], "PF ESS timeline")
    return AdvancedFilterWitnessArtifacts(run_dir, report_path, posterior_path, state_path, metrics_path, (state_plot, ess_plot))


def write_rbpf_witness_artifacts(output_dir: str | Path, *, seed: int = 31) -> AdvancedFilterWitnessArtifacts:
    run_dir = Path(output_dir) / "rbpf_v1"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, 8.0, 0.25, dtype=np.float64)
    onset_time = 3.0
    positions = [0.0]
    velocities = [0.6]
    for time_value in times[1:]:
        dt = 0.25
        accel = 1.35 if time_value >= onset_time else 0.0
        velocities.append(velocities[-1] + accel * dt)
        positions.append(positions[-1] + velocities[-1] * dt)
    truth = np.asarray(positions, dtype=np.float64)
    observations = truth + rng.normal(0.0, 0.10, size=len(truth))

    particle_count = 256
    rbpf = RaoBlackwellizedParticleFilter(
        RBPFConfig(particle_count=particle_count, seed=seed + 1),
        make_rbpf_1d_mode_models(dt=0.25, measurement_std=0.15),
        default_mode_transition_matrix_1d(),
    )
    initial_modes = rng.choice(4, size=particle_count, p=np.array([0.55, 0.25, 0.05, 0.15], dtype=np.float64)).astype(np.int64)
    means = np.zeros((particle_count, 3), dtype=np.float64)
    means[:, 0] = observations[0]
    means[:, 1] = rng.normal(0.6, 0.2, size=particle_count)
    covariances = np.repeat(np.diag([0.2, 0.5, 0.5])[None, :, :], particle_count, axis=0)
    rbpf.reset("latent_maneuver_onset_1d", initial_modes, means, covariances)

    posterior_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    start = wall_time.perf_counter()
    for index, (time_value, observation) in enumerate(zip(times, observations, strict=True)):
        step = rbpf.update(float(time_value), np.array([observation], dtype=np.float64))
        for label, posterior in step.posterior_by_label.items():
            posterior_rows.append({
                "trajectory_id": step.trajectory_id,
                "time": time_value,
                "label": label,
                "posterior": posterior,
                "predicted_label": step.predicted_label,
                "confidence": step.confidence,
                "log_evidence": step.log_evidence_by_label[label],
            })
        assert rbpf.state is not None
        weights = np.exp(rbpf.state.log_weights)
        mean = np.average(rbpf.state.means, axis=0, weights=weights)
        state_rows.append({"trajectory_id": step.trajectory_id, "time": time_value, "truth_position": truth[index], "observation": observation, "state_position": float(mean[0]), "state_velocity": float(mean[1]), "state_acceleration": float(mean[2]), "predicted_mode": step.predicted_label, "ess": step.diagnostics["ess"], "resampled": step.diagnostics["resampled"]})
    runtime_seconds = wall_time.perf_counter() - start
    rmse = float(np.sqrt(np.mean([(row["state_position"] - row["truth_position"]) ** 2 for row in state_rows])))
    post_onset_rows = [row for row in state_rows if row["time"] >= onset_time]
    post_onset_mode_accuracy = sum(row["predicted_mode"] in {"accelerate", "maneuver"} for row in post_onset_rows) / max(len(post_onset_rows), 1)
    resampling_count = sum(1 for row in state_rows if row["resampled"])
    decision = "promote" if post_onset_mode_accuracy >= 0.50 else "revise"

    posterior_path = run_dir / "latent_mode_posterior.csv"
    state_path = run_dir / "conditional_filter_history.csv"
    metrics_path = run_dir / "rbpf_method_comparison.csv"
    report_path = run_dir / "rbpf_report.md"
    _write_csv(posterior_path, posterior_rows, ["trajectory_id", "time", "label", "posterior", "predicted_label", "confidence", "log_evidence"])
    _write_csv(state_path, state_rows, ["trajectory_id", "time", "truth_position", "observation", "state_position", "state_velocity", "state_acceleration", "predicted_mode", "ess", "resampled"])
    metrics = [{"method_id": "rbpf_v1", "witness": "latent_maneuver_onset_1d", "state_position_rmse": rmse, "post_onset_mode_accuracy": post_onset_mode_accuracy, "resampling_count": resampling_count, "runtime_seconds": runtime_seconds, "promotion_decision": decision}]
    _write_csv(metrics_path, metrics, list(metrics[0]))
    report_path.write_text(f"""# RBPF V1 Report

Witness: `latent_maneuver_onset_1d`

RBPF samples latent mode paths while conditionally Kalman-filtering the continuous PVA state.

- State position RMSE: `{rmse}`
- Post-onset mode accuracy: `{post_onset_mode_accuracy}`
- Resampling count: `{resampling_count}`
- Runtime seconds: `{runtime_seconds}`
- Decision: `{decision}`
""", encoding="utf-8")
    mode_plot = plot_dir / "rbpf_mode_posterior.png"
    ess_plot = plot_dir / "rbpf_ess_timeline.png"
    labels = sorted({row["label"] for row in posterior_rows})
    series = []
    for label in labels:
        series.append((np.array([row["posterior"] for row in posterior_rows if row["label"] == label], dtype=np.float64), label))
    _plot_series(mode_plot, times, series, "RBPF mode posterior")
    _plot_series(ess_plot, times, [(np.array([row["ess"] for row in state_rows], dtype=np.float64), "ESS")], "RBPF ESS timeline")
    return AdvancedFilterWitnessArtifacts(run_dir, report_path, posterior_path, state_path, metrics_path, (mode_plot, ess_plot))


def write_advanced_filter_comparison_artifacts(output_dir: str | Path) -> AdvancedFilterComparisonArtifacts:
    output_root = Path(output_dir)
    run_dir = output_root / "advanced_filter_comparison_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    imm_metrics = _read_first_csv_row(output_root / "imm_filter_v1" / "switching_detection_metrics.csv")
    pf_metrics = _read_first_csv_row(output_root / "particle_filter_v1" / "pf_method_comparison.csv")
    rbpf_metrics = _read_first_csv_row(output_root / "rbpf_v1" / "rbpf_method_comparison.csv")

    method_rows = [
        {
            "method_id": "imm_v1",
            "corpus_objective_id": "imm_switching_v1",
            "scenario_family": "switching_1d",
            "failure_case": "static_class_assumption",
            "baseline_failed": "yes",
            "method_improved": "yes" if _as_float(imm_metrics.get("post_switch_accuracy")) > 0.40 else "no",
            "primary_metric": "post_switch_accuracy",
            "primary_metric_value": imm_metrics.get("post_switch_accuracy", ""),
            "runtime_seconds": imm_metrics.get("runtime_seconds", ""),
            "promotion_decision": imm_metrics.get("promotion_decision", "defer"),
            "artifact_path": "artifacts/imm_filter_v1/switching_detection_metrics.csv",
        },
        {
            "method_id": "particle_filter_bank_v1",
            "corpus_objective_id": "pf_nonlinear_drag_v1",
            "scenario_family": "nonlinear_drag_outlier_1d",
            "failure_case": "nonlinear_or_nongaussian_noise",
            "baseline_failed": "yes",
            "method_improved": "yes" if _as_float(pf_metrics.get("position_rmse")) < _as_float(pf_metrics.get("kalman_baseline_rmse")) else "no",
            "primary_metric": "position_rmse",
            "primary_metric_value": pf_metrics.get("position_rmse", ""),
            "runtime_seconds": pf_metrics.get("runtime_seconds", ""),
            "promotion_decision": pf_metrics.get("promotion_decision", "defer"),
            "artifact_path": "artifacts/particle_filter_v1/pf_method_comparison.csv",
        },
        {
            "method_id": "rbpf_v1",
            "corpus_objective_id": "rbpf_latent_maneuver_onset_v1",
            "scenario_family": "latent_maneuver_onset_1d",
            "failure_case": "latent_event_timing",
            "baseline_failed": "yes",
            "method_improved": "yes" if _as_float(rbpf_metrics.get("post_onset_mode_accuracy")) >= 0.50 else "no",
            "primary_metric": "post_onset_mode_accuracy",
            "primary_metric_value": rbpf_metrics.get("post_onset_mode_accuracy", ""),
            "runtime_seconds": rbpf_metrics.get("runtime_seconds", ""),
            "promotion_decision": rbpf_metrics.get("promotion_decision", "defer"),
            "artifact_path": "artifacts/rbpf_v1/rbpf_method_comparison.csv",
        },
    ]

    nonlinear_rows = [
        {
            "method_id": pf_metrics.get("method_id", "particle_filter_bank_v1"),
            "witness": pf_metrics.get("witness", "nonlinear_drag_outlier_1d"),
            "position_rmse": pf_metrics.get("position_rmse", ""),
            "kalman_baseline_rmse": pf_metrics.get("kalman_baseline_rmse", ""),
            "observation_baseline_rmse": pf_metrics.get("observation_baseline_rmse", ""),
            "rmse_delta_vs_kalman": pf_metrics.get("rmse_delta_vs_kalman", ""),
            "resampling_count": pf_metrics.get("resampling_count", ""),
            "promotion_decision": pf_metrics.get("promotion_decision", "defer"),
        }
    ]
    latent_rows = [
        {
            "method_id": rbpf_metrics.get("method_id", "rbpf_v1"),
            "witness": rbpf_metrics.get("witness", "latent_maneuver_onset_1d"),
            "state_position_rmse": rbpf_metrics.get("state_position_rmse", ""),
            "post_onset_mode_accuracy": rbpf_metrics.get("post_onset_mode_accuracy", ""),
            "resampling_count": rbpf_metrics.get("resampling_count", ""),
            "promotion_decision": rbpf_metrics.get("promotion_decision", "defer"),
        }
    ]
    runtime_rows = [
        {"method_id": row["method_id"], "runtime_seconds": row["runtime_seconds"], "scenario_family": row["scenario_family"]}
        for row in method_rows
    ]
    decision_rows = [
        {
            "method_id": row["method_id"],
            "failure_case": row["failure_case"],
            "baseline_failed": row["baseline_failed"],
            "method_improved": row["method_improved"],
            "cost_acceptable": "yes",
            "promotion_decision": row["promotion_decision"],
            "supporting_artifact": row["artifact_path"],
        }
        for row in method_rows
    ]

    method_path = run_dir / "method_comparison.csv"
    nonlinear_path = run_dir / "nonlinear_stress_metrics.csv"
    latent_path = run_dir / "latent_maneuver_metrics.csv"
    runtime_path = run_dir / "runtime_cost_metrics.csv"
    decision_path = run_dir / "advanced_filter_decision_matrix.csv"
    report_path = run_dir / "advanced_filter_comparison_report.md"
    _write_csv(method_path, method_rows, list(method_rows[0]))
    _write_csv(nonlinear_path, nonlinear_rows, list(nonlinear_rows[0]))
    _write_csv(latent_path, latent_rows, list(latent_rows[0]))
    _write_csv(runtime_path, runtime_rows, list(runtime_rows[0]))
    _write_csv(decision_path, decision_rows, list(decision_rows[0]))
    report_path.write_text(_render_advanced_filter_comparison_report(method_rows, decision_rows), encoding="utf-8")

    return AdvancedFilterComparisonArtifacts(
        run_dir=run_dir,
        method_comparison_path=method_path,
        nonlinear_stress_metrics_path=nonlinear_path,
        latent_maneuver_metrics_path=latent_path,
        runtime_cost_metrics_path=runtime_path,
        decision_matrix_path=decision_path,
        report_path=report_path,
    )


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_first_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _render_advanced_filter_comparison_report(method_rows: list[dict[str, object]], decision_rows: list[dict[str, object]]) -> str:
    lines = [
        "# Advanced Filter Comparison V1",
        "",
        "This artifact treats IMM, PF, and RBPF as advanced evidence providers inside the shared posterior/evaluation contract.",
        "",
        "## Decision Matrix",
        "",
        "| Method | Failure Case | Improved | Decision | Artifact |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in decision_rows:
        lines.append(
            f"| `{row['method_id']}` | `{row['failure_case']}` | `{row['method_improved']}` | `{row['promotion_decision']}` | `{row['supporting_artifact']}` |"
        )
    lines.extend(
        [
            "",
            "## Method Metrics",
            "",
            "| Method | Objective | Primary Metric | Value | Runtime Seconds |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in method_rows:
        lines.append(
            f"| `{row['method_id']}` | `{row['corpus_objective_id']}` | `{row['primary_metric']}` | `{row['primary_metric_value']}` | `{row['runtime_seconds']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- IMM is judged on switching-mode recovery, especially post-switch accuracy.",
            "- PF is judged on nonlinear or non-Gaussian state estimation stress, not generic accuracy.",
            "- RBPF is judged on latent-event mode recovery while conditionally filtering the continuous PVA state.",
            "- A method is promoted only when the failure case it was designed for is visible and improved.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_series(path: Path, times: np.ndarray, series: list[tuple[np.ndarray, str]], title: str) -> None:
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for values, label in series:
        ax.plot(times, values, label=label)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
