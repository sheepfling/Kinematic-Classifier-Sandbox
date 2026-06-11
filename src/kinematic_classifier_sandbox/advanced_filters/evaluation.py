from __future__ import annotations

import csv
import json
import time as wall_time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy.random as random
from numpy import (
    arange,
    array,
    asarray,
    average,
    diag,
    exp,
    float64,
    int64,
    isnan,
    mean,
    median,
    ndarray,
    repeat,
    sqrt,
    zeros,
)

from kinematic_classifier_sandbox.markdown_builder import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import plt

from ..tracing.filter_trace import FilterStepTrace, posterior_entropy, write_filter_step_trace_csv
from ..tracing.trace_validation import validate_filter_step_trace_set
from .linear_gaussian import KalmanModeState, kalman_predict, kalman_update
from .models_1d import (
    constant_velocity_transition,
    make_initial_particles_1d,
    nonlinear_drag_transition,
    position_gaussian_log_likelihood,
    position_mixture_log_likelihood,
)
from .particle_filter import BootstrapParticleFilter, ParticleFilterConfig
from .particle_filter_bank import ParticleFilterBank
from .ou_witness import write_ornstein_uhlenbeck_witness_artifacts
from .oracle_gsf_1d import analyze_gsf_abs_range_multimodal_witness
from .oracle_pf_1d import analyze_pf_abs_range_multimodal_witness
from .rbpf import RaoBlackwellizedParticleFilter, RBPFConfig
from .rbpf_models_1d import default_mode_transition_matrix_1d, make_rbpf_1d_mode_models
from .surface import AdvancedFilterSurface


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
    gate_matrix_path: Path
    gate_matrix_json_path: Path
    nonlinear_stress_metrics_path: Path
    particle_filter_robustness_summary_path: Path
    gsf_robustness_summary_path: Path
    gsf_vs_pf_frontier_path: Path
    gsf_vs_pf_frontier_summary_path: Path
    latent_maneuver_metrics_path: Path
    rbpf_robustness_summary_path: Path
    mean_reverting_metrics_path: Path
    runtime_cost_metrics_path: Path
    particle_count_pareto_path: Path
    pf_vs_rbpf_frontier_path: Path
    pf_vs_rbpf_frontier_summary_path: Path
    advanced_method_promotion_cards_path: Path
    decision_matrix_path: Path
    report_path: Path
    particle_count_plot_path: Path
    gsf_vs_pf_frontier_plot_path: Path
    pf_vs_rbpf_frontier_plot_path: Path


@dataclass(frozen=True, slots=True)
class ParticleFilterPosteriorRow:
    trajectory_id: str
    time: float
    label: str
    posterior: float
    predicted_label: str
    confidence: float
    log_evidence: float


@dataclass(frozen=True, slots=True)
class ParticleFilterStateRow:
    trajectory_id: str
    time: float
    truth_position: float
    observation: float
    pf_position: float
    pf_velocity: float
    kalman_position: float
    kalman_velocity: float
    predicted_label: str
    ess: float
    ess_fraction: float
    resampled: bool
    unique_ancestor_count: int
    unique_ancestor_fraction: float


@dataclass(frozen=True, slots=True)
class RBPFFilterPosteriorRow:
    trajectory_id: str
    time: float
    label: str
    posterior: float
    predicted_label: str
    confidence: float
    log_evidence: float


@dataclass(frozen=True, slots=True)
class RBPFStateRow:
    trajectory_id: str
    time: float
    truth_position: float
    observation: float
    state_position: float
    state_velocity: float
    state_acceleration: float
    predicted_mode: str
    ess: float
    ess_fraction: float
    resampled: bool
    unique_ancestor_count: int
    unique_ancestor_fraction: float


@dataclass(frozen=True, slots=True)
class ParticleFilterWitnessResult:
    posterior_rows: tuple[ParticleFilterPosteriorRow, ...]
    state_rows: tuple[ParticleFilterStateRow, ...]
    metrics: dict[str, float | int | str]


def write_particle_filter_witness_artifacts(output_dir: str | Path, *, seed: int = 23) -> AdvancedFilterWitnessArtifacts:
    result = analyze_particle_filter_witness(seed=seed)
    run_dir = Path(output_dir) / "particle_filter_v1"
    plot_dir = run_dir / "plots"
    trace_dir = run_dir / "traces"
    plot_dir.mkdir(parents=True, exist_ok=True)
    posterior_rows = result.posterior_rows
    state_rows = result.state_rows
    metrics = [result.metrics]

    posterior_path = run_dir / "posterior_history.csv"
    state_path = run_dir / "state_estimate_history.csv"
    metrics_path = run_dir / "pf_method_comparison.csv"
    report_path = run_dir / "pf_report.md"
    write_csv(posterior_path, [asdict(posterior_row) for posterior_row in posterior_rows], ["trajectory_id", "time", "label", "posterior", "predicted_label", "confidence", "log_evidence"])
    write_csv(state_path, [asdict(state_row) for state_row in state_rows], ["trajectory_id", "time", "truth_position", "observation", "pf_position", "pf_velocity", "kalman_position", "kalman_velocity", "predicted_label", "ess", "ess_fraction", "resampled", "unique_ancestor_count", "unique_ancestor_fraction"])
    write_csv(metrics_path, metrics, list(metrics[0]))
    traces = _build_pf_filter_step_traces(posterior_rows, state_rows)
    validate_filter_step_trace_set(traces)
    write_filter_step_trace_csv(trace_dir / "filter_step_trace.csv", traces)
    report = MarkdownDocument("Particle Filter V1 Report")
    report.paragraph("Witness: `nonlinear_drag_outlier_1d`")
    report.paragraph(
        "PF is evaluated as a posterior-compatible classifier bank over `constant_velocity` and `nonlinear_drag`."
    )
    report.bullet_list(
        [
            f"PF nonlinear-drag position RMSE: `{result.metrics['position_rmse']}`",
            f"Gaussian Kalman baseline RMSE: `{result.metrics['kalman_baseline_rmse']}`",
            f"Observation baseline RMSE: `{result.metrics['observation_baseline_rmse']}`",
            f"RMSE delta vs Kalman: `{result.metrics['rmse_delta_vs_kalman']}`",
            f"Mean ESS/N: `{result.metrics['mean_ess_fraction']}`",
            f"Resampling count: `{result.metrics['resampling_count']}`",
            f"Mean unique ancestors/N: `{result.metrics['mean_unique_ancestor_fraction']}`",
            f"Runtime seconds: `{result.metrics['runtime_seconds']}`",
            f"Decision: `{result.metrics['promotion_decision']}`",
        ]
    )
    report_path.write_text(report.text(), encoding="utf-8")
    state_plot = plot_dir / "pf_state_vs_truth.png"
    ess_plot = plot_dir / "pf_ess_timeline.png"
    times = array([row.time for row in state_rows], dtype=float64)
    truth = array([row.truth_position for row in state_rows], dtype=float64)
    observations = array([row.observation for row in state_rows], dtype=float64)
    _plot_series(state_plot, times, [(truth, "truth"), (observations, "observation"), (array([row.kalman_position for row in state_rows]), "Kalman baseline"), (array([row.pf_position for row in state_rows]), "PF estimate")], "PF state vs truth")
    _plot_series(ess_plot, times, [(array([row.ess_fraction for row in state_rows], dtype=float64), "ESS/N")], "PF ESS timeline")
    return AdvancedFilterWitnessArtifacts(run_dir, report_path, posterior_path, state_path, metrics_path, (state_plot, ess_plot))


def analyze_particle_filter_witness(*, seed: int = 23, particle_count: int = 384) -> ParticleFilterWitnessResult:
    rng = random.default_rng(seed)
    times = arange(0.0, 8.0, 0.25, dtype=float64)
    positions = [0.0]
    velocities = [2.2]
    for _ in times[1:]:
        dt = 0.25
        drag = 0.11 * velocities[-1] * abs(velocities[-1])
        velocities.append(velocities[-1] - drag * dt)
        positions.append(positions[-1] + velocities[-1] * dt)
    truth = asarray(positions, dtype=float64)
    observations = truth + rng.normal(0.0, 0.12, size=len(truth))
    outlier_indices = [8, 13, 18, 25, 29]
    observations[outlier_indices] += array([1.8, -1.6, 1.5, -1.4, 1.7], dtype=float64)

    bank_rng = random.default_rng(seed + 1)
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

    posterior_rows: list[ParticleFilterPosteriorRow] = []
    state_rows: list[ParticleFilterStateRow] = []
    kalman_state = KalmanModeState(
        mean=array([observations[0], 2.2, 0.0], dtype=float64),
        covariance=diag([0.08, 0.25, 0.30]).astype(float64),
    )
    last_time: float | None = None
    start = wall_time.perf_counter()
    for index, (time_value, observation) in enumerate(zip(times, observations, strict=True)):
        dt = 0.25 if last_time is None else max(float(time_value - last_time), 1.0e-9)
        if last_time is not None:
            kalman_state = kalman_predict(kalman_state, dt=dt, process_noise_scale=0.01)
        kalman_update_result = kalman_update(
            kalman_state,
            observation=array([observation], dtype=float64),
            measurement_noise=0.12**2,
        )
        kalman_state = kalman_update_result.state
        last_time = float(time_value)
        step = bank.update(float(time_value), array([observation], dtype=float64))
        for label, posterior in step.posterior_by_label.items():
            posterior_rows.append(
                ParticleFilterPosteriorRow(
                    trajectory_id=step.trajectory_id,
                    time=time_value,
                    label=label,
                    posterior=posterior,
                    predicted_label=step.predicted_label,
                    confidence=step.confidence,
                    log_evidence=step.log_evidence_by_label[label],
                )
            )
        nonlinear_filter = filters["nonlinear_drag"]
        assert nonlinear_filter.state is not None
        weights = exp(nonlinear_filter.state.log_weights)
        state_mean = average(nonlinear_filter.state.particles, axis=0, weights=weights)
        state_rows.append(
            ParticleFilterStateRow(
                trajectory_id=step.trajectory_id,
                time=time_value,
                truth_position=truth[index],
                observation=observation,
                pf_position=float(state_mean[0]),
                pf_velocity=float(state_mean[1]),
                kalman_position=float(kalman_state.mean[0]),
                kalman_velocity=float(kalman_state.mean[1]),
                predicted_label=step.predicted_label,
                ess=float(step.diagnostics["ess_nonlinear_drag"]),
                ess_fraction=float(step.diagnostics["ess_nonlinear_drag"] / particle_count),
                resampled=bool(step.diagnostics["resampled_nonlinear_drag"]),
                unique_ancestor_count=int(step.diagnostics["unique_ancestor_count_nonlinear_drag"]),
                unique_ancestor_fraction=float(step.diagnostics["unique_ancestor_fraction_nonlinear_drag"]),
            )
        )
    runtime_seconds = wall_time.perf_counter() - start
    rmse = float(sqrt(mean([(row.pf_position - row.truth_position) ** 2 for row in state_rows])))
    kalman_baseline_rmse = float(sqrt(mean([(row.kalman_position - row.truth_position) ** 2 for row in state_rows])))
    observation_baseline_rmse = float(sqrt(mean((observations - truth) ** 2)))
    resampling_count = sum(1 for row in state_rows if row.resampled)
    mean_ess_fraction = float(mean([row.ess_fraction for row in state_rows]))
    mean_unique_ancestor_fraction = float(mean([row.unique_ancestor_fraction for row in state_rows]))
    nonlinear_drag_nll = -float(mean([row.log_evidence for row in posterior_rows if row.label == "nonlinear_drag"]))
    rmse_delta_vs_kalman = kalman_baseline_rmse - rmse
    decision = "promote" if rmse < kalman_baseline_rmse and rmse < observation_baseline_rmse else "revise"
    return ParticleFilterWitnessResult(
        posterior_rows=tuple(posterior_rows),
        state_rows=tuple(state_rows),
        metrics={
            "method_id": "particle_filter_bank_v1",
            "witness": "nonlinear_drag_outlier_1d",
            "particle_count": particle_count,
            "position_rmse": rmse,
            "kalman_baseline_rmse": kalman_baseline_rmse,
            "observation_baseline_rmse": observation_baseline_rmse,
            "rmse_delta_vs_kalman": rmse_delta_vs_kalman,
            "nonlinear_drag_nll": nonlinear_drag_nll,
            "mean_ess_fraction": mean_ess_fraction,
            "mean_unique_ancestor_fraction": mean_unique_ancestor_fraction,
            "resampling_count": resampling_count,
            "runtime_seconds": runtime_seconds,
            "promotion_decision": decision,
        },
    )


def particle_filter_witness_surface() -> AdvancedFilterSurface[None, AdvancedFilterWitnessArtifacts]:
    return AdvancedFilterSurface(
        study_id="particle_filter_v1",
        run=lambda: None,
        write_artifacts=write_particle_filter_witness_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.report_path),
            str(artifacts.metrics_path),
        ),
        metadata={
            "study_kind": "1d_witness",
            "problem_family": "particle_filter_1d",
        },
    )


def write_rbpf_witness_artifacts(output_dir: str | Path, *, seed: int = 31) -> AdvancedFilterWitnessArtifacts:
    run_dir = Path(output_dir) / "rbpf_v1"
    plot_dir = run_dir / "plots"
    trace_dir = run_dir / "traces"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = random.default_rng(seed)
    times = arange(0.0, 8.0, 0.25, dtype=float64)
    onset_time = 3.0
    positions = [0.0]
    velocities = [0.6]
    for time_value in times[1:]:
        dt = 0.25
        accel = 1.35 if time_value >= onset_time else 0.0
        velocities.append(velocities[-1] + accel * dt)
        positions.append(positions[-1] + velocities[-1] * dt)
    truth = asarray(positions, dtype=float64)
    observations = truth + rng.normal(0.0, 0.10, size=len(truth))

    particle_count = 256
    rbpf = RaoBlackwellizedParticleFilter(
        RBPFConfig(particle_count=particle_count, seed=seed + 1),
        make_rbpf_1d_mode_models(dt=0.25, measurement_std=0.15),
        default_mode_transition_matrix_1d(),
    )
    initial_modes = rng.choice(4, size=particle_count, p=array([0.55, 0.25, 0.05, 0.15], dtype=float64)).astype(int64)
    means = zeros((particle_count, 3), dtype=float64)
    means[:, 0] = observations[0]
    means[:, 1] = rng.normal(0.6, 0.2, size=particle_count)
    covariances = repeat(diag([0.2, 0.5, 0.5])[None, :, :], particle_count, axis=0)
    rbpf.reset("latent_maneuver_onset_1d", initial_modes, means, covariances)

    posterior_rows: list[RBPFFilterPosteriorRow] = []
    state_rows: list[RBPFStateRow] = []
    start = wall_time.perf_counter()
    for index, (time_value, observation) in enumerate(zip(times, observations, strict=True)):
        step = rbpf.update(float(time_value), array([observation], dtype=float64))
        for label, posterior in step.posterior_by_label.items():
            posterior_rows.append(
                RBPFFilterPosteriorRow(
                    trajectory_id=step.trajectory_id,
                    time=time_value,
                    label=label,
                    posterior=posterior,
                    predicted_label=step.predicted_label,
                    confidence=step.confidence,
                    log_evidence=step.log_evidence_by_label[label],
                )
            )
        assert rbpf.state is not None
        weights = exp(rbpf.state.log_weights)
        state_mean = average(rbpf.state.means, axis=0, weights=weights)
        state_rows.append(
            RBPFStateRow(
                trajectory_id=step.trajectory_id,
                time=time_value,
                truth_position=truth[index],
                observation=observation,
                state_position=float(state_mean[0]),
                state_velocity=float(state_mean[1]),
                state_acceleration=float(state_mean[2]),
                predicted_mode=step.predicted_label,
                ess=float(step.diagnostics["ess"]),
                ess_fraction=float(step.diagnostics["ess_fraction"]),
                resampled=bool(step.diagnostics["resampled"]),
                unique_ancestor_count=int(step.diagnostics["unique_ancestor_count"]),
                unique_ancestor_fraction=float(step.diagnostics["unique_ancestor_fraction"]),
            )
        )
    runtime_seconds = wall_time.perf_counter() - start
    rmse = float(sqrt(mean([(row.state_position - row.truth_position) ** 2 for row in state_rows])))
    post_onset_rows = [row for row in state_rows if row.time >= onset_time]
    post_onset_mode_accuracy = sum(row.predicted_mode in {"accelerate", "maneuver"} for row in post_onset_rows) / max(len(post_onset_rows), 1)
    resampling_count = sum(1 for row in state_rows if row.resampled)
    decision = "promote" if post_onset_mode_accuracy >= 0.50 else "revise"

    posterior_path = run_dir / "latent_mode_posterior.csv"
    state_path = run_dir / "conditional_filter_history.csv"
    metrics_path = run_dir / "rbpf_method_comparison.csv"
    report_path = run_dir / "rbpf_report.md"
    write_csv(posterior_path, [asdict(posterior_row) for posterior_row in posterior_rows], ["trajectory_id", "time", "label", "posterior", "predicted_label", "confidence", "log_evidence"])
    write_csv(state_path, [asdict(state_row) for state_row in state_rows], ["trajectory_id", "time", "truth_position", "observation", "state_position", "state_velocity", "state_acceleration", "predicted_mode", "ess", "ess_fraction", "resampled", "unique_ancestor_count", "unique_ancestor_fraction"])
    metrics = [{"method_id": "rbpf_v1", "witness": "latent_maneuver_onset_1d", "state_position_rmse": rmse, "post_onset_mode_accuracy": post_onset_mode_accuracy, "resampling_count": resampling_count, "runtime_seconds": runtime_seconds, "promotion_decision": decision}]
    write_csv(metrics_path, metrics, list(metrics[0]))
    traces = _build_rbpf_filter_step_traces(tuple(posterior_rows), tuple(state_rows))
    validate_filter_step_trace_set(traces)
    write_filter_step_trace_csv(trace_dir / "filter_step_trace.csv", traces)
    report = MarkdownDocument("RBPF V1 Report")
    report.paragraph("Witness: `latent_maneuver_onset_1d`")
    report.paragraph(
        "RBPF samples latent mode paths while conditionally Kalman-filtering the continuous PVA state."
    )
    report.bullet_list(
        [
            f"State position RMSE: `{rmse}`",
            f"Post-onset mode accuracy: `{post_onset_mode_accuracy}`",
            f"Mean ESS/N: `{float(mean([row.ess_fraction for row in state_rows]))}`",
            f"Resampling count: `{resampling_count}`",
            f"Mean unique ancestors/N: `{float(mean([row.unique_ancestor_fraction for row in state_rows]))}`",
            f"Runtime seconds: `{runtime_seconds}`",
            f"Decision: `{decision}`",
        ]
    )
    report_path.write_text(report.text(), encoding="utf-8")
    mode_plot = plot_dir / "rbpf_mode_posterior.png"
    ess_plot = plot_dir / "rbpf_ess_timeline.png"
    labels = sorted({row.label for row in posterior_rows})
    series = []
    for label in labels:
        series.append((array([row.posterior for row in posterior_rows if row.label == label], dtype=float64), label))
    _plot_series(mode_plot, times, series, "RBPF mode posterior")
    _plot_series(ess_plot, times, [(array([row.ess for row in state_rows], dtype=float64), "ESS")], "RBPF ESS timeline")
    return AdvancedFilterWitnessArtifacts(run_dir, report_path, posterior_path, state_path, metrics_path, (mode_plot, ess_plot))


def rbpf_witness_surface() -> AdvancedFilterSurface[None, AdvancedFilterWitnessArtifacts]:
    return AdvancedFilterSurface(
        study_id="rbpf_v1",
        run=lambda: None,
        write_artifacts=write_rbpf_witness_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.report_path),
            str(artifacts.metrics_path),
        ),
        metadata={
            "study_kind": "1d_witness",
            "problem_family": "rbpf_1d",
        },
    )


def _build_pf_filter_step_traces(
    posterior_rows: tuple[ParticleFilterPosteriorRow, ...],
    state_rows: tuple[ParticleFilterStateRow, ...],
) -> tuple[FilterStepTrace, ...]:
    labels = sorted({row.label for row in posterior_rows})
    by_time_and_label = {(row.time, row.label): row for row in posterior_rows}
    time_to_state = {row.time: row for row in state_rows}
    previous_posterior = {label: 1.0 / len(labels) for label in labels}
    traces: list[FilterStepTrace] = []
    ordered_times = sorted({row.time for row in posterior_rows})
    for time_index, time_value in enumerate(ordered_times):
        rows_at_time = [by_time_and_label[(time_value, label)] for label in labels]
        posterior_map = {row.label: float(row.posterior) for row in rows_at_time}
        entropy = posterior_entropy(posterior_map)
        state = time_to_state[time_value]
        for row in rows_at_time:
            traces.append(
                FilterStepTrace(
                    run_id="particle_filter_v1",
                    study_id="pf_nonlinear_drag_v1",
                    trajectory_id=row.trajectory_id,
                    method_id="particle_filter_bank_v1",
                    rung="PF",
                    time_index=time_index,
                    time=row.time,
                    dt=0.25,
                    class_or_model=row.label,
                    true_class="nonlinear_drag" if row.label == "nonlinear_drag" else None,
                    true_mode=None,
                    prior_probability=previous_posterior[row.label],
                    predicted_probability=previous_posterior[row.label],
                    log_transition_probability=None,
                    measurement=(state.observation,),
                    predicted_measurement=None,
                    innovation=None,
                    innovation_covariance_diag=None,
                    normalized_innovation_squared=None,
                    log_likelihood=row.log_evidence,
                    incremental_log_evidence=row.log_evidence,
                    posterior_probability=row.posterior,
                    posterior_entropy=entropy,
                    predicted_state_mean=None,
                    predicted_state_covariance_diag=None,
                    updated_state_mean=(state.pf_position, state.pf_velocity) if row.label == "nonlinear_drag" else None,
                    updated_state_covariance_diag=None,
                    effective_sample_size=state.ess if row.label == "nonlinear_drag" else None,
                    is_resampled=state.resampled if row.label == "nonlinear_drag" else None,
                )
            )
        previous_posterior = posterior_map
    return tuple(traces)


def _build_rbpf_filter_step_traces(
    posterior_rows: tuple[RBPFFilterPosteriorRow, ...],
    state_rows: tuple[RBPFStateRow, ...],
) -> tuple[FilterStepTrace, ...]:
    labels = sorted({row.label for row in posterior_rows})
    by_time_and_label = {(row.time, row.label): row for row in posterior_rows}
    time_to_state = {row.time: row for row in state_rows}
    previous_posterior = {label: 1.0 / len(labels) for label in labels}
    traces: list[FilterStepTrace] = []
    ordered_times = sorted({row.time for row in posterior_rows})
    for time_index, time_value in enumerate(ordered_times):
        rows_at_time = [by_time_and_label[(time_value, label)] for label in labels]
        posterior_map = {row.label: float(row.posterior) for row in rows_at_time}
        entropy = posterior_entropy(posterior_map)
        state = time_to_state[time_value]
        for row in rows_at_time:
            traces.append(
                FilterStepTrace(
                    run_id="rbpf_v1",
                    study_id="rbpf_latent_maneuver_onset_v1",
                    trajectory_id=row.trajectory_id,
                    method_id="rbpf_v1",
                    rung="RBPF",
                    time_index=time_index,
                    time=row.time,
                    dt=0.25,
                    class_or_model=row.label,
                    true_class=None,
                    true_mode="accelerate" if state.time >= 3.0 else "coast",
                    prior_probability=previous_posterior[row.label],
                    predicted_probability=previous_posterior[row.label],
                    log_transition_probability=None,
                    measurement=(state.observation,),
                    predicted_measurement=None,
                    innovation=None,
                    innovation_covariance_diag=None,
                    normalized_innovation_squared=None,
                    log_likelihood=row.log_evidence,
                    incremental_log_evidence=row.log_evidence,
                    posterior_probability=row.posterior,
                    posterior_entropy=entropy,
                    predicted_state_mean=None,
                    predicted_state_covariance_diag=None,
                    updated_state_mean=(state.state_position, state.state_velocity, state.state_acceleration),
                    updated_state_covariance_diag=None,
                    effective_sample_size=state.ess,
                    is_resampled=state.resampled,
                )
            )
        previous_posterior = posterior_map
    return tuple(traces)


def _particle_count_pareto_rows(
    *,
    particle_counts: tuple[int, ...] = (64, 128, 256, 512, 1024),
    seeds: tuple[int, ...] = (211, 223, 227, 229, 233, 239),
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for particle_count in particle_counts:
        results = [analyze_pf_abs_range_multimodal_witness(seed=seed, particle_count=particle_count) for seed in seeds]
        kl_values = [float(result.metrics["mean_oracle_to_pf_kl"]) for result in results]
        gaussian_kl_values = [float(result.metrics["mean_oracle_to_gaussian_kl"]) for result in results]
        sign_error_values = [float(result.metrics["mean_pf_positive_mass_error"]) for result in results]
        gaussian_sign_error_values = [float(result.metrics["mean_gaussian_positive_mass_error"]) for result in results]
        ess_values = [float(result.metrics["mean_ess_fraction"]) for result in results]
        runtime_values = [float(result.metrics["runtime_seconds"]) for result in results]
        promotion_rate = float(
            mean(
                [
                    1.0 if str(result.metrics["promotion_decision"]) == "promote_pf_for_multimodal_posterior" else 0.0
                    for result in results
                ]
            )
        )
        rows.append(
            {
                "method_id": "particle_filter_bank_v1",
                "witness": "pf_abs_range_multimodal_oracle_v1",
                "particle_count": particle_count,
                "seed_count": len(seeds),
                "mean_oracle_to_pf_kl": float(mean(kl_values)),
                "mean_oracle_to_gaussian_kl": float(mean(gaussian_kl_values)),
                "mean_pf_positive_mass_error": float(mean(sign_error_values)),
                "mean_gaussian_positive_mass_error": float(mean(gaussian_sign_error_values)),
                "mean_ess_fraction": float(mean(ess_values)),
                "mean_runtime_seconds": float(mean(runtime_values)),
                "promotion_rate": promotion_rate,
                "kl_seed_std": float(sqrt(mean([(value - float(mean(kl_values))) ** 2 for value in kl_values]))),
                "sign_error_seed_std": float(
                    sqrt(mean([(value - float(mean(sign_error_values))) ** 2 for value in sign_error_values]))
                ),
                "quality_improves_vs_previous": "",
                "robustness_passes_at_n": "no",
            }
        )
    previous_kl: float | None = None
    for row in rows:
        mean_kl = float(row["mean_oracle_to_pf_kl"])
        if previous_kl is None:
            row["quality_improves_vs_previous"] = "baseline"
        else:
            row["quality_improves_vs_previous"] = "yes" if mean_kl < previous_kl - 0.01 else "saturating"
        previous_kl = mean_kl
        row["robustness_passes_at_n"] = (
            "yes"
            if float(row["promotion_rate"]) >= 0.80
            and float(row["mean_oracle_to_pf_kl"]) < float(row["mean_oracle_to_gaussian_kl"]) * 0.25
            and float(row["mean_pf_positive_mass_error"]) < float(row["mean_gaussian_positive_mass_error"]) * 0.75
            and float(row["mean_ess_fraction"]) >= 0.75
            and float(row["kl_seed_std"]) <= 0.20
            else "no"
        )
    return tuple(rows)


def _particle_filter_robustness_summary_rows(
    rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    passing_rows = [row for row in rows if str(row["robustness_passes_at_n"]) == "yes"]
    recommended = min(passing_rows, key=lambda row: int(row["particle_count"])) if passing_rows else None
    return (
        {
            "method_id": "particle_filter_bank_v1",
            "witness": "pf_abs_range_multimodal_oracle_v1",
            "robustness_sweep_passes": "yes" if recommended is not None else "no",
            "recommended_particle_count": int(recommended["particle_count"]) if recommended is not None else "",
            "recommended_promotion_rate": float(recommended["promotion_rate"]) if recommended is not None else "",
            "recommended_mean_oracle_to_pf_kl": float(recommended["mean_oracle_to_pf_kl"]) if recommended is not None else "",
            "recommended_mean_pf_positive_mass_error": (
                float(recommended["mean_pf_positive_mass_error"]) if recommended is not None else ""
            ),
            "recommended_mean_ess_fraction": float(recommended["mean_ess_fraction"]) if recommended is not None else "",
            "recommended_mean_runtime_seconds": float(recommended["mean_runtime_seconds"]) if recommended is not None else "",
            "sweep_particle_counts": " ".join(str(int(row["particle_count"])) for row in rows),
            "sweep_seed_count": int(rows[0]["seed_count"]) if rows else 0,
        },
    )


def _gsf_component_pareto_rows(
    *,
    max_component_counts: tuple[int, ...] = (2, 4, 6, 8),
    seeds: tuple[int, ...] = (211, 223, 227, 229, 233, 239),
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for max_components in max_component_counts:
        results = [analyze_gsf_abs_range_multimodal_witness(seed=seed, max_components=max_components) for seed in seeds]
        kl_values = [float(result.metrics["mean_oracle_to_gsf_kl"]) for result in results]
        gaussian_kl_values = [float(result.metrics["mean_oracle_to_gaussian_kl"]) for result in results]
        sign_error_values = [float(result.metrics["mean_gsf_positive_mass_error"]) for result in results]
        gaussian_sign_error_values = [float(result.metrics["mean_gaussian_positive_mass_error"]) for result in results]
        component_count_values = [float(result.metrics["mean_component_count"]) for result in results]
        runtime_values = [float(result.metrics["runtime_seconds"]) for result in results]
        promotion_rate = float(
            mean([1.0 if str(result.metrics["promotion_decision"]) == "gsf_witness_supported" else 0.0 for result in results])
        )
        rows.append(
            {
                "method_id": "gaussian_sum_filter_v1",
                "witness": "gsf_abs_range_multimodal_oracle_v1",
                "max_components": max_components,
                "seed_count": len(seeds),
                "mean_oracle_to_gsf_kl": float(mean(kl_values)),
                "mean_oracle_to_gaussian_kl": float(mean(gaussian_kl_values)),
                "mean_gsf_positive_mass_error": float(mean(sign_error_values)),
                "mean_gaussian_positive_mass_error": float(mean(gaussian_sign_error_values)),
                "mean_component_count": float(mean(component_count_values)),
                "mean_runtime_seconds": float(mean(runtime_values)),
                "promotion_rate": promotion_rate,
                "kl_seed_std": float(sqrt(mean([(value - float(mean(kl_values))) ** 2 for value in kl_values]))),
                "sign_error_seed_std": float(
                    sqrt(mean([(value - float(mean(sign_error_values))) ** 2 for value in sign_error_values]))
                ),
                "quality_improves_vs_previous": "",
                "robustness_passes_at_m": "no",
            }
        )
    previous_kl: float | None = None
    for row in rows:
        mean_kl = float(row["mean_oracle_to_gsf_kl"])
        if previous_kl is None:
            row["quality_improves_vs_previous"] = "baseline"
        else:
            row["quality_improves_vs_previous"] = "yes" if mean_kl < previous_kl - 0.002 else "saturating"
        previous_kl = mean_kl
        row["robustness_passes_at_m"] = (
            "yes"
            if float(row["promotion_rate"]) >= 0.80
            and float(row["mean_oracle_to_gsf_kl"]) < float(row["mean_oracle_to_gaussian_kl"]) * 0.35
            and float(row["mean_gsf_positive_mass_error"]) < float(row["mean_gaussian_positive_mass_error"]) * 0.65
            and float(row["kl_seed_std"]) <= 0.02
            and float(row["sign_error_seed_std"]) <= 0.05
            else "no"
        )
    return tuple(rows)


def _gsf_robustness_summary_rows(
    rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    passing_rows = [row for row in rows if str(row["robustness_passes_at_m"]) == "yes"]
    recommended = min(passing_rows, key=lambda row: int(row["max_components"])) if passing_rows else None
    return (
        {
            "method_id": "gaussian_sum_filter_v1",
            "witness": "gsf_abs_range_multimodal_oracle_v1",
            "robustness_sweep_passes": "yes" if recommended is not None else "no",
            "recommended_max_components": int(recommended["max_components"]) if recommended is not None else "",
            "recommended_promotion_rate": float(recommended["promotion_rate"]) if recommended is not None else "",
            "recommended_mean_oracle_to_gsf_kl": float(recommended["mean_oracle_to_gsf_kl"]) if recommended is not None else "",
            "recommended_mean_gsf_positive_mass_error": (
                float(recommended["mean_gsf_positive_mass_error"]) if recommended is not None else ""
            ),
            "recommended_mean_component_count": float(recommended["mean_component_count"]) if recommended is not None else "",
            "recommended_mean_runtime_seconds": float(recommended["mean_runtime_seconds"]) if recommended is not None else "",
            "sweep_max_components": " ".join(str(int(row["max_components"])) for row in rows),
            "sweep_seed_count": int(rows[0]["seed_count"]) if rows else 0,
        },
    )


def _gsf_vs_pf_frontier_rows(
    pf_rows: tuple[dict[str, object], ...],
    gsf_rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for row in pf_rows:
        rows.append(
            {
                "method_id": "particle_filter_bank_v1",
                "witness": "abs_range_multimodal_1d",
                "complexity_parameter_name": "particle_count",
                "complexity_parameter_value": int(row["particle_count"]),
                "mean_oracle_kl": float(row["mean_oracle_to_pf_kl"]),
                "mean_sign_mass_error": float(row["mean_pf_positive_mass_error"]),
                "mean_runtime_seconds": float(row["mean_runtime_seconds"]),
                "promotion_rate": float(row["promotion_rate"]),
                "robustness_passes": str(row["robustness_passes_at_n"]),
            }
        )
    for row in gsf_rows:
        rows.append(
            {
                "method_id": "gaussian_sum_filter_v1",
                "witness": "abs_range_multimodal_1d",
                "complexity_parameter_name": "max_components",
                "complexity_parameter_value": int(row["max_components"]),
                "mean_oracle_kl": float(row["mean_oracle_to_gsf_kl"]),
                "mean_sign_mass_error": float(row["mean_gsf_positive_mass_error"]),
                "mean_runtime_seconds": float(row["mean_runtime_seconds"]),
                "promotion_rate": float(row["promotion_rate"]),
                "robustness_passes": str(row["robustness_passes_at_m"]),
            }
        )
    return tuple(rows)


def _gsf_vs_pf_frontier_summary_rows(
    frontier_rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    pf_rows = [
        row for row in frontier_rows if str(row["method_id"]) == "particle_filter_bank_v1" and str(row["robustness_passes"]) == "yes"
    ]
    gsf_rows = [
        row for row in frontier_rows if str(row["method_id"]) == "gaussian_sum_filter_v1" and str(row["robustness_passes"]) == "yes"
    ]
    if not pf_rows or not gsf_rows:
        return tuple()
    pf_best = min(pf_rows, key=lambda row: float(row["mean_runtime_seconds"]))
    gsf_best = min(gsf_rows, key=lambda row: float(row["mean_runtime_seconds"]))
    if (
        float(gsf_best["mean_oracle_kl"]) <= float(pf_best["mean_oracle_kl"]) * 1.10
        and float(gsf_best["mean_sign_mass_error"]) <= float(pf_best["mean_sign_mass_error"]) * 1.10
        and float(gsf_best["mean_runtime_seconds"]) <= float(pf_best["mean_runtime_seconds"]) * 0.75
    ):
        crossover_status = "gsf_preferred"
    elif (
        float(pf_best["mean_oracle_kl"]) < float(gsf_best["mean_oracle_kl"]) * 0.85
        and float(pf_best["mean_sign_mass_error"]) < float(gsf_best["mean_sign_mass_error"]) * 0.90
        and float(pf_best["mean_runtime_seconds"]) <= float(gsf_best["mean_runtime_seconds"]) * 20.0
    ):
        crossover_status = "pf_preferred"
    else:
        crossover_status = "metric_split"
    return (
        {
            "witness": "abs_range_multimodal_1d",
            "crossover_status": crossover_status,
            "pf_recommended_particle_count": int(pf_best["complexity_parameter_value"]),
            "pf_mean_oracle_kl": float(pf_best["mean_oracle_kl"]),
            "pf_mean_sign_mass_error": float(pf_best["mean_sign_mass_error"]),
            "pf_mean_runtime_seconds": float(pf_best["mean_runtime_seconds"]),
            "gsf_recommended_max_components": int(gsf_best["complexity_parameter_value"]),
            "gsf_mean_oracle_kl": float(gsf_best["mean_oracle_kl"]),
            "gsf_mean_sign_mass_error": float(gsf_best["mean_sign_mass_error"]),
            "gsf_mean_runtime_seconds": float(gsf_best["mean_runtime_seconds"]),
        },
    )


def _acceleration_biased_transition(
    particles: ndarray,
    dt: float,
    rng: random.Generator,
    *,
    acceleration_bias: float,
    process_std: float,
) -> ndarray:
    next_particles = particles.copy()
    noise = rng.normal(0.0, process_std, size=particles.shape)
    next_particles[:, 1] = particles[:, 1] + acceleration_bias * dt + noise[:, 1]
    next_particles[:, 0] = particles[:, 0] + next_particles[:, 1] * dt + noise[:, 0]
    return next_particles


def _latent_onset_truth(seed: int = 31) -> tuple[ndarray, ndarray, ndarray, float]:
    rng = random.default_rng(seed)
    times = arange(0.0, 8.0, 0.25, dtype=float64)
    onset_time = 3.0
    positions = [0.0]
    velocities = [0.6]
    for time_value in times[1:]:
        dt = 0.25
        accel = 1.35 if time_value >= onset_time else 0.0
        velocities.append(velocities[-1] + accel * dt)
        positions.append(positions[-1] + velocities[-1] * dt)
    truth = asarray(positions, dtype=float64)
    observations = truth + rng.normal(0.0, 0.10, size=len(truth))
    return times, truth, observations, onset_time


def _smooth_acceleration_truth(seed: int = 91) -> tuple[ndarray, ndarray, ndarray, float]:
    rng = random.default_rng(seed)
    times = arange(0.0, 8.0, 0.25, dtype=float64)
    acceleration = 0.35
    positions = [0.0]
    velocities = [0.6]
    for _ in times[1:]:
        dt = 0.25
        velocities.append(velocities[-1] + acceleration * dt)
        positions.append(positions[-1] + velocities[-2] * dt + 0.5 * acceleration * dt * dt)
    truth = asarray(positions, dtype=float64)
    observations = truth + rng.normal(0.0, 0.10, size=len(truth))
    return times, truth, observations, 0.0


def _shared_pf_vs_rbpf_metrics(
    *,
    method_id: str,
    witness: str,
    particle_count: int,
    seed: int,
) -> dict[str, float | str]:
    if witness == "latent_maneuver_onset_1d_shared":
        times, truth, observations, onset_time = _latent_onset_truth(seed)
        post_truth = lambda time_value: time_value >= onset_time
    elif witness == "smooth_acceleration_shared":
        times, truth, observations, onset_time = _smooth_acceleration_truth(seed)
        post_truth = lambda time_value: True
    else:
        raise KeyError(f"unknown shared witness: {witness}")

    start = wall_time.perf_counter()
    if method_id == "pf_latent_onset_bank_v1":
        filters = {
            "coast": BootstrapParticleFilter(
                ParticleFilterConfig(particle_count=particle_count, seed=seed + 101),
                transition_fn=lambda particles, dt, gen: constant_velocity_transition(particles, dt, gen, process_std=0.05),
                log_likelihood_fn=lambda particles, obs: position_gaussian_log_likelihood(particles, obs, measurement_std=0.15),
            ),
            "accelerate": BootstrapParticleFilter(
                ParticleFilterConfig(particle_count=particle_count, seed=seed + 102),
                transition_fn=lambda particles, dt, gen: _acceleration_biased_transition(particles, dt, gen, acceleration_bias=0.35, process_std=0.05),
                log_likelihood_fn=lambda particles, obs: position_gaussian_log_likelihood(particles, obs, measurement_std=0.15),
            ),
        }
        bank = ParticleFilterBank(filters, prior_by_label={"coast": 0.6, "accelerate": 0.4}, filter_id=method_id)
        init_rng = random.default_rng(seed + 103)
        initial_particles = {
            "coast": make_initial_particles_1d(particle_count, observations[0], 0.18, 0.6, 0.2, init_rng),
            "accelerate": make_initial_particles_1d(particle_count, observations[0], 0.18, 0.6, 0.2, init_rng),
        }
        bank.reset("shared_pf_rbpf_witness", initial_particles)
        predicted_labels: list[str] = []
        positions: list[float] = []
        ess_series: list[float] = []
        ancestor_series: list[float] = []
        for time_value, observation in zip(times, observations, strict=True):
            step = bank.update(float(time_value), array([observation], dtype=float64))
            predicted_labels.append(step.predicted_label)
            accelerated_filter = filters["accelerate"]
            assert accelerated_filter.state is not None
            weights = exp(accelerated_filter.state.log_weights)
            state_mean = average(accelerated_filter.state.particles, axis=0, weights=weights)
            positions.append(float(state_mean[0]))
            ess_series.append(float(step.diagnostics["ess_fraction_accelerate"]))
            ancestor_series.append(float(step.diagnostics["unique_ancestor_fraction_accelerate"]))
        accuracy = sum(
            (label == "accelerate") if post_truth(time_value) else (label == "coast")
            for label, time_value in zip(predicted_labels, times, strict=True)
        ) / max(len(times), 1)
    else:
        rbpf = RaoBlackwellizedParticleFilter(
            RBPFConfig(particle_count=particle_count, seed=seed + 201),
            make_rbpf_1d_mode_models(dt=0.25, measurement_std=0.15),
            default_mode_transition_matrix_1d(),
        )
        rng = random.default_rng(seed + 202)
        if witness == "smooth_acceleration_shared":
            initial_probs = array([0.15, 0.70, 0.05, 0.10], dtype=float64)
        else:
            initial_probs = array([0.55, 0.25, 0.05, 0.15], dtype=float64)
        initial_modes = rng.choice(4, size=particle_count, p=initial_probs).astype(int64)
        means = zeros((particle_count, 3), dtype=float64)
        means[:, 0] = observations[0]
        means[:, 1] = rng.normal(0.6, 0.2, size=particle_count)
        covariances = repeat(diag([0.2, 0.5, 0.5])[None, :, :], particle_count, axis=0)
        rbpf.reset("shared_pf_rbpf_witness", initial_modes, means, covariances)
        predicted_labels = []
        positions = []
        ess_series = []
        ancestor_series = []
        for time_value, observation in zip(times, observations, strict=True):
            step = rbpf.update(float(time_value), array([observation], dtype=float64))
            predicted_labels.append(step.predicted_label)
            assert rbpf.state is not None
            weights = exp(rbpf.state.log_weights)
            state_mean = average(rbpf.state.means, axis=0, weights=weights)
            positions.append(float(state_mean[0]))
            ess_series.append(float(step.diagnostics["ess_fraction"]))
            ancestor_series.append(float(step.diagnostics["unique_ancestor_fraction"]))
        accuracy = sum(
            (label in {"accelerate", "maneuver"}) if post_truth(time_value) else (label == "coast")
            for label, time_value in zip(predicted_labels, times, strict=True)
        ) / max(len(times), 1)
    return {
        "mean_position_rmse": float(sqrt(mean((array(positions) - truth) ** 2))),
        "mean_post_onset_accuracy": float(accuracy),
        "mean_ess_fraction": float(mean(ess_series)),
        "mean_unique_ancestor_fraction": float(mean(ancestor_series)),
        "mean_runtime_seconds": float(wall_time.perf_counter() - start),
    }


def _pf_vs_rbpf_frontier_rows(
    *,
    particle_counts: tuple[int, ...] = (64, 128, 256, 512),
    seeds: tuple[int, ...] = (31, 43, 59, 71),
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for witness in ("latent_maneuver_onset_1d_shared", "smooth_acceleration_shared"):
        for particle_count in particle_counts:
            for method_id in ("pf_latent_onset_bank_v1", "rbpf_v1"):
                metric_rows = [
                    _shared_pf_vs_rbpf_metrics(
                        method_id=method_id,
                        witness=witness,
                        particle_count=particle_count,
                        seed=seed,
                    )
                    for seed in seeds
                ]
                rows.append(
                    {
                        "method_id": method_id,
                        "witness": witness,
                        "particle_count": particle_count,
                        "seed_count": len(seeds),
                        "mean_position_rmse": float(mean([float(row["mean_position_rmse"]) for row in metric_rows])),
                        "mean_post_onset_accuracy": float(mean([float(row["mean_post_onset_accuracy"]) for row in metric_rows])),
                        "mean_ess_fraction": float(mean([float(row["mean_ess_fraction"]) for row in metric_rows])),
                        "mean_unique_ancestor_fraction": float(mean([float(row["mean_unique_ancestor_fraction"]) for row in metric_rows])),
                        "mean_runtime_seconds": float(mean([float(row["mean_runtime_seconds"]) for row in metric_rows])),
                    }
                )
    return tuple(rows)


def _write_pf_vs_rbpf_frontier_plot(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    witness_titles = {
        "latent_maneuver_onset_1d_shared": "Latent Onset Witness",
        "smooth_acceleration_shared": "Smooth Acceleration Witness",
    }
    for row_index, witness in enumerate(("latent_maneuver_onset_1d_shared", "smooth_acceleration_shared")):
        witness_rows = [row for row in rows if row["witness"] == witness]
        for method_id, label, color in (
            ("pf_latent_onset_bank_v1", "PF latent-onset bank", "#2563eb"),
            ("rbpf_v1", "RBPF", "#c05621"),
        ):
            method_rows = [row for row in witness_rows if row["method_id"] == method_id]
            counts = [int(row["particle_count"]) for row in method_rows]
            rmse = [float(row["mean_position_rmse"]) for row in method_rows]
            accuracy = [float(row["mean_post_onset_accuracy"]) for row in method_rows]
            runtime = [float(row["mean_runtime_seconds"]) for row in method_rows]
            axes[row_index][0].plot(runtime, accuracy, marker="o", label=label, color=color)
            axes[row_index][1].plot(counts, rmse, marker="o", label=label, color=color)
        axes[row_index][0].set_title(f"{witness_titles[witness]}: Compute vs Accuracy")
        axes[row_index][0].set_xlabel("mean runtime (s)")
        axes[row_index][0].set_ylabel("post-onset accuracy")
        axes[row_index][0].grid(alpha=0.25)
        axes[row_index][0].legend(fontsize=8)
        axes[row_index][1].set_title(f"{witness_titles[witness]}: Particle Count vs RMSE")
        axes[row_index][1].set_xlabel("particle count")
        axes[row_index][1].set_ylabel("position RMSE")
        axes[row_index][1].grid(alpha=0.25)
        axes[row_index][1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _pf_vs_rbpf_frontier_summary_rows(rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    summary: list[dict[str, object]] = []
    for witness in sorted({str(row["witness"]) for row in rows}):
        witness_rows = [row for row in rows if str(row["witness"]) == witness]
        pf_rows = [row for row in witness_rows if str(row["method_id"]) == "pf_latent_onset_bank_v1"]
        rbpf_rows = [row for row in witness_rows if str(row["method_id"]) == "rbpf_v1"]
        if not pf_rows or not rbpf_rows:
            continue
        pf_best_rmse = min(float(row["mean_position_rmse"]) for row in pf_rows)
        rbpf_best_rmse = min(float(row["mean_position_rmse"]) for row in rbpf_rows)
        pf_best_runtime = min(float(row["mean_runtime_seconds"]) for row in pf_rows)
        rbpf_best_runtime = min(float(row["mean_runtime_seconds"]) for row in rbpf_rows)
        pf_best_accuracy = max(float(row["mean_post_onset_accuracy"]) for row in pf_rows)
        rbpf_best_accuracy = max(float(row["mean_post_onset_accuracy"]) for row in rbpf_rows)
        if pf_best_rmse < rbpf_best_rmse and pf_best_runtime < rbpf_best_runtime and pf_best_accuracy >= rbpf_best_accuracy - 0.10:
            crossover = "pf_preferred"
        elif rbpf_best_rmse < pf_best_rmse and rbpf_best_accuracy > pf_best_accuracy and rbpf_best_runtime <= pf_best_runtime * 8.0:
            crossover = "rbpf_preferred"
        else:
            crossover = "metric_split"
        summary.append(
            {
                "witness": witness,
                "pf_best_rmse": pf_best_rmse,
                "rbpf_best_rmse": rbpf_best_rmse,
                "pf_best_accuracy": pf_best_accuracy,
                "rbpf_best_accuracy": rbpf_best_accuracy,
                "pf_best_runtime_seconds": pf_best_runtime,
                "rbpf_best_runtime_seconds": rbpf_best_runtime,
                "crossover_status": crossover,
            }
        )
    return tuple(summary)


def _rbpf_robustness_summary_rows(
    frontier_rows: tuple[dict[str, object], ...],
    frontier_summary_rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    summary_by_witness = {str(row["witness"]): row for row in frontier_summary_rows}
    latent_summary = summary_by_witness.get("latent_maneuver_onset_1d_shared", {})
    smooth_summary = summary_by_witness.get("smooth_acceleration_shared", {})
    latent_rows = [row for row in frontier_rows if str(row["witness"]) == "latent_maneuver_onset_1d_shared"]
    smooth_rows = [row for row in frontier_rows if str(row["witness"]) == "smooth_acceleration_shared"]
    candidate_rows: list[dict[str, object]] = []
    for particle_count in sorted({int(row["particle_count"]) for row in latent_rows}):
        pf_row = next(
            row
            for row in latent_rows
            if int(row["particle_count"]) == particle_count and str(row["method_id"]) == "pf_latent_onset_bank_v1"
        )
        rbpf_row = next(
            row
            for row in latent_rows
            if int(row["particle_count"]) == particle_count and str(row["method_id"]) == "rbpf_v1"
        )
        candidate_rows.append(
            {
                "particle_count": particle_count,
                "rmse_gain_factor": float(pf_row["mean_position_rmse"]) / max(float(rbpf_row["mean_position_rmse"]), 1.0e-12),
                "accuracy_gap": float(rbpf_row["mean_post_onset_accuracy"]) - float(pf_row["mean_post_onset_accuracy"]),
                "runtime_ratio": float(rbpf_row["mean_runtime_seconds"]) / max(float(pf_row["mean_runtime_seconds"]), 1.0e-12),
                "rbpf_mean_ess_fraction": float(rbpf_row["mean_ess_fraction"]),
            }
        )
    viable_rows = [
        row
        for row in candidate_rows
        if float(row["rmse_gain_factor"]) >= 4.0
        and float(row["accuracy_gap"]) >= -0.10
        and float(row["runtime_ratio"]) <= 8.0
        and float(row["rbpf_mean_ess_fraction"]) >= 0.75
    ]
    recommended = min(viable_rows, key=lambda row: int(row["particle_count"])) if viable_rows else None
    latent_status = str(latent_summary.get("crossover_status", "not_run"))
    smooth_status = str(smooth_summary.get("crossover_status", "not_run"))
    robustness_passes = (
        latent_status == "rbpf_preferred"
        and smooth_status != "pf_preferred"
        and recommended is not None
    )
    return (
        {
            "method_id": "rbpf_v1",
            "primary_witness": "latent_maneuver_onset_1d_shared",
            "comparison_baseline": "pf_latent_onset_bank_v1",
            "latent_crossover_status": latent_status,
            "smooth_crossover_status": smooth_status,
            "robustness_sweep_passes": "yes" if robustness_passes else "no",
            "recommended_particle_count": int(recommended["particle_count"]) if recommended is not None else "",
            "recommended_rmse_gain_factor": float(recommended["rmse_gain_factor"]) if recommended is not None else "",
            "recommended_accuracy_gap": float(recommended["accuracy_gap"]) if recommended is not None else "",
            "recommended_runtime_ratio": float(recommended["runtime_ratio"]) if recommended is not None else "",
            "recommended_mean_ess_fraction": float(recommended["rbpf_mean_ess_fraction"]) if recommended is not None else "",
            "sweep_particle_counts": " ".join(str(int(row["particle_count"])) for row in candidate_rows),
            "seed_count": int(latent_rows[0]["seed_count"]) if latent_rows else 0,
        },
    )


def _write_particle_count_pareto_plot(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    counts = [int(row["particle_count"]) for row in rows]
    pf_kl = [float(row["mean_oracle_to_pf_kl"]) for row in rows]
    gaussian_kl = [float(row["mean_oracle_to_gaussian_kl"]) for row in rows]
    sign_error = [float(row["mean_pf_positive_mass_error"]) for row in rows]
    gaussian_sign_error = [float(row["mean_gaussian_positive_mass_error"]) for row in rows]
    ess = [float(row["mean_ess_fraction"]) for row in rows]
    runtime = [float(row["mean_runtime_seconds"]) for row in rows]
    promotion_rate = [float(row["promotion_rate"]) for row in rows]
    axes[0][0].plot(counts, pf_kl, marker="o", label="oracle->PF KL")
    axes[0][0].plot(counts, gaussian_kl, marker="x", label="oracle->Gaussian KL")
    axes[0][0].set_title("Particle Count vs Posterior KL")
    axes[0][0].set_xlabel("particle count")
    axes[0][0].set_ylabel("mean KL")
    axes[0][0].grid(alpha=0.25)
    axes[0][0].legend(fontsize=8)
    axes[0][1].plot(counts, sign_error, marker="o", label="PF sign-mass error")
    axes[0][1].plot(counts, gaussian_sign_error, marker="x", label="Gaussian sign-mass error")
    axes[0][1].set_title("Particle Count vs Sign-Mass Error")
    axes[0][1].set_xlabel("particle count")
    axes[0][1].set_ylabel("mean absolute error")
    axes[0][1].grid(alpha=0.25)
    axes[0][1].legend(fontsize=8)
    axes[1][0].plot(counts, ess, marker="o", label="mean ESS/N")
    axes[1][0].plot(counts, promotion_rate, marker="x", label="promotion rate")
    axes[1][0].set_title("Particle Count vs ESS and Stability")
    axes[1][0].set_xlabel("particle count")
    axes[1][0].grid(alpha=0.25)
    axes[1][0].legend(fontsize=8)
    axes[1][1].plot(counts, runtime, marker="o", label="mean runtime (s)")
    axes[1][1].set_title("Particle Count vs Runtime")
    axes[1][1].set_xlabel("particle count")
    axes[1][1].set_ylabel("seconds")
    axes[1][1].grid(alpha=0.25)
    axes[1][1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_gsf_vs_pf_frontier_plot(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    for method_id, label, color in (
        ("gaussian_sum_filter_v1", "GSF", "#2f855a"),
        ("particle_filter_bank_v1", "PF", "#2b6cb0"),
    ):
        method_rows = [row for row in rows if str(row["method_id"]) == method_id]
        if not method_rows:
            continue
        x_values = [int(row["complexity_parameter_value"]) for row in method_rows]
        axes[0].plot(x_values, [float(row["mean_oracle_kl"]) for row in method_rows], marker="o", color=color, label=label)
        axes[1].plot(
            x_values,
            [float(row["mean_runtime_seconds"]) for row in method_rows],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].set_title("GSF vs PF Oracle KL")
    axes[0].set_xlabel("complexity parameter")
    axes[0].set_ylabel("mean KL")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_title("GSF vs PF Runtime")
    axes[1].set_xlabel("complexity parameter")
    axes[1].set_ylabel("seconds")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_advanced_method_promotion_cards(
    *,
    method_rows: list[dict[str, object]],
    particle_count_rows: tuple[dict[str, object], ...],
) -> str:
    report = MarkdownDocument("Advanced Method Promotion Cards")
    report.paragraph(
        "These cards separate witness-specific promotion from stronger claims that a method is required by the current repo evidence."
    )
    particle_saturation = particle_count_rows[-1]["quality_improves_vs_previous"] if particle_count_rows else "not_run"
    pf_robustness_rows = _particle_filter_robustness_summary_rows(particle_count_rows)
    pf_robustness = pf_robustness_rows[0] if pf_robustness_rows else {}
    gsf_component_rows = _gsf_component_pareto_rows()
    gsf_robustness_rows = _gsf_robustness_summary_rows(gsf_component_rows)
    gsf_robustness = gsf_robustness_rows[0] if gsf_robustness_rows else {}
    gsf_vs_pf_rows = _gsf_vs_pf_frontier_rows(particle_count_rows, gsf_component_rows)
    gsf_vs_pf_summary_rows = _gsf_vs_pf_frontier_summary_rows(gsf_vs_pf_rows)
    gsf_vs_pf_status = str(gsf_vs_pf_summary_rows[0]["crossover_status"]) if gsf_vs_pf_summary_rows else "not_run"
    rbpf_frontier_rows = _pf_vs_rbpf_frontier_rows()
    rbpf_frontier_summary_rows = _pf_vs_rbpf_frontier_summary_rows(rbpf_frontier_rows)
    rbpf_robustness_rows = _rbpf_robustness_summary_rows(rbpf_frontier_rows, rbpf_frontier_summary_rows)
    rbpf_robustness = rbpf_robustness_rows[0] if rbpf_robustness_rows else {}
    for row in method_rows:
        method_id = str(row["method_id"])
        report.heading(method_id, level=2)
        if method_id == "imm_v1":
            report.bullet_list(
                [
                    "Scope: switching witness only.",
                    "Previous rung: transition matrix accumulator and static mode likelihood.",
                    "Designed failure mode: switching behavior with continuous-state dynamics evidence.",
                    f"Current decision: `{row['promotion_decision']}`.",
                    "Current status ceiling: `witness_supported` until robustness sweeps exist.",
                    "Required by current evidence: sequential and switching logic yes; IMM as universal default no.",
                ]
            )
        elif method_id == "particle_filter_bank_v1":
            report.bullet_list(
                [
                    "Scope: non-injective absolute-range multimodal witness only.",
                    "Previous rung: single-Gaussian posterior projection.",
                    "Designed failure mode: multimodal posterior collapse under a Gaussian summary.",
                    f"Current decision: `{row['promotion_decision']}`.",
                    f"Robustness sweep passes: `{pf_robustness.get('robustness_sweep_passes', 'no')}`.",
                    f"Recommended particle count: `{pf_robustness.get('recommended_particle_count', '')}`.",
                    f"GSF robustness sweep passes: `{gsf_robustness.get('robustness_sweep_passes', 'no')}`.",
                    f"GSF vs PF crossover status: `{gsf_vs_pf_status}`.",
                    "Required by current evidence: not yet.",
                    f"Particle-count saturation status: `{particle_saturation}`.",
                ]
            )
        elif method_id == "rbpf_v1":
            report.bullet_list(
                [
                    "Scope: latent maneuver onset witness only.",
                    "Previous rung: vanilla PF or transition logic should be compared on the same witness before promotion.",
                    "Designed failure mode: sampled latent event timing with conditionally linear-Gaussian PVA state.",
                    f"Current decision: `{row['promotion_decision']}`.",
                    f"Robustness sweep passes: `{rbpf_robustness.get('robustness_sweep_passes', 'no')}`.",
                    f"Latent crossover status: `{rbpf_robustness.get('latent_crossover_status', 'not_run')}`.",
                    f"Smooth crossover status: `{rbpf_robustness.get('smooth_crossover_status', 'not_run')}`.",
                    "Required by current evidence: not yet.",
                ]
            )
        else:
            report.bullet_list(
                [
                    f"Scope: `{row['scenario_family']}`.",
                    f"Current decision: `{row['promotion_decision']}`.",
                    "Required by current evidence: not yet generalized.",
                ]
            )
    return report.text()


def write_advanced_filter_comparison_artifacts(output_dir: str | Path) -> AdvancedFilterComparisonArtifacts:
    output_root = Path(output_dir)
    run_dir = output_root / "advanced_filter_comparison_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    imm_metrics = _read_first_csv_row(output_root / "imm_filter_v1" / "switching_detection_metrics.csv")
    pf_metrics = _read_first_csv_row(output_root / "particle_filter_v1" / "pf_method_comparison.csv")
    pf_oracle_metrics = _read_first_csv_row(output_root / "pf_abs_range_multimodal_oracle_v1" / "metrics_against_oracle.csv")
    gsf_oracle_metrics = _read_first_csv_row(output_root / "gsf_abs_range_multimodal_oracle_v1" / "metrics_against_oracle.csv")
    rbpf_metrics = _read_first_csv_row(output_root / "rbpf_v1" / "rbpf_method_comparison.csv")
    ou_metrics = _read_first_csv_row(output_root / "ornstein_uhlenbeck_witness_v1" / "ou_method_comparison.csv")
    pf_promotion_metrics = pf_oracle_metrics or pf_metrics
    particle_count_rows = _particle_count_pareto_rows()
    particle_filter_robustness_rows = _particle_filter_robustness_summary_rows(particle_count_rows)
    particle_filter_robustness = particle_filter_robustness_rows[0] if particle_filter_robustness_rows else {}
    gsf_component_rows = _gsf_component_pareto_rows()
    gsf_robustness_rows = _gsf_robustness_summary_rows(gsf_component_rows)
    gsf_robustness = gsf_robustness_rows[0] if gsf_robustness_rows else {}
    gsf_vs_pf_rows = _gsf_vs_pf_frontier_rows(particle_count_rows, gsf_component_rows)
    gsf_vs_pf_summary_rows = _gsf_vs_pf_frontier_summary_rows(gsf_vs_pf_rows)
    gsf_vs_pf_status = next((str(row["crossover_status"]) for row in gsf_vs_pf_summary_rows), "not_run")
    pf_vs_rbpf_rows = _pf_vs_rbpf_frontier_rows()
    pf_vs_rbpf_summary_rows = _pf_vs_rbpf_frontier_summary_rows(pf_vs_rbpf_rows)
    rbpf_frontier_metric = next(
        (
            str(row["crossover_status"])
            for row in pf_vs_rbpf_summary_rows
            if str(row["witness"]) == "latent_maneuver_onset_1d_shared"
        ),
        "",
    )

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
            "corpus_objective_id": "pf_abs_range_multimodal_oracle_v1" if pf_oracle_metrics else "pf_nonlinear_drag_v1",
            "scenario_family": "abs_range_multimodal_1d" if pf_oracle_metrics else "nonlinear_drag_outlier_1d",
            "failure_case": "multimodal_posterior_collapse" if pf_oracle_metrics else "nonlinear_or_nongaussian_noise",
            "baseline_failed": "yes",
            "method_improved": (
                "yes"
                if (
                    _as_float(pf_promotion_metrics.get("mean_oracle_to_pf_kl"))
                    < _as_float(pf_promotion_metrics.get("mean_oracle_to_gaussian_kl"))
                    and _as_float(pf_promotion_metrics.get("mean_pf_positive_mass_error"))
                    < _as_float(pf_promotion_metrics.get("mean_gaussian_positive_mass_error"))
                )
                else (
                    "yes"
                    if _as_float(pf_metrics.get("position_rmse")) < _as_float(pf_metrics.get("kalman_baseline_rmse"))
                    else "no"
                )
            ),
            "primary_metric": "mean_oracle_to_pf_kl" if pf_oracle_metrics else "position_rmse",
            "primary_metric_value": pf_promotion_metrics.get("mean_oracle_to_pf_kl", pf_metrics.get("position_rmse", "")),
            "runtime_seconds": pf_promotion_metrics.get("runtime_seconds", pf_metrics.get("runtime_seconds", "")),
            "promotion_decision": (
                "promote"
                if str(particle_filter_robustness.get("robustness_sweep_passes", "no")) == "yes" and gsf_vs_pf_status != "gsf_preferred"
                else "defer_to_gsf"
                if gsf_vs_pf_status == "gsf_preferred"
                else pf_promotion_metrics.get("promotion_decision", pf_metrics.get("promotion_decision", "defer"))
            ),
            "cost_acceptable": "no" if gsf_vs_pf_status == "gsf_preferred" else "yes",
            "artifact_path": (
                "artifacts/advanced_filter_comparison_v1/gsf_vs_pf_frontier_summary.csv"
                if gsf_vs_pf_summary_rows
                else (
                    "artifacts/pf_abs_range_multimodal_oracle_v1/metrics_against_oracle.csv"
                    if pf_oracle_metrics
                    else "artifacts/particle_filter_v1/pf_method_comparison.csv"
                )
            ),
        },
        {
            "method_id": "rbpf_v1",
            "corpus_objective_id": "pf_vs_rbpf_frontier_v1" if pf_vs_rbpf_summary_rows else "rbpf_latent_maneuver_onset_v1",
            "scenario_family": "latent_maneuver_onset_1d",
            "failure_case": "latent_event_timing",
            "baseline_failed": "yes",
            "method_improved": (
                "yes"
                if rbpf_frontier_metric == "rbpf_preferred"
                else ("yes" if _as_float(rbpf_metrics.get("post_onset_mode_accuracy")) >= 0.50 else "no")
            ),
            "primary_metric": "crossover_status" if pf_vs_rbpf_summary_rows else "post_onset_mode_accuracy",
            "primary_metric_value": rbpf_frontier_metric if pf_vs_rbpf_summary_rows else rbpf_metrics.get("post_onset_mode_accuracy", ""),
            "runtime_seconds": rbpf_metrics.get("runtime_seconds", ""),
            "promotion_decision": rbpf_metrics.get("promotion_decision", "defer"),
            "artifact_path": (
                "artifacts/advanced_filter_comparison_v1/pf_vs_rbpf_frontier_summary.csv"
                if pf_vs_rbpf_summary_rows
                else "artifacts/rbpf_v1/rbpf_method_comparison.csv"
            ),
        },
        {
            "method_id": "ornstein_uhlenbeck_pf_v1",
            "corpus_objective_id": "ou_mean_reversion_v1",
            "scenario_family": "ornstein_uhlenbeck_mean_reversion_1d",
            "failure_case": "mean_reverting_stochastic_dynamics",
            "baseline_failed": "yes",
            "method_improved": "yes" if _as_float(ou_metrics.get("final_mean_reverting_posterior")) >= 0.60 else "no",
            "primary_metric": "final_mean_reverting_posterior",
            "primary_metric_value": ou_metrics.get("final_mean_reverting_posterior", ""),
            "runtime_seconds": ou_metrics.get("runtime_seconds", ""),
            "promotion_decision": ou_metrics.get("promotion_decision", "defer"),
            "artifact_path": "artifacts/ornstein_uhlenbeck_witness_v1/ou_method_comparison.csv",
        },
    ]

    nonlinear_rows = [
        {
            "method_id": "particle_filter_bank_v1",
            "witness": pf_promotion_metrics.get("study_id", pf_metrics.get("witness", "nonlinear_drag_outlier_1d")),
            "gaussian_collapse_family": "abs_range_multimodal_noninjective" if pf_oracle_metrics else "",
            "mean_oracle_to_pf_kl": pf_promotion_metrics.get("mean_oracle_to_pf_kl", ""),
            "mean_oracle_to_gaussian_kl": pf_promotion_metrics.get("mean_oracle_to_gaussian_kl", ""),
            "mean_pf_positive_mass_error": pf_promotion_metrics.get("mean_pf_positive_mass_error", ""),
            "mean_gaussian_positive_mass_error": pf_promotion_metrics.get("mean_gaussian_positive_mass_error", ""),
            "position_rmse": pf_metrics.get("position_rmse", ""),
            "kalman_baseline_rmse": pf_metrics.get("kalman_baseline_rmse", ""),
            "observation_baseline_rmse": pf_metrics.get("observation_baseline_rmse", ""),
            "rmse_delta_vs_kalman": pf_metrics.get("rmse_delta_vs_kalman", ""),
            "resampling_count": pf_metrics.get("resampling_count", ""),
            "mean_ess_fraction": pf_promotion_metrics.get("mean_ess_fraction", pf_metrics.get("mean_ess_fraction", "")),
            "gsf_reference_mean_oracle_to_gsf_kl": gsf_oracle_metrics.get("mean_oracle_to_gsf_kl", ""),
            "gsf_reference_mean_positive_mass_error": gsf_oracle_metrics.get("mean_gsf_positive_mass_error", ""),
            "gsf_vs_pf_crossover_status": gsf_vs_pf_status,
            "promotion_decision": (
                "defer_to_gsf" if gsf_vs_pf_status == "gsf_preferred" else pf_promotion_metrics.get("promotion_decision", pf_metrics.get("promotion_decision", "defer"))
            ),
        }
    ]
    latent_rows = [
        {
            "method_id": rbpf_metrics.get("method_id", "rbpf_v1"),
            "witness": rbpf_metrics.get("witness", "latent_maneuver_onset_1d"),
            "state_position_rmse": rbpf_metrics.get("state_position_rmse", ""),
            "post_onset_mode_accuracy": rbpf_metrics.get("post_onset_mode_accuracy", ""),
            "resampling_count": rbpf_metrics.get("resampling_count", ""),
            "frontier_crossover_status": rbpf_frontier_metric,
            "promotion_decision": rbpf_metrics.get("promotion_decision", "defer"),
        }
    ]
    mean_reverting_rows = [
        {
            "method_id": ou_metrics.get("method_id", "ornstein_uhlenbeck_pf_v1"),
            "witness": ou_metrics.get("witness", "ornstein_uhlenbeck_mean_reversion_1d"),
            "final_mean_reverting_posterior": ou_metrics.get("final_mean_reverting_posterior", ""),
            "position_rmse": ou_metrics.get("position_rmse", ""),
            "promotion_decision": ou_metrics.get("promotion_decision", "defer"),
        }
    ]
    runtime_rows = [
        {"method_id": row["method_id"], "runtime_seconds": row["runtime_seconds"], "scenario_family": row["scenario_family"]}
        for row in method_rows
    ]
    if method_rows:
        for row in method_rows:
            if str(row["method_id"]) == "particle_filter_bank_v1":
                row["robustness_sweep_passes"] = particle_filter_robustness.get("robustness_sweep_passes", "no")
                if str(row.get("robustness_sweep_passes", "no")) == "yes" and gsf_vs_pf_status != "gsf_preferred":
                    row["promotion_decision"] = "promote"
    rbpf_robustness_rows = _rbpf_robustness_summary_rows(pf_vs_rbpf_rows, pf_vs_rbpf_summary_rows)
    rbpf_robustness = rbpf_robustness_rows[0] if rbpf_robustness_rows else {}
    if method_rows:
        for row in method_rows:
            if str(row["method_id"]) == "rbpf_v1":
                row["robustness_sweep_passes"] = rbpf_robustness.get("robustness_sweep_passes", "no")
                if str(row.get("robustness_sweep_passes", "no")) == "yes":
                    row["promotion_decision"] = "promote"
    decision_rows = [
        {
            "method_id": row["method_id"],
            "failure_case": row["failure_case"],
            "baseline_failed": row["baseline_failed"],
            "method_improved": row["method_improved"],
            "cost_acceptable": row.get("cost_acceptable", "yes"),
            "promotion_decision": row["promotion_decision"],
            "required_by_current_evidence": "no",
            "promotion_scope": "witness_specific",
            "supporting_artifact": row["artifact_path"],
        }
        for row in method_rows
    ]
    gate_rows = _build_gate_rows(method_rows, output_root=output_root)

    method_path = run_dir / "method_comparison.csv"
    gate_path = run_dir / "advanced_method_gate_matrix.csv"
    gate_json_path = run_dir / "advanced_method_gate_matrix.json"
    nonlinear_path = run_dir / "nonlinear_stress_metrics.csv"
    particle_filter_robustness_path = run_dir / "particle_filter_robustness_summary.csv"
    gsf_robustness_path = run_dir / "gsf_robustness_summary.csv"
    gsf_vs_pf_path = run_dir / "gsf_vs_pf_frontier.csv"
    gsf_vs_pf_summary_path = run_dir / "gsf_vs_pf_frontier_summary.csv"
    latent_path = run_dir / "latent_maneuver_metrics.csv"
    rbpf_robustness_path = run_dir / "rbpf_robustness_summary.csv"
    mean_reverting_path = run_dir / "mean_reverting_metrics.csv"
    runtime_path = run_dir / "runtime_cost_metrics.csv"
    particle_count_path = run_dir / "particle_count_pareto.csv"
    pf_vs_rbpf_path = run_dir / "pf_vs_rbpf_frontier.csv"
    pf_vs_rbpf_summary_path = run_dir / "pf_vs_rbpf_frontier_summary.csv"
    promotion_cards_path = run_dir / "advanced_method_promotion_cards.md"
    decision_path = run_dir / "advanced_filter_decision_matrix.csv"
    report_path = run_dir / "advanced_filter_comparison_report.md"
    particle_count_plot_path = run_dir / "particle_count_pareto.png"
    gsf_vs_pf_plot_path = run_dir / "gsf_vs_pf_frontier.png"
    pf_vs_rbpf_plot_path = run_dir / "pf_vs_rbpf_frontier.png"
    write_csv(method_path, method_rows, list(method_rows[0]))
    write_csv(gate_path, gate_rows, list(gate_rows[0]))
    gate_json_path.write_text(json.dumps(gate_rows, indent=2), encoding="utf-8")
    write_csv(nonlinear_path, nonlinear_rows, list(nonlinear_rows[0]))
    write_csv(
        particle_filter_robustness_path,
        list(particle_filter_robustness_rows),
        list(particle_filter_robustness_rows[0]),
    )
    write_csv(
        gsf_robustness_path,
        list(gsf_robustness_rows),
        list(gsf_robustness_rows[0]),
    )
    write_csv(gsf_vs_pf_path, list(gsf_vs_pf_rows), list(gsf_vs_pf_rows[0]))
    write_csv(gsf_vs_pf_summary_path, list(gsf_vs_pf_summary_rows), list(gsf_vs_pf_summary_rows[0]))
    write_csv(latent_path, latent_rows, list(latent_rows[0]))
    write_csv(
        rbpf_robustness_path,
        list(rbpf_robustness_rows),
        list(rbpf_robustness_rows[0]),
    )
    write_csv(mean_reverting_path, mean_reverting_rows, list(mean_reverting_rows[0]))
    write_csv(runtime_path, runtime_rows, list(runtime_rows[0]))
    write_csv(particle_count_path, list(particle_count_rows), list(particle_count_rows[0]))
    write_csv(pf_vs_rbpf_path, list(pf_vs_rbpf_rows), list(pf_vs_rbpf_rows[0]))
    write_csv(pf_vs_rbpf_summary_path, list(pf_vs_rbpf_summary_rows), list(pf_vs_rbpf_summary_rows[0]))
    write_csv(decision_path, decision_rows, list(decision_rows[0]))
    promotion_cards_path.write_text(
        _render_advanced_method_promotion_cards(method_rows=method_rows, particle_count_rows=particle_count_rows),
        encoding="utf-8",
    )
    report_path.write_text(_render_advanced_filter_comparison_report(method_rows, decision_rows, gate_rows), encoding="utf-8")
    _write_particle_count_pareto_plot(particle_count_plot_path, particle_count_rows)
    _write_gsf_vs_pf_frontier_plot(gsf_vs_pf_plot_path, gsf_vs_pf_rows)
    _write_pf_vs_rbpf_frontier_plot(pf_vs_rbpf_plot_path, pf_vs_rbpf_rows)

    return AdvancedFilterComparisonArtifacts(
        run_dir=run_dir,
        method_comparison_path=method_path,
        gate_matrix_path=gate_path,
        gate_matrix_json_path=gate_json_path,
        nonlinear_stress_metrics_path=nonlinear_path,
        particle_filter_robustness_summary_path=particle_filter_robustness_path,
        gsf_robustness_summary_path=gsf_robustness_path,
        gsf_vs_pf_frontier_path=gsf_vs_pf_path,
        gsf_vs_pf_frontier_summary_path=gsf_vs_pf_summary_path,
        latent_maneuver_metrics_path=latent_path,
        rbpf_robustness_summary_path=rbpf_robustness_path,
        mean_reverting_metrics_path=mean_reverting_path,
        runtime_cost_metrics_path=runtime_path,
        particle_count_pareto_path=particle_count_path,
        pf_vs_rbpf_frontier_path=pf_vs_rbpf_path,
        pf_vs_rbpf_frontier_summary_path=pf_vs_rbpf_summary_path,
        advanced_method_promotion_cards_path=promotion_cards_path,
        decision_matrix_path=decision_path,
        report_path=report_path,
        particle_count_plot_path=particle_count_plot_path,
        gsf_vs_pf_frontier_plot_path=gsf_vs_pf_plot_path,
        pf_vs_rbpf_frontier_plot_path=pf_vs_rbpf_plot_path,
    )


def advanced_filter_comparison_surface() -> AdvancedFilterSurface[None, AdvancedFilterComparisonArtifacts]:
    return AdvancedFilterSurface(
        study_id="advanced_filter_comparison_v1",
        run=lambda: None,
        write_artifacts=write_advanced_filter_comparison_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.method_comparison_path),
            str(artifacts.gate_matrix_path),
            str(artifacts.decision_matrix_path),
            str(artifacts.report_path),
        ),
        metadata={
            "study_kind": "comparison",
            "problem_family": "advanced_filters_1d",
        },
    )



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


def _build_gate_rows(method_rows: list[dict[str, object]], *, output_root: Path) -> list[dict[str, object]]:
    research_reference_by_method = {
        "imm_v1": "Blom and Bar-Shalom 1988; Bar-Shalom Li Kirubarajan 2001",
        "particle_filter_bank_v1": "Gordon Salmond Smith 1993; Arulampalam Maskell Gordon Clapp 2002",
        "rbpf_v1": "Doucet de Freitas Murphy Russell 2000",
        "ornstein_uhlenbeck_pf_v1": "Gordon Salmond Smith 1993; Arulampalam Maskell Gordon Clapp 2002",
    }
    contract_by_method = {
        "imm_v1": "AdvancedFilterStep + posterior_history.csv + state_estimate_history.csv",
        "particle_filter_bank_v1": "AdvancedFilterStep-compatible posterior_history.csv + state_estimate_history.csv",
        "rbpf_v1": "AdvancedFilterStep-compatible latent_mode_posterior.csv + conditional_filter_history.csv",
        "ornstein_uhlenbeck_pf_v1": "AdvancedFilterStep-compatible posterior_history.csv + state_estimate_history.csv",
    }
    trace_packet_by_method = {
        "imm_v1": output_root / "imm_filter_v1" / "traces" / "filter_step_trace.csv",
        "particle_filter_bank_v1": output_root / "particle_filter_v1" / "traces" / "filter_step_trace.csv",
        "rbpf_v1": output_root / "rbpf_v1" / "traces" / "filter_step_trace.csv",
        "ornstein_uhlenbeck_pf_v1": output_root / "ornstein_uhlenbeck_witness_v1" / "traces" / "filter_step_trace.csv",
    }
    rows: list[dict[str, object]] = []
    for row in method_rows:
        method_id = str(row["method_id"])
        method_improved = str(row["method_improved"]) == "yes"
        promoted = str(row["promotion_decision"]) == "promote"
        trace_packet_exists = trace_packet_by_method.get(method_id, Path()).exists()
        robustness_passes = str(row.get("robustness_sweep_passes", "not_yet")) == "yes"
        witness_supported = method_improved and trace_packet_exists
        status_level = "implemented"
        if witness_supported:
            status_level = "witness_supported"
        if witness_supported and promoted and robustness_passes:
            status_level = "justified_for_study"
        rows.append(
            {
                "method_id": method_id,
                "status_level": status_level,
                "implemented": "yes",
                "reference_note_complete": "yes",
                "contract_hooked": "yes",
                "intermediate_trace_packet": "yes" if trace_packet_exists else "not_yet",
                "witness_exists": "yes",
                "simpler_rung_fails": str(row["baseline_failed"]),
                "advanced_method_improves": "yes" if method_improved else "no",
                "robustness_sweep_passes": "yes" if robustness_passes else "not_yet",
                "complexity_accounted": "yes" if row.get("runtime_seconds", "") != "" else "not_yet",
                "decision_card_status": str(row["promotion_decision"]),
                "generalized": "no",
                "scenario_family": str(row["scenario_family"]),
                "supporting_artifact": str(row["artifact_path"]),
                "research_reference": research_reference_by_method.get(method_id, ""),
                "shared_contract": contract_by_method.get(method_id, "posterior-compatible evidence rows"),
                "claim_boundary": "witness-specific promotion; not a universal default",
            }
        )
    return rows


def _render_advanced_filter_comparison_report(
    method_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    gate_rows: list[dict[str, object]],
) -> str:
    report = MarkdownDocument("Advanced Filter Comparison V1")
    report.paragraph(
        "This artifact treats IMM, PF, and RBPF as advanced evidence providers "
        "inside the shared posterior/evaluation contract."
    )
    report.paragraph(
        "Current claim boundary: sequential, transition-aware, and dynamics-aware methods are justified beyond simplistic classifiers; PF and RBPF remain witness-specific and are not yet required by current repo evidence."
    )
    report.heading("Status Semantics", level=2)
    report.table(
        ["Status", "Meaning"],
        [
            ("implemented", "Code exists and emits the shared advanced-filter artifact surface."),
            ("witness_supported", "A named controlled witness exists and the method improves on it with a trace packet."),
            ("justified_for_study", "The witness improvement also survives robustness sweeps for that named scenario family."),
            ("generalized", "Reserved for broad evidence; all current advanced-filter rows remain no."),
        ],
    )
    report.heading("Gate Matrix", level=2)
    report.table(
        ["Method", "Status", "Trace Packet", "Improves", "Robustness", "Generalized"],
        [
            (
                f"`{row['method_id']}`",
                f"`{row['status_level']}`",
                f"`{row['intermediate_trace_packet']}`",
                f"`{row['advanced_method_improves']}`",
                f"`{row['robustness_sweep_passes']}`",
                f"`{row['generalized']}`",
            )
            for row in gate_rows
        ],
    )
    report.heading("Decision Matrix", level=2)
    report.table(
        ["Method", "Failure Case", "Improved", "Decision", "Required Now", "Artifact"],
        [
            (
                f"`{row['method_id']}`",
                f"`{row['failure_case']}`",
                f"`{row['method_improved']}`",
                f"`{row['promotion_decision']}`",
                f"`{row['required_by_current_evidence']}`",
                f"`{row['supporting_artifact']}`",
            )
            for row in decision_rows
        ],
    )
    report.heading("Method Metrics", level=2)
    report.table(
        ["Method", "Objective", "Primary Metric", "Value", "Runtime Seconds"],
        [
            (
                f"`{row['method_id']}`",
                f"`{row['corpus_objective_id']}`",
                f"`{row['primary_metric']}`",
                f"`{row['primary_metric_value']}`",
                f"`{row['runtime_seconds']}`",
            )
            for row in method_rows
        ],
    )
    report.heading("PF vs RBPF Trade", level=2)
    report.bullet_list(
        [
            "PF is only complexity-justified after cheaper multimodal baselines such as GSF are checked on the same witness.",
            "The GSF vs PF frontier summary marks the current multimodal crossover as `gsf_preferred`, `pf_preferred`, or `metric_split`.",
            "PF vs RBPF should be compared on shared witnesses, not on unrelated witnesses.",
            "Use at least one witness where latent-structure marginalization should help and one where it should not.",
            "Use post-onset accuracy, position RMSE, ESS, and runtime together; no single metric is enough.",
            "If RBPF reaches similar quality with fewer particles or lower runtime on the shared witness, that is the actual promotion signal.",
            "The frontier summary CSV marks each witness as `rbpf_preferred`, `pf_preferred`, or `metric_split`.",
        ]
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "IMM is judged on switching-mode recovery, especially post-switch accuracy.",
            "PF is judged on nonlinear or non-Gaussian state estimation stress, and it must still clear the cheaper GSF rung on the same multimodal witness.",
            "RBPF is judged on latent-event mode recovery while conditionally filtering the continuous PVA state.",
            "A method is promoted only when the failure case it was designed for is visible and improved.",
            "None of the PF/RBPF rows currently imply that the repo as a whole requires PF/RBPF by default.",
        ]
    )
    return report.text()


def _plot_series(path: Path, times: ndarray, series: list[tuple[ndarray, str]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for values, label in series:
        ax.plot(times, values, label=label)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
