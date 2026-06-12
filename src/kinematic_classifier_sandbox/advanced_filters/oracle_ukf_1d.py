from __future__ import annotations

import math
import time as wall_time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy.random as random
from numpy import arange, float64, mean, sqrt

from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import plt

from .surface import AdvancedFilterSurface


def _normal_pdf(x: float, mean_value: float, variance: float) -> float:
    variance = max(float(variance), 1.0e-12)
    return math.exp(-0.5 * ((float(x) - float(mean_value)) ** 2) / variance) / math.sqrt(2.0 * math.pi * variance)


def _measurement_fn(position: float, *, offset: float) -> float:
    return math.sqrt(float(position) * float(position) + offset * offset)


@dataclass(frozen=True, slots=True)
class UKFOracleGridConfig:
    grid_min: float = 0.0
    grid_max: float = 3.0
    grid_step: float = 0.02


@dataclass(frozen=True, slots=True)
class NonlinearTruthRow:
    time: float
    truth_position: float


@dataclass(frozen=True, slots=True)
class NonlinearMeasurementRow:
    time: float
    measurement: float


@dataclass(frozen=True, slots=True)
class UKFOraclePosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class UKFMethodPosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class UKFKalmanPosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class UKFStateEstimateRow:
    time: float
    truth_position: float
    measurement: float
    oracle_mean: float
    oracle_variance: float
    ukf_mean: float
    ukf_variance: float
    kalman_mean: float
    kalman_variance: float
    oracle_to_ukf_kl: float
    oracle_to_kalman_kl: float
    ukf_contains_truth_95: bool
    kalman_contains_truth_95: bool


@dataclass(frozen=True, slots=True)
class UKFOracleWitnessResult:
    truth_rows: tuple[NonlinearTruthRow, ...]
    measurement_rows: tuple[NonlinearMeasurementRow, ...]
    oracle_posterior_rows: tuple[UKFOraclePosteriorRow, ...]
    method_posterior_rows: tuple[UKFMethodPosteriorRow, ...]
    kalman_posterior_rows: tuple[UKFKalmanPosteriorRow, ...]
    state_rows: tuple[UKFStateEstimateRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class UKFOracleWitnessArtifacts:
    run_dir: Path
    truth_path: Path
    measurement_path: Path
    grid_oracle_posterior_path: Path
    method_posterior_path: Path
    kalman_baseline_posterior_path: Path
    state_estimate_history_path: Path
    summary_path: Path
    metrics_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _grid_contains_truth_95(grid_points: list[float], posterior: list[float], truth_value: float) -> bool:
    cumulative = 0.0
    lower = grid_points[0]
    upper = grid_points[-1]
    for position, probability in zip(grid_points, posterior, strict=True):
        cumulative += probability
        if cumulative >= 0.025:
            lower = position
            break
    cumulative = 0.0
    for position, probability in zip(reversed(grid_points), reversed(posterior), strict=True):
        cumulative += probability
        if cumulative >= 0.025:
            upper = position
            break
    return lower <= truth_value <= upper


def _discrete_gaussian_density(grid_points: list[float], mean_value: float, variance: float) -> list[float]:
    density = [_normal_pdf(position, mean_value, variance) for position in grid_points]
    normalizer = max(sum(density), 1.0e-15)
    return [float(value / normalizer) for value in density]


def _ukf_predict(mean_value: float, variance: float, process_variance: float) -> tuple[float, float]:
    return float(mean_value), float(variance + process_variance)


def _ukf_update(
    predicted_mean: float,
    predicted_variance: float,
    measurement: float,
    *,
    measurement_variance: float,
    offset: float,
    alpha: float = 1.0e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[float, float]:
    state_dim = 1
    lambda_value = alpha * alpha * (state_dim + kappa) - state_dim
    scaling = state_dim + lambda_value
    sigma_offset = math.sqrt(max(scaling * predicted_variance, 1.0e-12))
    sigma_points = (
        float(predicted_mean),
        float(predicted_mean + sigma_offset),
        float(max(predicted_mean - sigma_offset, 0.0)),
    )
    weight_mean_0 = lambda_value / scaling
    weight_cov_0 = weight_mean_0 + (1.0 - alpha * alpha + beta)
    weight_i = 1.0 / (2.0 * scaling)
    weights_mean = (weight_mean_0, weight_i, weight_i)
    weights_cov = (weight_cov_0, weight_i, weight_i)
    transformed = tuple(_measurement_fn(point, offset=offset) for point in sigma_points)
    predicted_measurement = float(sum(weight * value for weight, value in zip(weights_mean, transformed, strict=True)))
    innovation_variance = measurement_variance + float(
        sum(
            weight * (value - predicted_measurement) ** 2
            for weight, value in zip(weights_cov, transformed, strict=True)
        )
    )
    cross_covariance = float(
        sum(
            weight * (state_value - predicted_mean) * (measurement_value - predicted_measurement)
            for weight, state_value, measurement_value in zip(weights_cov, sigma_points, transformed, strict=True)
        )
    )
    kalman_gain = cross_covariance / max(innovation_variance, 1.0e-12)
    innovation = float(measurement - predicted_measurement)
    updated_mean = max(float(predicted_mean + kalman_gain * innovation), 0.0)
    updated_variance = max(float(predicted_variance - kalman_gain * innovation_variance * kalman_gain), 1.0e-9)
    return updated_mean, updated_variance


def analyze_ukf_nonlinear_unimodal_witness(
    *,
    seed: int = 307,
    grid: UKFOracleGridConfig = UKFOracleGridConfig(),
    measurement_offset: float = 0.75,
) -> UKFOracleWitnessResult:
    rng = random.default_rng(seed)
    times = tuple(float(time_value) for time_value in arange(0.0, 6.0, 0.25, dtype=float64))
    process_std = 0.06
    measurement_std = 0.08
    prior_mean = 1.05
    prior_variance = 0.22**2
    grid_points = [
        float(grid.grid_min + grid.grid_step * index)
        for index in range(int(round((grid.grid_max - grid.grid_min) / grid.grid_step)) + 1)
    ]

    truth_values = [1.10]
    for _ in times[1:]:
        truth_values.append(float(max(truth_values[-1] + rng.normal(0.0, process_std), 0.05)))
    measurement_values = [
        float(_measurement_fn(truth, offset=measurement_offset) + rng.normal(0.0, measurement_std))
        for truth in truth_values
    ]

    truth_rows = tuple(
        NonlinearTruthRow(time=time_value, truth_position=truth_value)
        for time_value, truth_value in zip(times, truth_values, strict=True)
    )
    measurement_rows = tuple(
        NonlinearMeasurementRow(time=time_value, measurement=measurement_value)
        for time_value, measurement_value in zip(times, measurement_values, strict=True)
    )

    oracle_prior = [_normal_pdf(position, prior_mean, prior_variance) for position in grid_points]
    oracle_prior_sum = max(sum(oracle_prior), 1.0e-15)
    oracle_posterior = [float(value / oracle_prior_sum) for value in oracle_prior]
    transition_matrix = [
        [_normal_pdf(next_position, previous_position, process_std**2) for previous_position in grid_points]
        for next_position in grid_points
    ]
    for column_index in range(len(grid_points)):
        column_sum = max(sum(row[column_index] for row in transition_matrix), 1.0e-15)
        for row in transition_matrix:
            row[column_index] = float(row[column_index] / column_sum)

    ukf_mean = prior_mean
    ukf_variance = prior_variance
    kalman_mean = prior_mean
    kalman_variance = prior_variance
    oracle_posterior_rows: list[UKFOraclePosteriorRow] = []
    ukf_posterior_rows: list[UKFMethodPosteriorRow] = []
    kalman_posterior_rows: list[UKFKalmanPosteriorRow] = []
    state_rows: list[UKFStateEstimateRow] = []
    start = wall_time.perf_counter()

    for time_value, truth_value, measurement_value in zip(times, truth_values, measurement_values, strict=True):
        predicted_oracle = [
            float(sum(transition_matrix[row_index][column_index] * oracle_posterior[column_index] for column_index in range(len(grid_points))))
            for row_index in range(len(grid_points))
        ]
        likelihood = [
            _normal_pdf(measurement_value, _measurement_fn(position, offset=measurement_offset), measurement_std**2)
            for position in grid_points
        ]
        updated_oracle = [float(predicted * like) for predicted, like in zip(predicted_oracle, likelihood, strict=True)]
        normalizer = max(sum(updated_oracle), 1.0e-15)
        oracle_posterior = [float(value / normalizer) for value in updated_oracle]

        ukf_predicted_mean, ukf_predicted_variance = _ukf_predict(ukf_mean, ukf_variance, process_std**2)
        ukf_mean, ukf_variance = _ukf_update(
            ukf_predicted_mean,
            ukf_predicted_variance,
            measurement_value,
            measurement_variance=measurement_std**2,
            offset=measurement_offset,
        )

        predicted_kalman_variance = kalman_variance + process_std**2
        predicted_measurement_variance = predicted_kalman_variance + measurement_std**2
        kalman_gain = predicted_kalman_variance / predicted_measurement_variance
        innovation = measurement_value - kalman_mean
        kalman_mean = max(float(kalman_mean + kalman_gain * innovation), 0.0)
        kalman_variance = max(float((1.0 - kalman_gain) * predicted_kalman_variance), 1.0e-9)

        oracle_mean = float(sum(position * probability for position, probability in zip(grid_points, oracle_posterior, strict=True)))
        oracle_variance = float(
            sum(((position - oracle_mean) ** 2) * probability for position, probability in zip(grid_points, oracle_posterior, strict=True))
        )
        ukf_density = _discrete_gaussian_density(grid_points, ukf_mean, ukf_variance)
        kalman_density = _discrete_gaussian_density(grid_points, kalman_mean, kalman_variance)
        oracle_to_ukf_kl = float(
            sum(
                probability * math.log(max(probability, 1.0e-300) / max(ukf_density[index], 1.0e-300))
                for index, probability in enumerate(oracle_posterior)
            )
        )
        oracle_to_kalman_kl = float(
            sum(
                probability * math.log(max(probability, 1.0e-300) / max(kalman_density[index], 1.0e-300))
                for index, probability in enumerate(oracle_posterior)
            )
        )
        state_rows.append(
            UKFStateEstimateRow(
                time=time_value,
                truth_position=truth_value,
                measurement=measurement_value,
                oracle_mean=oracle_mean,
                oracle_variance=oracle_variance,
                ukf_mean=ukf_mean,
                ukf_variance=ukf_variance,
                kalman_mean=kalman_mean,
                kalman_variance=kalman_variance,
                oracle_to_ukf_kl=oracle_to_ukf_kl,
                oracle_to_kalman_kl=oracle_to_kalman_kl,
                ukf_contains_truth_95=abs(truth_value - ukf_mean) <= 1.96 * math.sqrt(max(ukf_variance, 1.0e-12)),
                kalman_contains_truth_95=abs(truth_value - kalman_mean) <= 1.96 * math.sqrt(max(kalman_variance, 1.0e-12)),
            )
        )
        for position, oracle_probability, ukf_probability, kalman_probability in zip(
            grid_points, oracle_posterior, ukf_density, kalman_density, strict=True
        ):
            oracle_posterior_rows.append(
                UKFOraclePosteriorRow(time=time_value, position=float(position), posterior_probability=float(oracle_probability))
            )
            ukf_posterior_rows.append(
                UKFMethodPosteriorRow(time=time_value, position=float(position), posterior_probability=float(ukf_probability))
            )
            kalman_posterior_rows.append(
                UKFKalmanPosteriorRow(time=time_value, position=float(position), posterior_probability=float(kalman_probability))
            )

    runtime_seconds = wall_time.perf_counter() - start
    oracle_rmse = float(sqrt(mean([(row.oracle_mean - row.truth_position) ** 2 for row in state_rows])))
    ukf_rmse = float(sqrt(mean([(row.ukf_mean - row.truth_position) ** 2 for row in state_rows])))
    kalman_rmse = float(sqrt(mean([(row.kalman_mean - row.truth_position) ** 2 for row in state_rows])))
    mean_oracle_to_ukf_kl = float(mean([row.oracle_to_ukf_kl for row in state_rows]))
    mean_oracle_to_kalman_kl = float(mean([row.oracle_to_kalman_kl for row in state_rows]))
    ukf_coverage = float(mean([1.0 if row.ukf_contains_truth_95 else 0.0 for row in state_rows]))
    kalman_coverage = float(mean([1.0 if row.kalman_contains_truth_95 else 0.0 for row in state_rows]))
    promotion_decision = (
        "promote_ukf_for_nonlinear_unimodal_measurement"
        if mean_oracle_to_ukf_kl < mean_oracle_to_kalman_kl * 0.35 and ukf_rmse < kalman_rmse * 0.80
        else "revise_ukf_witness"
    )
    metrics = {
        "study_id": "ukf_nonlinear_unimodal_oracle_v1",
        "seed": seed,
        "step_count": len(times),
        "oracle_rmse": oracle_rmse,
        "ukf_rmse": ukf_rmse,
        "kalman_rmse": kalman_rmse,
        "mean_oracle_to_ukf_kl": mean_oracle_to_ukf_kl,
        "mean_oracle_to_kalman_kl": mean_oracle_to_kalman_kl,
        "ukf_coverage_95": ukf_coverage,
        "kalman_coverage_95": kalman_coverage,
        "measurement_offset": measurement_offset,
        "runtime_seconds": runtime_seconds,
        "promotion_decision": promotion_decision,
    }
    return UKFOracleWitnessResult(
        truth_rows=truth_rows,
        measurement_rows=measurement_rows,
        oracle_posterior_rows=tuple(oracle_posterior_rows),
        method_posterior_rows=tuple(ukf_posterior_rows),
        kalman_posterior_rows=tuple(kalman_posterior_rows),
        state_rows=tuple(state_rows),
        metrics=metrics,
    )


def write_ukf_nonlinear_unimodal_witness_artifacts(
    output_dir: str | Path,
    *,
    result: UKFOracleWitnessResult | None = None,
) -> UKFOracleWitnessArtifacts:
    analysis = result or analyze_ukf_nonlinear_unimodal_witness()
    run_dir = Path(output_dir) / "ukf_nonlinear_unimodal_oracle_v1"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    truth_path = run_dir / "truth_trajectory.csv"
    measurement_path = run_dir / "measurements.csv"
    oracle_posterior_path = run_dir / "grid_oracle_posterior_history.csv"
    method_posterior_path = run_dir / "method_posterior_history.csv"
    kalman_posterior_path = run_dir / "kalman_baseline_posterior_history.csv"
    state_path = run_dir / "state_estimate_history.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics_against_oracle.csv"
    decision_card_path = run_dir / "decision_card.md"
    final_overlay_plot_path = plot_dir / "final_posterior_overlay.png"
    kl_timeline_plot_path = plot_dir / "oracle_kl_timeline.png"
    state_panel_plot_path = plot_dir / "state_mean_panel.png"

    write_csv(truth_path, [asdict(row) for row in analysis.truth_rows], ["time", "truth_position"])
    write_csv(measurement_path, [asdict(row) for row in analysis.measurement_rows], ["time", "measurement"])
    write_csv(oracle_posterior_path, [asdict(row) for row in analysis.oracle_posterior_rows], ["time", "position", "posterior_probability"])
    write_csv(method_posterior_path, [asdict(row) for row in analysis.method_posterior_rows], ["time", "position", "posterior_probability"])
    write_csv(kalman_posterior_path, [asdict(row) for row in analysis.kalman_posterior_rows], ["time", "position", "posterior_probability"])
    write_csv(
        state_path,
        [asdict(row) for row in analysis.state_rows],
        [
            "time",
            "truth_position",
            "measurement",
            "oracle_mean",
            "oracle_variance",
            "ukf_mean",
            "ukf_variance",
            "kalman_mean",
            "kalman_variance",
            "oracle_to_ukf_kl",
            "oracle_to_kalman_kl",
            "ukf_contains_truth_95",
            "kalman_contains_truth_95",
        ],
    )
    write_comparison_summary_csv(summary_path, [analysis.metrics])
    write_csv(metrics_path, [analysis.metrics], list(analysis.metrics))
    decision_card_path.write_text(_render_ukf_decision_card(analysis), encoding="utf-8")
    _write_ukf_plots(analysis, final_overlay_plot_path, kl_timeline_plot_path, state_panel_plot_path)
    return UKFOracleWitnessArtifacts(
        run_dir=run_dir,
        truth_path=truth_path,
        measurement_path=measurement_path,
        grid_oracle_posterior_path=oracle_posterior_path,
        method_posterior_path=method_posterior_path,
        kalman_baseline_posterior_path=kalman_posterior_path,
        state_estimate_history_path=state_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        decision_card_path=decision_card_path,
        plot_paths=(final_overlay_plot_path, kl_timeline_plot_path, state_panel_plot_path),
    )


def ukf_nonlinear_unimodal_witness_surface() -> AdvancedFilterSurface[UKFOracleWitnessResult, UKFOracleWitnessArtifacts]:
    return AdvancedFilterSurface(
        study_id="ukf_nonlinear_unimodal_oracle_v1",
        run=analyze_ukf_nonlinear_unimodal_witness,
        write_artifacts=write_ukf_nonlinear_unimodal_witness_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.metrics_path),
            str(artifacts.decision_card_path),
        ),
        metadata={
            "study_kind": "1d_oracle_positive_witness",
            "problem_family": "nonlinear_unimodal_sensor",
        },
    )


def _render_ukf_decision_card(result: UKFOracleWitnessResult) -> str:
    report = MarkdownDocument("UKF Nonlinear Unimodal Oracle Witness")
    report.paragraph(
        "This witness asks whether nonlinear but still unimodal measurement geometry can be handled by a Gaussian approximation before escalating to mixtures or particles."
    )
    report.bullet_list(
        [
            f"UKF mean oracle->UKF KL: `{result.metrics['mean_oracle_to_ukf_kl']}`",
            f"Kalman proxy mean oracle->Kalman KL: `{result.metrics['mean_oracle_to_kalman_kl']}`",
            f"UKF RMSE: `{result.metrics['ukf_rmse']}`",
            f"Kalman proxy RMSE: `{result.metrics['kalman_rmse']}`",
            f"UKF 95% coverage: `{result.metrics['ukf_coverage_95']}`",
            f"Kalman proxy 95% coverage: `{result.metrics['kalman_coverage_95']}`",
            f"Decision: `{result.metrics['promotion_decision']}`",
        ]
    )
    report.paragraph(
        "Interpretation: if UKF stays close to the oracle on a nonlinear but unimodal witness, PF should not be promoted from this failure mode alone."
    )
    return report.text()


def _write_ukf_plots(
    result: UKFOracleWitnessResult,
    final_overlay_plot_path: Path,
    kl_timeline_plot_path: Path,
    state_panel_plot_path: Path,
) -> None:
    final_time = result.state_rows[-1].time
    oracle_rows = [row for row in result.oracle_posterior_rows if row.time == final_time]
    ukf_rows = [row for row in result.method_posterior_rows if row.time == final_time]
    kalman_rows = [row for row in result.kalman_posterior_rows if row.time == final_time]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([row.position for row in oracle_rows], [row.posterior_probability for row in oracle_rows], label="grid oracle")
    ax.plot([row.position for row in ukf_rows], [row.posterior_probability for row in ukf_rows], label="UKF")
    ax.plot([row.position for row in kalman_rows], [row.posterior_probability for row in kalman_rows], label="Kalman proxy")
    ax.set_title("Final posterior overlay")
    ax.set_xlabel("position")
    ax.set_ylabel("posterior probability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(final_overlay_plot_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([row.time for row in result.state_rows], [row.oracle_to_ukf_kl for row in result.state_rows], label="oracle->UKF KL")
    ax.plot([row.time for row in result.state_rows], [row.oracle_to_kalman_kl for row in result.state_rows], label="oracle->Kalman KL")
    ax.set_title("Oracle divergence timeline")
    ax.set_xlabel("time")
    ax.set_ylabel("KL divergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(kl_timeline_plot_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([row.time for row in result.state_rows], [row.truth_position for row in result.state_rows], label="truth", color="black", linewidth=1.4)
    ax.plot([row.time for row in result.state_rows], [row.oracle_mean for row in result.state_rows], label="oracle mean")
    ax.plot([row.time for row in result.state_rows], [row.ukf_mean for row in result.state_rows], label="UKF mean")
    ax.plot([row.time for row in result.state_rows], [row.kalman_mean for row in result.state_rows], label="Kalman proxy mean")
    ax.set_title("State mean panel")
    ax.set_xlabel("time")
    ax.set_ylabel("position")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(state_panel_plot_path)
    plt.close(fig)
