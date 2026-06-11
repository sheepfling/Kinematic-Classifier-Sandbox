from __future__ import annotations

import math
import time as wall_time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy.random as random
from numpy import arange, array, cumsum, float64, mean, sqrt

from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import write_comparison_summary_csv
from kinematic_classifier_sandbox.markdown_builder import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import plt

from .surface import AdvancedFilterSurface


def _normal_pdf(x: float, mean_value: float, variance: float) -> float:
    variance = max(float(variance), 1.0e-12)
    return math.exp(-0.5 * ((float(x) - float(mean_value)) ** 2) / variance) / math.sqrt(2.0 * math.pi * variance)


@dataclass(frozen=True, slots=True)
class OracleGridConfig:
    grid_min: float = -3.0
    grid_max: float = 3.0
    grid_step: float = 0.02


@dataclass(frozen=True, slots=True)
class LinearGaussianTruthRow:
    time: float
    truth_position: float


@dataclass(frozen=True, slots=True)
class LinearGaussianMeasurementRow:
    time: float
    measurement: float


@dataclass(frozen=True, slots=True)
class OraclePosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class MethodPosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class OracleStateEstimateRow:
    time: float
    truth_position: float
    measurement: float
    oracle_mean: float
    oracle_variance: float
    kalman_mean: float
    kalman_variance: float
    posterior_kl_oracle_to_kalman: float
    oracle_entropy: float
    oracle_contains_truth_95: bool
    kalman_contains_truth_95: bool


@dataclass(frozen=True, slots=True)
class LinearGaussianNegativeControlResult:
    truth_rows: tuple[LinearGaussianTruthRow, ...]
    measurement_rows: tuple[LinearGaussianMeasurementRow, ...]
    oracle_posterior_rows: tuple[OraclePosteriorRow, ...]
    method_posterior_rows: tuple[MethodPosteriorRow, ...]
    state_rows: tuple[OracleStateEstimateRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class LinearGaussianNegativeControlArtifacts:
    run_dir: Path
    truth_path: Path
    measurement_path: Path
    grid_oracle_posterior_path: Path
    method_posterior_path: Path
    state_estimate_history_path: Path
    summary_path: Path
    metrics_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def analyze_linear_gaussian_negative_control(
    *,
    seed: int = 101,
    grid: OracleGridConfig = OracleGridConfig(),
) -> LinearGaussianNegativeControlResult:
    rng = random.default_rng(seed)
    times = tuple(float(time_value) for time_value in arange(0.0, 6.0, 0.25, dtype=float64))
    process_std = 0.09
    measurement_std = 0.14
    prior_mean = 0.0
    prior_variance = 0.50**2
    grid_points = array(
        [grid.grid_min + grid.grid_step * index for index in range(int(round((grid.grid_max - grid.grid_min) / grid.grid_step)) + 1)],
        dtype=float64,
    )

    truth_values = [0.35]
    for _ in times[1:]:
        truth_values.append(float(truth_values[-1] + rng.normal(0.0, process_std)))
    measurement_values = [float(truth + rng.normal(0.0, measurement_std)) for truth in truth_values]

    truth_rows = tuple(
        LinearGaussianTruthRow(time=time_value, truth_position=truth_value)
        for time_value, truth_value in zip(times, truth_values, strict=True)
    )
    measurement_rows = tuple(
        LinearGaussianMeasurementRow(time=time_value, measurement=measurement_value)
        for time_value, measurement_value in zip(times, measurement_values, strict=True)
    )

    oracle_prior = array([_normal_pdf(position, prior_mean, prior_variance) for position in grid_points], dtype=float64)
    oracle_prior /= oracle_prior.sum()
    transition_matrix = array(
        [[_normal_pdf(next_position, previous_position, process_std**2) for previous_position in grid_points] for next_position in grid_points],
        dtype=float64,
    )
    transition_matrix /= transition_matrix.sum(axis=0, keepdims=True)

    kalman_mean = prior_mean
    kalman_variance = prior_variance
    oracle_posterior = oracle_prior
    oracle_posterior_rows: list[OraclePosteriorRow] = []
    method_posterior_rows: list[MethodPosteriorRow] = []
    state_rows: list[OracleStateEstimateRow] = []
    oracle_nll_values: list[float] = []
    kalman_nll_values: list[float] = []

    start = wall_time.perf_counter()
    for time_value, truth_value, measurement_value in zip(times, truth_values, measurement_values, strict=True):
        predicted_oracle = transition_matrix @ oracle_posterior
        likelihood = array(
            [_normal_pdf(measurement_value, position, measurement_std**2) for position in grid_points],
            dtype=float64,
        )
        oracle_evidence = float((predicted_oracle * likelihood).sum())
        oracle_posterior = predicted_oracle * likelihood
        oracle_posterior /= max(float(oracle_posterior.sum()), 1.0e-15)
        oracle_nll_values.append(-math.log(max(oracle_evidence, 1.0e-300)))

        predicted_kalman_variance = kalman_variance + process_std**2
        predicted_measurement_variance = predicted_kalman_variance + measurement_std**2
        kalman_gain = predicted_kalman_variance / predicted_measurement_variance
        innovation = measurement_value - kalman_mean
        kalman_mean = kalman_mean + kalman_gain * innovation
        kalman_variance = (1.0 - kalman_gain) * predicted_kalman_variance
        kalman_nll_values.append(-math.log(max(_normal_pdf(measurement_value, kalman_mean, kalman_variance + measurement_std**2), 1.0e-300)))

        oracle_mean = float((grid_points * oracle_posterior).sum())
        oracle_variance = float((((grid_points - oracle_mean) ** 2) * oracle_posterior).sum())
        discrete_kalman = array(
            [_normal_pdf(position, kalman_mean, kalman_variance) for position in grid_points],
            dtype=float64,
        )
        discrete_kalman /= max(float(discrete_kalman.sum()), 1.0e-15)
        posterior_kl = float(
            sum(
                float(probability) * math.log(max(float(probability), 1.0e-300) / max(float(discrete_kalman[index]), 1.0e-300))
                for index, probability in enumerate(oracle_posterior)
            )
        )
        oracle_entropy = float(
            -sum(float(probability) * math.log(max(float(probability), 1.0e-300)) for probability in oracle_posterior)
        )
        oracle_contains_truth = _grid_contains_truth_95(grid_points, oracle_posterior, truth_value)
        kalman_contains_truth = abs(truth_value - kalman_mean) <= 1.96 * math.sqrt(max(kalman_variance, 1.0e-12))
        state_rows.append(
            OracleStateEstimateRow(
                time=time_value,
                truth_position=truth_value,
                measurement=measurement_value,
                oracle_mean=oracle_mean,
                oracle_variance=oracle_variance,
                kalman_mean=kalman_mean,
                kalman_variance=kalman_variance,
                posterior_kl_oracle_to_kalman=posterior_kl,
                oracle_entropy=oracle_entropy,
                oracle_contains_truth_95=oracle_contains_truth,
                kalman_contains_truth_95=kalman_contains_truth,
            )
        )
        for position, probability, kalman_probability in zip(grid_points, oracle_posterior, discrete_kalman, strict=True):
            oracle_posterior_rows.append(
                OraclePosteriorRow(
                    time=time_value,
                    position=float(position),
                    posterior_probability=float(probability),
                )
            )
            method_posterior_rows.append(
                MethodPosteriorRow(
                    time=time_value,
                    position=float(position),
                    posterior_probability=float(kalman_probability),
                )
            )
    runtime_seconds = wall_time.perf_counter() - start

    oracle_rmse = float(sqrt(mean([(row.oracle_mean - row.truth_position) ** 2 for row in state_rows])))
    kalman_rmse = float(sqrt(mean([(row.kalman_mean - row.truth_position) ** 2 for row in state_rows])))
    observation_rmse = float(sqrt(mean([(measurement - truth) ** 2 for measurement, truth in zip(measurement_values, truth_values, strict=True)])))
    mean_kl = float(mean([row.posterior_kl_oracle_to_kalman for row in state_rows]))
    mean_entropy = float(mean([row.oracle_entropy for row in state_rows]))
    oracle_coverage = float(mean([1.0 if row.oracle_contains_truth_95 else 0.0 for row in state_rows]))
    kalman_coverage = float(mean([1.0 if row.kalman_contains_truth_95 else 0.0 for row in state_rows]))
    promotion_decision = (
        "do_not_escalate_beyond_kalman"
        if kalman_rmse <= observation_rmse and mean_kl <= 0.02 and abs(kalman_rmse - oracle_rmse) <= 0.02
        else "investigate_mismatch"
    )
    metrics = {
        "study_id": "linear_gaussian_negative_control_v1",
        "seed": seed,
        "step_count": len(times),
        "oracle_rmse": oracle_rmse,
        "kalman_rmse": kalman_rmse,
        "observation_rmse": observation_rmse,
        "mean_oracle_to_kalman_kl": mean_kl,
        "mean_oracle_entropy": mean_entropy,
        "oracle_mean_nll": float(mean(oracle_nll_values)),
        "kalman_mean_nll": float(mean(kalman_nll_values)),
        "oracle_95_coverage": oracle_coverage,
        "kalman_95_coverage": kalman_coverage,
        "runtime_seconds": runtime_seconds,
        "promotion_decision": promotion_decision,
    }
    return LinearGaussianNegativeControlResult(
        truth_rows=truth_rows,
        measurement_rows=measurement_rows,
        oracle_posterior_rows=tuple(oracle_posterior_rows),
        method_posterior_rows=tuple(method_posterior_rows),
        state_rows=tuple(state_rows),
        metrics=metrics,
    )


def write_linear_gaussian_negative_control_artifacts(
    output_dir: str | Path,
    *,
    result: LinearGaussianNegativeControlResult | None = None,
) -> LinearGaussianNegativeControlArtifacts:
    analysis = result or analyze_linear_gaussian_negative_control()
    run_dir = Path(output_dir) / "linear_gaussian_negative_control_v1"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    truth_path = run_dir / "truth_trajectory.csv"
    measurement_path = run_dir / "measurements.csv"
    oracle_posterior_path = run_dir / "grid_oracle_posterior_history.csv"
    method_posterior_path = run_dir / "method_posterior_history.csv"
    state_path = run_dir / "state_estimate_history.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics_against_oracle.csv"
    decision_card_path = run_dir / "decision_card.md"
    overlay_plot_path = plot_dir / "final_posterior_overlay.png"
    mean_plot_path = plot_dir / "posterior_mean_timeline.png"

    write_csv(truth_path, [asdict(row) for row in analysis.truth_rows], ["time", "truth_position"])
    write_csv(measurement_path, [asdict(row) for row in analysis.measurement_rows], ["time", "measurement"])
    write_csv(
        oracle_posterior_path,
        [asdict(row) for row in analysis.oracle_posterior_rows],
        ["time", "position", "posterior_probability"],
    )
    write_csv(
        method_posterior_path,
        [asdict(row) for row in analysis.method_posterior_rows],
        ["time", "position", "posterior_probability"],
    )
    write_csv(
        state_path,
        [asdict(row) for row in analysis.state_rows],
        [
            "time",
            "truth_position",
            "measurement",
            "oracle_mean",
            "oracle_variance",
            "kalman_mean",
            "kalman_variance",
            "posterior_kl_oracle_to_kalman",
            "oracle_entropy",
            "oracle_contains_truth_95",
            "kalman_contains_truth_95",
        ],
    )
    write_comparison_summary_csv(summary_path, [analysis.metrics])
    write_csv(metrics_path, [analysis.metrics], list(analysis.metrics))
    decision_card_path.write_text(_render_negative_control_decision_card(analysis), encoding="utf-8")
    _write_negative_control_plots(analysis, overlay_plot_path, mean_plot_path)
    return LinearGaussianNegativeControlArtifacts(
        run_dir=run_dir,
        truth_path=truth_path,
        measurement_path=measurement_path,
        grid_oracle_posterior_path=oracle_posterior_path,
        method_posterior_path=method_posterior_path,
        state_estimate_history_path=state_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        decision_card_path=decision_card_path,
        plot_paths=(overlay_plot_path, mean_plot_path),
    )


def linear_gaussian_negative_control_surface() -> AdvancedFilterSurface[LinearGaussianNegativeControlResult, LinearGaussianNegativeControlArtifacts]:
    return AdvancedFilterSurface(
        study_id="linear_gaussian_negative_control_v1",
        run=analyze_linear_gaussian_negative_control,
        write_artifacts=write_linear_gaussian_negative_control_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.metrics_path),
            str(artifacts.decision_card_path),
        ),
        metadata={
            "study_kind": "1d_oracle_negative_control",
            "problem_family": "linear_gaussian_oracle",
        },
    )


def _grid_contains_truth_95(grid_points: array, posterior: array, truth_value: float) -> bool:
    cumulative = cumsum(posterior)
    lower_index = next(index for index, probability in enumerate(cumulative) if float(probability) >= 0.025)
    upper_index = next(index for index, probability in enumerate(cumulative) if float(probability) >= 0.975)
    return float(grid_points[lower_index]) <= float(truth_value) <= float(grid_points[upper_index])


def _render_negative_control_decision_card(result: LinearGaussianNegativeControlResult) -> str:
    report = MarkdownDocument("Linear Gaussian Negative Control")
    report.paragraph(
        "This witness is designed to prove restraint. The posterior is linear-Gaussian, so a Kalman filter should match the grid oracle closely enough that escalation to PF or RBPF is not justified."
    )
    report.bullet_list(
        [
            f"Kalman RMSE: `{result.metrics['kalman_rmse']}`",
            f"Oracle RMSE: `{result.metrics['oracle_rmse']}`",
            f"Observation RMSE: `{result.metrics['observation_rmse']}`",
            f"Mean oracle->Kalman KL: `{result.metrics['mean_oracle_to_kalman_kl']}`",
            f"Oracle 95% coverage: `{result.metrics['oracle_95_coverage']}`",
            f"Kalman 95% coverage: `{result.metrics['kalman_95_coverage']}`",
            f"Decision: `{result.metrics['promotion_decision']}`",
        ]
    )
    report.paragraph(
        "Interpretation: if this card ever stops saying `do_not_escalate_beyond_kalman`, the oracle or baseline implementation needs investigation before any stronger method is promoted."
    )
    return report.text()


def _write_negative_control_plots(
    result: LinearGaussianNegativeControlResult,
    overlay_plot_path: Path,
    mean_plot_path: Path,
) -> None:
    final_time = result.state_rows[-1].time
    oracle_rows = [row for row in result.oracle_posterior_rows if row.time == final_time]
    method_rows = [row for row in result.method_posterior_rows if row.time == final_time]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(
        [row.position for row in oracle_rows],
        [row.posterior_probability for row in oracle_rows],
        label="grid oracle",
    )
    ax.plot(
        [row.position for row in method_rows],
        [row.posterior_probability for row in method_rows],
        label="Kalman Gaussian",
    )
    ax.set_title("Final posterior overlay")
    ax.set_xlabel("position")
    ax.set_ylabel("posterior probability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(overlay_plot_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([row.time for row in result.state_rows], [row.truth_position for row in result.state_rows], label="truth")
    ax.plot([row.time for row in result.state_rows], [row.measurement for row in result.state_rows], label="measurement", alpha=0.5)
    ax.plot([row.time for row in result.state_rows], [row.oracle_mean for row in result.state_rows], label="oracle mean")
    ax.plot([row.time for row in result.state_rows], [row.kalman_mean for row in result.state_rows], label="Kalman mean")
    ax.set_title("Oracle vs Kalman posterior means")
    ax.set_xlabel("time")
    ax.set_ylabel("position")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(mean_plot_path)
    plt.close(fig)
