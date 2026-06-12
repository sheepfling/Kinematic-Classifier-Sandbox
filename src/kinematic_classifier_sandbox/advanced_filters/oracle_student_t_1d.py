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


def _student_t_pdf(x: float, mean_value: float, scale: float, degrees_of_freedom: float) -> float:
    scale = max(float(scale), 1.0e-12)
    nu = max(float(degrees_of_freedom), 1.0e-6)
    z = (float(x) - float(mean_value)) / scale
    log_norm = (
        math.lgamma((nu + 1.0) * 0.5)
        - math.lgamma(nu * 0.5)
        - 0.5 * math.log(nu * math.pi)
        - math.log(scale)
    )
    log_kernel = -0.5 * (nu + 1.0) * math.log1p((z * z) / nu)
    return math.exp(log_norm + log_kernel)


@dataclass(frozen=True, slots=True)
class StudentTOracleGridConfig:
    grid_min: float = -3.0
    grid_max: float = 3.0
    grid_step: float = 0.02


@dataclass(frozen=True, slots=True)
class StudentTTruthRow:
    time: float
    truth_position: float


@dataclass(frozen=True, slots=True)
class StudentTMeasurementRow:
    time: float
    measurement: float


@dataclass(frozen=True, slots=True)
class StudentTOraclePosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class StudentTRobustPosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class StudentTGaussianPosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class StudentTStateEstimateRow:
    time: float
    truth_position: float
    measurement: float
    oracle_mean: float
    oracle_variance: float
    robust_mean: float
    robust_variance: float
    gaussian_mean: float
    gaussian_variance: float
    oracle_to_robust_kl: float
    oracle_to_gaussian_kl: float
    robust_contains_truth_95: bool
    gaussian_contains_truth_95: bool
    innovation_scale_factor: float


@dataclass(frozen=True, slots=True)
class StudentTOracleWitnessResult:
    truth_rows: tuple[StudentTTruthRow, ...]
    measurement_rows: tuple[StudentTMeasurementRow, ...]
    oracle_posterior_rows: tuple[StudentTOraclePosteriorRow, ...]
    robust_posterior_rows: tuple[StudentTRobustPosteriorRow, ...]
    gaussian_posterior_rows: tuple[StudentTGaussianPosteriorRow, ...]
    state_rows: tuple[StudentTStateEstimateRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class StudentTOracleWitnessArtifacts:
    run_dir: Path
    truth_path: Path
    measurement_path: Path
    grid_oracle_posterior_path: Path
    robust_posterior_path: Path
    gaussian_posterior_path: Path
    state_estimate_history_path: Path
    summary_path: Path
    metrics_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _discrete_gaussian_density(grid_points: list[float], mean_value: float, variance: float) -> list[float]:
    density = [_normal_pdf(position, mean_value, variance) for position in grid_points]
    normalizer = max(sum(density), 1.0e-15)
    return [float(value / normalizer) for value in density]


def analyze_student_t_heavy_tail_witness(
    *,
    seed: int = 409,
    grid: StudentTOracleGridConfig = StudentTOracleGridConfig(),
) -> StudentTOracleWitnessResult:
    rng = random.default_rng(seed)
    times = tuple(float(time_value) for time_value in arange(0.0, 6.0, 0.25, dtype=float64))
    process_std = 0.08
    measurement_scale = 0.10
    degrees_of_freedom = 3.0
    prior_mean = 0.2
    prior_variance = 0.45**2
    grid_points = [
        float(grid.grid_min + grid.grid_step * index)
        for index in range(int(round((grid.grid_max - grid.grid_min) / grid.grid_step)) + 1)
    ]

    truth_values = [0.3]
    for _ in times[1:]:
        truth_values.append(float(truth_values[-1] + rng.normal(0.0, process_std)))
    measurement_values = [
        float(truth + measurement_scale * rng.standard_t(df=degrees_of_freedom))
        for truth in truth_values
    ]

    truth_rows = tuple(
        StudentTTruthRow(time=time_value, truth_position=truth_value)
        for time_value, truth_value in zip(times, truth_values, strict=True)
    )
    measurement_rows = tuple(
        StudentTMeasurementRow(time=time_value, measurement=measurement_value)
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

    robust_mean = prior_mean
    robust_variance = prior_variance
    gaussian_mean = prior_mean
    gaussian_variance = prior_variance
    oracle_posterior_rows: list[StudentTOraclePosteriorRow] = []
    robust_posterior_rows: list[StudentTRobustPosteriorRow] = []
    gaussian_posterior_rows: list[StudentTGaussianPosteriorRow] = []
    state_rows: list[StudentTStateEstimateRow] = []
    start = wall_time.perf_counter()

    measurement_variance_proxy = measurement_scale * measurement_scale
    for time_value, truth_value, measurement_value in zip(times, truth_values, measurement_values, strict=True):
        predicted_oracle = [
            float(sum(transition_matrix[row_index][column_index] * oracle_posterior[column_index] for column_index in range(len(grid_points))))
            for row_index in range(len(grid_points))
        ]
        likelihood = [
            _student_t_pdf(measurement_value, position, measurement_scale, degrees_of_freedom)
            for position in grid_points
        ]
        updated_oracle = [float(predicted * like) for predicted, like in zip(predicted_oracle, likelihood, strict=True)]
        normalizer = max(sum(updated_oracle), 1.0e-15)
        oracle_posterior = [float(value / normalizer) for value in updated_oracle]

        predicted_robust_variance = robust_variance + process_std**2
        robust_innovation = float(measurement_value - robust_mean)
        robust_innovation_variance = float(predicted_robust_variance + measurement_variance_proxy)
        nis = (robust_innovation * robust_innovation) / max(robust_innovation_variance, 1.0e-12)
        scale_factor = float((degrees_of_freedom + 1.0) / (degrees_of_freedom + nis))
        effective_measurement_variance = measurement_variance_proxy / max(scale_factor, 1.0e-6)
        robust_gain = predicted_robust_variance / max(predicted_robust_variance + effective_measurement_variance, 1.0e-12)
        robust_mean = float(robust_mean + robust_gain * robust_innovation)
        robust_variance = max(float((1.0 - robust_gain) * predicted_robust_variance), 1.0e-9)

        predicted_gaussian_variance = gaussian_variance + process_std**2
        gaussian_gain = predicted_gaussian_variance / max(predicted_gaussian_variance + measurement_variance_proxy, 1.0e-12)
        gaussian_innovation = float(measurement_value - gaussian_mean)
        gaussian_mean = float(gaussian_mean + gaussian_gain * gaussian_innovation)
        gaussian_variance = max(float((1.0 - gaussian_gain) * predicted_gaussian_variance), 1.0e-9)

        oracle_mean = float(sum(position * probability for position, probability in zip(grid_points, oracle_posterior, strict=True)))
        oracle_variance = float(
            sum(((position - oracle_mean) ** 2) * probability for position, probability in zip(grid_points, oracle_posterior, strict=True))
        )
        robust_density = _discrete_gaussian_density(grid_points, robust_mean, robust_variance)
        gaussian_density = _discrete_gaussian_density(grid_points, gaussian_mean, gaussian_variance)
        oracle_to_robust_kl = float(
            sum(
                probability * math.log(max(probability, 1.0e-300) / max(robust_density[index], 1.0e-300))
                for index, probability in enumerate(oracle_posterior)
            )
        )
        oracle_to_gaussian_kl = float(
            sum(
                probability * math.log(max(probability, 1.0e-300) / max(gaussian_density[index], 1.0e-300))
                for index, probability in enumerate(oracle_posterior)
            )
        )
        state_rows.append(
            StudentTStateEstimateRow(
                time=time_value,
                truth_position=truth_value,
                measurement=measurement_value,
                oracle_mean=oracle_mean,
                oracle_variance=oracle_variance,
                robust_mean=robust_mean,
                robust_variance=robust_variance,
                gaussian_mean=gaussian_mean,
                gaussian_variance=gaussian_variance,
                oracle_to_robust_kl=oracle_to_robust_kl,
                oracle_to_gaussian_kl=oracle_to_gaussian_kl,
                robust_contains_truth_95=abs(truth_value - robust_mean) <= 1.96 * math.sqrt(max(robust_variance, 1.0e-12)),
                gaussian_contains_truth_95=abs(truth_value - gaussian_mean) <= 1.96 * math.sqrt(max(gaussian_variance, 1.0e-12)),
                innovation_scale_factor=scale_factor,
            )
        )
        for position, oracle_probability, robust_probability, gaussian_probability in zip(
            grid_points, oracle_posterior, robust_density, gaussian_density, strict=True
        ):
            oracle_posterior_rows.append(
                StudentTOraclePosteriorRow(time=time_value, position=float(position), posterior_probability=float(oracle_probability))
            )
            robust_posterior_rows.append(
                StudentTRobustPosteriorRow(time=time_value, position=float(position), posterior_probability=float(robust_probability))
            )
            gaussian_posterior_rows.append(
                StudentTGaussianPosteriorRow(time=time_value, position=float(position), posterior_probability=float(gaussian_probability))
            )
    runtime_seconds = wall_time.perf_counter() - start

    oracle_rmse = float(sqrt(mean([(row.oracle_mean - row.truth_position) ** 2 for row in state_rows])))
    robust_rmse = float(sqrt(mean([(row.robust_mean - row.truth_position) ** 2 for row in state_rows])))
    gaussian_rmse = float(sqrt(mean([(row.gaussian_mean - row.truth_position) ** 2 for row in state_rows])))
    mean_oracle_to_robust_kl = float(mean([row.oracle_to_robust_kl for row in state_rows]))
    mean_oracle_to_gaussian_kl = float(mean([row.oracle_to_gaussian_kl for row in state_rows]))
    robust_coverage = float(mean([1.0 if row.robust_contains_truth_95 else 0.0 for row in state_rows]))
    gaussian_coverage = float(mean([1.0 if row.gaussian_contains_truth_95 else 0.0 for row in state_rows]))
    mean_innovation_scale_factor = float(mean([row.innovation_scale_factor for row in state_rows]))
    promotion_decision = (
        "promote_student_t_for_heavy_tail_measurements"
        if mean_oracle_to_robust_kl < mean_oracle_to_gaussian_kl * 0.40
        and robust_rmse < gaussian_rmse * 0.85
        and robust_coverage >= gaussian_coverage
        else "revise_student_t_witness"
    )
    metrics = {
        "study_id": "student_t_heavy_tail_oracle_v1",
        "seed": seed,
        "step_count": len(times),
        "oracle_rmse": oracle_rmse,
        "robust_rmse": robust_rmse,
        "gaussian_rmse": gaussian_rmse,
        "mean_oracle_to_robust_kl": mean_oracle_to_robust_kl,
        "mean_oracle_to_gaussian_kl": mean_oracle_to_gaussian_kl,
        "robust_coverage_95": robust_coverage,
        "gaussian_coverage_95": gaussian_coverage,
        "mean_innovation_scale_factor": mean_innovation_scale_factor,
        "degrees_of_freedom": degrees_of_freedom,
        "runtime_seconds": runtime_seconds,
        "promotion_decision": promotion_decision,
    }
    return StudentTOracleWitnessResult(
        truth_rows=truth_rows,
        measurement_rows=measurement_rows,
        oracle_posterior_rows=tuple(oracle_posterior_rows),
        robust_posterior_rows=tuple(robust_posterior_rows),
        gaussian_posterior_rows=tuple(gaussian_posterior_rows),
        state_rows=tuple(state_rows),
        metrics=metrics,
    )


def write_student_t_heavy_tail_witness_artifacts(
    output_dir: str | Path,
    *,
    result: StudentTOracleWitnessResult | None = None,
) -> StudentTOracleWitnessArtifacts:
    analysis = result or analyze_student_t_heavy_tail_witness()
    run_dir = Path(output_dir) / "student_t_heavy_tail_oracle_v1"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    truth_path = run_dir / "truth_trajectory.csv"
    measurement_path = run_dir / "measurements.csv"
    oracle_posterior_path = run_dir / "grid_oracle_posterior_history.csv"
    robust_posterior_path = run_dir / "robust_posterior_history.csv"
    gaussian_posterior_path = run_dir / "gaussian_baseline_posterior_history.csv"
    state_path = run_dir / "state_estimate_history.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics_against_oracle.csv"
    decision_card_path = run_dir / "decision_card.md"
    final_overlay_plot_path = plot_dir / "final_posterior_overlay.png"
    kl_timeline_plot_path = plot_dir / "oracle_kl_timeline.png"
    scale_factor_plot_path = plot_dir / "innovation_scale_factor_timeline.png"

    write_csv(truth_path, [asdict(row) for row in analysis.truth_rows], ["time", "truth_position"])
    write_csv(measurement_path, [asdict(row) for row in analysis.measurement_rows], ["time", "measurement"])
    write_csv(oracle_posterior_path, [asdict(row) for row in analysis.oracle_posterior_rows], ["time", "position", "posterior_probability"])
    write_csv(robust_posterior_path, [asdict(row) for row in analysis.robust_posterior_rows], ["time", "position", "posterior_probability"])
    write_csv(gaussian_posterior_path, [asdict(row) for row in analysis.gaussian_posterior_rows], ["time", "position", "posterior_probability"])
    write_csv(
        state_path,
        [asdict(row) for row in analysis.state_rows],
        [
            "time",
            "truth_position",
            "measurement",
            "oracle_mean",
            "oracle_variance",
            "robust_mean",
            "robust_variance",
            "gaussian_mean",
            "gaussian_variance",
            "oracle_to_robust_kl",
            "oracle_to_gaussian_kl",
            "robust_contains_truth_95",
            "gaussian_contains_truth_95",
            "innovation_scale_factor",
        ],
    )
    write_comparison_summary_csv(summary_path, [analysis.metrics])
    write_csv(metrics_path, [analysis.metrics], list(analysis.metrics))
    decision_card_path.write_text(_render_student_t_decision_card(analysis), encoding="utf-8")
    _write_student_t_plots(analysis, final_overlay_plot_path, kl_timeline_plot_path, scale_factor_plot_path)
    return StudentTOracleWitnessArtifacts(
        run_dir=run_dir,
        truth_path=truth_path,
        measurement_path=measurement_path,
        grid_oracle_posterior_path=oracle_posterior_path,
        robust_posterior_path=robust_posterior_path,
        gaussian_posterior_path=gaussian_posterior_path,
        state_estimate_history_path=state_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        decision_card_path=decision_card_path,
        plot_paths=(final_overlay_plot_path, kl_timeline_plot_path, scale_factor_plot_path),
    )


def student_t_heavy_tail_witness_surface() -> AdvancedFilterSurface[StudentTOracleWitnessResult, StudentTOracleWitnessArtifacts]:
    return AdvancedFilterSurface(
        study_id="student_t_heavy_tail_oracle_v1",
        run=analyze_student_t_heavy_tail_witness,
        write_artifacts=write_student_t_heavy_tail_witness_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.metrics_path),
            str(artifacts.decision_card_path),
        ),
        metadata={
            "study_kind": "1d_oracle_positive_witness",
            "problem_family": "heavy_tail_outlier_tracking",
        },
    )


def _render_student_t_decision_card(result: StudentTOracleWitnessResult) -> str:
    report = MarkdownDocument("Student-t Heavy-Tail Oracle Witness")
    report.paragraph(
        "This witness asks whether heavy-tailed measurement noise can be handled by a robust Gaussian filter before escalating to particle methods."
    )
    report.bullet_list(
        [
            f"Robust mean oracle->robust KL: `{result.metrics['mean_oracle_to_robust_kl']}`",
            f"Gaussian mean oracle->Gaussian KL: `{result.metrics['mean_oracle_to_gaussian_kl']}`",
            f"Robust RMSE: `{result.metrics['robust_rmse']}`",
            f"Gaussian RMSE: `{result.metrics['gaussian_rmse']}`",
            f"Robust 95% coverage: `{result.metrics['robust_coverage_95']}`",
            f"Gaussian 95% coverage: `{result.metrics['gaussian_coverage_95']}`",
            f"Mean innovation scale factor: `{result.metrics['mean_innovation_scale_factor']}`",
            f"Decision: `{result.metrics['promotion_decision']}`",
        ]
    )
    report.paragraph(
        "Interpretation: if a Student-t style update restores posterior fidelity and coverage on a linear heavy-tail witness, PF should not be promoted from outliers alone."
    )
    return report.text()


def _write_student_t_plots(
    result: StudentTOracleWitnessResult,
    final_overlay_plot_path: Path,
    kl_timeline_plot_path: Path,
    scale_factor_plot_path: Path,
) -> None:
    final_time = result.state_rows[-1].time
    oracle_rows = [row for row in result.oracle_posterior_rows if row.time == final_time]
    robust_rows = [row for row in result.robust_posterior_rows if row.time == final_time]
    gaussian_rows = [row for row in result.gaussian_posterior_rows if row.time == final_time]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([row.position for row in oracle_rows], [row.posterior_probability for row in oracle_rows], label="grid oracle")
    ax.plot([row.position for row in robust_rows], [row.posterior_probability for row in robust_rows], label="Student-t robust")
    ax.plot([row.position for row in gaussian_rows], [row.posterior_probability for row in gaussian_rows], label="Gaussian Kalman")
    ax.set_title("Final posterior overlay")
    ax.set_xlabel("position")
    ax.set_ylabel("posterior probability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(final_overlay_plot_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([row.time for row in result.state_rows], [row.oracle_to_robust_kl for row in result.state_rows], label="oracle->robust KL")
    ax.plot([row.time for row in result.state_rows], [row.oracle_to_gaussian_kl for row in result.state_rows], label="oracle->Gaussian KL")
    ax.set_title("Oracle divergence timeline")
    ax.set_xlabel("time")
    ax.set_ylabel("KL divergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(kl_timeline_plot_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([row.time for row in result.state_rows], [row.innovation_scale_factor for row in result.state_rows], label="innovation scale factor")
    ax.set_title("Robust innovation scaling")
    ax.set_xlabel("time")
    ax.set_ylabel("scale factor")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(scale_factor_plot_path)
    plt.close(fig)
