from __future__ import annotations

import math
import time as wall_time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy.random as random
from numpy import arange, array, float64, mean, sqrt

from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import plt

from .oracle_pf_1d import OraclePFGridConfig, analyze_pf_abs_range_multimodal_witness
from .surface import AdvancedFilterSurface


def _normal_pdf(x: float, mean_value: float, variance: float) -> float:
    variance = max(float(variance), 1.0e-12)
    return math.exp(-0.5 * ((float(x) - float(mean_value)) ** 2) / variance) / math.sqrt(2.0 * math.pi * variance)


def _normal_cdf(x: float, mean_value: float, variance: float) -> float:
    std = math.sqrt(max(float(variance), 1.0e-12))
    z = (float(x) - float(mean_value)) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


@dataclass(frozen=True, slots=True)
class GSFComponent:
    weight: float
    mean: float
    variance: float


@dataclass(frozen=True, slots=True)
class GSFTruthRow:
    time: float
    truth_position: float


@dataclass(frozen=True, slots=True)
class GSFMeasurementRow:
    time: float
    measurement: float


@dataclass(frozen=True, slots=True)
class GSFOraclePosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class GSFMethodPosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class GSFGaussianPosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class GSFComponentRow:
    time: float
    component_index: int
    weight: float
    mean: float
    variance: float
    sign_label: str


@dataclass(frozen=True, slots=True)
class GSFStateEstimateRow:
    time: float
    truth_position: float
    measurement: float
    oracle_mean: float
    oracle_map: float
    oracle_positive_mass: float
    gsf_mean: float
    gsf_positive_mass: float
    gaussian_mean: float
    gaussian_positive_mass: float
    oracle_to_gsf_kl: float
    oracle_to_gaussian_kl: float
    active_component_count: int


@dataclass(frozen=True, slots=True)
class GSFOracleWitnessResult:
    truth_rows: tuple[GSFTruthRow, ...]
    measurement_rows: tuple[GSFMeasurementRow, ...]
    oracle_posterior_rows: tuple[GSFOraclePosteriorRow, ...]
    method_posterior_rows: tuple[GSFMethodPosteriorRow, ...]
    gaussian_posterior_rows: tuple[GSFGaussianPosteriorRow, ...]
    component_rows: tuple[GSFComponentRow, ...]
    state_rows: tuple[GSFStateEstimateRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class GSFOracleWitnessArtifacts:
    run_dir: Path
    truth_path: Path
    measurement_path: Path
    grid_oracle_posterior_path: Path
    method_posterior_path: Path
    gaussian_baseline_posterior_path: Path
    component_history_path: Path
    state_estimate_history_path: Path
    summary_path: Path
    metrics_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _normalize_components(components: list[GSFComponent]) -> list[GSFComponent]:
    total = sum(max(component.weight, 0.0) for component in components)
    if total <= 1.0e-15:
        return [GSFComponent(weight=1.0 / max(len(components), 1), mean=component.mean, variance=component.variance) for component in components]
    return [
        GSFComponent(weight=float(component.weight / total), mean=float(component.mean), variance=max(float(component.variance), 1.0e-9))
        for component in components
    ]


def _prune_components(components: list[GSFComponent], *, max_components: int) -> list[GSFComponent]:
    ranked = sorted(components, key=lambda component: component.weight, reverse=True)[:max_components]
    return _normalize_components(ranked)


def analyze_gsf_abs_range_multimodal_witness(
    *,
    seed: int = 211,
    max_components: int = 4,
    grid: OraclePFGridConfig = OraclePFGridConfig(),
) -> GSFOracleWitnessResult:
    rng = random.default_rng(seed)
    times = tuple(float(time_value) for time_value in arange(0.0, 5.0, 0.25, dtype=float64))
    truth_values = [1.1]
    for _ in times[1:]:
        truth_values.append(float(truth_values[-1] + rng.normal(0.0, 0.08)))
    measurement_values = [float(abs(truth) + rng.normal(0.0, 0.12)) for truth in truth_values]

    truth_rows = tuple(
        GSFTruthRow(time=time_value, truth_position=truth_value)
        for time_value, truth_value in zip(times, truth_values, strict=True)
    )
    measurement_rows = tuple(
        GSFMeasurementRow(time=time_value, measurement=measurement_value)
        for time_value, measurement_value in zip(times, measurement_values, strict=True)
    )

    grid_points = array(
        [grid.grid_min + grid.grid_step * index for index in range(int(round((grid.grid_max - grid.grid_min) / grid.grid_step)) + 1)],
        dtype=float64,
    )
    oracle_prior = array([_normal_pdf(position, 0.25, 0.9**2) for position in grid_points], dtype=float64)
    oracle_prior /= oracle_prior.sum()
    transition_matrix = array(
        [[_normal_pdf(next_position, previous_position, 0.08**2) for previous_position in grid_points] for next_position in grid_points],
        dtype=float64,
    )
    transition_matrix /= transition_matrix.sum(axis=0, keepdims=True)

    oracle_posterior = oracle_prior
    gaussian_mean = 0.25
    gaussian_variance = 0.9**2
    components = [GSFComponent(weight=1.0, mean=0.25, variance=0.9**2)]

    oracle_posterior_rows: list[GSFOraclePosteriorRow] = []
    method_posterior_rows: list[GSFMethodPosteriorRow] = []
    gaussian_posterior_rows: list[GSFGaussianPosteriorRow] = []
    component_rows: list[GSFComponentRow] = []
    state_rows: list[GSFStateEstimateRow] = []
    oracle_nll_values: list[float] = []
    start = wall_time.perf_counter()

    process_variance = 0.08**2
    measurement_variance = 0.12**2
    for time_value, truth_value, measurement_value in zip(times, truth_values, measurement_values, strict=True):
        predicted_oracle = transition_matrix @ oracle_posterior
        likelihood = array(
            [_normal_pdf(measurement_value, abs(position), measurement_variance) for position in grid_points],
            dtype=float64,
        )
        oracle_evidence = float((predicted_oracle * likelihood).sum())
        oracle_posterior = predicted_oracle * likelihood
        oracle_posterior /= max(float(oracle_posterior.sum()), 1.0e-15)
        oracle_nll_values.append(-math.log(max(oracle_evidence, 1.0e-300)))

        predicted_gaussian_variance = gaussian_variance + process_variance
        gaussian_predicted_density = array(
            [_normal_pdf(position, gaussian_mean, predicted_gaussian_variance) for position in grid_points],
            dtype=float64,
        )
        gaussian_predicted_density /= max(float(gaussian_predicted_density.sum()), 1.0e-15)
        gaussian_updated = gaussian_predicted_density * likelihood
        gaussian_updated /= max(float(gaussian_updated.sum()), 1.0e-15)
        gaussian_mean = float((grid_points * gaussian_updated).sum())
        gaussian_variance = float((((grid_points - gaussian_mean) ** 2) * gaussian_updated).sum())
        gaussian_density = array(
            [_normal_pdf(position, gaussian_mean, gaussian_variance) for position in grid_points],
            dtype=float64,
        )
        gaussian_density /= max(float(gaussian_density.sum()), 1.0e-15)

        branched_components: list[GSFComponent] = []
        for component in components:
            predicted_mean = float(component.mean)
            predicted_variance = float(component.variance + process_variance)
            sign_support = {
                1.0: max(1.0 - _normal_cdf(0.0, predicted_mean, predicted_variance), 1.0e-9),
                -1.0: max(_normal_cdf(0.0, predicted_mean, predicted_variance), 1.0e-9),
            }
            innovation_variance = predicted_variance + measurement_variance
            kalman_gain = predicted_variance / innovation_variance
            updated_variance = max((1.0 - kalman_gain) * predicted_variance, 1.0e-9)
            for sign in (1.0, -1.0):
                innovation = measurement_value - sign * predicted_mean
                branch_likelihood = _normal_pdf(measurement_value, sign * predicted_mean, innovation_variance)
                updated_mean = predicted_mean + sign * kalman_gain * innovation
                branched_components.append(
                    GSFComponent(
                        weight=float(component.weight * sign_support[sign] * branch_likelihood),
                        mean=float(updated_mean),
                        variance=float(updated_variance),
                    )
                )
        components = _prune_components(branched_components, max_components=max_components)

        gsf_density = array(
            [
                float(sum(component.weight * _normal_pdf(position, component.mean, component.variance) for component in components))
                for position in grid_points
            ],
            dtype=float64,
        )
        gsf_density /= max(float(gsf_density.sum()), 1.0e-15)

        oracle_mean = float((grid_points * oracle_posterior).sum())
        oracle_map = float(grid_points[int(oracle_posterior.argmax())])
        oracle_positive_mass = float(sum(float(probability) for position, probability in zip(grid_points, oracle_posterior, strict=True) if position >= 0.0))
        gsf_mean = float(sum(component.weight * component.mean for component in components))
        gsf_positive_mass = float(
            sum(component.weight * (1.0 - _normal_cdf(0.0, component.mean, component.variance)) for component in components)
        )
        gaussian_positive_mass = float(sum(float(probability) for position, probability in zip(grid_points, gaussian_density, strict=True) if position >= 0.0))
        oracle_to_gsf_kl = float(
            sum(
                float(probability) * math.log(max(float(probability), 1.0e-300) / max(float(gsf_density[index]), 1.0e-300))
                for index, probability in enumerate(oracle_posterior)
            )
        )
        oracle_to_gaussian_kl = float(
            sum(
                float(probability) * math.log(max(float(probability), 1.0e-300) / max(float(gaussian_density[index]), 1.0e-300))
                for index, probability in enumerate(oracle_posterior)
            )
        )
        state_rows.append(
            GSFStateEstimateRow(
                time=time_value,
                truth_position=truth_value,
                measurement=measurement_value,
                oracle_mean=oracle_mean,
                oracle_map=oracle_map,
                oracle_positive_mass=oracle_positive_mass,
                gsf_mean=gsf_mean,
                gsf_positive_mass=gsf_positive_mass,
                gaussian_mean=gaussian_mean,
                gaussian_positive_mass=gaussian_positive_mass,
                oracle_to_gsf_kl=oracle_to_gsf_kl,
                oracle_to_gaussian_kl=oracle_to_gaussian_kl,
                active_component_count=len(components),
            )
        )
        for component_index, component in enumerate(components):
            component_rows.append(
                GSFComponentRow(
                    time=time_value,
                    component_index=component_index,
                    weight=float(component.weight),
                    mean=float(component.mean),
                    variance=float(component.variance),
                    sign_label="positive" if component.mean >= 0.0 else "negative",
                )
            )
        for position, oracle_probability, gsf_probability, gaussian_probability in zip(
            grid_points,
            oracle_posterior,
            gsf_density,
            gaussian_density,
            strict=True,
        ):
            oracle_posterior_rows.append(
                GSFOraclePosteriorRow(
                    time=time_value,
                    position=float(position),
                    posterior_probability=float(oracle_probability),
                )
            )
            method_posterior_rows.append(
                GSFMethodPosteriorRow(
                    time=time_value,
                    position=float(position),
                    posterior_probability=float(gsf_probability),
                )
            )
            gaussian_posterior_rows.append(
                GSFGaussianPosteriorRow(
                    time=time_value,
                    position=float(position),
                    posterior_probability=float(gaussian_probability),
                )
            )
    runtime_seconds = wall_time.perf_counter() - start

    pf_reference = analyze_pf_abs_range_multimodal_witness(seed=seed, particle_count=512, grid=grid)
    mean_gsf_kl = float(mean([row.oracle_to_gsf_kl for row in state_rows]))
    mean_gaussian_kl = float(mean([row.oracle_to_gaussian_kl for row in state_rows]))
    gsf_rmse = float(sqrt(mean([(row.gsf_mean - row.truth_position) ** 2 for row in state_rows])))
    gaussian_rmse = float(sqrt(mean([(row.gaussian_mean - row.truth_position) ** 2 for row in state_rows])))
    oracle_rmse = float(sqrt(mean([(row.oracle_mean - row.truth_position) ** 2 for row in state_rows])))
    gsf_positive_mass_error = float(mean([abs(row.gsf_positive_mass - row.oracle_positive_mass) for row in state_rows]))
    gaussian_positive_mass_error = float(mean([abs(row.gaussian_positive_mass - row.oracle_positive_mass) for row in state_rows]))
    mean_component_count = float(mean([row.active_component_count for row in state_rows]))
    decision = (
        "gsf_witness_supported"
        if mean_gsf_kl < mean_gaussian_kl * 0.35 and gsf_positive_mass_error < gaussian_positive_mass_error * 0.65
        else "revise_gsf_witness"
    )
    metrics = {
        "study_id": "gsf_abs_range_multimodal_oracle_v1",
        "seed": seed,
        "max_components": max_components,
        "step_count": len(times),
        "oracle_rmse": oracle_rmse,
        "gsf_rmse": gsf_rmse,
        "gaussian_rmse": gaussian_rmse,
        "mean_oracle_to_gsf_kl": mean_gsf_kl,
        "mean_oracle_to_gaussian_kl": mean_gaussian_kl,
        "mean_gsf_positive_mass_error": gsf_positive_mass_error,
        "mean_gaussian_positive_mass_error": gaussian_positive_mass_error,
        "mean_component_count": mean_component_count,
        "runtime_seconds": runtime_seconds,
        "pf_reference_mean_oracle_to_pf_kl": float(pf_reference.metrics["mean_oracle_to_pf_kl"]),
        "pf_reference_mean_positive_mass_error": float(pf_reference.metrics["mean_pf_positive_mass_error"]),
        "gsf_to_pf_kl_ratio": float(mean_gsf_kl / max(float(pf_reference.metrics["mean_oracle_to_pf_kl"]), 1.0e-12)),
        "gsf_to_pf_sign_error_ratio": float(gsf_positive_mass_error / max(float(pf_reference.metrics["mean_pf_positive_mass_error"]), 1.0e-12)),
        "promotion_decision": decision,
    }
    return GSFOracleWitnessResult(
        truth_rows=truth_rows,
        measurement_rows=measurement_rows,
        oracle_posterior_rows=tuple(oracle_posterior_rows),
        method_posterior_rows=tuple(method_posterior_rows),
        gaussian_posterior_rows=tuple(gaussian_posterior_rows),
        component_rows=tuple(component_rows),
        state_rows=tuple(state_rows),
        metrics=metrics,
    )


def write_gsf_abs_range_multimodal_witness_artifacts(
    output_dir: str | Path,
    *,
    result: GSFOracleWitnessResult | None = None,
) -> GSFOracleWitnessArtifacts:
    analysis = result or analyze_gsf_abs_range_multimodal_witness()
    run_dir = Path(output_dir) / "gsf_abs_range_multimodal_oracle_v1"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    truth_path = run_dir / "truth_trajectory.csv"
    measurement_path = run_dir / "measurements.csv"
    oracle_posterior_path = run_dir / "grid_oracle_posterior_history.csv"
    method_posterior_path = run_dir / "method_posterior_history.csv"
    gaussian_posterior_path = run_dir / "gaussian_baseline_posterior_history.csv"
    component_history_path = run_dir / "component_history.csv"
    state_path = run_dir / "state_estimate_history.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics_against_oracle.csv"
    decision_card_path = run_dir / "decision_card.md"
    final_overlay_plot_path = plot_dir / "final_posterior_overlay.png"
    kl_timeline_plot_path = plot_dir / "oracle_kl_timeline.png"
    component_timeline_plot_path = plot_dir / "component_timeline.png"

    write_csv(truth_path, [asdict(row) for row in analysis.truth_rows], ["time", "truth_position"])
    write_csv(measurement_path, [asdict(row) for row in analysis.measurement_rows], ["time", "measurement"])
    write_csv(oracle_posterior_path, [asdict(row) for row in analysis.oracle_posterior_rows], ["time", "position", "posterior_probability"])
    write_csv(method_posterior_path, [asdict(row) for row in analysis.method_posterior_rows], ["time", "position", "posterior_probability"])
    write_csv(gaussian_posterior_path, [asdict(row) for row in analysis.gaussian_posterior_rows], ["time", "position", "posterior_probability"])
    write_csv(component_history_path, [asdict(row) for row in analysis.component_rows], ["time", "component_index", "weight", "mean", "variance", "sign_label"])
    write_csv(
        state_path,
        [asdict(row) for row in analysis.state_rows],
        [
            "time",
            "truth_position",
            "measurement",
            "oracle_mean",
            "oracle_map",
            "oracle_positive_mass",
            "gsf_mean",
            "gsf_positive_mass",
            "gaussian_mean",
            "gaussian_positive_mass",
            "oracle_to_gsf_kl",
            "oracle_to_gaussian_kl",
            "active_component_count",
        ],
    )
    write_comparison_summary_csv(summary_path, [analysis.metrics])
    write_csv(metrics_path, [analysis.metrics], list(analysis.metrics))
    decision_card_path.write_text(_render_gsf_decision_card(analysis), encoding="utf-8")
    _write_gsf_plots(analysis, final_overlay_plot_path, kl_timeline_plot_path, component_timeline_plot_path)
    return GSFOracleWitnessArtifacts(
        run_dir=run_dir,
        truth_path=truth_path,
        measurement_path=measurement_path,
        grid_oracle_posterior_path=oracle_posterior_path,
        method_posterior_path=method_posterior_path,
        gaussian_baseline_posterior_path=gaussian_posterior_path,
        component_history_path=component_history_path,
        state_estimate_history_path=state_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        decision_card_path=decision_card_path,
        plot_paths=(final_overlay_plot_path, kl_timeline_plot_path, component_timeline_plot_path),
    )


def gsf_abs_range_multimodal_witness_surface() -> AdvancedFilterSurface[GSFOracleWitnessResult, GSFOracleWitnessArtifacts]:
    return AdvancedFilterSurface(
        study_id="gsf_abs_range_multimodal_oracle_v1",
        run=analyze_gsf_abs_range_multimodal_witness,
        write_artifacts=write_gsf_abs_range_multimodal_witness_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.metrics_path),
            str(artifacts.decision_card_path),
        ),
        metadata={
            "study_kind": "1d_oracle_positive_witness",
            "problem_family": "gsf_abs_range_multimodal",
        },
    )


def _render_gsf_decision_card(result: GSFOracleWitnessResult) -> str:
    report = MarkdownDocument("GSF Abs-Range Multimodal Oracle Witness")
    report.paragraph(
        "This witness asks whether a small Gaussian mixture can close enough of the multimodal posterior gap that escalation to particles is no longer automatic."
    )
    report.bullet_list(
        [
            f"GSF mean oracle->GSF KL: `{result.metrics['mean_oracle_to_gsf_kl']}`",
            f"Gaussian mean oracle->Gaussian KL: `{result.metrics['mean_oracle_to_gaussian_kl']}`",
            f"GSF positive-mass error: `{result.metrics['mean_gsf_positive_mass_error']}`",
            f"Gaussian positive-mass error: `{result.metrics['mean_gaussian_positive_mass_error']}`",
            f"GSF to PF KL ratio: `{result.metrics['gsf_to_pf_kl_ratio']}`",
            f"GSF to PF sign-error ratio: `{result.metrics['gsf_to_pf_sign_error_ratio']}`",
            f"Mean component count: `{result.metrics['mean_component_count']}`",
            f"Decision: `{result.metrics['promotion_decision']}`",
        ]
    )
    report.paragraph(
        "Interpretation: a GSF witness can support mixture-based escalation above a single Gaussian, but it only blocks PF promotion if the mixture closes enough of the oracle gap at materially lower complexity."
    )
    return report.text()


def _write_gsf_plots(
    result: GSFOracleWitnessResult,
    final_overlay_plot_path: Path,
    kl_timeline_plot_path: Path,
    component_timeline_plot_path: Path,
) -> None:
    final_time = result.state_rows[-1].time
    oracle_rows = [row for row in result.oracle_posterior_rows if row.time == final_time]
    gsf_rows = [row for row in result.method_posterior_rows if row.time == final_time]
    gaussian_rows = [row for row in result.gaussian_posterior_rows if row.time == final_time]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([row.position for row in oracle_rows], [row.posterior_probability for row in oracle_rows], label="grid oracle")
    ax.plot([row.position for row in gsf_rows], [row.posterior_probability for row in gsf_rows], label="GSF")
    ax.plot([row.position for row in gaussian_rows], [row.posterior_probability for row in gaussian_rows], label="single Gaussian")
    ax.set_title("Final posterior overlay")
    ax.set_xlabel("position")
    ax.set_ylabel("posterior probability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(final_overlay_plot_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([row.time for row in result.state_rows], [row.oracle_to_gsf_kl for row in result.state_rows], label="oracle->GSF KL")
    ax.plot([row.time for row in result.state_rows], [row.oracle_to_gaussian_kl for row in result.state_rows], label="oracle->Gaussian KL")
    ax.set_title("Oracle divergence timeline")
    ax.set_xlabel("time")
    ax.set_ylabel("KL divergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(kl_timeline_plot_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for component_index in sorted({row.component_index for row in result.component_rows}):
        rows = [row for row in result.component_rows if row.component_index == component_index]
        ax.scatter([row.time for row in rows], [row.mean for row in rows], s=[max(16.0, 180.0 * row.weight) for row in rows], alpha=0.6, label=f"component {component_index}")
    ax.plot([row.time for row in result.state_rows], [row.truth_position for row in result.state_rows], color="black", linewidth=1.2, label="truth")
    ax.set_title("GSF component timeline")
    ax.set_xlabel("time")
    ax.set_ylabel("component mean")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(component_timeline_plot_path)
    plt.close(fig)
