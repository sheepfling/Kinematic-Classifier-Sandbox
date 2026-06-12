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

from .particle_filter import BootstrapParticleFilter, ParticleFilterConfig
from .surface import AdvancedFilterSurface


def _normal_pdf(x: float, mean_value: float, variance: float) -> float:
    variance = max(float(variance), 1.0e-12)
    return math.exp(-0.5 * ((float(x) - float(mean_value)) ** 2) / variance) / math.sqrt(2.0 * math.pi * variance)


def _abs_measurement_log_likelihood(particles, observation):
    measurement = float(observation[0])
    measurement_std = 0.12
    values = []
    for particle in particles:
        residual = measurement - abs(float(particle[0]))
        variance = measurement_std**2
        values.append(-0.5 * (math.log(2.0 * math.pi * variance) + residual**2 / variance))
    return array(values, dtype=float64)


def _random_walk_transition(particles, dt, rng):
    del dt
    next_particles = particles.copy()
    noise = rng.normal(0.0, 0.08, size=len(particles))
    next_particles[:, 0] = next_particles[:, 0] + noise
    return next_particles


@dataclass(frozen=True, slots=True)
class OraclePFGridConfig:
    grid_min: float = -2.5
    grid_max: float = 2.5
    grid_step: float = 0.02


@dataclass(frozen=True, slots=True)
class PFOracleTruthRow:
    time: float
    truth_position: float


@dataclass(frozen=True, slots=True)
class PFOracleMeasurementRow:
    time: float
    measurement: float


@dataclass(frozen=True, slots=True)
class PFOraclePosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class PFMethodPosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class PFGaussianPosteriorRow:
    time: float
    position: float
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class PFOracleStateEstimateRow:
    time: float
    truth_position: float
    measurement: float
    oracle_mean: float
    oracle_map: float
    oracle_positive_mass: float
    pf_mean: float
    pf_positive_mass: float
    gaussian_mean: float
    gaussian_positive_mass: float
    oracle_to_pf_kl: float
    oracle_to_gaussian_kl: float
    pf_ess_fraction: float
    pf_resampled: bool


@dataclass(frozen=True, slots=True)
class PFOracleParticleDiagnosticRow:
    time: float
    ess_fraction: float
    unique_ancestor_fraction: float
    weight_entropy: float
    resampled: bool


@dataclass(frozen=True, slots=True)
class PFOracleWitnessResult:
    truth_rows: tuple[PFOracleTruthRow, ...]
    measurement_rows: tuple[PFOracleMeasurementRow, ...]
    oracle_posterior_rows: tuple[PFOraclePosteriorRow, ...]
    method_posterior_rows: tuple[PFMethodPosteriorRow, ...]
    gaussian_posterior_rows: tuple[PFGaussianPosteriorRow, ...]
    state_rows: tuple[PFOracleStateEstimateRow, ...]
    particle_diagnostic_rows: tuple[PFOracleParticleDiagnosticRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class PFOracleWitnessArtifacts:
    run_dir: Path
    truth_path: Path
    measurement_path: Path
    grid_oracle_posterior_path: Path
    method_posterior_path: Path
    gaussian_baseline_posterior_path: Path
    state_estimate_history_path: Path
    particle_diagnostics_path: Path
    summary_path: Path
    metrics_path: Path
    decision_card_path: Path
    gaussian_collapse_panel_path: Path
    plot_paths: tuple[Path, ...]


def analyze_pf_abs_range_multimodal_witness(
    *,
    seed: int = 211,
    particle_count: int = 768,
    grid: OraclePFGridConfig = OraclePFGridConfig(),
) -> PFOracleWitnessResult:
    rng = random.default_rng(seed)
    times = tuple(float(time_value) for time_value in arange(0.0, 5.0, 0.25, dtype=float64))
    truth_values = [1.1]
    for _ in times[1:]:
        truth_values.append(float(truth_values[-1] + rng.normal(0.0, 0.08)))
    measurement_values = [float(abs(truth) + rng.normal(0.0, 0.12)) for truth in truth_values]
    truth_rows = tuple(
        PFOracleTruthRow(time=time_value, truth_position=truth_value)
        for time_value, truth_value in zip(times, truth_values, strict=True)
    )
    measurement_rows = tuple(
        PFOracleMeasurementRow(time=time_value, measurement=measurement_value)
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

    pf_rng = random.default_rng(seed + 1)
    initial_particles = array([[value] for value in pf_rng.normal(0.25, 0.9, size=particle_count)], dtype=float64)
    pf = BootstrapParticleFilter(
        ParticleFilterConfig(particle_count=particle_count, seed=seed + 2),
        transition_fn=_random_walk_transition,
        log_likelihood_fn=_abs_measurement_log_likelihood,
    )
    pf.reset("pf_abs_range_multimodal_v1", initial_particles)

    oracle_posterior = oracle_prior
    gaussian_mean = 0.25
    gaussian_variance = 0.9**2
    oracle_posterior_rows: list[PFOraclePosteriorRow] = []
    method_posterior_rows: list[PFMethodPosteriorRow] = []
    gaussian_posterior_rows: list[PFGaussianPosteriorRow] = []
    state_rows: list[PFOracleStateEstimateRow] = []
    particle_rows: list[PFOracleParticleDiagnosticRow] = []
    oracle_nll_values: list[float] = []
    pf_nll_values: list[float] = []
    start = wall_time.perf_counter()

    for time_value, truth_value, measurement_value in zip(times, truth_values, measurement_values, strict=True):
        predicted_oracle = transition_matrix @ oracle_posterior
        likelihood = array(
            [_normal_pdf(measurement_value, abs(position), 0.12**2) for position in grid_points],
            dtype=float64,
        )
        oracle_evidence = float((predicted_oracle * likelihood).sum())
        oracle_posterior = predicted_oracle * likelihood
        oracle_posterior /= max(float(oracle_posterior.sum()), 1.0e-15)
        oracle_nll_values.append(-math.log(max(oracle_evidence, 1.0e-300)))

        predicted_gaussian_variance = gaussian_variance + 0.08**2
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

        pf_step = pf.update(time_value, array([measurement_value], dtype=float64))
        assert pf.state is not None
        weights = array([math.exp(float(value)) for value in pf.state.log_weights], dtype=float64)
        particles = pf.state.particles[:, 0]
        bandwidth = 0.07
        pf_density = array(
            [
                float(
                    sum(weight * _normal_pdf(position, particle, bandwidth**2) for particle, weight in zip(particles, weights, strict=True))
                )
                for position in grid_points
            ],
            dtype=float64,
        )
        pf_density /= max(float(pf_density.sum()), 1.0e-15)
        pf_nll_values.append(-float(pf_step.log_marginal_likelihood))

        oracle_mean = float((grid_points * oracle_posterior).sum())
        oracle_map = float(grid_points[int(oracle_posterior.argmax())])
        oracle_positive_mass = float(sum(float(probability) for position, probability in zip(grid_points, oracle_posterior, strict=True) if position >= 0.0))
        pf_mean = float(pf_step.state_mean[0])
        pf_positive_mass = float(sum(weight for particle, weight in zip(particles, weights, strict=True) if particle >= 0.0))
        gaussian_positive_mass = float(sum(float(probability) for position, probability in zip(grid_points, gaussian_density, strict=True) if position >= 0.0))
        oracle_to_pf_kl = float(
            sum(
                float(probability) * math.log(max(float(probability), 1.0e-300) / max(float(pf_density[index]), 1.0e-300))
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
            PFOracleStateEstimateRow(
                time=time_value,
                truth_position=truth_value,
                measurement=measurement_value,
                oracle_mean=oracle_mean,
                oracle_map=oracle_map,
                oracle_positive_mass=oracle_positive_mass,
                pf_mean=pf_mean,
                pf_positive_mass=pf_positive_mass,
                gaussian_mean=gaussian_mean,
                gaussian_positive_mass=gaussian_positive_mass,
                oracle_to_pf_kl=oracle_to_pf_kl,
                oracle_to_gaussian_kl=oracle_to_gaussian_kl,
                pf_ess_fraction=pf_step.ess_fraction,
                pf_resampled=pf_step.resampled,
            )
        )
        particle_rows.append(
            PFOracleParticleDiagnosticRow(
                time=time_value,
                ess_fraction=pf_step.ess_fraction,
                unique_ancestor_fraction=pf_step.unique_ancestor_fraction,
                weight_entropy=pf_step.weight_entropy,
                resampled=pf_step.resampled,
            )
        )
        for position, oracle_probability, pf_probability, gaussian_probability in zip(
            grid_points,
            oracle_posterior,
            pf_density,
            gaussian_density,
            strict=True,
        ):
            oracle_posterior_rows.append(
                PFOraclePosteriorRow(
                    time=time_value,
                    position=float(position),
                    posterior_probability=float(oracle_probability),
                )
            )
            method_posterior_rows.append(
                PFMethodPosteriorRow(
                    time=time_value,
                    position=float(position),
                    posterior_probability=float(pf_probability),
                )
            )
            gaussian_posterior_rows.append(
                PFGaussianPosteriorRow(
                    time=time_value,
                    position=float(position),
                    posterior_probability=float(gaussian_probability),
                )
            )
    runtime_seconds = wall_time.perf_counter() - start

    mean_pf_kl = float(mean([row.oracle_to_pf_kl for row in state_rows]))
    mean_gaussian_kl = float(mean([row.oracle_to_gaussian_kl for row in state_rows]))
    pf_rmse = float(sqrt(mean([(row.pf_mean - row.truth_position) ** 2 for row in state_rows])))
    gaussian_rmse = float(sqrt(mean([(row.gaussian_mean - row.truth_position) ** 2 for row in state_rows])))
    oracle_rmse = float(sqrt(mean([(row.oracle_mean - row.truth_position) ** 2 for row in state_rows])))
    pf_positive_mass_error = float(mean([abs(row.pf_positive_mass - row.oracle_positive_mass) for row in state_rows]))
    gaussian_positive_mass_error = float(mean([abs(row.gaussian_positive_mass - row.oracle_positive_mass) for row in state_rows]))
    mean_ess_fraction = float(mean([row.ess_fraction for row in particle_rows]))
    promotion_decision = (
        "promote_pf_for_multimodal_posterior"
        if mean_pf_kl < mean_gaussian_kl * 0.65 and pf_positive_mass_error < gaussian_positive_mass_error * 0.65
        else "revise_pf_witness"
    )
    metrics = {
        "study_id": "pf_abs_range_multimodal_oracle_v1",
        "seed": seed,
        "particle_count": particle_count,
        "step_count": len(times),
        "oracle_rmse": oracle_rmse,
        "pf_rmse": pf_rmse,
        "gaussian_rmse": gaussian_rmse,
        "mean_oracle_to_pf_kl": mean_pf_kl,
        "mean_oracle_to_gaussian_kl": mean_gaussian_kl,
        "mean_pf_positive_mass_error": pf_positive_mass_error,
        "mean_gaussian_positive_mass_error": gaussian_positive_mass_error,
        "oracle_mean_nll": float(mean(oracle_nll_values)),
        "pf_mean_nll": float(mean(pf_nll_values)),
        "mean_ess_fraction": mean_ess_fraction,
        "runtime_seconds": runtime_seconds,
        "promotion_decision": promotion_decision,
    }
    return PFOracleWitnessResult(
        truth_rows=truth_rows,
        measurement_rows=measurement_rows,
        oracle_posterior_rows=tuple(oracle_posterior_rows),
        method_posterior_rows=tuple(method_posterior_rows),
        gaussian_posterior_rows=tuple(gaussian_posterior_rows),
        state_rows=tuple(state_rows),
        particle_diagnostic_rows=tuple(particle_rows),
        metrics=metrics,
    )


def write_pf_abs_range_multimodal_witness_artifacts(
    output_dir: str | Path,
    *,
    result: PFOracleWitnessResult | None = None,
) -> PFOracleWitnessArtifacts:
    analysis = result or analyze_pf_abs_range_multimodal_witness()
    run_dir = Path(output_dir) / "pf_abs_range_multimodal_oracle_v1"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    truth_path = run_dir / "truth_trajectory.csv"
    measurement_path = run_dir / "measurements.csv"
    oracle_posterior_path = run_dir / "grid_oracle_posterior_history.csv"
    method_posterior_path = run_dir / "method_posterior_history.csv"
    gaussian_posterior_path = run_dir / "gaussian_baseline_posterior_history.csv"
    state_path = run_dir / "state_estimate_history.csv"
    particle_diagnostics_path = run_dir / "particle_diagnostics.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics_against_oracle.csv"
    decision_card_path = run_dir / "decision_card.md"
    final_overlay_plot_path = plot_dir / "final_posterior_overlay.png"
    kl_timeline_plot_path = plot_dir / "oracle_kl_timeline.png"
    gaussian_collapse_panel_path = plot_dir / "gaussian_collapse_panel.png"

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
        gaussian_posterior_path,
        [asdict(row) for row in analysis.gaussian_posterior_rows],
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
            "oracle_map",
            "oracle_positive_mass",
            "pf_mean",
            "pf_positive_mass",
            "gaussian_mean",
            "gaussian_positive_mass",
            "oracle_to_pf_kl",
            "oracle_to_gaussian_kl",
            "pf_ess_fraction",
            "pf_resampled",
        ],
    )
    write_csv(
        particle_diagnostics_path,
        [asdict(row) for row in analysis.particle_diagnostic_rows],
        ["time", "ess_fraction", "unique_ancestor_fraction", "weight_entropy", "resampled"],
    )
    write_comparison_summary_csv(summary_path, [analysis.metrics])
    write_csv(metrics_path, [analysis.metrics], list(analysis.metrics))
    decision_card_path.write_text(_render_pf_decision_card(analysis), encoding="utf-8")
    _write_pf_plots(
        analysis,
        final_overlay_plot_path,
        kl_timeline_plot_path,
        gaussian_collapse_panel_path,
    )
    return PFOracleWitnessArtifacts(
        run_dir=run_dir,
        truth_path=truth_path,
        measurement_path=measurement_path,
        grid_oracle_posterior_path=oracle_posterior_path,
        method_posterior_path=method_posterior_path,
        gaussian_baseline_posterior_path=gaussian_posterior_path,
        state_estimate_history_path=state_path,
        particle_diagnostics_path=particle_diagnostics_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        decision_card_path=decision_card_path,
        gaussian_collapse_panel_path=gaussian_collapse_panel_path,
        plot_paths=(final_overlay_plot_path, kl_timeline_plot_path, gaussian_collapse_panel_path),
    )


def pf_abs_range_multimodal_witness_surface() -> AdvancedFilterSurface[PFOracleWitnessResult, PFOracleWitnessArtifacts]:
    return AdvancedFilterSurface(
        study_id="pf_abs_range_multimodal_oracle_v1",
        run=analyze_pf_abs_range_multimodal_witness,
        write_artifacts=write_pf_abs_range_multimodal_witness_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.metrics_path),
            str(artifacts.decision_card_path),
        ),
        metadata={
            "study_kind": "1d_oracle_positive_witness",
            "problem_family": "pf_abs_range_multimodal",
        },
    )


def _render_pf_decision_card(result: PFOracleWitnessResult) -> str:
    report = MarkdownDocument("PF Abs-Range Multimodal Oracle Witness")
    report.paragraph(
        "This witness is designed to justify PF for a named failure mode. The measurement is non-injective, so the posterior can carry separated positive and negative position mass that a single-Gaussian projection cannot represent faithfully."
    )
    report.bullet_list(
        [
            f"PF mean oracle->PF KL: `{result.metrics['mean_oracle_to_pf_kl']}`",
            f"Gaussian mean oracle->Gaussian KL: `{result.metrics['mean_oracle_to_gaussian_kl']}`",
            f"PF positive-mass error: `{result.metrics['mean_pf_positive_mass_error']}`",
            f"Gaussian positive-mass error: `{result.metrics['mean_gaussian_positive_mass_error']}`",
            f"PF RMSE: `{result.metrics['pf_rmse']}`",
            f"Gaussian RMSE: `{result.metrics['gaussian_rmse']}`",
            f"Mean ESS/N: `{result.metrics['mean_ess_fraction']}`",
            f"Decision: `{result.metrics['promotion_decision']}`",
        ]
    )
    report.paragraph(
        "Interpretation: this is a representational promotion card, not a universal scorecard. PF is promoted only because the oracle shows multimodal posterior structure and the PF approximation preserves that structure materially better than a single-Gaussian projection."
    )
    return report.text()


def _write_pf_plots(
    result: PFOracleWitnessResult,
    final_overlay_plot_path: Path,
    kl_timeline_plot_path: Path,
    gaussian_collapse_panel_path: Path,
) -> None:
    final_time = result.state_rows[-1].time
    oracle_rows = [row for row in result.oracle_posterior_rows if row.time == final_time]
    pf_rows = [row for row in result.method_posterior_rows if row.time == final_time]
    gaussian_rows = [row for row in result.gaussian_posterior_rows if row.time == final_time]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([row.position for row in oracle_rows], [row.posterior_probability for row in oracle_rows], label="grid oracle")
    ax.plot([row.position for row in pf_rows], [row.posterior_probability for row in pf_rows], label="PF")
    ax.plot([row.position for row in gaussian_rows], [row.posterior_probability for row in gaussian_rows], label="single Gaussian")
    ax.set_title("Final posterior overlay")
    ax.set_xlabel("position")
    ax.set_ylabel("posterior probability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(final_overlay_plot_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([row.time for row in result.state_rows], [row.oracle_to_pf_kl for row in result.state_rows], label="oracle->PF KL")
    ax.plot(
        [row.time for row in result.state_rows],
        [row.oracle_to_gaussian_kl for row in result.state_rows],
        label="oracle->Gaussian KL",
    )
    ax.set_title("Oracle divergence timeline")
    ax.set_xlabel("time")
    ax.set_ylabel("KL divergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(kl_timeline_plot_path)
    plt.close(fig)

    final_row = result.state_rows[-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)
    axes[0].plot([row.position for row in oracle_rows], [row.posterior_probability for row in oracle_rows], label="grid oracle")
    axes[0].plot([row.position for row in pf_rows], [row.posterior_probability for row in pf_rows], label="PF")
    axes[0].plot([row.position for row in gaussian_rows], [row.posterior_probability for row in gaussian_rows], label="single Gaussian")
    axes[0].set_title("Final multimodal posterior")
    axes[0].set_xlabel("position")
    axes[0].set_ylabel("posterior probability")
    axes[0].legend(fontsize=7)

    times = [row.time for row in result.state_rows]
    axes[1].plot(times, [row.oracle_positive_mass for row in result.state_rows], label="oracle positive mass")
    axes[1].plot(times, [row.pf_positive_mass for row in result.state_rows], label="PF positive mass")
    axes[1].plot(times, [row.gaussian_positive_mass for row in result.state_rows], label="Gaussian positive mass")
    axes[1].set_title("Sign-mass recovery")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("mass on x >= 0")
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].legend(fontsize=7)

    names = ["oracle->PF KL", "oracle->Gaussian KL", "PF sign error", "Gaussian sign error"]
    values = [
        float(result.metrics["mean_oracle_to_pf_kl"]),
        float(result.metrics["mean_oracle_to_gaussian_kl"]),
        float(result.metrics["mean_pf_positive_mass_error"]),
        float(result.metrics["mean_gaussian_positive_mass_error"]),
    ]
    axes[2].barh(range(len(names)), values, color=["#1f77b4", "#7f7f7f", "#2ca02c", "#d62728"])
    axes[2].set_yticks(range(len(names)), names)
    axes[2].invert_yaxis()
    axes[2].set_title("Gaussian collapse summary")
    axes[2].text(
        0.02,
        0.02,
        f"final oracle map={final_row.oracle_map:.2f}\nfinal PF mean={final_row.pf_mean:.2f}\nfinal Gaussian mean={final_row.gaussian_mean:.2f}",
        transform=axes[2].transAxes,
        fontsize=7,
        va="bottom",
    )
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(gaussian_collapse_panel_path)
    plt.close(fig)
