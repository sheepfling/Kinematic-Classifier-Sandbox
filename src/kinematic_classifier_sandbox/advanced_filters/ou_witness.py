from __future__ import annotations

import time as wall_time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy.random as random
from numpy import arange, array, asarray, average, exp, float64, sqrt
from numpy import mean as nmean

from ..markdown_builder import MarkdownDocument
from ..utils.io import write_csv
from ..utils.plotting import plt
from .models_1d import (
    constant_velocity_transition,
    make_initial_particles_1d,
    position_gaussian_log_likelihood,
)
from .particle_filter import BootstrapParticleFilter, ParticleFilterConfig
from .particle_filter_bank import ParticleFilterBank
from .surface import AdvancedFilterSurface


@dataclass(frozen=True, slots=True)
class OrnsteinUhlenbeckPosteriorRow:
    trajectory_id: str
    time: float
    label: str
    posterior: float
    predicted_label: str
    confidence: float
    log_evidence: float


@dataclass(frozen=True, slots=True)
class OrnsteinUhlenbeckStateRow:
    trajectory_id: str
    time: float
    truth_position: float
    truth_velocity: float
    observation: float
    pf_position: float
    pf_velocity: float
    predicted_label: str
    ess: float
    resampled: bool


@dataclass(frozen=True, slots=True)
class OrnsteinUhlenbeckWitnessResult:
    trajectory_id: str
    posterior_rows: tuple[OrnsteinUhlenbeckPosteriorRow, ...]
    state_rows: tuple[OrnsteinUhlenbeckStateRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class OrnsteinUhlenbeckWitnessArtifacts:
    run_dir: Path
    report_path: Path
    posterior_history_path: Path
    state_estimate_history_path: Path
    metrics_path: Path
    plot_paths: tuple[Path, ...]


def _ou_velocity_transition(
    particles,
    dt: float,
    rng: random.Generator,
    mean_velocity: float = 0.35,
    theta: float = 1.25,
    process_std: float = 0.09,
):
    next_particles = particles.copy()
    velocity_noise = rng.normal(0.0, process_std, size=len(particles))
    next_velocity = particles[:, 1] + theta * (mean_velocity - particles[:, 1]) * dt + velocity_noise
    next_position = particles[:, 0] + next_velocity * dt
    next_particles[:, 0] = next_position
    next_particles[:, 1] = next_velocity
    return next_particles


def analyze_ornstein_uhlenbeck_witness(*, seed: int = 37) -> OrnsteinUhlenbeckWitnessResult:
    rng = random.default_rng(seed)
    trajectory_id = "ornstein_uhlenbeck_mean_reversion_1d"
    times = arange(0.0, 8.0, 0.25, dtype=float64)
    velocities = [1.65]
    positions = [0.0]
    for _ in times[1:]:
        dt = 0.25
        noise = float(rng.normal(0.0, 0.08))
        next_velocity = velocities[-1] + 1.15 * (0.35 - velocities[-1]) * dt + noise
        velocities.append(next_velocity)
        positions.append(positions[-1] + next_velocity * dt)
    truth_position = asarray(positions, dtype=float64)
    truth_velocity = asarray(velocities, dtype=float64)
    observations = truth_position + rng.normal(0.0, 0.10, size=len(truth_position))

    particle_count = 256
    filters = {
        "constant_velocity": BootstrapParticleFilter(
            ParticleFilterConfig(particle_count=particle_count, seed=seed + 1),
            transition_fn=lambda particles, dt, gen: constant_velocity_transition(particles, dt, gen, process_std=0.04),
            log_likelihood_fn=lambda particles, obs: position_gaussian_log_likelihood(particles, obs, measurement_std=0.12),
        ),
        "mean_reverting_velocity": BootstrapParticleFilter(
            ParticleFilterConfig(particle_count=particle_count, seed=seed + 2),
            transition_fn=lambda particles, dt, gen: _ou_velocity_transition(particles, dt, gen),
            log_likelihood_fn=lambda particles, obs: position_gaussian_log_likelihood(particles, obs, measurement_std=0.12),
        ),
    }
    bank = ParticleFilterBank(
        filters,
        prior_by_label={"constant_velocity": 0.5, "mean_reverting_velocity": 0.5},
        filter_id="ornstein_uhlenbeck_pf_v1",
    )
    init_rng = random.default_rng(seed + 3)
    initial_particles = {
        "constant_velocity": make_initial_particles_1d(particle_count, observations[0], 0.18, 1.5, 0.25, init_rng),
        "mean_reverting_velocity": make_initial_particles_1d(particle_count, observations[0], 0.18, 1.5, 0.25, init_rng),
    }
    bank.reset(trajectory_id, initial_particles)

    posterior_rows: list[OrnsteinUhlenbeckPosteriorRow] = []
    state_rows: list[OrnsteinUhlenbeckStateRow] = []
    start = wall_time.perf_counter()
    for index, (time_value, observation) in enumerate(zip(times, observations, strict=True)):
        step = bank.update(float(time_value), array([float(observation)], dtype=float64))
        for label, posterior in step.posterior_by_label.items():
            posterior_rows.append(
                OrnsteinUhlenbeckPosteriorRow(
                    trajectory_id=trajectory_id,
                    time=float(time_value),
                    label=label,
                    posterior=float(posterior),
                    predicted_label=step.predicted_label,
                    confidence=float(step.confidence),
                    log_evidence=float(step.log_evidence_by_label[label]),
                )
            )
        selected_filter = filters["mean_reverting_velocity"]
        assert selected_filter.state is not None
        weights = exp(selected_filter.state.log_weights)
        state_mean = average(selected_filter.state.particles, axis=0, weights=weights)
        state_rows.append(
            OrnsteinUhlenbeckStateRow(
                trajectory_id=trajectory_id,
                time=float(time_value),
                truth_position=float(truth_position[index]),
                truth_velocity=float(truth_velocity[index]),
                observation=float(observation),
                pf_position=float(state_mean[0]),
                pf_velocity=float(state_mean[1]),
                predicted_label=step.predicted_label,
                ess=float(step.diagnostics["ess_mean_reverting_velocity"]),
                resampled=bool(step.diagnostics["resampled_mean_reverting_velocity"]),
            )
        )
    runtime_seconds = wall_time.perf_counter() - start
    final_mean_reverting = next(
        row.posterior
        for row in reversed(posterior_rows)
        if row.label == "mean_reverting_velocity"
    )
    rmse = float(
        sqrt(
            nmean(
                [(row.pf_position - row.truth_position) ** 2 for row in state_rows]
            )
        )
    )
    metrics = {
        "method_id": "ornstein_uhlenbeck_pf_v1",
        "witness": trajectory_id,
        "final_mean_reverting_posterior": float(final_mean_reverting),
        "position_rmse": rmse,
        "runtime_seconds": runtime_seconds,
        "promotion_decision": "promote" if final_mean_reverting >= 0.60 else "revise",
    }
    return OrnsteinUhlenbeckWitnessResult(
        trajectory_id=trajectory_id,
        posterior_rows=tuple(posterior_rows),
        state_rows=tuple(state_rows),
        metrics=metrics,
    )


def write_ornstein_uhlenbeck_witness_artifacts(
    output_dir: str | Path,
    *,
    result: OrnsteinUhlenbeckWitnessResult | None = None,
    seed: int = 37,
) -> OrnsteinUhlenbeckWitnessArtifacts:
    witness = result or analyze_ornstein_uhlenbeck_witness(seed=seed)
    run_dir = Path(output_dir) / "ornstein_uhlenbeck_witness_v1"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = run_dir / "posterior_history.csv"
    state_path = run_dir / "state_estimate_history.csv"
    metrics_path = run_dir / "ou_method_comparison.csv"
    report_path = run_dir / "ou_report.md"
    write_csv(
        posterior_path,
        [asdict(row) for row in witness.posterior_rows],
        ["trajectory_id", "time", "label", "posterior", "predicted_label", "confidence", "log_evidence"],
    )
    write_csv(
        state_path,
        [asdict(row) for row in witness.state_rows],
        ["trajectory_id", "time", "truth_position", "truth_velocity", "observation", "pf_position", "pf_velocity", "predicted_label", "ess", "resampled"],
    )
    write_csv(metrics_path, [witness.metrics], list(witness.metrics.keys()))
    report = MarkdownDocument("Ornstein-Uhlenbeck Witness Report")
    report.paragraph(
        "Witness: `ornstein_uhlenbeck_mean_reversion_1d` with a particle-filter bank that compares constant-velocity and mean-reverting velocity hypotheses."
    )
    report.bullet_list(
        [
            f"Final mean-reverting posterior: `{float(witness.metrics['final_mean_reverting_posterior']):.3f}`",
            f"Position RMSE: `{float(witness.metrics['position_rmse']):.3f}`",
            f"Runtime seconds: `{float(witness.metrics['runtime_seconds']):.3f}`",
            f"Decision: `{witness.metrics['promotion_decision']}`",
        ]
    )
    report_path.write_text(report.text(), encoding="utf-8")
    state_plot = plot_dir / "ou_state_vs_truth.png"
    posterior_plot = plot_dir / "ou_mode_posterior.png"
    times = [row.time for row in witness.state_rows]
    truth_position = [row.truth_position for row in witness.state_rows]
    observations = [row.observation for row in witness.state_rows]
    pf_position = [row.pf_position for row in witness.state_rows]
    plt.figure(figsize=(8, 4))
    plt.plot(times, truth_position, label="truth")
    plt.plot(times, observations, label="observation")
    plt.plot(times, pf_position, label="OU PF estimate")
    plt.title("OU witness state vs truth")
    plt.xlabel("time")
    plt.ylabel("position")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(state_plot, dpi=160)
    plt.close()
    plt.figure(figsize=(8, 4))
    for label in sorted({row.label for row in witness.posterior_rows}):
        label_rows = [row.posterior for row in witness.posterior_rows if row.label == label]
        plt.plot(times, label_rows, label=label)
    plt.title("OU witness posterior timeline")
    plt.xlabel("time")
    plt.ylabel("posterior")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(posterior_plot, dpi=160)
    plt.close()
    return OrnsteinUhlenbeckWitnessArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        posterior_history_path=posterior_path,
        state_estimate_history_path=state_path,
        metrics_path=metrics_path,
        plot_paths=(state_plot, posterior_plot),
    )


def ornstein_uhlenbeck_witness_surface() -> AdvancedFilterSurface[OrnsteinUhlenbeckWitnessResult, OrnsteinUhlenbeckWitnessArtifacts]:
    return AdvancedFilterSurface(
        study_id="ornstein_uhlenbeck_witness_v1",
        run=analyze_ornstein_uhlenbeck_witness,
        write_artifacts=write_ornstein_uhlenbeck_witness_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.report_path),
            str(artifacts.metrics_path),
        ),
        metadata={
            "study_kind": "1d_witness",
            "problem_family": "ornstein_uhlenbeck_1d",
        },
    )
