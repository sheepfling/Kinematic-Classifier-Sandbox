from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png

from ..markdown_builder import MarkdownDocument
from ..runtime_paths import prepare_matplotlib
from .short_horizon_identifiability import ShortHorizonIdentifiabilityResult


@dataclass(frozen=True, slots=True)
class ShortHorizonIdentifiabilityArtifacts:
    run_dir: Path
    report_path: Path
    time_series_path: Path
    noise_sweep_path: Path
    duration_thresholds_path: Path
    time_plot_png_path: Path
    noise_plot_png_path: Path
    duration_plot_png_path: Path


def render_short_horizon_identifiability_report(result: ShortHorizonIdentifiabilityResult) -> str:
    final_row = result.times[-1]
    nominal_row = next(row for row in result.noise_sweep if abs(row.measurement_sigma - result.nominal_measurement_sigma) < 1e-9)
    nominal_duration_row = next(
        row for row in result.duration_thresholds if abs(row.measurement_sigma - result.nominal_measurement_sigma) < 1e-9
    )
    report = MarkdownDocument("Short-Horizon Identifiability")
    report.paragraph(
        "This artifact makes the `short_noisy` limit explicit for constant velocity versus constant acceleration "
        "over the shared 4-sample horizon."
    )
    report.heading("Key Numbers", level=2)
    report.bullet_list(
        [
            f"Nominal measurement sigma: `{result.nominal_measurement_sigma:.2f}`",
            f"Final-step absolute position gap: `{final_row.absolute_gap:.3f}`",
            f"Final-step normalized gap at nominal noise: `{final_row.normalized_gap_at_nominal_noise:.3f}`",
            f"Mean normalized gap at nominal noise: `{nominal_row.mean_normalized_gap:.3f}`",
            f"First time to reach 1 sigma at nominal noise: `{nominal_duration_row.first_time_at_1sigma}`",
            f"First time to reach 2 sigma at nominal noise: `{nominal_duration_row.first_time_at_2sigma}`",
        ]
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "The Kalman bank already infers latent velocity and acceleration internally.",
            "This study asks a simpler identifiability question: how far apart are the two class-predicted position sequences, relative to the measurement noise, over the short noisy horizon?",
            "When the normalized gap stays near or below one sigma for much of the sequence, no filter can separate the classes confidently from position alone without becoming prior- or model-dominated.",
            "If we want to improve `short_noisy`, we likely need either more elapsed time, lower noise, or an actually stronger measurement stream rather than more reuse of the same positions.",
        ]
    )
    return report.text()


def _render_time_plot(result: ShortHorizonIdentifiabilityResult):
    plt = prepare_matplotlib()
    times = [row.time for row in result.times]
    cv = [row.constant_velocity_position for row in result.times]
    ca = [row.constant_acceleration_position for row in result.times]
    normalized_gap = [row.normalized_gap_at_nominal_noise for row in result.times]

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.8), sharex=True)
    axes[0].plot(times, cv, label="constant_velocity", color="#2563eb", linewidth=2.0)
    axes[0].plot(times, ca, label="constant_acceleration", color="#dc2626", linewidth=2.0)
    axes[0].set_ylabel("position")
    axes[0].set_title("Short-Noisy Class Trajectories", loc="left", fontsize=12, fontweight="bold")
    axes[0].grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False)

    axes[1].plot(times, normalized_gap, color="#0f766e", linewidth=2.0, marker="o")
    axes[1].axhline(1.0, color="#d97706", linestyle="--", linewidth=1.2, label="1 sigma gap")
    axes[1].set_ylabel("gap / sigma")
    axes[1].set_xlabel("time")
    axes[1].set_title("Normalized Position Separation", loc="left", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    return fig


def _render_noise_plot(result: ShortHorizonIdentifiabilityResult):
    plt = prepare_matplotlib()
    sigmas = [row.measurement_sigma for row in result.noise_sweep]
    mean_gap = [row.mean_normalized_gap for row in result.noise_sweep]
    final_gap = [row.final_step_normalized_gap for row in result.noise_sweep]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(sigmas, mean_gap, color="#2563eb", linewidth=2.0, marker="o", label="mean normalized gap")
    ax.plot(sigmas, final_gap, color="#dc2626", linewidth=2.0, marker="s", label="final-step normalized gap")
    ax.axhline(1.0, color="#d97706", linestyle="--", linewidth=1.2, label="1 sigma gap")
    ax.set_xlabel("measurement sigma")
    ax.set_ylabel("gap / sigma")
    ax.set_title("Short-Horizon Separation vs Noise", loc="left", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_duration_plot(result: ShortHorizonIdentifiabilityResult):
    plt = prepare_matplotlib()
    sigmas = [row.measurement_sigma for row in result.duration_thresholds]
    first_1sigma = [row.first_time_at_1sigma if row.first_time_at_1sigma is not None else float("nan") for row in result.duration_thresholds]
    first_2sigma = [row.first_time_at_2sigma if row.first_time_at_2sigma is not None else float("nan") for row in result.duration_thresholds]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(sigmas, first_1sigma, color="#2563eb", linewidth=2.0, marker="o", label="time to 1 sigma")
    ax.plot(sigmas, first_2sigma, color="#dc2626", linewidth=2.0, marker="s", label="time to 2 sigma")
    ax.set_xlabel("measurement sigma")
    ax.set_ylabel("elapsed time")
    ax.set_title("Required Duration for Separation", loc="left", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def write_short_horizon_identifiability_artifacts(
    output_root: str | Path,
    *,
    result: ShortHorizonIdentifiabilityResult | None = None,
) -> ShortHorizonIdentifiabilityArtifacts:
    base_path = Path(output_root)
    run_dir = base_path / "short_horizon_identifiability_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    if result is None:
        from .short_horizon_identifiability import analyze_short_horizon_identifiability

        result = analyze_short_horizon_identifiability()

    analysis = result

    report_path = run_dir / "short_horizon_identifiability_report.md"
    time_series_path = run_dir / "short_horizon_time_series.csv"
    noise_sweep_path = run_dir / "short_horizon_noise_sweep.csv"
    duration_thresholds_path = run_dir / "short_horizon_duration_thresholds.csv"
    time_plot_png_path = run_dir / "short_horizon_time_series.png"
    noise_plot_png_path = run_dir / "short_horizon_noise_sweep.png"
    duration_plot_png_path = run_dir / "short_horizon_duration_thresholds.png"

    report_path.write_text(render_short_horizon_identifiability_report(analysis), encoding="utf-8")
    write_csv(
        time_series_path,
        [
            {
                "time": row.time,
                "constant_velocity_position": row.constant_velocity_position,
                "constant_acceleration_position": row.constant_acceleration_position,
                "absolute_gap": row.absolute_gap,
                "normalized_gap_at_nominal_noise": row.normalized_gap_at_nominal_noise,
            }
            for row in analysis.times
        ],
        [
            "time",
            "constant_velocity_position",
            "constant_acceleration_position",
            "absolute_gap",
            "normalized_gap_at_nominal_noise",
        ],
    )
    write_csv(
        noise_sweep_path,
        [
            {
                "measurement_sigma": row.measurement_sigma,
                "mean_normalized_gap": row.mean_normalized_gap,
                "max_normalized_gap": row.max_normalized_gap,
                "final_step_normalized_gap": row.final_step_normalized_gap,
            }
            for row in analysis.noise_sweep
        ],
        [
            "measurement_sigma",
            "mean_normalized_gap",
            "max_normalized_gap",
            "final_step_normalized_gap",
        ],
    )
    write_csv(
        duration_thresholds_path,
        [
            {
                "measurement_sigma": row.measurement_sigma,
                "first_time_at_1sigma": row.first_time_at_1sigma,
                "first_time_at_2sigma": row.first_time_at_2sigma,
            }
            for row in analysis.duration_thresholds
        ],
        [
            "measurement_sigma",
            "first_time_at_1sigma",
            "first_time_at_2sigma",
        ],
    )

    time_figure = _render_time_plot(analysis)
    try:
        time_plot_png_path.write_bytes(_figure_to_png(time_figure))
    finally:
        time_figure.clf()
        time_figure = None

    noise_figure = _render_noise_plot(analysis)
    try:
        noise_plot_png_path.write_bytes(_figure_to_png(noise_figure))
    finally:
        noise_figure.clf()
        noise_figure = None

    duration_figure = _render_duration_plot(analysis)
    try:
        duration_plot_png_path.write_bytes(_figure_to_png(duration_figure))
    finally:
        duration_figure.clf()
        duration_figure = None

    return ShortHorizonIdentifiabilityArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        time_series_path=time_series_path,
        noise_sweep_path=noise_sweep_path,
        duration_thresholds_path=duration_thresholds_path,
        time_plot_png_path=time_plot_png_path,
        noise_plot_png_path=noise_plot_png_path,
        duration_plot_png_path=duration_plot_png_path,
    )


__all__ = [
    "ShortHorizonIdentifiabilityArtifacts",
    "render_short_horizon_identifiability_report",
    "write_short_horizon_identifiability_artifacts",
]
