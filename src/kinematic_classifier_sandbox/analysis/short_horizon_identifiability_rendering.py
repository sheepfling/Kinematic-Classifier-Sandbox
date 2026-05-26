from __future__ import annotations

from ..markdown_builder import MarkdownDocument
from ..utils.plotting import plt
from .short_horizon_identifiability_contracts import ShortHorizonIdentifiabilityResult


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


__all__ = [
    "_render_duration_plot",
    "_render_noise_plot",
    "_render_time_plot",
    "render_short_horizon_identifiability_report",
]
