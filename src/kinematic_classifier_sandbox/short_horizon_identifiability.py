from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path

from .common_dataset_comparison import SCENARIO_TIMES, _scenario_dynamics


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass(frozen=True, slots=True)
class ShortHorizonTimeRow:
    time: float
    constant_velocity_position: float
    constant_acceleration_position: float
    absolute_gap: float
    normalized_gap_at_nominal_noise: float


@dataclass(frozen=True, slots=True)
class ShortHorizonNoiseRow:
    measurement_sigma: float
    mean_normalized_gap: float
    max_normalized_gap: float
    final_step_normalized_gap: float


@dataclass(frozen=True, slots=True)
class ShortHorizonDurationThresholdRow:
    measurement_sigma: float
    first_time_at_1sigma: float | None
    first_time_at_2sigma: float | None


@dataclass(frozen=True, slots=True)
class ShortHorizonIdentifiabilityResult:
    nominal_measurement_sigma: float
    times: tuple[ShortHorizonTimeRow, ...]
    noise_sweep: tuple[ShortHorizonNoiseRow, ...]
    duration_thresholds: tuple[ShortHorizonDurationThresholdRow, ...]


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


def _position_at_time(class_name: str, time: float, scenario_name: str) -> float:
    velocity0, acceleration = _scenario_dynamics(scenario_name, class_name)
    return velocity0 * time + 0.5 * acceleration * time * time


def analyze_short_horizon_identifiability() -> ShortHorizonIdentifiabilityResult:
    scenario_name = "short_noisy"
    times = SCENARIO_TIMES[scenario_name]
    nominal_sigma = 0.28

    time_rows: list[ShortHorizonTimeRow] = []
    raw_gaps: list[float] = []
    for time in times:
        cv_position = _position_at_time("constant_velocity", time, scenario_name)
        ca_position = _position_at_time("constant_acceleration", time, scenario_name)
        absolute_gap = abs(ca_position - cv_position)
        raw_gaps.append(absolute_gap)
        time_rows.append(
            ShortHorizonTimeRow(
                time=time,
                constant_velocity_position=cv_position,
                constant_acceleration_position=ca_position,
                absolute_gap=absolute_gap,
                normalized_gap_at_nominal_noise=absolute_gap / nominal_sigma,
            )
        )

    sigma_values = (0.10, 0.16, 0.22, 0.28, 0.34, 0.40, 0.50)
    noise_sweep_rows: list[ShortHorizonNoiseRow] = []
    for sigma in sigma_values:
        normalized = [gap / sigma for gap in raw_gaps]
        noise_sweep_rows.append(
            ShortHorizonNoiseRow(
                measurement_sigma=sigma,
                mean_normalized_gap=sum(normalized) / len(normalized),
                max_normalized_gap=max(normalized),
                final_step_normalized_gap=normalized[-1],
            )
        )
    duration_times = tuple(0.5 * step for step in range(13))
    duration_gaps = [
        abs(
            _position_at_time("constant_acceleration", time, scenario_name)
            - _position_at_time("constant_velocity", time, scenario_name)
        )
        for time in duration_times
    ]
    duration_threshold_rows: list[ShortHorizonDurationThresholdRow] = []
    for sigma in sigma_values:
        normalized = [gap / sigma for gap in duration_gaps]
        first_1sigma = next((time for time, value in zip(duration_times, normalized) if value >= 1.0), None)
        first_2sigma = next((time for time, value in zip(duration_times, normalized) if value >= 2.0), None)
        duration_threshold_rows.append(
            ShortHorizonDurationThresholdRow(
                measurement_sigma=sigma,
                first_time_at_1sigma=first_1sigma,
                first_time_at_2sigma=first_2sigma,
            )
        )

    return ShortHorizonIdentifiabilityResult(
        nominal_measurement_sigma=nominal_sigma,
        times=tuple(time_rows),
        noise_sweep=tuple(noise_sweep_rows),
        duration_thresholds=tuple(duration_threshold_rows),
    )


def render_short_horizon_identifiability_report(result: ShortHorizonIdentifiabilityResult) -> str:
    final_row = result.times[-1]
    nominal_row = next(row for row in result.noise_sweep if abs(row.measurement_sigma - result.nominal_measurement_sigma) < 1e-9)
    nominal_duration_row = next(
        row for row in result.duration_thresholds if abs(row.measurement_sigma - result.nominal_measurement_sigma) < 1e-9
    )
    lines = [
        "# Short-Horizon Identifiability",
        "",
        "This artifact makes the `short_noisy` limit explicit for constant velocity versus constant acceleration over the shared 4-sample horizon.",
        "",
        "## Key Numbers",
        "",
        f"- Nominal measurement sigma: `{result.nominal_measurement_sigma:.2f}`",
        f"- Final-step absolute position gap: `{final_row.absolute_gap:.3f}`",
        f"- Final-step normalized gap at nominal noise: `{final_row.normalized_gap_at_nominal_noise:.3f}`",
        f"- Mean normalized gap at nominal noise: `{nominal_row.mean_normalized_gap:.3f}`",
        f"- First time to reach 1 sigma at nominal noise: `{nominal_duration_row.first_time_at_1sigma}`",
        f"- First time to reach 2 sigma at nominal noise: `{nominal_duration_row.first_time_at_2sigma}`",
        "",
        "## Interpretation",
        "",
        "- The Kalman bank already infers latent velocity and acceleration internally.",
        "- This study asks a simpler identifiability question: how far apart are the two class-predicted position sequences, relative to the measurement noise, over the short noisy horizon?",
        "- When the normalized gap stays near or below one sigma for much of the sequence, no filter can separate the classes confidently from position alone without becoming prior- or model-dominated.",
        "- If we want to improve `short_noisy`, we likely need either more elapsed time, lower noise, or an actually stronger measurement stream rather than more reuse of the same positions.",
    ]
    return "\n".join(lines)


def _render_time_plot(result: ShortHorizonIdentifiabilityResult):
    plt = _prepare_matplotlib()
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
    plt = _prepare_matplotlib()
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
    plt = _prepare_matplotlib()
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
    analysis = result or analyze_short_horizon_identifiability()

    report_path = run_dir / "short_horizon_identifiability_report.md"
    time_series_path = run_dir / "short_horizon_time_series.csv"
    noise_sweep_path = run_dir / "short_horizon_noise_sweep.csv"
    duration_thresholds_path = run_dir / "short_horizon_duration_thresholds.csv"
    time_plot_png_path = run_dir / "short_horizon_time_series.png"
    noise_plot_png_path = run_dir / "short_horizon_noise_sweep.png"
    duration_plot_png_path = run_dir / "short_horizon_duration_thresholds.png"

    report_path.write_text(render_short_horizon_identifiability_report(analysis), encoding="utf-8")
    _write_csv(
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
    _write_csv(
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
    _write_csv(
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
    time_figure.savefig(time_plot_png_path, format="png", dpi=160, bbox_inches="tight")
    time_figure.clf()

    noise_figure = _render_noise_plot(analysis)
    noise_figure.savefig(noise_plot_png_path, format="png", dpi=160, bbox_inches="tight")
    noise_figure.clf()

    duration_figure = _render_duration_plot(analysis)
    duration_figure.savefig(duration_plot_png_path, format="png", dpi=160, bbox_inches="tight")
    duration_figure.clf()

    plt = _prepare_matplotlib()
    plt.close(time_figure)
    plt.close(noise_figure)
    plt.close(duration_figure)

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
