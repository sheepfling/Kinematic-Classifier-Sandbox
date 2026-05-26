from __future__ import annotations

from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png

from .short_horizon_identifiability_contracts import (
    ShortHorizonIdentifiabilityArtifacts,
    ShortHorizonIdentifiabilityResult,
)
from .short_horizon_identifiability_rendering import (
    _render_duration_plot,
    _render_noise_plot,
    _render_time_plot,
    render_short_horizon_identifiability_report,
)


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

    noise_figure = _render_noise_plot(analysis)
    try:
        noise_plot_png_path.write_bytes(_figure_to_png(noise_figure))
    finally:
        noise_figure.clf()

    duration_figure = _render_duration_plot(analysis)
    try:
        duration_plot_png_path.write_bytes(_figure_to_png(duration_figure))
    finally:
        duration_figure.clf()

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
