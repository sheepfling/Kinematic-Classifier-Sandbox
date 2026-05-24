from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path
import random

from .common_dataset_comparison import SharedDynamicsTrajectory, generate_shared_dynamics_dataset
from .kalman_filter_bank import KalmanClassificationRun, KalmanModelSpec, KalmanTrajectory, run_kalman_filter_bank


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
class VelocityAidedRow:
    measurement_mode: str
    overall_accuracy: float
    endpoint_match_accuracy: float
    short_accuracy: float
    short_noisy_accuracy: float
    outlier_accuracy: float


@dataclass(frozen=True, slots=True)
class VelocityAidedTrace:
    measurement_mode: str
    trajectory_id: str
    scenario_name: str
    true_class: str
    final_predicted_class: str
    final_confidence: float
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    velocity_measurements: tuple[float, ...]
    true_class_posterior: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VelocityAidedComparisonResult:
    rows: tuple[VelocityAidedRow, ...]
    traces: tuple[VelocityAidedTrace, ...]


@dataclass(frozen=True, slots=True)
class VelocityAidedComparisonArtifacts:
    run_dir: Path
    report_path: Path
    summary_csv_path: Path
    trace_csv_path: Path
    heatmap_png_path: Path
    diagnostics_png_path: Path


def _shared_kalman_trajectory(trajectory: SharedDynamicsTrajectory) -> KalmanTrajectory:
    return KalmanTrajectory(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        seed=trajectory.seed,
        times=trajectory.times,
        measurements=trajectory.measurements,
        true_position=trajectory.true_position,
        true_velocity=trajectory.true_velocity,
        true_acceleration=trajectory.true_acceleration,
    )


def _kalman_shared_model_specs() -> tuple[KalmanModelSpec, ...]:
    return (
        KalmanModelSpec(
            name="constant_velocity_quiet",
            class_name="constant_velocity",
            state_dim=2,
            process_sigma=0.14,
            measurement_sigma=0.20,
            initial_covariance_scale=5.0,
            prior_weight=0.375,
        ),
        KalmanModelSpec(
            name="constant_velocity_rough",
            class_name="constant_velocity",
            state_dim=2,
            process_sigma=0.22,
            measurement_sigma=0.20,
            initial_covariance_scale=5.5,
            prior_weight=0.125,
        ),
        KalmanModelSpec(
            name="constant_acceleration_quiet",
            class_name="constant_acceleration",
            state_dim=3,
            process_sigma=0.24,
            measurement_sigma=0.20,
            initial_covariance_scale=6.0,
            prior_weight=0.375,
        ),
        KalmanModelSpec(
            name="constant_acceleration_rough",
            class_name="constant_acceleration",
            state_dim=3,
            process_sigma=0.34,
            measurement_sigma=0.20,
            initial_covariance_scale=6.5,
            prior_weight=0.125,
        ),
    )


def _synthesized_velocity_measurements(
    trajectory: SharedDynamicsTrajectory,
    *,
    sigma: float,
) -> tuple[float, ...]:
    rng = random.Random(trajectory.seed + 9001)
    return tuple(value + rng.gauss(0.0, sigma) for value in trajectory.true_velocity)


def _run_mode(
    trajectory: SharedDynamicsTrajectory,
    *,
    measurement_mode: str,
) -> tuple[KalmanClassificationRun, tuple[float, ...]]:
    velocity_sigma = 0.12
    velocity_measurements = _synthesized_velocity_measurements(trajectory, sigma=velocity_sigma)
    kwargs = {
        "robust_measurement_update": True,
        "adaptive_process_noise": True,
        "derived_velocity_observation": False,
        "derived_acceleration_observation": False,
    }
    if measurement_mode == "position_only":
        velocity_measurements_used = tuple(0.0 for _ in trajectory.times)
    elif measurement_mode == "position_plus_direct_velocity":
        kwargs["velocity_measurements"] = velocity_measurements
        kwargs["velocity_measurement_sigma"] = velocity_sigma
        velocity_measurements_used = velocity_measurements
    else:
        raise ValueError(f"Unsupported measurement mode: {measurement_mode}")
    run = run_kalman_filter_bank(
        _shared_kalman_trajectory(trajectory),
        _kalman_shared_model_specs(),
        **kwargs,
    )
    return run, velocity_measurements_used


def _mode_accuracy(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    measurement_mode: str,
) -> VelocityAidedRow:
    runs = [_run_mode(trajectory, measurement_mode=measurement_mode)[0] for trajectory in trajectories]

    def _accuracy(scenario_name: str | None = None) -> float:
        selected = [
            run for run, trajectory in zip(runs, trajectories)
            if scenario_name is None or trajectory.scenario_name == scenario_name
        ]
        return sum(1.0 if run.final_predicted_class == run.true_class else 0.0 for run in selected) / len(selected)

    return VelocityAidedRow(
        measurement_mode=measurement_mode,
        overall_accuracy=_accuracy(),
        endpoint_match_accuracy=_accuracy("endpoint_match"),
        short_accuracy=_accuracy("short"),
        short_noisy_accuracy=_accuracy("short_noisy"),
        outlier_accuracy=_accuracy("outlier"),
    )


def analyze_velocity_aided_kalman_comparison(*, seed: int = 7, trajectories_per_case: int = 8) -> VelocityAidedComparisonResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    measurement_modes = ("position_only", "position_plus_direct_velocity")
    rows = tuple(_mode_accuracy(trajectories, measurement_mode=mode) for mode in measurement_modes)
    representative_trajectories = []
    for scenario_name, class_name in (
        ("short_noisy", "constant_acceleration"),
        ("endpoint_match", "constant_acceleration"),
        ("outlier", "constant_velocity"),
    ):
        representative_trajectories.append(
            next(
                trajectory
                for trajectory in trajectories
                if trajectory.scenario_name == scenario_name and trajectory.true_class == class_name
            )
        )
    traces: list[VelocityAidedTrace] = []
    for trajectory in representative_trajectories:
        for mode in measurement_modes:
            run, velocity_measurements = _run_mode(trajectory, measurement_mode=mode)
            traces.append(
                VelocityAidedTrace(
                    measurement_mode=mode,
                    trajectory_id=trajectory.trajectory_id,
                    scenario_name=trajectory.scenario_name,
                    true_class=trajectory.true_class,
                    final_predicted_class=run.final_predicted_class,
                    final_confidence=run.final_confidence,
                    times=trajectory.times,
                    measurements=trajectory.measurements,
                    velocity_measurements=velocity_measurements,
                    true_class_posterior=tuple(step.posterior_weights[trajectory.true_class] for step in run.steps),
                )
            )
    return VelocityAidedComparisonResult(rows=rows, traces=tuple(traces))


def render_velocity_aided_kalman_comparison_report(result: VelocityAidedComparisonResult) -> str:
    lines = [
        "# Velocity-Aided Kalman Comparison",
        "",
        "This artifact compares the same robust/adaptive Kalman family under two sensor stacks on the shared corpus.",
        "",
        "| measurement_mode | overall | endpoint_match | short | short_noisy | outlier |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.rows:
        lines.append(
            f"| {row.measurement_mode} | {row.overall_accuracy:.3f} | {row.endpoint_match_accuracy:.3f} | {row.short_accuracy:.3f} | {row.short_noisy_accuracy:.3f} | {row.outlier_accuracy:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `position_only` is the baseline position-measurement Kalman bank.",
            "- `position_plus_direct_velocity` adds an actual velocity sensor stream rather than a pseudo-observation derived from the same positions.",
            "- This isolates the value of genuinely stronger sensing from cleverer reuse of the same data.",
        ]
    )
    return "\n".join(lines)


def _render_heatmap(result: VelocityAidedComparisonResult):
    plt = _prepare_matplotlib()
    fields = ("overall_accuracy", "endpoint_match_accuracy", "short_accuracy", "short_noisy_accuracy", "outlier_accuracy")
    matrix = [[getattr(row, field) for field in fields] for row in result.rows]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Velocity-Aided Kalman Comparison", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(["overall", "endpoint", "short", "short_noisy", "outlier"], rotation=20, ha="right")
    ax.set_yticks(range(len(result.rows)))
    ax.set_yticklabels([row.measurement_mode for row in result.rows])
    for row_index, row in enumerate(result.rows):
        for col_index, field in enumerate(fields):
            ax.text(col_index, row_index, f"{getattr(row, field):.2f}", ha="center", va="center", fontsize=9, color="#0f172a")
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    fig.tight_layout()
    return fig


def _render_diagnostics(result: VelocityAidedComparisonResult):
    plt = _prepare_matplotlib()
    scenario_names = ("short_noisy", "endpoint_match", "outlier")
    fig, axes = plt.subplots(len(scenario_names), 1, figsize=(9.6, 8.8), sharex=False)
    color_map = {
        "position_only": "#2563eb",
        "position_plus_direct_velocity": "#dc2626",
    }
    for axis, scenario_name in zip(axes, scenario_names):
        scenario_traces = [trace for trace in result.traces if trace.scenario_name == scenario_name]
        for trace in scenario_traces:
            axis.plot(
                trace.times,
                trace.true_class_posterior,
                label=trace.measurement_mode,
                linewidth=2.0,
                color=color_map[trace.measurement_mode],
            )
        example = scenario_traces[0]
        axis.scatter(example.times, example.measurements, color="#111827", s=14, alpha=0.55, label="position z")
        if any(any(abs(value) > 1e-9 for value in trace.velocity_measurements) for trace in scenario_traces):
            axis2 = axis.twinx()
            velocity_trace = next(trace for trace in scenario_traces if trace.measurement_mode == "position_plus_direct_velocity")
            axis2.plot(
                velocity_trace.times,
                velocity_trace.velocity_measurements,
                color="#16a34a",
                linestyle="--",
                linewidth=1.6,
                alpha=0.8,
                label="velocity z",
            )
            axis2.set_ylabel("velocity")
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("true-class posterior")
        axis.set_title(f"{scenario_name} ({example.true_class})", loc="left", fontsize=12, fontweight="bold")
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[-1].set_xlabel("time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return fig


def write_velocity_aided_kalman_comparison_artifacts(
    output_root: str | Path,
    *,
    result: VelocityAidedComparisonResult | None = None,
) -> VelocityAidedComparisonArtifacts:
    base_path = Path(output_root)
    run_dir = base_path / "velocity_aided_kalman_comparison_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    comparison = result or analyze_velocity_aided_kalman_comparison()

    report_path = run_dir / "velocity_aided_kalman_comparison_report.md"
    summary_csv_path = run_dir / "velocity_aided_kalman_summary.csv"
    trace_csv_path = run_dir / "velocity_aided_kalman_trace_summary.csv"
    heatmap_png_path = run_dir / "velocity_aided_kalman_heatmap.png"
    diagnostics_png_path = run_dir / "velocity_aided_kalman_diagnostics.png"

    report_path.write_text(render_velocity_aided_kalman_comparison_report(comparison), encoding="utf-8")
    _write_csv(
        summary_csv_path,
        [
            {
                "measurement_mode": row.measurement_mode,
                "overall_accuracy": row.overall_accuracy,
                "endpoint_match_accuracy": row.endpoint_match_accuracy,
                "short_accuracy": row.short_accuracy,
                "short_noisy_accuracy": row.short_noisy_accuracy,
                "outlier_accuracy": row.outlier_accuracy,
            }
            for row in comparison.rows
        ],
        [
            "measurement_mode",
            "overall_accuracy",
            "endpoint_match_accuracy",
            "short_accuracy",
            "short_noisy_accuracy",
            "outlier_accuracy",
        ],
    )
    _write_csv(
        trace_csv_path,
        [
            {
                "measurement_mode": trace.measurement_mode,
                "trajectory_id": trace.trajectory_id,
                "scenario_name": trace.scenario_name,
                "true_class": trace.true_class,
                "final_predicted_class": trace.final_predicted_class,
                "final_confidence": trace.final_confidence,
                "times": " ".join(f"{value:.3f}" for value in trace.times),
                "measurements": " ".join(f"{value:.3f}" for value in trace.measurements),
                "velocity_measurements": " ".join(f"{value:.3f}" for value in trace.velocity_measurements),
                "true_class_posterior": " ".join(f"{value:.3f}" for value in trace.true_class_posterior),
            }
            for trace in comparison.traces
        ],
        [
            "measurement_mode",
            "trajectory_id",
            "scenario_name",
            "true_class",
            "final_predicted_class",
            "final_confidence",
            "times",
            "measurements",
            "velocity_measurements",
            "true_class_posterior",
        ],
    )

    heatmap_figure = _render_heatmap(comparison)
    heatmap_figure.savefig(heatmap_png_path, format="png", dpi=160, bbox_inches="tight")
    heatmap_figure.clf()

    diagnostics_figure = _render_diagnostics(comparison)
    diagnostics_figure.savefig(diagnostics_png_path, format="png", dpi=160, bbox_inches="tight")
    diagnostics_figure.clf()

    plt = _prepare_matplotlib()
    plt.close(heatmap_figure)
    plt.close(diagnostics_figure)

    return VelocityAidedComparisonArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        trace_csv_path=trace_csv_path,
        heatmap_png_path=heatmap_png_path,
        diagnostics_png_path=diagnostics_png_path,
    )
