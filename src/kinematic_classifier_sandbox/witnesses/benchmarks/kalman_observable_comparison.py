from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ..analysis.common_dataset_comparison import (
    SharedDynamicsTrajectory,
    generate_shared_dynamics_dataset,
)
from ..markdown_builder import MarkdownDocument
from ..utils.plotting import plt
from .kalman_filter_bank import (
    KalmanClassificationRun,
    KalmanModelSpec,
    KalmanTrajectory,
    run_kalman_filter_bank,
)


@dataclass(frozen=True, slots=True)
class KalmanObservableRow:
    observable_mode: str
    overall_accuracy: float
    endpoint_match_accuracy: float
    short_accuracy: float
    short_noisy_accuracy: float
    outlier_accuracy: float


@dataclass(frozen=True, slots=True)
class KalmanObservableTrace:
    observable_mode: str
    trajectory_id: str
    scenario_name: str
    true_class: str
    final_predicted_class: str
    final_confidence: float
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_position: tuple[float, ...]
    true_class_posterior: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class KalmanObservableComparisonResult:
    rows: tuple[KalmanObservableRow, ...]
    traces: tuple[KalmanObservableTrace, ...]


@dataclass(frozen=True, slots=True)
class KalmanObservableComparisonArtifacts:
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


def _observable_flags(observable_mode: str) -> tuple[bool, bool]:
    if observable_mode == "position_only":
        return False, False
    if observable_mode == "position_plus_velocity":
        return True, False
    if observable_mode == "position_plus_velocity_acceleration":
        return True, True
    raise ValueError(f"Unsupported observable mode: {observable_mode}")


def _run_observable_mode(
    trajectory: SharedDynamicsTrajectory,
    *,
    observable_mode: str,
) -> KalmanClassificationRun:
    derived_velocity_observation, derived_acceleration_observation = _observable_flags(observable_mode)
    return run_kalman_filter_bank(
        _shared_kalman_trajectory(trajectory),
        _kalman_shared_model_specs(),
        robust_measurement_update=True,
        adaptive_process_noise=True,
        derived_velocity_observation=derived_velocity_observation,
        derived_acceleration_observation=derived_acceleration_observation,
    )


def _observable_accuracy(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    observable_mode: str,
) -> KalmanObservableRow:
    runs = [_run_observable_mode(trajectory, observable_mode=observable_mode) for trajectory in trajectories]

    def _accuracy(scenario_name: str | None = None) -> float:
        selected = [
            run for run, trajectory in zip(runs, trajectories)
            if scenario_name is None or trajectory.scenario_name == scenario_name
        ]
        return sum(1.0 if run.final_predicted_class == run.true_class else 0.0 for run in selected) / len(selected)

    return KalmanObservableRow(
        observable_mode=observable_mode,
        overall_accuracy=_accuracy(),
        endpoint_match_accuracy=_accuracy("endpoint_match"),
        short_accuracy=_accuracy("short"),
        short_noisy_accuracy=_accuracy("short_noisy"),
        outlier_accuracy=_accuracy("outlier"),
    )


def _build_trace(
    trajectory: SharedDynamicsTrajectory,
    run: KalmanClassificationRun,
    *,
    observable_mode: str,
) -> KalmanObservableTrace:
    return KalmanObservableTrace(
        observable_mode=observable_mode,
        trajectory_id=trajectory.trajectory_id,
        scenario_name=trajectory.scenario_name,
        true_class=trajectory.true_class,
        final_predicted_class=run.final_predicted_class,
        final_confidence=run.final_confidence,
        times=trajectory.times,
        measurements=trajectory.measurements,
        true_position=trajectory.true_position,
        true_class_posterior=tuple(step.posterior_weights[trajectory.true_class] for step in run.steps),
    )


def analyze_kalman_observable_comparison(*, seed: int = 7, trajectories_per_case: int = 8) -> KalmanObservableComparisonResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    observable_modes = (
        "position_only",
        "position_plus_velocity",
        "position_plus_velocity_acceleration",
    )
    rows = tuple(
        _observable_accuracy(
            trajectories,
            observable_mode=observable_mode,
        )
        for observable_mode in observable_modes
    )
    representative_trajectories = []
    for scenario_name, class_name in (
        ("endpoint_match", "constant_acceleration"),
        ("short_noisy", "constant_acceleration"),
        ("outlier", "constant_velocity"),
    ):
        representative_trajectories.append(
            next(
                trajectory
                for trajectory in trajectories
                if trajectory.scenario_name == scenario_name and trajectory.true_class == class_name
            )
        )
    traces: list[KalmanObservableTrace] = []
    for trajectory in representative_trajectories:
        for observable_mode in observable_modes:
            run = _run_observable_mode(trajectory, observable_mode=observable_mode)
            traces.append(_build_trace(trajectory, run, observable_mode=observable_mode))
    return KalmanObservableComparisonResult(rows=rows, traces=tuple(traces))


def render_kalman_observable_comparison_report(result: KalmanObservableComparisonResult) -> str:
    report = MarkdownDocument("Kalman Observable Comparison")
    report.paragraph(
        "This artifact compares three observable stacks for the same robust/adaptive Kalman family on the shared dynamics corpus."
    )
    report.table(
        ["observable_mode", "overall", "endpoint_match", "short", "short_noisy", "outlier"],
        [
            (
                row.observable_mode,
                f"{row.overall_accuracy:.3f}",
                f"{row.endpoint_match_accuracy:.3f}",
                f"{row.short_accuracy:.3f}",
                f"{row.short_noisy_accuracy:.3f}",
                f"{row.outlier_accuracy:.3f}",
            )
            for row in result.rows
        ],
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "`position_only` uses the original scalar position measurement only.",
            "`position_plus_velocity` adds a tempered local-slope pseudo-observation from the last 3 samples.",
            "`position_plus_velocity_acceleration` also adds a tempered local-curvature pseudo-observation from the last 3 samples.",
            "These pseudo-observations are derived from the same raw measurements, so their likelihood contributions are intentionally downweighted.",
        ]
    )
    return report.text()


def _render_heatmap(result: KalmanObservableComparisonResult):
    fields = (
        "overall_accuracy",
        "endpoint_match_accuracy",
        "short_accuracy",
        "short_noisy_accuracy",
        "outlier_accuracy",
    )
    matrix = [[getattr(row, field) for field in fields] for row in result.rows]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Kalman Observable Comparison", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(["overall", "endpoint", "short", "short_noisy", "outlier"], rotation=20, ha="right")
    ax.set_yticks(range(len(result.rows)))
    ax.set_yticklabels([row.observable_mode for row in result.rows])
    for row_index, row in enumerate(result.rows):
        for col_index, field in enumerate(fields):
            ax.text(col_index, row_index, f"{getattr(row, field):.2f}", ha="center", va="center", fontsize=9, color="#0f172a")
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    fig.tight_layout()
    return fig


def _render_diagnostics(result: KalmanObservableComparisonResult):
    scenario_names = ("endpoint_match", "short_noisy", "outlier")
    fig, axes = plt.subplots(len(scenario_names), 1, figsize=(9.6, 8.8), sharex=False)
    color_map = {
        "position_only": "#2563eb",
        "position_plus_velocity": "#d97706",
        "position_plus_velocity_acceleration": "#dc2626",
    }
    for axis, scenario_name in zip(axes, scenario_names):
        scenario_traces = [trace for trace in result.traces if trace.scenario_name == scenario_name]
        for trace in scenario_traces:
            axis.plot(
                trace.times,
                trace.true_class_posterior,
                label=trace.observable_mode,
                linewidth=2.0,
                color=color_map[trace.observable_mode],
            )
        example = scenario_traces[0]
        axis.scatter(example.times, example.measurements, color="#111827", s=14, alpha=0.55, label="measurements")
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("true-class posterior")
        axis.set_title(
            f"{scenario_name} ({example.true_class})",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[-1].set_xlabel("time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return fig


def write_kalman_observable_comparison_artifacts(
    output_root: str | Path,
    *,
    result: KalmanObservableComparisonResult | None = None,
) -> KalmanObservableComparisonArtifacts:
    base_path = Path(output_root)
    run_dir = base_path / "kalman_observable_comparison_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    comparison = result or analyze_kalman_observable_comparison()

    report_path = run_dir / "kalman_observable_comparison_report.md"
    summary_csv_path = run_dir / "kalman_observable_summary.csv"
    trace_csv_path = run_dir / "kalman_observable_trace_summary.csv"
    heatmap_png_path = run_dir / "kalman_observable_heatmap.png"
    diagnostics_png_path = run_dir / "kalman_observable_diagnostics.png"

    report_path.write_text(render_kalman_observable_comparison_report(comparison), encoding="utf-8")
    write_csv(
        summary_csv_path,
        [
            {
                "observable_mode": row.observable_mode,
                "overall_accuracy": row.overall_accuracy,
                "endpoint_match_accuracy": row.endpoint_match_accuracy,
                "short_accuracy": row.short_accuracy,
                "short_noisy_accuracy": row.short_noisy_accuracy,
                "outlier_accuracy": row.outlier_accuracy,
            }
            for row in comparison.rows
        ],
        [
            "observable_mode",
            "overall_accuracy",
            "endpoint_match_accuracy",
            "short_accuracy",
            "short_noisy_accuracy",
            "outlier_accuracy",
        ],
    )
    write_csv(
        trace_csv_path,
        [
            {
                "observable_mode": trace.observable_mode,
                "trajectory_id": trace.trajectory_id,
                "scenario_name": trace.scenario_name,
                "true_class": trace.true_class,
                "final_predicted_class": trace.final_predicted_class,
                "final_confidence": trace.final_confidence,
                "times": " ".join(f"{value:.3f}" for value in trace.times),
                "measurements": " ".join(f"{value:.3f}" for value in trace.measurements),
                "true_position": " ".join(f"{value:.3f}" for value in trace.true_position),
                "true_class_posterior": " ".join(f"{value:.3f}" for value in trace.true_class_posterior),
            }
            for trace in comparison.traces
        ],
        [
            "observable_mode",
            "trajectory_id",
            "scenario_name",
            "true_class",
            "final_predicted_class",
            "final_confidence",
            "times",
            "measurements",
            "true_position",
            "true_class_posterior",
        ],
    )

    heatmap_figure = _render_heatmap(comparison)
    heatmap_figure.savefig(heatmap_png_path, format="png", dpi=160, bbox_inches="tight")
    heatmap_figure.clf()
    plt.close(heatmap_figure)

    diagnostics_figure = _render_diagnostics(comparison)
    diagnostics_figure.savefig(diagnostics_png_path, format="png", dpi=160, bbox_inches="tight")
    diagnostics_figure.clf()
    plt.close(diagnostics_figure)

    return KalmanObservableComparisonArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        trace_csv_path=trace_csv_path,
        heatmap_png_path=heatmap_png_path,
        diagnostics_png_path=diagnostics_png_path,
    )
