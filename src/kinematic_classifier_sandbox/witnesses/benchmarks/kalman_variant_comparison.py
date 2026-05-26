from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png

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
class KalmanVariantRow:
    variant_name: str
    overall_accuracy: float
    endpoint_match_accuracy: float
    short_accuracy: float
    short_noisy_accuracy: float
    outlier_accuracy: float


@dataclass(frozen=True, slots=True)
class KalmanVariantScenarioTrace:
    variant_name: str
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
class KalmanVariantComparisonResult:
    rows: tuple[KalmanVariantRow, ...]
    traces: tuple[KalmanVariantScenarioTrace, ...]


@dataclass(frozen=True, slots=True)
class KalmanVariantComparisonArtifacts:
    run_dir: Path
    report_path: Path
    summary_csv_path: Path
    trace_csv_path: Path
    heatmap_png_path: Path
    diagnostics_png_path: Path
    delta_png_path: Path


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


def _variant_accuracy(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    robust_measurement_update: bool,
    adaptive_process_noise: bool,
) -> KalmanVariantRow:
    model_specs = _kalman_shared_model_specs()
    runs = [
        run_kalman_filter_bank(
            _shared_kalman_trajectory(trajectory),
            model_specs,
            robust_measurement_update=robust_measurement_update,
            adaptive_process_noise=adaptive_process_noise,
            derived_velocity_observation=True,
            derived_acceleration_observation=True,
        )
        for trajectory in trajectories
    ]

    def _accuracy(scenario_name: str | None = None) -> float:
        selected = [
            run for run, trajectory in zip(runs, trajectories)
            if scenario_name is None or trajectory.scenario_name == scenario_name
        ]
        return sum(1.0 if run.final_predicted_class == run.true_class else 0.0 for run in selected) / len(selected)

    name = _variant_name(
        robust_measurement_update=robust_measurement_update,
        adaptive_process_noise=adaptive_process_noise,
    )
    return KalmanVariantRow(
        variant_name=name,
        overall_accuracy=_accuracy(),
        endpoint_match_accuracy=_accuracy("endpoint_match"),
        short_accuracy=_accuracy("short"),
        short_noisy_accuracy=_accuracy("short_noisy"),
        outlier_accuracy=_accuracy("outlier"),
    )


def _variant_name(*, robust_measurement_update: bool, adaptive_process_noise: bool) -> str:
    if not robust_measurement_update and not adaptive_process_noise:
        return "plain"
    if robust_measurement_update and not adaptive_process_noise:
        return "robust_measurement"
    return "robust_plus_adaptive_process"


def _run_variant(
    trajectory: SharedDynamicsTrajectory,
    *,
    robust_measurement_update: bool,
    adaptive_process_noise: bool,
) -> KalmanClassificationRun:
    return run_kalman_filter_bank(
        _shared_kalman_trajectory(trajectory),
        _kalman_shared_model_specs(),
        robust_measurement_update=robust_measurement_update,
        adaptive_process_noise=adaptive_process_noise,
        derived_velocity_observation=True,
        derived_acceleration_observation=True,
    )


def _build_trace(
    trajectory: SharedDynamicsTrajectory,
    run: KalmanClassificationRun,
    *,
    variant_name: str,
) -> KalmanVariantScenarioTrace:
    return KalmanVariantScenarioTrace(
        variant_name=variant_name,
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


def analyze_kalman_variant_comparison(*, seed: int = 7, trajectories_per_case: int = 8) -> KalmanVariantComparisonResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    rows = (
        _variant_accuracy(
            trajectories,
            robust_measurement_update=False,
            adaptive_process_noise=False,
        ),
        _variant_accuracy(
            trajectories,
            robust_measurement_update=True,
            adaptive_process_noise=False,
        ),
        _variant_accuracy(
            trajectories,
            robust_measurement_update=True,
            adaptive_process_noise=True,
        ),
    )
    representative_trajectories = []
    for scenario_name, class_name in (("short_noisy", "constant_acceleration"), ("outlier", "constant_velocity")):
        representative_trajectories.append(
            next(
                trajectory
                for trajectory in trajectories
                if trajectory.scenario_name == scenario_name and trajectory.true_class == class_name
            )
        )
    traces: list[KalmanVariantScenarioTrace] = []
    for trajectory in representative_trajectories:
        for robust_measurement_update, adaptive_process_noise in (
            (False, False),
            (True, False),
            (True, True),
        ):
            name = _variant_name(
                robust_measurement_update=robust_measurement_update,
                adaptive_process_noise=adaptive_process_noise,
            )
            run = _run_variant(
                trajectory,
                robust_measurement_update=robust_measurement_update,
                adaptive_process_noise=adaptive_process_noise,
            )
            traces.append(_build_trace(trajectory, run, variant_name=name))
    return KalmanVariantComparisonResult(rows=rows, traces=tuple(traces))


def render_kalman_variant_comparison_report(result: KalmanVariantComparisonResult) -> str:
    report = MarkdownDocument("Kalman Variant Comparison")
    report.paragraph(
        "This artifact compares three Kalman-family variants on the shared dynamics corpus: plain, "
        "robust-measurement-only, and robust-plus-adaptive-process."
    )
    report.table(
        ["variant", "overall", "endpoint_match", "short", "short_noisy", "outlier"],
        [
            (
                row.variant_name,
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
            "`plain` isolates the baseline linear-Gaussian bank.",
            "`robust_measurement` adds innovation-based measurement variance inflation only.",
            "`robust_plus_adaptive_process` also adapts process noise over time from repeated innovation energy.",
        ]
    )
    return report.text()


def _render_heatmap(result: KalmanVariantComparisonResult):
    fields = ("overall_accuracy", "endpoint_match_accuracy", "short_accuracy", "short_noisy_accuracy", "outlier_accuracy")
    matrix = [[getattr(row, field) for field in fields] for row in result.rows]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Kalman Variant Comparison", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(["overall", "endpoint", "short", "short_noisy", "outlier"], rotation=20, ha="right")
    ax.set_yticks(range(len(result.rows)))
    ax.set_yticklabels([row.variant_name for row in result.rows])
    for row_index, row in enumerate(result.rows):
        for col_index, field in enumerate(fields):
            ax.text(col_index, row_index, f"{getattr(row, field):.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="accuracy")
    fig.tight_layout()
    return fig


def _render_diagnostics(result: KalmanVariantComparisonResult):
    scenario_names = ["short_noisy", "outlier"]
    variant_order = ["plain", "robust_measurement", "robust_plus_adaptive_process"]
    colors = {
        "plain": "#2563eb",
        "robust_measurement": "#d97706",
        "robust_plus_adaptive_process": "#16a34a",
    }
    traces = {(trace.scenario_name, trace.variant_name): trace for trace in result.traces}
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=False)
    for row_index, scenario_name in enumerate(scenario_names):
        measurement_ax, posterior_ax = axes[row_index]
        reference = traces[(scenario_name, "plain")]
        measurement_ax.plot(reference.times, reference.measurements, color="#111827", linewidth=2.0, marker="o", label="measurement")
        measurement_ax.plot(reference.times, reference.true_position, color="#9ca3af", linewidth=1.6, linestyle="--", label="true position")
        measurement_ax.set_title(f"{scenario_name} measurement", loc="left", fontsize=12, fontweight="bold")
        measurement_ax.grid(True, alpha=0.25)
        measurement_ax.legend(frameon=False, fontsize=8)
        for variant_name in variant_order:
            trace = traces[(scenario_name, variant_name)]
            posterior_ax.plot(
                list(range(len(trace.true_class_posterior))),
                trace.true_class_posterior,
                color=colors[variant_name],
                linewidth=2.1,
                label=variant_name,
            )
        posterior_ax.set_ylim(0.0, 1.0)
        posterior_ax.set_title(f"{scenario_name} true-class posterior", loc="left", fontsize=12, fontweight="bold")
        posterior_ax.grid(True, alpha=0.25)
        posterior_ax.legend(frameon=False, fontsize=8)
        measurement_ax.set_ylabel("position")
        posterior_ax.set_ylabel("posterior")
        posterior_ax.set_xlabel("step")
    fig.suptitle("Kalman Variant Scenario Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _render_delta_vs_plain(result: KalmanVariantComparisonResult):
    fields = [
        ("endpoint_match_accuracy", "endpoint"),
        ("short_accuracy", "short"),
        ("short_noisy_accuracy", "short+noise"),
        ("outlier_accuracy", "outlier"),
    ]
    baseline = next(row for row in result.rows if row.variant_name == "plain")
    variants = [row for row in result.rows if row.variant_name != "plain"]
    palette = {
        "robust_measurement": "#d97706",
        "robust_plus_adaptive_process": "#16a34a",
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    x = list(range(len(fields)))
    width = 0.34
    for offset, row in zip((-width / 2, width / 2), variants):
        deltas = [getattr(row, field) - getattr(baseline, field) for field, _ in fields]
        ax.bar(
            [value + offset for value in x],
            deltas,
            width=width,
            color=palette[row.variant_name],
            label=row.variant_name,
        )
        for xpos, delta in zip([value + offset for value in x], deltas):
            ax.text(xpos, delta + (0.015 if delta >= 0.0 else -0.03), f"{delta:+.2f}", ha="center", va="bottom" if delta >= 0.0 else "top", fontsize=8)
    ax.axhline(0.0, color="#6b7280", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in fields], rotation=15, ha="right")
    ax.set_ylabel("accuracy delta vs plain")
    ax.set_title("Kalman Variant Gains vs Plain Baseline", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _figure_to_svg(fig) -> str:
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_kalman_variant_comparison_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    result: KalmanVariantComparisonResult | None = None,
) -> KalmanVariantComparisonArtifacts:
    comparison = result or analyze_kalman_variant_comparison(seed=seed)
    output_root = Path(output_dir)
    run_dir = output_root / "kalman_variant_comparison_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "kalman_variant_comparison_report.md"
    summary_csv_path = run_dir / "kalman_variant_summary.csv"
    trace_csv_path = run_dir / "kalman_variant_trace_summary.csv"
    heatmap_png_path = run_dir / "kalman_variant_heatmap.png"
    diagnostics_png_path = run_dir / "kalman_variant_diagnostics.png"
    delta_png_path = run_dir / "kalman_variant_delta_vs_plain.png"

    report_path.write_text(render_kalman_variant_comparison_report(comparison), encoding="utf-8")
    write_csv(
        summary_csv_path,
        [
            {
                "variant_name": row.variant_name,
                "overall_accuracy": row.overall_accuracy,
                "endpoint_match_accuracy": row.endpoint_match_accuracy,
                "short_accuracy": row.short_accuracy,
                "short_noisy_accuracy": row.short_noisy_accuracy,
                "outlier_accuracy": row.outlier_accuracy,
            }
            for row in comparison.rows
        ],
        [
            "variant_name",
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
                "variant_name": trace.variant_name,
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
            "variant_name",
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
    heatmap_png_path.write_bytes(_figure_to_png(_render_heatmap(comparison)))
    diagnostics_png_path.write_bytes(_figure_to_png(_render_diagnostics(comparison)))
    delta_png_path.write_bytes(_figure_to_png(_render_delta_vs_plain(comparison)))
    return KalmanVariantComparisonArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        trace_csv_path=trace_csv_path,
        heatmap_png_path=heatmap_png_path,
        diagnostics_png_path=diagnostics_png_path,
        delta_png_path=delta_png_path,
    )
