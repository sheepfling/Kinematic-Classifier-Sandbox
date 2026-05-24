from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Callable

from .common_dataset_comparison import default_shared_classifier_adapters
from .kalman_filter_bank import run_kalman_bank_benchmark
from .velocity_aided_kalman_comparison import analyze_velocity_aided_kalman_comparison
from .pointwise_baseline import run_pointwise_benchmark
from .prior_sensitivity_analysis import (
    analyze_pointwise_prior_sensitivity,
    analyze_prior_sensitivity,
    analyze_windowed_prior_sensitivity,
)
from .sequential_bayes_accumulator import run_accumulator_benchmark
from .windowed_baseline import run_windowed_benchmark


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


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class TechniqueComparisonRow:
    method_name: str
    sensor_regime_id: str
    overall_accuracy: float
    prior_flip_fraction: float
    median_flip_threshold: float | None
    easy_accuracy: float | None
    boundary_accuracy: float | None
    outlier_accuracy: float | None
    transition_accuracy: float | None
    long_history_accuracy: float | None
    irregular_dt_accuracy: float | None
    acceleration_accuracy: float | None
    uses_temporal_history: float
    model_based: float
    irregular_dt_native: float
    outlier_aware: float
    stronger_sensor_stream: float


@dataclass(frozen=True, slots=True)
class TechniqueComparisonResult:
    rows: tuple[TechniqueComparisonRow, ...]


@dataclass(frozen=True, slots=True)
class TechniqueComparisonArtifacts:
    run_dir: Path
    report_path: Path
    summary_csv_path: Path
    scenario_csv_path: Path
    capability_csv_path: Path
    metric_heatmap_png_path: Path
    scatter_png_path: Path
    capability_png_path: Path


@dataclass(frozen=True, slots=True)
class TechniqueDefinition:
    method_name: str
    sensor_regime_id: str
    build_row: Callable[[int], TechniqueComparisonRow]


def _pointwise_row(seed: int) -> TechniqueComparisonRow:
    result = run_pointwise_benchmark(seed=seed)
    prior = analyze_pointwise_prior_sensitivity(seed=seed)
    easy_runs = [run for run in result.runs if run.scenario_name == "easy"]
    overlap_runs = [run for run in result.runs if run.scenario_name == "overlap"]
    easy_accuracy = _safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in easy_runs])
    boundary_accuracy = _safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in overlap_runs])
    return TechniqueComparisonRow(
        method_name="pointwise",
        sensor_regime_id="position_only",
        overall_accuracy=result.summary.final_accuracy,
        prior_flip_fraction=prior.summary.flipped_by_small_prior_fraction,
        median_flip_threshold=prior.summary.median_smallest_prior_shift_to_flip,
        easy_accuracy=easy_accuracy,
        boundary_accuracy=boundary_accuracy,
        outlier_accuracy=None,
        transition_accuracy=None,
        long_history_accuracy=None,
        irregular_dt_accuracy=None,
        acceleration_accuracy=None,
        uses_temporal_history=0.0,
        model_based=0.0,
        irregular_dt_native=0.0,
        outlier_aware=0.0,
        stronger_sensor_stream=0.0,
    )


def _windowed_rows(seed: int) -> tuple[TechniqueComparisonRow, TechniqueComparisonRow]:
    result = run_windowed_benchmark(seed=seed)
    prior_raw = analyze_windowed_prior_sensitivity(seed=seed, feature_mode="raw")
    prior_robust = analyze_windowed_prior_sensitivity(seed=seed, feature_mode="robust")

    def _scenario_accuracy(runs, matchers: tuple[str, ...]) -> float | None:
        selected = [run for run in runs if any(token in run.scenario_name for token in matchers)]
        return _safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in selected])

    raw_row = TechniqueComparisonRow(
        method_name="windowed_raw",
        sensor_regime_id="position_only",
        overall_accuracy=result.summary.raw_final_accuracy,
        prior_flip_fraction=prior_raw.summary.flipped_by_small_prior_fraction,
        median_flip_threshold=prior_raw.summary.median_smallest_prior_shift_to_flip,
        easy_accuracy=_scenario_accuracy(result.raw_runs, ("clean",)),
        boundary_accuracy=None,
        outlier_accuracy=_scenario_accuracy(result.raw_runs, ("spike", "dip")),
        transition_accuracy=None,
        long_history_accuracy=_scenario_accuracy(result.raw_runs, ("long",)),
        irregular_dt_accuracy=None,
        acceleration_accuracy=None,
        uses_temporal_history=1.0,
        model_based=0.0,
        irregular_dt_native=1.0,
        outlier_aware=0.0,
        stronger_sensor_stream=0.0,
    )
    robust_row = TechniqueComparisonRow(
        method_name="windowed_robust",
        sensor_regime_id="position_only",
        overall_accuracy=result.summary.robust_final_accuracy,
        prior_flip_fraction=prior_robust.summary.flipped_by_small_prior_fraction,
        median_flip_threshold=prior_robust.summary.median_smallest_prior_shift_to_flip,
        easy_accuracy=_scenario_accuracy(result.robust_runs, ("clean",)),
        boundary_accuracy=None,
        outlier_accuracy=_scenario_accuracy(result.robust_runs, ("spike", "dip")),
        transition_accuracy=None,
        long_history_accuracy=_scenario_accuracy(result.robust_runs, ("long",)),
        irregular_dt_accuracy=None,
        acceleration_accuracy=None,
        uses_temporal_history=1.0,
        model_based=0.0,
        irregular_dt_native=1.0,
        outlier_aware=1.0,
        stronger_sensor_stream=0.0,
    )
    return raw_row, robust_row


def _accumulator_row(seed: int) -> TechniqueComparisonRow:
    result = run_accumulator_benchmark(seed=seed)
    prior = analyze_prior_sensitivity(seed=seed)
    easy_runs = [run for run in result.runs if run.scenario_name == "easy"]
    ambiguous_runs = [run for run in result.runs if run.scenario_name == "ambiguous"]
    transition_runs = [run for run in result.runs if run.scenario_name == "late_flip"]
    return TechniqueComparisonRow(
        method_name="accumulator",
        sensor_regime_id="position_only",
        overall_accuracy=result.summary.final_accuracy,
        prior_flip_fraction=prior.summary.flipped_by_small_prior_fraction,
        median_flip_threshold=prior.summary.median_smallest_prior_shift_to_flip,
        easy_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in easy_runs]),
        boundary_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in ambiguous_runs]),
        outlier_accuracy=None,
        transition_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in transition_runs]),
        long_history_accuracy=None,
        irregular_dt_accuracy=None,
        acceleration_accuracy=None,
        uses_temporal_history=1.0,
        model_based=0.0,
        irregular_dt_native=0.0,
        outlier_aware=0.0,
        stronger_sensor_stream=0.0,
    )


def _kalman_row(seed: int) -> TechniqueComparisonRow:
    result = run_kalman_bank_benchmark(seed=seed)
    easy_runs = [run for run in result.runs if run.scenario_name in {"stationary_regular", "constant_velocity_regular"}]
    irregular_runs = [run for run in result.runs if run.scenario_name == "constant_velocity_irregular"]
    acceleration_runs = [run for run in result.runs if run.scenario_name == "constant_acceleration_regular"]
    return TechniqueComparisonRow(
        method_name="kalman_bank",
        sensor_regime_id="position_only",
        overall_accuracy=result.summary.final_accuracy,
        prior_flip_fraction=0.0,
        median_flip_threshold=None,
        easy_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in easy_runs]),
        boundary_accuracy=None,
        outlier_accuracy=None,
        transition_accuracy=None,
        long_history_accuracy=None,
        irregular_dt_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in irregular_runs]),
        acceleration_accuracy=_safe_mean([1.0 if run.final_predicted_class == run.true_class else 0.0 for run in acceleration_runs]),
        uses_temporal_history=1.0,
        model_based=1.0,
        irregular_dt_native=1.0,
        outlier_aware=0.0,
        stronger_sensor_stream=0.0,
    )


def _kalman_velocity_aided_row(seed: int) -> TechniqueComparisonRow:
    result = analyze_velocity_aided_kalman_comparison(seed=seed)
    position_only = next(row for row in result.rows if row.measurement_mode == "position_only")
    velocity_aided = next(row for row in result.rows if row.measurement_mode == "position_plus_direct_velocity")
    return TechniqueComparisonRow(
        method_name="kalman_bank_velocity_aided",
        sensor_regime_id="position_plus_direct_velocity",
        overall_accuracy=velocity_aided.overall_accuracy,
        prior_flip_fraction=0.0,
        median_flip_threshold=None,
        easy_accuracy=None,
        boundary_accuracy=velocity_aided.short_noisy_accuracy,
        outlier_accuracy=velocity_aided.outlier_accuracy,
        transition_accuracy=None,
        long_history_accuracy=None,
        irregular_dt_accuracy=velocity_aided.endpoint_match_accuracy,
        acceleration_accuracy=velocity_aided.short_accuracy,
        uses_temporal_history=1.0,
        model_based=1.0,
        irregular_dt_native=1.0,
        outlier_aware=1.0 if velocity_aided.outlier_accuracy > position_only.outlier_accuracy else 0.0,
        stronger_sensor_stream=1.0,
    )


def default_technique_definitions() -> tuple[TechniqueDefinition, ...]:
    shared_adapters = {adapter.method_name: adapter for adapter in default_shared_classifier_adapters()}
    return (
        TechniqueDefinition(
            method_name="pointwise",
            sensor_regime_id=shared_adapters["pointwise"].sensor_regime_id,
            build_row=_pointwise_row,
        ),
        TechniqueDefinition(
            method_name="windowed_raw",
            sensor_regime_id=shared_adapters["windowed_raw"].sensor_regime_id,
            build_row=lambda seed: _windowed_rows(seed)[0],
        ),
        TechniqueDefinition(
            method_name="windowed_robust",
            sensor_regime_id=shared_adapters["windowed_robust"].sensor_regime_id,
            build_row=lambda seed: _windowed_rows(seed)[1],
        ),
        TechniqueDefinition(
            method_name="accumulator",
            sensor_regime_id=shared_adapters["accumulator"].sensor_regime_id,
            build_row=_accumulator_row,
        ),
        TechniqueDefinition(
            method_name="kalman_bank",
            sensor_regime_id=shared_adapters["kalman_bank"].sensor_regime_id,
            build_row=_kalman_row,
        ),
        TechniqueDefinition(
            method_name="kalman_bank_velocity_aided",
            sensor_regime_id=shared_adapters["kalman_bank_velocity_aided"].sensor_regime_id,
            build_row=_kalman_velocity_aided_row,
        ),
    )


def analyze_technique_comparison(*, seed: int = 7) -> TechniqueComparisonResult:
    definitions = default_technique_definitions()
    windowed_cache: tuple[TechniqueComparisonRow, TechniqueComparisonRow] | None = None
    rows: list[TechniqueComparisonRow] = []
    for definition in definitions:
        if definition.method_name.startswith("windowed_"):
            if windowed_cache is None:
                windowed_cache = _windowed_rows(seed)
            row = windowed_cache[0] if definition.method_name == "windowed_raw" else windowed_cache[1]
        else:
            row = definition.build_row(seed)
        rows.append(row)
    return TechniqueComparisonResult(rows=tuple(rows))


def render_technique_comparison_report(result: TechniqueComparisonResult) -> str:
    scenario_columns = (
        "easy_accuracy",
        "boundary_accuracy",
        "outlier_accuracy",
        "transition_accuracy",
        "long_history_accuracy",
        "irregular_dt_accuracy",
        "acceleration_accuracy",
    )
    lines = [
        "# Technique Comparison Study",
        "",
        "This package compares the current baseline techniques using their existing benchmark suites, prior-sensitivity diagnostics, and scenario-family stress cases. The numbers are useful for debugging and roadmap decisions, but they are not yet a single common-dataset bakeoff.",
        "",
        "## Summary Metrics",
        "",
        "| method | sensor_regime | overall_accuracy | prior_flip_fraction | median_flip_threshold |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in result.rows:
        median_text = "n/a" if row.median_flip_threshold is None else f"{row.median_flip_threshold:.2f}"
        lines.append(
            f"| {row.method_name} | {row.sensor_regime_id} | {row.overall_accuracy:.3f} | {row.prior_flip_fraction:.3f} | {median_text} |"
        )
    lines.extend(
        [
            "",
            "## Scenario-Family Accuracy",
            "",
            "| method | easy | boundary | outlier | transition | long_history | irregular_dt | acceleration |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result.rows:
        values = []
        for name in scenario_columns:
            value = getattr(row, name)
            values.append("n/a" if value is None else f"{value:.3f}")
        lines.append(f"| {row.method_name} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Capability Flags",
            "",
            "| method | temporal_history | model_based | irregular_dt_native | outlier_aware | stronger_sensor_stream |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result.rows:
        lines.append(
            f"| {row.method_name} | {row.uses_temporal_history:.1f} | {row.model_based:.1f} | {row.irregular_dt_native:.1f} | {row.outlier_aware:.1f} | {row.stronger_sensor_stream:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use this report to see where a method is strong, where it is brittle, and which failure mode is due to priors versus model mismatch.",
            "- The windowed methods are strong on outlier and long-history comparisons but do not supply model-based irregular-dt prediction.",
            "- The Kalman bank is the first explicitly model-based classifier in the ladder and is the right baseline before considering IMM.",
            "- `kalman_bank_velocity_aided` should be read as a separate sensor regime, not as a fair same-sensor improvement over the position-only techniques.",
        ]
    )
    return "\n".join(lines)


def _metric_value(row: TechniqueComparisonRow, field: str) -> float | None:
    value = getattr(row, field)
    return value


def _render_metric_heatmap(result: TechniqueComparisonResult):
    plt = _prepare_matplotlib()
    metric_fields = (
        "overall_accuracy",
        "prior_flip_fraction",
        "easy_accuracy",
        "boundary_accuracy",
        "outlier_accuracy",
        "transition_accuracy",
        "long_history_accuracy",
        "irregular_dt_accuracy",
        "acceleration_accuracy",
    )
    matrix = []
    for row in result.rows:
        matrix.append([float("nan") if _metric_value(row, field) is None else float(_metric_value(row, field)) for field in metric_fields])
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    colormap = plt.get_cmap("YlGnBu").copy()
    colormap.set_bad(color="#e5e7eb")
    image = ax.imshow(matrix, aspect="auto", cmap=colormap, vmin=0.0, vmax=1.0)
    ax.set_title("Technique Metric Heatmap", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(metric_fields)))
    ax.set_xticklabels([field.removesuffix("_accuracy") for field in metric_fields], rotation=35, ha="right")
    ax.set_yticks(range(len(result.rows)))
    ax.set_yticklabels([row.method_name for row in result.rows])
    for row_index, row in enumerate(result.rows):
        for col_index, field in enumerate(metric_fields):
            value = _metric_value(row, field)
            ax.text(col_index, row_index, "n/a" if value is None else f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="metric value")
    fig.tight_layout()
    return fig


def _render_accuracy_fragility_scatter(result: TechniqueComparisonResult):
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    colors = {
        "pointwise": "#2563eb",
        "windowed_raw": "#dc2626",
        "windowed_robust": "#16a34a",
        "accumulator": "#d97706",
        "kalman_bank": "#7c3aed",
        "kalman_bank_velocity_aided": "#0f766e",
    }
    for row in result.rows:
        ax.scatter(row.prior_flip_fraction, row.overall_accuracy, s=90, color=colors[row.method_name], label=row.method_name)
        ax.text(row.prior_flip_fraction + 0.01, row.overall_accuracy + 0.005, row.method_name, fontsize=8)
    ax.set_title("Accuracy vs Prior Fragility", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("fraction flipped by small prior perturbation")
    ax.set_ylabel("overall accuracy")
    ax.set_xlim(-0.01, max(row.prior_flip_fraction for row in result.rows) + 0.10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _render_capability_heatmap(result: TechniqueComparisonResult):
    plt = _prepare_matplotlib()
    capability_fields = ("uses_temporal_history", "model_based", "irregular_dt_native", "outlier_aware", "stronger_sensor_stream")
    matrix = [[float(getattr(row, field)) for field in capability_fields] for row in result.rows]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    image = ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0)
    ax.set_title("Technique Capability Matrix", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(capability_fields)))
    ax.set_xticklabels(["temporal", "model", "irregular_dt", "outlier", "sensor+"], rotation=25, ha="right")
    ax.set_yticks(range(len(result.rows)))
    ax.set_yticklabels([row.method_name for row in result.rows])
    for row_index, row in enumerate(result.rows):
        for col_index, field in enumerate(capability_fields):
            ax.text(col_index, row_index, f"{getattr(row, field):.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="capability score")
    fig.tight_layout()
    return fig


def _figure_to_svg(fig) -> str:
    plt = _prepare_matplotlib()
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _figure_to_png(fig) -> bytes:
    plt = _prepare_matplotlib()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_technique_comparison_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    result: TechniqueComparisonResult | None = None,
) -> TechniqueComparisonArtifacts:
    comparison = result or analyze_technique_comparison(seed=seed)
    output_root = Path(output_dir)
    run_dir = output_root / "technique_comparison_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "technique_comparison_report.md"
    summary_csv_path = run_dir / "technique_summary.csv"
    scenario_csv_path = run_dir / "technique_scenario_metrics.csv"
    capability_csv_path = run_dir / "technique_capabilities.csv"
    metric_heatmap_png_path = run_dir / "technique_metric_heatmap.png"
    scatter_png_path = run_dir / "accuracy_vs_prior_fragility.png"
    capability_png_path = run_dir / "technique_capability_heatmap.png"

    report_path.write_text(render_technique_comparison_report(comparison), encoding="utf-8")
    _write_csv(
        summary_csv_path,
        [
            {
                "method_name": row.method_name,
                "sensor_regime_id": row.sensor_regime_id,
                "overall_accuracy": row.overall_accuracy,
                "prior_flip_fraction": row.prior_flip_fraction,
                "median_flip_threshold": row.median_flip_threshold,
            }
            for row in comparison.rows
        ],
        ["method_name", "sensor_regime_id", "overall_accuracy", "prior_flip_fraction", "median_flip_threshold"],
    )
    _write_csv(
        scenario_csv_path,
        [
            {
                "method_name": row.method_name,
                "easy_accuracy": row.easy_accuracy,
                "boundary_accuracy": row.boundary_accuracy,
                "outlier_accuracy": row.outlier_accuracy,
                "transition_accuracy": row.transition_accuracy,
                "long_history_accuracy": row.long_history_accuracy,
                "irregular_dt_accuracy": row.irregular_dt_accuracy,
                "acceleration_accuracy": row.acceleration_accuracy,
            }
            for row in comparison.rows
        ],
        [
            "method_name",
            "easy_accuracy",
            "boundary_accuracy",
            "outlier_accuracy",
            "transition_accuracy",
            "long_history_accuracy",
            "irregular_dt_accuracy",
            "acceleration_accuracy",
        ],
    )
    _write_csv(
        capability_csv_path,
        [
            {
                "method_name": row.method_name,
                "uses_temporal_history": row.uses_temporal_history,
                "model_based": row.model_based,
                "irregular_dt_native": row.irregular_dt_native,
                "outlier_aware": row.outlier_aware,
            }
            for row in comparison.rows
        ],
        ["method_name", "uses_temporal_history", "model_based", "irregular_dt_native", "outlier_aware"],
    )
    metric_heatmap_png_path.write_bytes(_figure_to_png(_render_metric_heatmap(comparison)))
    scatter_png_path.write_bytes(_figure_to_png(_render_accuracy_fragility_scatter(comparison)))
    capability_png_path.write_bytes(_figure_to_png(_render_capability_heatmap(comparison)))

    return TechniqueComparisonArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        scenario_csv_path=scenario_csv_path,
        capability_csv_path=capability_csv_path,
        metric_heatmap_png_path=metric_heatmap_png_path,
        scatter_png_path=scatter_png_path,
        capability_png_path=capability_png_path,
    )
