from __future__ import annotations

import json
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ..utils.plotting import _figure_to_png
from ..utils.plotting import plt
from ..utils.plotting import render_labeled_heatmap
from ..validation.shared_evaluation import sensor_regime_summary_rows
from .common_dataset_comparison_contracts import CommonComparisonArtifacts, CommonComparisonResult
from .common_dataset_comparison_reporting import render_common_dataset_comparison_report


CLASS_NAMES = ("constant_velocity", "constant_acceleration")


def _sensor_regime_metadata() -> tuple[dict[str, object], ...]:
    return (
        {
            "sensor_regime_id": "position_only",
            "description": "Position measurements only; derived pseudo-observations may be allowed but no independent auxiliary sensor is present.",
            "same_sensor_fairness_bucket": "position_only",
            "supported_measurement_dims": [1],
            "supported_coordinate_frames": ["scalar_line"],
        },
        {
            "sensor_regime_id": "position_plus_direct_velocity",
            "description": "Position measurements plus an independent direct velocity sensor stream.",
            "same_sensor_fairness_bucket": "position_plus_direct_velocity",
            "supported_measurement_dims": [1],
            "supported_coordinate_frames": ["scalar_line"],
        },
    )


def _sensor_regime_summary_rows(result: CommonComparisonResult) -> list[dict[str, object]]:
    return sensor_regime_summary_rows(result.runs)


def _supported_rows(result: CommonComparisonResult):
    return [row for row in result.rows if row.overall_accuracy is not None]


def _render_common_metric_heatmap(result: CommonComparisonResult):
    fields = (
        "overall_accuracy",
        "easy_accuracy",
        "irregular_accuracy",
        "endpoint_match_accuracy",
        "short_accuracy",
        "noisy_accuracy",
        "outlier_accuracy",
        "prior_flip_fraction",
    )
    rows = _supported_rows(result)
    matrix = [[float("nan") if getattr(row, field) is None else getattr(row, field) for field in fields] for row in rows]
    return render_labeled_heatmap(
        matrix,
        [row.method_name for row in rows],
        ["overall", "easy", "irregular", "endpoint", "short", "short_noisy", "outlier", "prior_flip"],
        title="Common-Dataset Method Metrics",
        cmap="YlGnBu",
        figsize=(10.6, 4.8),
        aspect="auto",
        colorbar_label="metric value",
        vmin=0.0,
        vmax=1.0,
    )


def _render_common_confusion_bars(result: CommonComparisonResult):
    rows = _supported_rows(result)
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    method_names = [row.method_name for row in rows]
    overall = [row.overall_accuracy or 0.0 for row in rows]
    irregular = [row.irregular_accuracy or 0.0 for row in rows]
    endpoint = [row.endpoint_match_accuracy or 0.0 for row in rows]
    short = [row.short_accuracy or 0.0 for row in rows]
    noisy = [row.noisy_accuracy or 0.0 for row in rows]
    x = list(range(len(method_names)))
    ax.bar([value - 0.36 for value in x], overall, width=0.14, label="overall", color="#2563eb")
    ax.bar([value - 0.18 for value in x], irregular, width=0.14, label="irregular", color="#16a34a")
    ax.bar(x, endpoint, width=0.14, label="endpoint_match", color="#7c3aed")
    ax.bar([value + 0.18 for value in x], short, width=0.14, label="short", color="#dc2626")
    ax.bar([value + 0.36 for value in x], noisy, width=0.14, label="short_noisy", color="#d97706")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Shared-Corpus Accuracy by Method", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_dataset_balance(result: CommonComparisonResult):
    scenario_names = sorted({trajectory.scenario_name for trajectory in result.trajectories})
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    class_counts = {
        class_name: [sum(1 for trajectory in result.trajectories if trajectory.scenario_name == scenario_name and trajectory.true_class == class_name) for scenario_name in scenario_names]
        for class_name in CLASS_NAMES
    }
    x = list(range(len(scenario_names)))
    width = 0.34
    ax.bar([value - width / 2 for value in x], class_counts["constant_velocity"], width=width, label="constant_velocity", color="#2563eb")
    ax.bar([value + width / 2 for value in x], class_counts["constant_acceleration"], width=width, label="constant_acceleration", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=20, ha="right")
    ax.set_ylabel("trajectory count")
    ax.set_title("Dataset Class Balance by Scenario", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_covariate_audit(result: CommonComparisonResult):
    scenario_names = sorted({trajectory.scenario_name for trajectory in result.trajectories})
    metrics = ("duration", "sample_count", "mean_dt", "measurement_rmse")
    matrix: list[list[float]] = []
    for scenario_name in scenario_names:
        selected = [trajectory for trajectory in result.trajectories if trajectory.scenario_name == scenario_name]
        durations = [trajectory.times[-1] - trajectory.times[0] for trajectory in selected]
        sample_counts = [float(len(trajectory.times)) for trajectory in selected]
        mean_dts = [
            sum(trajectory.times[index] - trajectory.times[index - 1] for index in range(1, len(trajectory.times))) / max(len(trajectory.times) - 1, 1)
            for trajectory in selected
        ]
        rmses = [
            (sum((measurement - truth) ** 2 for measurement, truth in zip(trajectory.measurements, trajectory.true_position)) / max(len(trajectory.measurements), 1)) ** 0.5
            for trajectory in selected
        ]
        matrix.append([sum(durations) / len(durations), sum(sample_counts) / len(sample_counts), sum(mean_dts) / len(mean_dts), sum(rmses) / len(rmses)])
    return render_labeled_heatmap(matrix, scenario_names, list(metrics), title="Scenario Covariate Audit", cmap="YlOrBr", figsize=(8.8, 4.8), aspect="auto", colorbar_label="mean value")


def _render_scenario_profile(result: CommonComparisonResult):
    rows = _supported_rows(result)
    scenario_order = ["easy", "irregular", "endpoint_match", "short", "short_noisy", "outlier"]
    palette = {row.method_name: color for row, color in zip(rows, ("#2563eb", "#dc2626", "#d97706", "#16a34a", "#7c3aed", "#0f766e"), strict=False)}
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    x = list(range(len(scenario_order)))
    for row in rows:
        ys = [row.easy_accuracy or 0.0, row.irregular_accuracy or 0.0, row.endpoint_match_accuracy or 0.0, row.short_accuracy or 0.0, row.noisy_accuracy or 0.0, row.outlier_accuracy or 0.0]
        ax.plot(x, ys, marker="o", linewidth=2.2, markersize=6.0, color=palette[row.method_name], label=row.method_name)
    ax.set_xticks(x)
    ax.set_xticklabels(["easy", "irregular", "endpoint", "short", "short+noise", "outlier"], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Scenario Accuracy Profile by Method", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    return fig


def _render_prior_flip_tradeoff(result: CommonComparisonResult):
    rows = [row for row in _supported_rows(result) if row.prior_flip_fraction is not None]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    colors = ("#2563eb", "#dc2626", "#d97706", "#16a34a", "#7c3aed", "#0f766e")
    for index, row in enumerate(rows):
        ax.scatter(row.prior_flip_fraction, row.overall_accuracy, s=120, color=colors[index % len(colors)], alpha=0.9)
        ax.text((row.prior_flip_fraction or 0.0) + 0.01, row.overall_accuracy or 0.0, row.method_name, fontsize=8.5, color=colors[index % len(colors)], va="center")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("prior flip fraction")
    ax.set_ylabel("overall accuracy")
    ax.set_title("Accuracy vs Prior Sensitivity", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _render_trajectory_examples(result: CommonComparisonResult):
    scenario_names = sorted({trajectory.scenario_name for trajectory in result.trajectories})
    fig, axes = plt.subplots(len(scenario_names), 1, figsize=(10.5, 2.5 * len(scenario_names)), sharex=False)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    palette = {"constant_velocity": "#2563eb", "constant_acceleration": "#dc2626"}
    for ax, scenario_name in zip(axes_list, scenario_names):
        selected = [trajectory for trajectory in result.trajectories if trajectory.scenario_name == scenario_name]
        representatives = {class_name: next(trajectory for trajectory in selected if trajectory.true_class == class_name) for class_name in CLASS_NAMES}
        for class_name, trajectory in representatives.items():
            ax.plot(trajectory.times, trajectory.true_position, color=palette[class_name], linewidth=2.2, label=f"{class_name} true")
            ax.scatter(trajectory.times, trajectory.measurements, color=palette[class_name], s=18, alpha=0.75, marker="o")
        ax.set_title(scenario_name, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("position")
        ax.grid(True, alpha=0.25)
    if axes_list:
        axes_list[0].legend(frameon=False, ncol=2, loc="upper left")
        axes_list[-1].set_xlabel("time")
    fig.suptitle("Representative Trajectories by Scenario", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _render_method_confusion_heatmaps(result: CommonComparisonResult):
    method_names = [row.method_name for row in _supported_rows(result)]
    fig, axes = plt.subplots(1, len(method_names), figsize=(3.2 * len(method_names), 4.2))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    class_names = list(CLASS_NAMES)
    for ax, method_name in zip(axes_list, method_names):
        method_runs = [run for run in result.runs if run.method_name == method_name]
        matrix = [
            [sum(1 for run in method_runs if run.true_class == true_class and run.final_predicted_class == predicted_class) for predicted_class in class_names]
            for true_class in class_names
        ]
        image = ax.imshow(matrix, aspect="auto", cmap="Blues")
        ax.set_title(method_name, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names, fontsize=8)
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Final Confusion by Method", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def write_common_dataset_comparison_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    result: CommonComparisonResult | None = None,
) -> CommonComparisonArtifacts:
    if result is None:
        from .common_dataset_comparison import analyze_common_dataset_comparison

        comparison = analyze_common_dataset_comparison(seed=seed)
    else:
        comparison = result
    output_root = Path(output_dir)
    run_dir = output_root / "common_dataset_comparison_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "common_dataset_comparison_report.md"
    trajectory_path = run_dir / "shared_trajectories.csv"
    run_summary_path = run_dir / "method_run_summary.csv"
    method_summary_path = run_dir / "method_summary.csv"
    sensor_regimes_path = run_dir / "sensor_regimes.json"
    sensor_regime_metrics_path = run_dir / "metrics_by_sensor_regime.csv"
    heatmap_png_path = run_dir / "common_dataset_metric_heatmap.png"
    confusion_png_path = run_dir / "shared_accuracy_bars.png"
    plots_dir = run_dir / "plots"
    overview_dir = plots_dir / "overview"
    single_trajectory_dir = plots_dir / "single_trajectory"
    confusion_dir = plots_dir / "confusion"
    diagnostics_dir = plots_dir / "diagnostics"
    overview_dir.mkdir(parents=True, exist_ok=True)
    single_trajectory_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    overview_balance_png_path = overview_dir / "dataset_class_balance.png"
    overview_covariates_png_path = overview_dir / "covariate_leakage_audit.png"
    scenario_profile_png_path = diagnostics_dir / "scenario_accuracy_profile.png"
    prior_sensitivity_png_path = diagnostics_dir / "accuracy_vs_prior_sensitivity.png"
    trajectory_examples_png_path = single_trajectory_dir / "trajectory_examples_by_scenario.png"
    final_confusion_png_path = confusion_dir / "final_confusion_by_method.png"

    report_path.write_text(render_common_dataset_comparison_report(comparison), encoding="utf-8")
    write_csv(
        trajectory_path,
        [
            {
                "trajectory_id": trajectory.trajectory_id,
                "true_class": trajectory.true_class,
                "scenario_name": trajectory.scenario_name,
                "seed": trajectory.seed,
                "measurement_dim": trajectory.measurement_dim,
                "coordinate_frame": trajectory.coordinate_frame,
                "times": " ".join(f"{value:.3f}" for value in trajectory.times),
                "measurements": " ".join(f"{value:.3f}" for value in trajectory.measurements),
            }
            for trajectory in comparison.trajectories
        ],
        ["trajectory_id", "true_class", "scenario_name", "seed", "measurement_dim", "coordinate_frame", "times", "measurements"],
    )
    write_csv(
        run_summary_path,
        [
            {
                "method_name": run.method_name,
                "sensor_regime_id": run.sensor_regime_id,
                "measurement_dim": run.measurement_dim,
                "coordinate_frame": run.coordinate_frame,
                "trajectory_id": run.trajectory_id,
                "true_class": run.true_class,
                "scenario_name": run.scenario_name,
                "final_predicted_class": run.final_predicted_class,
                "final_confidence": run.final_confidence,
                **{f"posterior_{class_name}": run.final_weights[class_name] for class_name in CLASS_NAMES},
            }
            for run in comparison.runs
        ],
        ["method_name", "sensor_regime_id", "measurement_dim", "coordinate_frame", "trajectory_id", "true_class", "scenario_name", "final_predicted_class", "final_confidence", "posterior_constant_velocity", "posterior_constant_acceleration"],
    )
    write_csv(
        method_summary_path,
        [
            {
                "method_name": row.method_name,
                "sensor_regime_id": row.sensor_regime_id,
                "applicability_status": row.applicability_status,
                "primary_evaluation_family": row.primary_evaluation_family,
                "overall_accuracy": row.overall_accuracy,
                "easy_accuracy": row.easy_accuracy,
                "irregular_accuracy": row.irregular_accuracy,
                "endpoint_match_accuracy": row.endpoint_match_accuracy,
                "short_accuracy": row.short_accuracy,
                "noisy_accuracy": row.noisy_accuracy,
                "outlier_accuracy": row.outlier_accuracy,
                "prior_flip_fraction": row.prior_flip_fraction,
                "witness_artifact": row.witness_artifact,
            }
            for row in comparison.rows
        ],
        ["method_name", "sensor_regime_id", "applicability_status", "primary_evaluation_family", "overall_accuracy", "easy_accuracy", "irregular_accuracy", "endpoint_match_accuracy", "short_accuracy", "noisy_accuracy", "outlier_accuracy", "prior_flip_fraction", "witness_artifact"],
    )
    sensor_regimes_path.write_text(json.dumps(_sensor_regime_metadata(), indent=2), encoding="utf-8")
    write_csv(sensor_regime_metrics_path, _sensor_regime_summary_rows(comparison), ["sensor_regime_id", "num_predictions", "mean_accuracy", "mean_confidence", "measurement_dims", "coordinate_frames", "methods"])
    heatmap_png_path.write_bytes(_figure_to_png(_render_common_metric_heatmap(comparison)))
    confusion_png_path.write_bytes(_figure_to_png(_render_common_confusion_bars(comparison)))
    overview_balance_png_path.write_bytes(_figure_to_png(_render_dataset_balance(comparison)))
    overview_covariates_png_path.write_bytes(_figure_to_png(_render_covariate_audit(comparison)))
    scenario_profile_png_path.write_bytes(_figure_to_png(_render_scenario_profile(comparison)))
    prior_sensitivity_png_path.write_bytes(_figure_to_png(_render_prior_flip_tradeoff(comparison)))
    trajectory_examples_png_path.write_bytes(_figure_to_png(_render_trajectory_examples(comparison)))
    final_confusion_png_path.write_bytes(_figure_to_png(_render_method_confusion_heatmaps(comparison)))
    return CommonComparisonArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        trajectory_path=trajectory_path,
        run_summary_path=run_summary_path,
        method_summary_path=method_summary_path,
        sensor_regimes_path=sensor_regimes_path,
        sensor_regime_metrics_path=sensor_regime_metrics_path,
        heatmap_png_path=heatmap_png_path,
        confusion_png_path=confusion_png_path,
        plots_dir=plots_dir,
        overview_balance_png_path=overview_balance_png_path,
        overview_covariates_png_path=overview_covariates_png_path,
        scenario_profile_png_path=scenario_profile_png_path,
        prior_sensitivity_png_path=prior_sensitivity_png_path,
        trajectory_examples_png_path=trajectory_examples_png_path,
        final_confusion_png_path=final_confusion_png_path,
    )


__all__ = ["write_common_dataset_comparison_artifacts"]
