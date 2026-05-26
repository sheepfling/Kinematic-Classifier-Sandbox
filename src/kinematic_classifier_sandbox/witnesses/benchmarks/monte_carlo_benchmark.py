from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

from kinematic_classifier_sandbox.utils.io import write_csv

from ..markdown_builder import MarkdownDocument
from ..runtime_paths import prepare_matplotlib
from ..utils.plotting import plt
from ..utils.math import _mean, _percentile, _safe_log
from .sequential_bayes_accumulator import AccumulatorBenchmarkResult, run_accumulator_benchmark


def _normalize_rows(confusion: dict[str, dict[str, int]], class_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for true_class in class_names:
        row = confusion[true_class]
        total = sum(row.values())
        normalized[true_class] = {
            predicted: (row[predicted] / total if total else 0.0) for predicted in class_names + ("unknown",)
        }
    return normalized


def _class_metrics(confusion: dict[str, dict[str, int]], class_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    total_samples = sum(sum(row.values()) for row in confusion.values())
    for class_name in class_names:
        tp = confusion[class_name][class_name]
        fn = sum(confusion[class_name].values()) - tp
        fp = sum(confusion[other][class_name] for other in class_names if other != class_name)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[class_name].values()),
        }
    balanced_accuracy = _mean([metrics[name]["recall"] for name in class_names])
    macro_f1 = _mean([metrics[name]["f1"] for name in class_names])
    final_accuracy = sum(confusion[name][name] for name in class_names) / max(total_samples, 1)
    return {
        "per_class": metrics,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "final_accuracy": final_accuracy,
    }


def _confusion_to_rows(confusion: dict[str, dict[str, int]], class_names: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for true_class in class_names:
        row = {"true_class": true_class}
        for predicted in class_names + ("unknown",):
            row[predicted] = confusion[true_class][predicted]
        row["total"] = sum(confusion[true_class].values())
        rows.append(row)
    return rows


def _rows_to_confusion(rows: list[dict[str, object]], class_names: tuple[str, ...]) -> dict[str, dict[str, int]]:
    confusion = {name: {predicted: 0 for predicted in class_names + ("unknown",)} for name in class_names}
    for row in rows:
        true_class = str(row["true_class"])
        for predicted in class_names + ("unknown",):
            confusion[true_class][predicted] = int(row[predicted])
    return confusion


def _final_confusion_from_runs(result: AccumulatorBenchmarkResult) -> dict[str, dict[str, int]]:
    class_names = tuple(result.summary.confusion_counts)
    confusion = {name: {predicted: 0 for predicted in class_names + ("unknown",)} for name in class_names}
    for run in result.runs:
        confusion[run.true_class][run.final_predicted_class] += 1
    return confusion


def _gated_confusion_from_runs(
    result: AccumulatorBenchmarkResult,
    *,
    gate_threshold: float,
) -> dict[str, dict[str, int]]:
    class_names = tuple(result.summary.confusion_counts)
    confusion = {name: {predicted: 0 for predicted in class_names + ("unknown",)} for name in class_names}
    for run in result.runs:
        final_step = run.steps[-1]
        predicted = final_step.predicted_class if final_step.confidence >= gate_threshold else "unknown"
        confusion[run.true_class][predicted] += 1
    return confusion


def _aggregate_time_series(result: AccumulatorBenchmarkResult) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    max_steps = max((len(run.steps) for run in result.runs), default=0)
    metrics_by_time: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for step_index in range(max_steps):
        true_posteriors: list[float] = []
        confidences: list[float] = []
        correctness: list[float] = []
        margins: list[float] = []
        for run in result.runs:
            if step_index >= len(run.steps):
                continue
            step = run.steps[step_index]
            posterior_true = step.posterior_weights[run.true_class]
            true_posteriors.append(posterior_true)
            confidences.append(step.confidence)
            correctness.append(1.0 if step.predicted_class == run.true_class else 0.0)
            ordered = sorted(step.posterior_weights.values(), reverse=True)
            margins.append(ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0])
            calibration_rows.append(
                {
                    "trajectory_id": run.trajectory_id,
                    "scenario_name": run.scenario_name,
                    "step": step_index,
                    "time": step.time,
                    "confidence": step.confidence,
                    "correct": 1 if step.predicted_class == run.true_class else 0,
                    "predicted_class": step.predicted_class,
                    "true_class": run.true_class,
                    "abstained": 1 if step.abstained else 0,
                    "posterior_true_class": posterior_true,
                }
            )
        if not true_posteriors:
            continue
        sorted_true_posteriors = sorted(true_posteriors)
        metrics_by_time.append(
            {
                "step": step_index,
                "time": result.runs[0].steps[step_index].time if result.runs and step_index < len(result.runs[0].steps) else float(step_index),
                "sample_count": len(true_posteriors),
                "accuracy": _mean(correctness),
                "mean_true_class_posterior": _mean(true_posteriors),
                "median_true_class_posterior": median(true_posteriors),
                "q10_true_class_posterior": _percentile(sorted_true_posteriors, 0.10),
                "q25_true_class_posterior": _percentile(sorted_true_posteriors, 0.25),
                "q75_true_class_posterior": _percentile(sorted_true_posteriors, 0.75),
                "q90_true_class_posterior": _percentile(sorted_true_posteriors, 0.90),
                "mean_confidence": _mean(confidences),
                "mean_margin": _mean(margins),
                "abstain_rate": _mean([1.0 if run.steps[step_index].abstained else 0.0 for run in result.runs if step_index < len(run.steps)]),
            }
        )
    return metrics_by_time, calibration_rows


def _threshold_rows(
    result: AccumulatorBenchmarkResult,
    *,
    gate_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    time_to_confidence_rows: list[dict[str, object]] = []
    time_to_correct_rows: list[dict[str, object]] = []
    for run in result.runs:
        time_to_confidence = None
        time_to_correct = None
        for step in run.steps:
            if time_to_confidence is None and step.confidence >= gate_threshold:
                time_to_confidence = step.time
            if time_to_correct is None and step.predicted_class == run.true_class:
                time_to_correct = step.time
            if time_to_confidence is not None and time_to_correct is not None:
                break
        time_to_confidence_rows.append(
            {
                "trajectory_id": run.trajectory_id,
                "scenario_name": run.scenario_name,
                "true_class": run.true_class,
                "time_to_confidence": time_to_confidence if time_to_confidence is not None else "",
                "reached_confidence": 1 if time_to_confidence is not None else 0,
                "final_confidence": run.final_confidence,
                "final_class": run.final_predicted_class,
            }
        )
        time_to_correct_rows.append(
            {
                "trajectory_id": run.trajectory_id,
                "scenario_name": run.scenario_name,
                "true_class": run.true_class,
                "time_to_correct_classification": time_to_correct if time_to_correct is not None else "",
                "reached_correct": 1 if time_to_correct is not None else 0,
                "final_confidence": run.final_confidence,
                "final_class": run.final_predicted_class,
            }
        )
    return time_to_confidence_rows, time_to_correct_rows


def _calibration_bins(
    calibration_rows: list[dict[str, object]],
    *,
    bin_count: int = 10,
) -> tuple[list[dict[str, object]], float]:
    bins: list[list[dict[str, object]]] = [[] for _ in range(bin_count)]
    for row in calibration_rows:
        confidence = float(row["confidence"])
        index = min(int(confidence * bin_count), bin_count - 1)
        bins[index].append(row)
    output_rows: list[dict[str, object]] = []
    total = len(calibration_rows)
    ece = 0.0
    for index, rows in enumerate(bins):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if rows:
            accuracies = [float(row["correct"]) for row in rows]
            confidences = [float(row["confidence"]) for row in rows]
            accuracy = _mean(accuracies)
            confidence = _mean(confidences)
            gap = abs(accuracy - confidence)
            weight = len(rows) / max(total, 1)
            ece += weight * gap
        else:
            accuracy = 0.0
            confidence = 0.0
            gap = 0.0
        output_rows.append(
            {
                "bin_index": index,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": len(rows),
                "accuracy": accuracy,
                "mean_confidence": confidence,
                "gap": gap,
            }
        )
    return output_rows, ece


def _multiclass_brier(probabilities: dict[str, float], true_class: str, class_names: tuple[str, ...]) -> float:
    return sum((probabilities[name] - (1.0 if name == true_class else 0.0)) ** 2 for name in class_names) / max(len(class_names), 1)


def _summarize(
    result: AccumulatorBenchmarkResult,
    *,
    gate_threshold: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
    dict[str, object],
]:
    class_names = tuple(result.summary.confusion_counts)
    metrics_by_time, calibration_rows = _aggregate_time_series(result)
    time_to_confidence_rows, time_to_correct_rows = _threshold_rows(result, gate_threshold=gate_threshold)
    calibration_bins, ece = _calibration_bins(calibration_rows)
    final_confusion = _final_confusion_from_runs(result)
    gated_confusion = _gated_confusion_from_runs(result, gate_threshold=gate_threshold)

    total_steps = len(calibration_rows)
    brier_scores: list[float] = []
    nll_scores: list[float] = []
    per_class_brier: dict[str, list[float]] = {name: [] for name in class_names}
    per_class_nll: dict[str, list[float]] = {name: [] for name in class_names}
    for run in result.runs:
        for step in run.steps:
            brier = _multiclass_brier(step.posterior_weights, run.true_class, class_names)
            nll = -_safe_log(step.posterior_weights[run.true_class])
            brier_scores.append(brier)
            nll_scores.append(nll)
            per_class_brier[run.true_class].append(brier)
            per_class_nll[run.true_class].append(nll)

    time_to_confidence_values = [float(row["time_to_confidence"]) for row in time_to_confidence_rows if row["time_to_confidence"] != ""]
    time_to_correct_values = [float(row["time_to_correct_classification"]) for row in time_to_correct_rows if row["time_to_correct_classification"] != ""]
    runs_reaching_confidence = sum(1 for row in time_to_confidence_rows if row["reached_confidence"])
    runs_reaching_correct = sum(1 for row in time_to_correct_rows if row["reached_correct"])
    runs_reaching_both = sum(
        1
        for confidence_row, correct_row in zip(time_to_confidence_rows, time_to_correct_rows, strict=True)
        if confidence_row["reached_confidence"] and correct_row["reached_correct"]
    )
    final_confusion_metrics = _class_metrics(final_confusion, class_names)
    gated_confusion_metrics = _class_metrics(gated_confusion, class_names)
    summary = {
        "total_trajectories": len(result.runs),
        "total_steps": total_steps,
        "confidence_gate_threshold": gate_threshold,
        "final_accuracy": final_confusion_metrics["final_accuracy"],
        "balanced_accuracy": final_confusion_metrics["balanced_accuracy"],
        "macro_f1": final_confusion_metrics["macro_f1"],
        "gated_accuracy": gated_confusion_metrics["final_accuracy"],
        "gated_balanced_accuracy": gated_confusion_metrics["balanced_accuracy"],
        "gated_macro_f1": gated_confusion_metrics["macro_f1"],
        "abstain_rate": sum(1 for run in result.runs if run.final_predicted_class == "unknown") / max(len(result.runs), 1),
        "confidence_reached_rate": runs_reaching_confidence / max(len(result.runs), 1),
        "correct_and_confident_rate": runs_reaching_both / max(len(result.runs), 1),
        "correct_reached_rate": runs_reaching_correct / max(len(result.runs), 1),
        "mean_brier_score": _mean(brier_scores),
        "mean_nll": _mean(nll_scores),
        "expected_calibration_error": ece,
        "mean_time_to_confidence": _mean(time_to_confidence_values) if time_to_confidence_values else None,
        "median_time_to_confidence": median(time_to_confidence_values) if time_to_confidence_values else None,
        "mean_time_to_correct_classification": _mean(time_to_correct_values) if time_to_correct_values else None,
        "median_time_to_correct_classification": median(time_to_correct_values) if time_to_correct_values else None,
        "per_class": {
            name: {
                "brier_score": _mean(per_class_brier[name]) if per_class_brier[name] else None,
                "nll": _mean(per_class_nll[name]) if per_class_nll[name] else None,
                **final_confusion_metrics["per_class"][name],
            }
            for name in class_names
        },
    }
    return (
        metrics_by_time,
        time_to_confidence_rows,
        time_to_correct_rows,
        calibration_bins,
        final_confusion,
        gated_confusion,
        summary,
    )


@dataclass(frozen=True, slots=True)
class MonteCarloBenchmarkSummary:
    total_trajectories: int
    total_steps: int
    confidence_gate_threshold: float
    final_accuracy: float
    balanced_accuracy: float
    macro_f1: float
    gated_accuracy: float
    gated_balanced_accuracy: float
    gated_macro_f1: float
    abstain_rate: float
    confidence_reached_rate: float
    correct_and_confident_rate: float
    correct_reached_rate: float
    mean_brier_score: float
    mean_nll: float
    expected_calibration_error: float
    mean_time_to_confidence: float | None
    median_time_to_confidence: float | None
    mean_time_to_correct_classification: float | None
    median_time_to_correct_classification: float | None
    per_class: dict[str, dict[str, float | None]]


@dataclass(frozen=True, slots=True)
class MonteCarloBenchmarkResult:
    accumulator_result: AccumulatorBenchmarkResult
    summary: MonteCarloBenchmarkSummary
    metrics_by_time: tuple[dict[str, object], ...]
    time_to_confidence: tuple[dict[str, object], ...]
    time_to_correct_classification: tuple[dict[str, object], ...]
    calibration_bins: tuple[dict[str, object], ...]
    final_confusion: dict[str, dict[str, int]]
    confidence_gated_confusion: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class MonteCarloBenchmarkArtifacts:
    run_dir: Path
    report_path: Path
    summary_json_path: Path
    metrics_by_time_path: Path
    time_to_confidence_path: Path
    time_to_correct_classification_path: Path
    calibration_bins_path: Path
    confusion_final_path: Path
    confusion_confidence_gated_path: Path
    plot_accuracy_png_path: Path
    plot_posterior_png_path: Path
    plot_time_to_confidence_png_path: Path
    plot_time_to_correct_png_path: Path
    plot_calibration_png_path: Path
    plot_confusion_png_path: Path


def analyze_accumulator_monte_carlo(
    result: AccumulatorBenchmarkResult,
    *,
    confidence_gate_threshold: float = 0.85,
) -> MonteCarloBenchmarkResult:
    (
        metrics_by_time,
        time_to_confidence_rows,
        time_to_correct_rows,
        calibration_bins,
        final_confusion,
        gated_confusion,
        summary_dict,
    ) = _summarize(result, gate_threshold=confidence_gate_threshold)
    summary = MonteCarloBenchmarkSummary(**summary_dict)
    return MonteCarloBenchmarkResult(
        accumulator_result=result,
        summary=summary,
        metrics_by_time=tuple(metrics_by_time),
        time_to_confidence=tuple(time_to_confidence_rows),
        time_to_correct_classification=tuple(time_to_correct_rows),
        calibration_bins=tuple(calibration_bins),
        final_confusion=final_confusion,
        confidence_gated_confusion=gated_confusion,
    )


def run_accumulator_monte_carlo_benchmark(
    *,
    seed: int = 7,
    forgetting_factor: float = 1.0,
    confidence_threshold: float = 0.75,
    trajectories_per_class: int = 12,
    class_specs=None,
    confidence_gate_threshold: float = 0.85,
) -> MonteCarloBenchmarkResult:
    accumulator_result = run_accumulator_benchmark(
        seed=seed,
        forgetting_factor=forgetting_factor,
        confidence_threshold=confidence_threshold,
        trajectories_per_class=trajectories_per_class,
        class_specs=class_specs,
    )
    return analyze_accumulator_monte_carlo(
        accumulator_result,
        confidence_gate_threshold=confidence_gate_threshold,
    )


def _render_report(result: MonteCarloBenchmarkResult) -> str:
    summary = result.summary
    class_names = tuple(result.final_confusion)
    report = MarkdownDocument("Monte Carlo Accumulator Report")
    report.paragraph("This report aggregates time-indexed evidence from the sequential Bayesian accumulator.")
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Trajectories: {summary.total_trajectories}",
            f"Steps per trajectory: {summary.total_steps // max(summary.total_trajectories, 1)}",
            f"Final accuracy: {summary.final_accuracy:.3f}",
            f"Balanced accuracy: {summary.balanced_accuracy:.3f}",
            f"Macro F1: {summary.macro_f1:.3f}",
            f"Gated accuracy: {summary.gated_accuracy:.3f}",
            f"Abstain rate: {summary.abstain_rate:.3f}",
            f"Confidence gate threshold: {summary.confidence_gate_threshold:.2f}",
            f"Confidence reached rate: {summary.confidence_reached_rate:.3f}",
            f"Correct reached rate: {summary.correct_reached_rate:.3f}",
            f"Correct and confident rate: {summary.correct_and_confident_rate:.3f}",
            f"Expected calibration error: {summary.expected_calibration_error:.3f}",
            f"Mean Brier score: {summary.mean_brier_score:.3f}",
            f"Mean NLL: {summary.mean_nll:.3f}",
        ]
    )
    report.heading("Time Behavior", level=2)
    report.table(
        ["metric", "value"],
        [
            (
                "mean time to confidence",
                summary.mean_time_to_confidence if summary.mean_time_to_confidence is not None else "n/a",
            ),
            (
                "median time to confidence",
                summary.median_time_to_confidence if summary.median_time_to_confidence is not None else "n/a",
            ),
            (
                "mean time to correct classification",
                summary.mean_time_to_correct_classification
                if summary.mean_time_to_correct_classification is not None
                else "n/a",
            ),
            (
                "median time to correct classification",
                summary.median_time_to_correct_classification
                if summary.median_time_to_correct_classification is not None
                else "n/a",
            ),
        ],
    )
    report.heading("Per-Class Summary", level=2)
    report.table(
        ["class", "precision", "recall", "f1", "brier", "nll", "support"],
        [
            (
                class_name,
                f"{summary.per_class[class_name]['precision']:.3f}",
                f"{summary.per_class[class_name]['recall']:.3f}",
                f"{summary.per_class[class_name]['f1']:.3f}",
                f"{summary.per_class[class_name]['brier_score']:.3f}",
                f"{summary.per_class[class_name]['nll']:.3f}",
                f"{int(summary.per_class[class_name]['support'])}",
            )
            for class_name in class_names
        ],
    )
    report.heading("Calibration Bins", level=2)
    report.table(
        ["bin", "count", "accuracy", "mean confidence", "gap"],
        [
            (
                int(row["bin_index"]),
                int(row["count"]),
                f"{float(row['accuracy']):.3f}",
                f"{float(row['mean_confidence']):.3f}",
                f"{float(row['gap']):.3f}",
            )
            for row in result.calibration_bins
        ],
    )
    report.heading("Acceptance Notes", level=2)
    report.bullet_list(
        [
            "Accuracy and posterior mass should increase with time on easy trajectories.",
            "Ambiguous trajectories should show slower confidence growth and lower calibration sharpness.",
            "Confidence-gated predictions separate confident error from uncertain abstention.",
        ]
    )
    return report.text()


def _build_accuracy_figure(result: MonteCarloBenchmarkResult):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    times = [row["time"] for row in result.metrics_by_time]
    accuracy = [row["accuracy"] for row in result.metrics_by_time]
    ax.plot(times, accuracy, color="#2563eb", linewidth=2.4, marker="o", markersize=4)
    ax.set_title("Accuracy vs Time", loc="left", fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _build_posterior_figure(result: MonteCarloBenchmarkResult):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    times = [row["time"] for row in result.metrics_by_time]
    mean_posterior = [row["mean_true_class_posterior"] for row in result.metrics_by_time]
    lower = [row["q25_true_class_posterior"] for row in result.metrics_by_time]
    upper = [row["q75_true_class_posterior"] for row in result.metrics_by_time]
    ax.plot(times, mean_posterior, color="#7c3aed", linewidth=2.4, marker="o", markersize=4)
    ax.fill_between(times, lower, upper, color="#7c3aed", alpha=0.15, linewidth=0.0)
    ax.set_title("True-Class Posterior Quantiles", loc="left", fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("posterior")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _build_histogram_figure(
    rows: tuple[dict[str, object], ...],
    *,
    key: str,
    title: str,
    xlabel: str,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    values = [float(row[key]) for row in rows if row[key] != ""]
    ax.hist(values, bins=min(10, max(len(values), 1)), color="#2563eb", alpha=0.85, edgecolor="white")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    return fig


def _build_calibration_figure(result: MonteCarloBenchmarkResult):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    xs = [row["mean_confidence"] for row in result.calibration_bins if row["count"]]
    ys = [row["accuracy"] for row in result.calibration_bins if row["count"]]
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1.5)
    axes[0].plot(xs, ys, marker="o", color="#16a34a", linewidth=2.0)
    axes[0].set_title("Calibration Curve", loc="left", fontweight="bold")
    axes[0].set_xlabel("mean confidence")
    axes[0].set_ylabel("empirical accuracy")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.2)
    widths = [row["bin_upper"] - row["bin_lower"] for row in result.calibration_bins]
    centers = [(row["bin_lower"] + row["bin_upper"]) / 2.0 for row in result.calibration_bins]
    counts = [row["count"] for row in result.calibration_bins]
    axes[1].bar(centers, counts, width=widths, color="#2563eb", alpha=0.85, edgecolor="white")
    axes[1].set_title("Confidence Histogram", loc="left", fontweight="bold")
    axes[1].set_xlabel("confidence bin center")
    axes[1].set_ylabel("count")
    axes[1].grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    return fig


def _build_confusion_figure(result: MonteCarloBenchmarkResult):
    class_names = tuple(result.final_confusion)
    final_normalized = _normalize_rows(result.final_confusion, class_names)
    gated_normalized = _normalize_rows(result.confidence_gated_confusion, class_names)
    labels = class_names + ("unknown",)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for axis, matrix, title in (
        (axes[0], final_normalized, "Final Confusion"),
        (axes[1], gated_normalized, "Confidence-Gated Confusion"),
    ):
        data = [[matrix[true][predicted] for predicted in labels] for true in class_names]
        image = axis.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_yticks(range(len(class_names)))
        axis.set_yticklabels(class_names)
        for row_index, row in enumerate(data):
            for col_index, value in enumerate(row):
                axis.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _render_figure_svg(fig) -> str:
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _render_figure_png(fig) -> bytes:
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_monte_carlo_report(result: MonteCarloBenchmarkResult) -> str:
    return _render_report(result)


def render_monte_carlo_accuracy_svg(result: MonteCarloBenchmarkResult) -> str:
    return _render_figure_svg(_build_accuracy_figure(result))


def render_monte_carlo_accuracy_png_bytes(result: MonteCarloBenchmarkResult) -> bytes:
    return _render_figure_png(_build_accuracy_figure(result))


def render_monte_carlo_posterior_svg(result: MonteCarloBenchmarkResult) -> str:
    return _render_figure_svg(_build_posterior_figure(result))


def render_monte_carlo_posterior_png_bytes(result: MonteCarloBenchmarkResult) -> bytes:
    return _render_figure_png(_build_posterior_figure(result))


def render_monte_carlo_time_to_confidence_svg(result: MonteCarloBenchmarkResult) -> str:
    return _render_figure_svg(
        _build_histogram_figure(
            result.time_to_confidence,
            key="time_to_confidence",
            title="Time to Confidence",
            xlabel="time",
        )
    )


def render_monte_carlo_time_to_confidence_png_bytes(result: MonteCarloBenchmarkResult) -> bytes:
    return _render_figure_png(
        _build_histogram_figure(
            result.time_to_confidence,
            key="time_to_confidence",
            title="Time to Confidence",
            xlabel="time",
        )
    )


def render_monte_carlo_time_to_correct_svg(result: MonteCarloBenchmarkResult) -> str:
    return _render_figure_svg(
        _build_histogram_figure(
            result.time_to_correct_classification,
            key="time_to_correct_classification",
            title="Time to Correct Classification",
            xlabel="time",
        )
    )


def render_monte_carlo_time_to_correct_png_bytes(result: MonteCarloBenchmarkResult) -> bytes:
    return _render_figure_png(
        _build_histogram_figure(
            result.time_to_correct_classification,
            key="time_to_correct_classification",
            title="Time to Correct Classification",
            xlabel="time",
        )
    )


def render_monte_carlo_calibration_svg(result: MonteCarloBenchmarkResult) -> str:
    return _render_figure_svg(_build_calibration_figure(result))


def render_monte_carlo_calibration_png_bytes(result: MonteCarloBenchmarkResult) -> bytes:
    return _render_figure_png(_build_calibration_figure(result))


def render_monte_carlo_confusion_svg(result: MonteCarloBenchmarkResult) -> str:
    return _render_figure_svg(_build_confusion_figure(result))


def render_monte_carlo_confusion_png_bytes(result: MonteCarloBenchmarkResult) -> bytes:
    return _render_figure_png(_build_confusion_figure(result))


def write_monte_carlo_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    forgetting_factor: float = 1.0,
    confidence_threshold: float = 0.75,
    trajectories_per_class: int = 12,
    confidence_gate_threshold: float = 0.85,
    result: MonteCarloBenchmarkResult | None = None,
) -> MonteCarloBenchmarkArtifacts:
    benchmark_result = result or run_accumulator_monte_carlo_benchmark(
        seed=seed,
        forgetting_factor=forgetting_factor,
        confidence_threshold=confidence_threshold,
        trajectories_per_class=trajectories_per_class,
        confidence_gate_threshold=confidence_gate_threshold,
    )
    output_root = Path(output_dir)
    run_dir = output_root / "monte_carlo_accumulator"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "monte_carlo_report.md"
    summary_json_path = run_dir / "monte_carlo_summary.json"
    metrics_by_time_path = run_dir / "metrics_by_time.csv"
    time_to_confidence_path = run_dir / "time_to_confidence.csv"
    time_to_correct_classification_path = run_dir / "time_to_correct_classification.csv"
    calibration_bins_path = run_dir / "calibration_bins.csv"
    confusion_final_path = run_dir / "confusion_final.csv"
    confusion_confidence_gated_path = run_dir / "confusion_confidence_gated.csv"
    plot_accuracy_png_path = run_dir / "accuracy_vs_time.png"
    plot_posterior_png_path = run_dir / "true_class_posterior_quantiles.png"
    plot_time_to_confidence_png_path = run_dir / "time_to_confidence.png"
    plot_time_to_correct_png_path = run_dir / "time_to_correct_classification.png"
    plot_calibration_png_path = run_dir / "calibration_curve.png"
    plot_confusion_png_path = run_dir / "confusion_matrices.png"

    report_path.write_text(render_monte_carlo_report(benchmark_result), encoding="utf-8")
    summary_json_path.write_text(json.dumps(asdict(benchmark_result.summary), indent=2, sort_keys=True), encoding="utf-8")
    plot_accuracy_png_path.write_bytes(render_monte_carlo_accuracy_png_bytes(benchmark_result))
    plot_posterior_png_path.write_bytes(render_monte_carlo_posterior_png_bytes(benchmark_result))
    plot_time_to_confidence_png_path.write_bytes(render_monte_carlo_time_to_confidence_png_bytes(benchmark_result))
    plot_time_to_correct_png_path.write_bytes(render_monte_carlo_time_to_correct_png_bytes(benchmark_result))
    plot_calibration_png_path.write_bytes(render_monte_carlo_calibration_png_bytes(benchmark_result))
    plot_confusion_png_path.write_bytes(render_monte_carlo_confusion_png_bytes(benchmark_result))

    class_names = tuple(benchmark_result.final_confusion)
    metrics_fieldnames = [
        "step",
        "time",
        "sample_count",
        "accuracy",
        "mean_true_class_posterior",
        "median_true_class_posterior",
        "q10_true_class_posterior",
        "q25_true_class_posterior",
        "q75_true_class_posterior",
        "q90_true_class_posterior",
        "mean_confidence",
        "mean_margin",
        "abstain_rate",
    ]
    write_csv(metrics_by_time_path, [dict(row) for row in benchmark_result.metrics_by_time], metrics_fieldnames)
    write_csv(
        time_to_confidence_path,
        [dict(row) for row in benchmark_result.time_to_confidence],
        ["trajectory_id", "scenario_name", "true_class", "time_to_confidence", "reached_confidence", "final_confidence", "final_class"],
    )
    write_csv(
        time_to_correct_classification_path,
        [dict(row) for row in benchmark_result.time_to_correct_classification],
        ["trajectory_id", "scenario_name", "true_class", "time_to_correct_classification", "reached_correct", "final_confidence", "final_class"],
    )
    write_csv(
        calibration_bins_path,
        [dict(row) for row in benchmark_result.calibration_bins],
        ["bin_index", "bin_lower", "bin_upper", "count", "accuracy", "mean_confidence", "gap"],
    )
    write_csv(confusion_final_path, _confusion_to_rows(benchmark_result.final_confusion, class_names), ["true_class", *class_names, "unknown", "total"])
    write_csv(
        confusion_confidence_gated_path,
        _confusion_to_rows(benchmark_result.confidence_gated_confusion, class_names),
        ["true_class", *class_names, "unknown", "total"],
    )

    config_path = run_dir / "monte_carlo_config.yaml"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: monte_carlo_accumulator",
                f"  seed: {seed}",
                "classifier:",
                f"  forgetting_factor: {forgetting_factor}",
                f"  confidence_threshold: {confidence_threshold}",
                f"  confidence_gate_threshold: {confidence_gate_threshold}",
                "dataset:",
                f"  trajectories_per_class: {trajectories_per_class}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "trajectories_per_class": trajectories_per_class,
                "class_names": list(class_names),
                "trajectory_count": benchmark_result.summary.total_trajectories,
                "steps_per_trajectory": benchmark_result.summary.total_steps // max(benchmark_result.summary.total_trajectories, 1),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return MonteCarloBenchmarkArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_json_path=summary_json_path,
        metrics_by_time_path=metrics_by_time_path,
        time_to_confidence_path=time_to_confidence_path,
        time_to_correct_classification_path=time_to_correct_classification_path,
        calibration_bins_path=calibration_bins_path,
        confusion_final_path=confusion_final_path,
        confusion_confidence_gated_path=confusion_confidence_gated_path,
        plot_accuracy_png_path=plot_accuracy_png_path,
        plot_posterior_png_path=plot_posterior_png_path,
        plot_time_to_confidence_png_path=plot_time_to_confidence_png_path,
        plot_time_to_correct_png_path=plot_time_to_correct_png_path,
        plot_calibration_png_path=plot_calibration_png_path,
        plot_confusion_png_path=plot_confusion_png_path,
    )
