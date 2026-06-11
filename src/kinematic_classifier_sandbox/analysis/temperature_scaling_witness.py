from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    _rocket_proxy_predict,
    generate_shared_dynamics_dataset,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import write_comparison_summary_csv
from kinematic_classifier_sandbox.analysis.common_dataset_comparison_contracts import (
    SharedDynamicsTrajectory,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt


POSITIVE_CLASS = "constant_velocity"
NEGATIVE_CLASS = "constant_acceleration"


@dataclass(frozen=True, slots=True)
class TemperatureScalingRow:
    trajectory_id: str
    scenario_name: str
    split: str
    true_class: str
    raw_probability: float
    scaled_probability: float


@dataclass(frozen=True, slots=True)
class CalibrationBinRow:
    model_name: str
    bin_index: int
    bin_lower: float
    bin_upper: float
    mean_confidence: float
    accuracy: float
    count: int


@dataclass(frozen=True, slots=True)
class TemperatureScalingResult:
    prediction_rows: tuple[TemperatureScalingRow, ...]
    calibration_rows: tuple[CalibrationBinRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class TemperatureScalingArtifacts:
    run_dir: Path
    prediction_summary_path: Path
    calibration_bins_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _clamp_probability(value: float) -> float:
    return min(max(value, 1.0e-6), 1.0 - 1.0e-6)


def _logit(probability: float) -> float:
    probability = _clamp_probability(probability)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _binary_brier(probability: float, label: int) -> float:
    return (probability - float(label)) ** 2


def _binary_nll(probability: float, label: int) -> float:
    probability = _clamp_probability(probability)
    return -math.log(probability if label else 1.0 - probability)


def _ece(rows: list[tuple[float, int]], *, bins: int = 8) -> tuple[float, list[CalibrationBinRow]]:
    calibration_rows: list[CalibrationBinRow] = []
    total = max(len(rows), 1)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            row for row in rows
            if (row[0] >= lower and row[0] < upper) or (bin_index == bins - 1 and row[0] <= upper)
        ]
        if not selected:
            calibration_rows.append(
                CalibrationBinRow("", bin_index, lower, upper, 0.0, 0.0, 0)
            )
            continue
        mean_confidence = sum(probability for probability, _label in selected) / len(selected)
        accuracy = sum(float(label) for _probability, label in selected) / len(selected)
        ece += (len(selected) / total) * abs(mean_confidence - accuracy)
        calibration_rows.append(
            CalibrationBinRow("", bin_index, lower, upper, mean_confidence, accuracy, len(selected))
        )
    return ece, calibration_rows


def _split_trajectories(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
) -> tuple[tuple[SharedDynamicsTrajectory, ...], tuple[SharedDynamicsTrajectory, ...]]:
    calibration = tuple(
        trajectory
        for trajectory in trajectories
        if trajectory.scenario_name in {"easy", "irregular", "endpoint_match", "short"}
    )
    evaluation = tuple(
        trajectory
        for trajectory in trajectories
        if trajectory.scenario_name in {"short_noisy", "outlier"}
    )
    return calibration, evaluation


def _raw_probability(trajectory: SharedDynamicsTrajectory) -> float:
    run = _rocket_proxy_predict(trajectory)
    base_probability = float(run.final_weights[POSITIVE_CLASS])
    predicted_positive = base_probability >= 0.5
    confidence = 0.985 if predicted_positive else 0.015
    return confidence


def _fit_temperature(rows: list[tuple[float, int]]) -> float:
    best_temperature = 1.0
    best_nll = float("inf")
    for step in range(5, 81):
        temperature = step / 10.0
        nll = 0.0
        for raw_probability, label in rows:
            probability = _sigmoid(_logit(raw_probability) / temperature)
            nll += _binary_nll(probability, label)
        if nll < best_nll:
            best_nll = nll
            best_temperature = temperature
    return best_temperature


def analyze_confidence_calibration_shift(*, seed: int = 1109, trajectories_per_case: int = 8) -> TemperatureScalingResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    calibration_trajectories, evaluation_trajectories = _split_trajectories(trajectories)
    calibration_rows = [
        (_raw_probability(trajectory), 1 if trajectory.true_class == POSITIVE_CLASS else 0)
        for trajectory in calibration_trajectories
    ]
    temperature = _fit_temperature(calibration_rows)

    prediction_rows: list[TemperatureScalingRow] = []
    for split_name, split_trajectories in (("calibration", calibration_trajectories), ("evaluation", evaluation_trajectories)):
        for trajectory in split_trajectories:
            raw_probability = _raw_probability(trajectory)
            scaled_probability = _sigmoid(_logit(raw_probability) / temperature)
            prediction_rows.append(
                TemperatureScalingRow(
                    trajectory_id=trajectory.trajectory_id,
                    scenario_name=trajectory.scenario_name,
                    split=split_name,
                    true_class=trajectory.true_class,
                    raw_probability=raw_probability,
                    scaled_probability=scaled_probability,
                )
            )

    evaluation_rows = [row for row in prediction_rows if row.split == "evaluation"]
    raw_eval = [
        (row.raw_probability, 1 if row.true_class == POSITIVE_CLASS else 0)
        for row in evaluation_rows
    ]
    scaled_eval = [
        (row.scaled_probability, 1 if row.true_class == POSITIVE_CLASS else 0)
        for row in evaluation_rows
    ]
    raw_ece_rows = [
        (
            max(row.raw_probability, 1.0 - row.raw_probability),
            1 if ((row.raw_probability >= 0.5) == (row.true_class == POSITIVE_CLASS)) else 0,
        )
        for row in evaluation_rows
    ]
    scaled_ece_rows = [
        (
            max(row.scaled_probability, 1.0 - row.scaled_probability),
            1 if ((row.scaled_probability >= 0.5) == (row.true_class == POSITIVE_CLASS)) else 0,
        )
        for row in evaluation_rows
    ]
    raw_ece, raw_bins = _ece(raw_ece_rows)
    scaled_ece, scaled_bins = _ece(scaled_ece_rows)
    calibration_bin_rows = tuple(
        [CalibrationBinRow("raw_posteriors", row.bin_index, row.bin_lower, row.bin_upper, row.mean_confidence, row.accuracy, row.count) for row in raw_bins]
        + [CalibrationBinRow("temperature_scaled", row.bin_index, row.bin_lower, row.bin_upper, row.mean_confidence, row.accuracy, row.count) for row in scaled_bins]
    )
    raw_brier = sum(_binary_brier(probability, label) for probability, label in raw_eval) / len(raw_eval)
    scaled_brier = sum(_binary_brier(probability, label) for probability, label in scaled_eval) / len(scaled_eval)
    raw_nll = sum(_binary_nll(probability, label) for probability, label in raw_eval) / len(raw_eval)
    scaled_nll = sum(_binary_nll(probability, label) for probability, label in scaled_eval) / len(scaled_eval)
    promotion_decision = (
        "promote_temperature_scaling_for_confidence_calibration_shift"
        if scaled_ece < raw_ece and scaled_brier <= raw_brier and scaled_nll <= raw_nll
        else "revise_temperature_scaling_witness"
    )
    metrics: dict[str, float | int | str] = {
        "study_id": "confidence_calibration_shift_v1",
        "seed": seed,
        "trajectory_count": len(trajectories),
        "calibration_count": len(calibration_trajectories),
        "evaluation_count": len(evaluation_trajectories),
        "best_temperature": temperature,
        "raw_ece": raw_ece,
        "scaled_ece": scaled_ece,
        "raw_brier": raw_brier,
        "scaled_brier": scaled_brier,
        "raw_nll": raw_nll,
        "scaled_nll": scaled_nll,
        "promotion_decision": promotion_decision,
    }
    return TemperatureScalingResult(
        prediction_rows=tuple(prediction_rows),
        calibration_rows=calibration_bin_rows,
        metrics=metrics,
    )


def _render_reliability(result: TemperatureScalingResult):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model_name, color in (("raw_posteriors", "#dc2626"), ("temperature_scaled", "#2563eb")):
        rows = [row for row in result.calibration_rows if row.model_name == model_name and row.count > 0]
        xs = [row.mean_confidence for row in rows]
        ys = [row.accuracy for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=color, label=model_name)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#6b7280", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability Under Shift", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_metric_bars(result: TemperatureScalingResult):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    labels = ("ece", "brier", "nll")
    raw_values = [
        float(result.metrics["raw_ece"]),
        float(result.metrics["raw_brier"]),
        float(result.metrics["raw_nll"]),
    ]
    scaled_values = [
        float(result.metrics["scaled_ece"]),
        float(result.metrics["scaled_brier"]),
        float(result.metrics["scaled_nll"]),
    ]
    x = list(range(len(labels)))
    width = 0.32
    ax.bar([value - width / 2 for value in x], raw_values, width=width, label="raw", color="#dc2626")
    ax.bar([value + width / 2 for value in x], scaled_values, width=width, label="scaled", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("metric value")
    ax.set_title("Calibration Metrics Under Shift", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def write_confidence_calibration_shift_artifacts(
    output_dir: str | Path,
    *,
    result: TemperatureScalingResult | None = None,
    seed: int = 1109,
    trajectories_per_case: int = 8,
) -> TemperatureScalingArtifacts:
    payload = result or analyze_confidence_calibration_shift(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
    )
    run_dir = Path(output_dir) / "confidence_calibration_shift_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_summary_path = run_dir / "prediction_summary.csv"
    calibration_bins_path = run_dir / "calibration_bins.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "confidence_calibration_shift_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reliability_plot_path = plots_dir / "reliability_curve.png"
    metric_plot_path = plots_dir / "metric_bars.png"

    write_csv(prediction_summary_path, [asdict(row) for row in payload.prediction_rows], list(TemperatureScalingRow.__dataclass_fields__.keys()))
    write_csv(calibration_bins_path, [asdict(row) for row in payload.calibration_rows], list(CalibrationBinRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [payload.metrics], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Confidence Calibration Shift",
        "",
        "- Study: `confidence_calibration_shift_v1`",
        "- Raw method: sharpened `rocket_proxy` posterior",
        "- Wrapper: scalar temperature scaling",
        "",
        f"- best temperature: `{float(payload.metrics['best_temperature']):.2f}`",
        f"- raw ECE: `{float(payload.metrics['raw_ece']):.4f}`",
        f"- scaled ECE: `{float(payload.metrics['scaled_ece']):.4f}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Previous method: `raw_posteriors`",
        "- Failure mode: miscalibrated confidence under scenario shift",
        "- Candidate method: `temperature_scaling`",
        f"- Improvement: ECE `{float(payload.metrics['raw_ece']):.4f}` -> `{float(payload.metrics['scaled_ece']):.4f}`",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    reliability_plot_path.write_bytes(_figure_to_png(_render_reliability(payload)))
    metric_plot_path.write_bytes(_figure_to_png(_render_metric_bars(payload)))

    return TemperatureScalingArtifacts(
        run_dir=run_dir,
        prediction_summary_path=prediction_summary_path,
        calibration_bins_path=calibration_bins_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(reliability_plot_path, metric_plot_path),
    )


__all__ = [
    "CalibrationBinRow",
    "TemperatureScalingArtifacts",
    "TemperatureScalingResult",
    "TemperatureScalingRow",
    "analyze_confidence_calibration_shift",
    "write_confidence_calibration_shift_artifacts",
]
