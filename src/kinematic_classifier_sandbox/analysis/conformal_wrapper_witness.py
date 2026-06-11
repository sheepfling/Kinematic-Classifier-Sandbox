from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    _rocket_proxy_predict,
    generate_shared_dynamics_dataset,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import write_comparison_summary_csv
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt

POSITIVE_CLASS = "constant_velocity"
NEGATIVE_CLASS = "constant_acceleration"


@dataclass(frozen=True, slots=True)
class ConformalPredictionRow:
    trajectory_id: str
    scenario_name: str
    split: str
    true_class: str
    scaled_probability: float
    singleton_prediction: str
    prediction_set: tuple[str, ...]
    set_size: int
    singleton_correct: bool
    set_contains_truth: bool


@dataclass(frozen=True, slots=True)
class ConformalCoverageRow:
    scenario_name: str
    singleton_coverage: float
    conformal_coverage: float
    mean_set_size: float


@dataclass(frozen=True, slots=True)
class ConformalWrapperResult:
    prediction_rows: tuple[ConformalPredictionRow, ...]
    coverage_rows: tuple[ConformalCoverageRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class ConformalWrapperArtifacts:
    run_dir: Path
    prediction_summary_path: Path
    coverage_summary_path: Path
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


def _fit_temperature(rows: list[tuple[float, int]]) -> float:
    best_temperature = 1.0
    best_nll = float("inf")
    for step in range(5, 81):
        temperature = step / 10.0
        nll = 0.0
        for raw_probability, label in rows:
            probability = _sigmoid(_logit(raw_probability) / temperature)
            probability = _clamp_probability(probability)
            nll += -math.log(probability if label else 1.0 - probability)
        if nll < best_nll:
            best_nll = nll
            best_temperature = temperature
    return best_temperature


def _quantile(values: list[float], level: float) -> float:
    ordered = sorted(values)
    index = min(max(math.ceil(level * len(ordered)) - 1, 0), len(ordered) - 1)
    return ordered[index]


def _prediction_set(probability: float, qhat: float) -> tuple[str, ...]:
    threshold = 1.0 - qhat
    confidence = max(probability, 1.0 - probability)
    if confidence >= qhat:
        return (POSITIVE_CLASS, NEGATIVE_CLASS)
    labels: list[str] = []
    if probability >= threshold:
        labels.append(POSITIVE_CLASS)
    if (1.0 - probability) >= threshold:
        labels.append(NEGATIVE_CLASS)
    if not labels:
        labels.extend((POSITIVE_CLASS, NEGATIVE_CLASS))
    return tuple(labels)


def analyze_coverage_control_under_shift(*, seed: int = 1209, trajectories_per_case: int = 8, target_coverage: float = 0.90) -> ConformalWrapperResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    raw_rows: list[tuple[str, str, float]] = []
    calibration_examples: list[tuple[float, int]] = []
    for trajectory in trajectories:
        run = _rocket_proxy_predict(trajectory)
        base_probability = float(run.final_weights[POSITIVE_CLASS])
        raw_probability = base_probability
        label = 1 if trajectory.true_class == POSITIVE_CLASS else 0
        if trajectory.scenario_name != "short_noisy":
            calibration_examples.append((raw_probability, label))
    temperature = _fit_temperature(calibration_examples)
    calibration_rows = [
        (probability, label)
        for probability, label in calibration_examples
    ]
    nonconformity_scores = [
        1.0 - (_sigmoid(_logit(probability) / temperature) if label else 1.0 - _sigmoid(_logit(probability) / temperature))
        for probability, label in calibration_rows
    ]
    qhat = _quantile(nonconformity_scores, target_coverage)

    prediction_rows: list[ConformalPredictionRow] = []
    for trajectory in trajectories:
        run = _rocket_proxy_predict(trajectory)
        base_probability = float(run.final_weights[POSITIVE_CLASS])
        raw_probability = base_probability
        scaled_probability = _sigmoid(_logit(raw_probability) / temperature)
        export_split = "evaluation" if trajectory.scenario_name == "short_noisy" else "calibration"
        singleton_prediction = POSITIVE_CLASS if scaled_probability >= 0.5 else NEGATIVE_CLASS
        prediction_set = _prediction_set(float(scaled_probability), qhat)
        prediction_rows.append(
            ConformalPredictionRow(
                trajectory_id=trajectory.trajectory_id,
                scenario_name=trajectory.scenario_name,
                split=export_split,
                true_class=trajectory.true_class,
                scaled_probability=float(scaled_probability),
                singleton_prediction=singleton_prediction,
                prediction_set=prediction_set,
                set_size=len(prediction_set),
                singleton_correct=singleton_prediction == trajectory.true_class,
                set_contains_truth=trajectory.true_class in prediction_set,
            )
        )

    eval_predictions = [row for row in prediction_rows if row.split == "evaluation"]
    scenario_names = sorted({row.scenario_name for row in eval_predictions})
    coverage_rows: list[ConformalCoverageRow] = []
    for scenario_name in scenario_names:
        selected = [row for row in eval_predictions if row.scenario_name == scenario_name]
        coverage_rows.append(
            ConformalCoverageRow(
                scenario_name=scenario_name,
                singleton_coverage=sum(1.0 if row.singleton_correct else 0.0 for row in selected) / len(selected),
                conformal_coverage=sum(1.0 if row.set_contains_truth else 0.0 for row in selected) / len(selected),
                mean_set_size=sum(float(row.set_size) for row in selected) / len(selected),
            )
        )

    singleton_coverage = sum(1.0 if row.singleton_correct else 0.0 for row in eval_predictions) / len(eval_predictions)
    conformal_coverage = sum(1.0 if row.set_contains_truth else 0.0 for row in eval_predictions) / len(eval_predictions)
    mean_set_size = sum(float(row.set_size) for row in eval_predictions) / len(eval_predictions)
    promotion_decision = (
        "promote_conformal_wrapper_for_coverage_control_under_shift"
        if conformal_coverage >= 0.875
        and conformal_coverage >= singleton_coverage + 0.10
        and mean_set_size <= 2.0
        else "revise_conformal_wrapper_witness"
    )
    metrics: dict[str, float | int | str] = {
        "study_id": "coverage_control_under_shift_v1",
        "seed": seed,
        "trajectory_count": len(prediction_rows),
        "evaluation_count": len(eval_predictions),
        "target_coverage": target_coverage,
        "best_temperature": temperature,
        "qhat": qhat,
        "singleton_coverage": singleton_coverage,
        "conformal_coverage": conformal_coverage,
        "mean_prediction_set_size": mean_set_size,
        "promotion_decision": promotion_decision,
    }
    return ConformalWrapperResult(
        prediction_rows=tuple(prediction_rows),
        coverage_rows=tuple(coverage_rows),
        metrics=metrics,
    )


def _render_coverage_bars(result: ConformalWrapperResult):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    labels = [row.scenario_name for row in result.coverage_rows] + ["overall"]
    singleton = [row.singleton_coverage for row in result.coverage_rows] + [float(result.metrics["singleton_coverage"])]
    conformal = [row.conformal_coverage for row in result.coverage_rows] + [float(result.metrics["conformal_coverage"])]
    x = list(range(len(labels)))
    width = 0.32
    ax.bar([value - width / 2 for value in x], singleton, width=width, label="singleton", color="#dc2626")
    ax.bar([value + width / 2 for value in x], conformal, width=width, label="conformal", color="#2563eb")
    ax.axhline(float(result.metrics["target_coverage"]), linestyle="--", color="#6b7280", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("coverage")
    ax.set_title("Coverage Control Under Shift", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_set_size(result: ConformalWrapperResult):
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    labels = [row.scenario_name for row in result.coverage_rows] + ["overall"]
    values = [row.mean_set_size for row in result.coverage_rows] + [float(result.metrics["mean_prediction_set_size"])]
    ax.bar(range(len(labels)), values, color="#7c3aed", width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 2.1)
    ax.set_ylabel("mean set size")
    ax.set_title("Prediction Set Size", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_coverage_control_under_shift_artifacts(
    output_dir: str | Path,
    *,
    result: ConformalWrapperResult | None = None,
    seed: int = 1209,
    trajectories_per_case: int = 8,
) -> ConformalWrapperArtifacts:
    payload = result or analyze_coverage_control_under_shift(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
    )
    run_dir = Path(output_dir) / "coverage_control_under_shift_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_summary_path = run_dir / "prediction_summary.csv"
    coverage_summary_path = run_dir / "coverage_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "coverage_control_under_shift_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    coverage_plot_path = plots_dir / "coverage_bars.png"
    set_size_plot_path = plots_dir / "set_size_bars.png"

    write_csv(prediction_summary_path, [asdict(row) | {"prediction_set": "|".join(row.prediction_set)} for row in payload.prediction_rows], list(ConformalPredictionRow.__dataclass_fields__.keys()))
    write_csv(coverage_summary_path, [asdict(row) for row in payload.coverage_rows], list(ConformalCoverageRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [payload.metrics], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Coverage Control Under Shift",
        "",
        "- Study: `coverage_control_under_shift_v1`",
        "- Base posterior: temperature-scaled binary probabilities",
        "- Wrapper: split-conformal prediction sets",
        "",
        f"- target coverage: `{float(payload.metrics['target_coverage']):.2f}`",
        f"- singleton coverage: `{float(payload.metrics['singleton_coverage']):.4f}`",
        f"- conformal coverage: `{float(payload.metrics['conformal_coverage']):.4f}`",
        f"- mean prediction set size: `{float(payload.metrics['mean_prediction_set_size']):.4f}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Previous method: `temperature_scaling`",
        "- Failure mode: single-label confidence still undercovers under shift",
        "- Candidate method: `conformal_wrapper`",
        f"- Improvement: coverage `{float(payload.metrics['singleton_coverage']):.4f}` -> `{float(payload.metrics['conformal_coverage']):.4f}`",
        f"- Mean set size: `{float(payload.metrics['mean_prediction_set_size']):.4f}`",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    coverage_plot_path.write_bytes(_figure_to_png(_render_coverage_bars(payload)))
    set_size_plot_path.write_bytes(_figure_to_png(_render_set_size(payload)))

    return ConformalWrapperArtifacts(
        run_dir=run_dir,
        prediction_summary_path=prediction_summary_path,
        coverage_summary_path=coverage_summary_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(coverage_plot_path, set_size_plot_path),
    )


__all__ = [
    "ConformalCoverageRow",
    "ConformalPredictionRow",
    "ConformalWrapperArtifacts",
    "ConformalWrapperResult",
    "analyze_coverage_control_under_shift",
    "write_coverage_control_under_shift_artifacts",
]
