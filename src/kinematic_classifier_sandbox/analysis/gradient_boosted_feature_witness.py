from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import write_comparison_summary_csv
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt


CLASS_NAMES = ("early_push_late_brake", "early_brake_late_push")
TRAIN_LABELS = {CLASS_NAMES[0]: 1, CLASS_NAMES[1]: -1}


@dataclass(frozen=True, slots=True)
class FeatureHeadroomTrajectory:
    trajectory_id: str
    true_class: str
    split: str
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_position: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FeatureHeadroomRow:
    trajectory_id: str
    true_class: str
    split: str
    slope: float
    quadratic_proxy: float
    early_residual_mean: float
    late_residual_mean: float
    residual_energy: float


@dataclass(frozen=True, slots=True)
class BoostedStump:
    round_index: int
    feature_name: str
    threshold: float
    polarity: int
    alpha: float
    weighted_error: float


@dataclass(frozen=True, slots=True)
class FeatureHeadroomPredictionRow:
    trajectory_id: str
    true_class: str
    split: str
    windowed_predicted_class: str
    windowed_confidence: float
    boosted_predicted_class: str
    boosted_confidence: float
    boosted_margin: float


@dataclass(frozen=True, slots=True)
class FeatureHeadroomResult:
    trajectories: tuple[FeatureHeadroomTrajectory, ...]
    feature_rows: tuple[FeatureHeadroomRow, ...]
    stumps: tuple[BoostedStump, ...]
    prediction_rows: tuple[FeatureHeadroomPredictionRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class FeatureHeadroomArtifacts:
    run_dir: Path
    trajectory_path: Path
    feature_matrix_path: Path
    stump_summary_path: Path
    prediction_summary_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _normalize_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
    max_score = max(log_scores.values())
    scaled = {label: math.exp(value - max_score) for label, value in log_scores.items()}
    norm = max(sum(scaled.values()), 1.0e-12)
    return {label: value / norm for label, value in scaled.items()}


def _segment_pattern(class_name: str) -> tuple[float, ...]:
    if class_name == "early_push_late_brake":
        return (1.2, 1.2, -1.2, -1.2)
    return (-1.2, -1.2, 1.2, 1.2)


def _make_trajectory(*, class_name: str, example_index: int, seed: int) -> FeatureHeadroomTrajectory:
    rng = random.Random(seed)
    times = tuple(float(index) for index in range(13))
    base_velocity = 1.0
    residual_velocity = [0.0] * (len(times) - 1)
    pattern = _segment_pattern(class_name)
    for segment_index, delta in enumerate(pattern):
        start = segment_index * 3
        for offset in range(3):
            residual_velocity[start + offset] += delta
    true_position = [0.0]
    for velocity_offset in residual_velocity:
        true_position.append(true_position[-1] + base_velocity + velocity_offset)
    measurement_sigma = 0.18
    measurements = tuple(value + rng.gauss(0.0, measurement_sigma) for value in true_position)
    split = "train" if example_index < 8 else "test"
    return FeatureHeadroomTrajectory(
        trajectory_id=f"{class_name}_{example_index}",
        true_class=class_name,
        split=split,
        times=times,
        measurements=measurements,
        true_position=tuple(true_position),
    )


def _extract_features(trajectory: FeatureHeadroomTrajectory) -> FeatureHeadroomRow:
    measurements = trajectory.measurements
    velocities = [measurements[index + 1] - measurements[index] for index in range(len(measurements) - 1)]
    slope = (measurements[-1] - measurements[0]) / max(len(measurements) - 1, 1)
    midpoint = len(measurements) // 2
    quadratic_proxy = measurements[-1] - 2.0 * measurements[midpoint] + measurements[0]
    centered = [value - slope for value in velocities]
    early_residual_mean = sum(centered[:6]) / 6.0
    late_residual_mean = sum(centered[6:]) / 6.0
    residual_energy = sum(abs(value) for value in centered) / len(centered)
    return FeatureHeadroomRow(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        split=trajectory.split,
        slope=float(slope),
        quadratic_proxy=float(quadratic_proxy),
        early_residual_mean=float(early_residual_mean),
        late_residual_mean=float(late_residual_mean),
        residual_energy=float(residual_energy),
    )


def _windowed_predict(row: FeatureHeadroomRow) -> tuple[str, float]:
    log_scores = {
        CLASS_NAMES[0]: -0.5 * ((row.slope - 1.0) / 0.20) ** 2 - 0.5 * (row.quadratic_proxy / 0.90) ** 2,
        CLASS_NAMES[1]: -0.5 * ((row.slope - 1.0) / 0.20) ** 2 - 0.5 * (row.quadratic_proxy / 0.90) ** 2,
    }
    weights = _normalize_log_scores(log_scores)
    predicted = max(weights, key=weights.get)
    return predicted, float(weights[predicted])


def _stump_prediction(value: float, *, threshold: float, polarity: int) -> int:
    prediction = 1 if value >= threshold else -1
    return prediction if polarity > 0 else -prediction


def _candidate_thresholds(values: list[float]) -> list[float]:
    ordered = sorted(values)
    thresholds = [ordered[0] - 1.0e-6, ordered[-1] + 1.0e-6]
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        thresholds.append(0.5 * (left + right))
    return thresholds


def _train_boosted_stumps(rows: tuple[FeatureHeadroomRow, ...], *, rounds: int = 4) -> tuple[BoostedStump, ...]:
    train_rows = [row for row in rows if row.split == "train"]
    weights = [1.0 / len(train_rows)] * len(train_rows)
    feature_names = ("early_residual_mean", "late_residual_mean", "residual_energy")
    stumps: list[BoostedStump] = []
    labels = [TRAIN_LABELS[row.true_class] for row in train_rows]

    for round_index in range(rounds):
        best_feature = feature_names[0]
        best_threshold = 0.0
        best_polarity = 1
        best_predictions = [1] * len(train_rows)
        best_error = float("inf")
        for feature_name in feature_names:
            values = [float(getattr(row, feature_name)) for row in train_rows]
            for threshold in _candidate_thresholds(values):
                for polarity in (1, -1):
                    predictions = [_stump_prediction(value, threshold=threshold, polarity=polarity) for value in values]
                    error = sum(weight for weight, prediction, label in zip(weights, predictions, labels, strict=True) if prediction != label)
                    if error < best_error:
                        best_error = error
                        best_feature = feature_name
                        best_threshold = threshold
                        best_polarity = polarity
                        best_predictions = predictions
        clipped_error = min(max(best_error, 1.0e-9), 1.0 - 1.0e-9)
        alpha = 0.5 * math.log((1.0 - clipped_error) / clipped_error)
        updated = [
            weight * math.exp(-alpha * label * prediction)
            for weight, label, prediction in zip(weights, labels, best_predictions, strict=True)
        ]
        norm = max(sum(updated), 1.0e-12)
        weights = [value / norm for value in updated]
        stumps.append(
            BoostedStump(
                round_index=round_index,
                feature_name=best_feature,
                threshold=float(best_threshold),
                polarity=best_polarity,
                alpha=float(alpha),
                weighted_error=float(best_error),
            )
        )
    return tuple(stumps)


def _predict_boosted(row: FeatureHeadroomRow, stumps: tuple[BoostedStump, ...]) -> tuple[str, float, float]:
    margin = 0.0
    for stump in stumps:
        value = float(getattr(row, stump.feature_name))
        margin += stump.alpha * _stump_prediction(value, threshold=stump.threshold, polarity=stump.polarity)
    probability = 1.0 / (1.0 + math.exp(-2.0 * margin))
    predicted = CLASS_NAMES[0] if margin >= 0.0 else CLASS_NAMES[1]
    confidence = probability if predicted == CLASS_NAMES[0] else 1.0 - probability
    return predicted, float(confidence), float(margin)


def analyze_feature_headroom_frontier(*, seed: int = 811, trajectories_per_class: int = 12) -> FeatureHeadroomResult:
    trajectories = tuple(
        _make_trajectory(
            class_name=class_name,
            example_index=example_index,
            seed=seed + class_index * 100 + example_index,
        )
        for class_index, class_name in enumerate(CLASS_NAMES)
        for example_index in range(trajectories_per_class)
    )
    feature_rows = tuple(_extract_features(trajectory) for trajectory in trajectories)
    stumps = _train_boosted_stumps(feature_rows)
    prediction_rows: list[FeatureHeadroomPredictionRow] = []
    for row in feature_rows:
        windowed_predicted_class, windowed_confidence = _windowed_predict(row)
        boosted_predicted_class, boosted_confidence, boosted_margin = _predict_boosted(row, stumps)
        prediction_rows.append(
            FeatureHeadroomPredictionRow(
                trajectory_id=row.trajectory_id,
                true_class=row.true_class,
                split=row.split,
                windowed_predicted_class=windowed_predicted_class,
                windowed_confidence=windowed_confidence,
                boosted_predicted_class=boosted_predicted_class,
                boosted_confidence=boosted_confidence,
                boosted_margin=boosted_margin,
            )
        )
    test_rows = [row for row in prediction_rows if row.split == "test"]
    train_rows = [row for row in prediction_rows if row.split == "train"]
    windowed_test_accuracy = sum(1.0 if row.windowed_predicted_class == row.true_class else 0.0 for row in test_rows) / len(test_rows)
    boosted_test_accuracy = sum(1.0 if row.boosted_predicted_class == row.true_class else 0.0 for row in test_rows) / len(test_rows)
    boosted_train_accuracy = sum(1.0 if row.boosted_predicted_class == row.true_class else 0.0 for row in train_rows) / len(train_rows)
    mean_boosted_margin = sum(abs(row.boosted_margin) for row in test_rows) / len(test_rows)
    promotion_decision = (
        "promote_gradient_boosted_features_for_feature_headroom"
        if boosted_test_accuracy >= 0.85
        and boosted_test_accuracy >= windowed_test_accuracy + 0.25
        and boosted_train_accuracy >= 0.85
        else "revise_gradient_boosted_feature_witness"
    )
    metrics: dict[str, float | int | str] = {
        "study_id": "feature_headroom_frontier_v1",
        "seed": seed,
        "trajectory_count": len(trajectories),
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "windowed_test_accuracy": float(windowed_test_accuracy),
        "boosted_test_accuracy": float(boosted_test_accuracy),
        "boosted_train_accuracy": float(boosted_train_accuracy),
        "mean_boosted_test_margin": float(mean_boosted_margin),
        "promotion_decision": promotion_decision,
    }
    return FeatureHeadroomResult(
        trajectories=trajectories,
        feature_rows=feature_rows,
        stumps=stumps,
        prediction_rows=tuple(prediction_rows),
        metrics=metrics,
    )


def _render_trajectory_examples(result: FeatureHeadroomResult):
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.0), sharex=True)
    palette = {CLASS_NAMES[0]: "#2563eb", CLASS_NAMES[1]: "#dc2626"}
    for axis, class_name in zip(axes, CLASS_NAMES, strict=True):
        trajectory = next(
            row for row in result.trajectories if row.true_class == class_name and row.split == "test"
        )
        axis.plot(trajectory.times, trajectory.true_position, color=palette[class_name], linewidth=2.0, label="truth")
        axis.scatter(trajectory.times, trajectory.measurements, color=palette[class_name], s=18, alpha=0.75, label="measurements")
        axis.axvspan(0.0, 6.0, color="#86efac", alpha=0.12)
        axis.axvspan(6.0, 12.0, color="#fca5a5", alpha=0.12)
        axis.set_title(class_name, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("position")
        axis.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    axes[-1].set_xlabel("time")
    fig.suptitle("Feature Headroom Witness Trajectories", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _render_feature_scatter(result: FeatureHeadroomResult):
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    palette = {CLASS_NAMES[0]: "#2563eb", CLASS_NAMES[1]: "#dc2626"}
    for row in result.feature_rows:
        ax.scatter(row.early_residual_mean, row.late_residual_mean, color=palette[row.true_class], s=34, alpha=0.85)
    ax.set_xlabel("early residual mean")
    ax.set_ylabel("late residual mean")
    ax.set_title("Engineered Feature Headroom", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    return fig


def _render_metric_bars(result: FeatureHeadroomResult):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels = ("windowed_test_accuracy", "boosted_test_accuracy", "boosted_train_accuracy")
    values = [float(result.metrics[label]) for label in labels]
    ax.bar(range(len(labels)), values, color=("#9ca3af", "#2563eb", "#16a34a"), width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(("windowed test", "boosted test", "boosted train"), rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Feature-Learner Frontier", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_feature_headroom_frontier_artifacts(
    output_dir: str | Path,
    *,
    result: FeatureHeadroomResult | None = None,
    seed: int = 811,
    trajectories_per_class: int = 12,
) -> FeatureHeadroomArtifacts:
    witness = result or analyze_feature_headroom_frontier(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
    )
    run_dir = Path(output_dir) / "feature_headroom_frontier_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = run_dir / "trajectory_summary.csv"
    feature_matrix_path = run_dir / "feature_matrix.csv"
    stump_summary_path = run_dir / "boosted_stump_summary.csv"
    prediction_summary_path = run_dir / "prediction_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "feature_headroom_frontier_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    trajectory_plot_path = plots_dir / "trajectory_examples.png"
    scatter_plot_path = plots_dir / "engineered_feature_scatter.png"
    metric_plot_path = plots_dir / "metric_bars.png"

    write_csv(
        trajectory_path,
        [
            {
                "trajectory_id": row.trajectory_id,
                "true_class": row.true_class,
                "split": row.split,
                "times": "|".join(f"{value:.1f}" for value in row.times),
                "measurements": "|".join(f"{value:.4f}" for value in row.measurements),
            }
            for row in witness.trajectories
        ],
        ["trajectory_id", "true_class", "split", "times", "measurements"],
    )
    write_csv(feature_matrix_path, [asdict(row) for row in witness.feature_rows], list(FeatureHeadroomRow.__dataclass_fields__.keys()))
    write_csv(stump_summary_path, [asdict(row) for row in witness.stumps], list(BoostedStump.__dataclass_fields__.keys()))
    write_csv(prediction_summary_path, [asdict(row) for row in witness.prediction_rows], list(FeatureHeadroomPredictionRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [witness.metrics], filename="summary.csv")
    write_csv(metrics_path, [witness.metrics], list(witness.metrics.keys()))

    report_lines = [
        "# Gradient Boosted Features",
        "",
        "- Study: `feature_headroom_frontier_v1`",
        "- Simpler baseline: global windowed summary",
        "- Candidate method: boosted decision stumps on engineered features",
        "",
        "## Metrics",
        "",
        f"- windowed test accuracy: `{float(witness.metrics['windowed_test_accuracy']):.3f}`",
        f"- boosted test accuracy: `{float(witness.metrics['boosted_test_accuracy']):.3f}`",
        f"- boosted train accuracy: `{float(witness.metrics['boosted_train_accuracy']):.3f}`",
        f"- decision: `{witness.metrics['promotion_decision']}`",
        "",
        "## Claim Boundary",
        "",
        "This witness justifies a low-risk nonlinear learner on engineered features.",
        "It does not claim full external gradient-boosting parity or broad superiority over stronger TSC families.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Previous method: `windowed`",
        "- Failure mode: nonlinear headroom inside engineered feature space",
        "- Candidate method: `gradient_boosted_features`",
        f"- Improvement: accuracy `{float(witness.metrics['windowed_test_accuracy']):.3f}` -> `{float(witness.metrics['boosted_test_accuracy']):.3f}`",
        f"- Complexity: `{len(witness.stumps)}` boosted stumps over explicit engineered features",
        f"- Decision: `{witness.metrics['promotion_decision']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    trajectory_plot_path.write_bytes(_figure_to_png(_render_trajectory_examples(witness)))
    scatter_plot_path.write_bytes(_figure_to_png(_render_feature_scatter(witness)))
    metric_plot_path.write_bytes(_figure_to_png(_render_metric_bars(witness)))

    return FeatureHeadroomArtifacts(
        run_dir=run_dir,
        trajectory_path=trajectory_path,
        feature_matrix_path=feature_matrix_path,
        stump_summary_path=stump_summary_path,
        prediction_summary_path=prediction_summary_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(trajectory_plot_path, scatter_plot_path, metric_plot_path),
    )


__all__ = [
    "BoostedStump",
    "FeatureHeadroomArtifacts",
    "FeatureHeadroomPredictionRow",
    "FeatureHeadroomResult",
    "FeatureHeadroomRow",
    "FeatureHeadroomTrajectory",
    "analyze_feature_headroom_frontier",
    "write_feature_headroom_frontier_artifacts",
]
