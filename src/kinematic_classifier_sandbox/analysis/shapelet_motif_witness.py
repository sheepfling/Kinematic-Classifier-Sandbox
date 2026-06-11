from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import write_comparison_summary_csv
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt


CLASS_NAMES = ("burst_then_brake", "brake_then_burst")
_TEMPLATE_BY_CLASS = {
    "burst_then_brake": (1.2, 1.2, -1.2, -1.2),
    "brake_then_burst": (-1.2, -1.2, 1.2, 1.2),
}


@dataclass(frozen=True, slots=True)
class ShapeletTrajectory:
    trajectory_id: str
    true_class: str
    motif_start_index: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_position: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ShapeletPredictionRow:
    trajectory_id: str
    true_class: str
    windowed_predicted_class: str
    windowed_confidence: float
    shapelet_predicted_class: str
    shapelet_confidence: float
    shapelet_best_match_start_index: int
    shapelet_alignment_error: int


@dataclass(frozen=True, slots=True)
class ShapeletActivationRow:
    trajectory_id: str
    true_class: str
    template_class: str
    start_index: int
    distance: float


@dataclass(frozen=True, slots=True)
class ShapeletMotifWitnessResult:
    trajectories: tuple[ShapeletTrajectory, ...]
    prediction_rows: tuple[ShapeletPredictionRow, ...]
    activation_rows: tuple[ShapeletActivationRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class ShapeletMotifWitnessArtifacts:
    run_dir: Path
    trajectory_path: Path
    prediction_path: Path
    activation_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _normalize_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
    max_score = max(log_scores.values())
    weights = {label: math.exp(score - max_score) for label, score in log_scores.items()}
    normalizer = max(sum(weights.values()), 1.0e-12)
    return {label: value / normalizer for label, value in weights.items()}


def _motif_residuals(class_name: str) -> tuple[float, ...]:
    return _TEMPLATE_BY_CLASS[class_name]


def _make_trajectory(*, class_name: str, example_index: int, seed: int) -> ShapeletTrajectory:
    rng = random.Random(seed)
    times = tuple(float(index) for index in range(13))
    base_velocity = 1.0
    base_position = [0.0]
    for _ in range(len(times) - 1):
        base_position.append(base_position[-1] + base_velocity)
    residual_velocity = [0.0] * (len(times) - 1)
    motif_start_index = 3 + (example_index % 3)
    for offset, delta in enumerate(_motif_residuals(class_name)):
        residual_velocity[motif_start_index + offset] += delta
    position = [0.0]
    for step_index in range(len(times) - 1):
        position.append(position[-1] + base_velocity + residual_velocity[step_index])
    measurement_sigma = 0.14
    measurements = tuple(value + rng.gauss(0.0, measurement_sigma) for value in position)
    return ShapeletTrajectory(
        trajectory_id=f"{class_name}_{example_index}",
        true_class=class_name,
        motif_start_index=motif_start_index,
        times=times,
        measurements=measurements,
        true_position=tuple(position),
    )


def _velocity_residual_window(measurements: tuple[float, ...], start_index: int, window_length: int) -> tuple[float, ...]:
    velocities = [measurements[index + 1] - measurements[index] for index in range(len(measurements) - 1)]
    mean_velocity = sum(velocities) / len(velocities)
    return tuple(velocities[start_index + offset] - mean_velocity for offset in range(window_length))


def _windowed_predict(trajectory: ShapeletTrajectory) -> tuple[str, float]:
    velocities = [trajectory.measurements[index + 1] - trajectory.measurements[index] for index in range(len(trajectory.measurements) - 1)]
    mean_velocity = sum(velocities) / len(velocities)
    variation_energy = sum(abs(value - mean_velocity) for value in velocities) / len(velocities)
    span = trajectory.measurements[-1] - trajectory.measurements[0]
    log_scores = {
        "burst_then_brake": -0.5 * ((mean_velocity - 1.0) / 0.20) ** 2 - 0.5 * ((variation_energy - 0.55) / 0.18) ** 2 - 0.5 * ((span - 12.0) / 0.40) ** 2,
        "brake_then_burst": -0.5 * ((mean_velocity - 1.0) / 0.20) ** 2 - 0.5 * ((variation_energy - 0.55) / 0.18) ** 2 - 0.5 * ((span - 12.0) / 0.40) ** 2,
    }
    weights = _normalize_log_scores(log_scores)
    predicted = max(weights, key=weights.get)
    return predicted, float(weights[predicted])


def _shapelet_predict(trajectory: ShapeletTrajectory) -> tuple[str, float, int, list[ShapeletActivationRow]]:
    template_length = len(next(iter(_TEMPLATE_BY_CLASS.values())))
    log_scores: dict[str, float] = {}
    best_start_index_by_template: dict[str, int] = {}
    activation_rows: list[ShapeletActivationRow] = []
    for template_class, template in _TEMPLATE_BY_CLASS.items():
        best_distance = float("inf")
        best_start_index = 0
        for start_index in range(len(trajectory.measurements) - 1 - template_length + 1):
            observed = _velocity_residual_window(trajectory.measurements, start_index, template_length)
            distance = float(sum((observed_value - template_value) ** 2 for observed_value, template_value in zip(observed, template, strict=True)))
            activation_rows.append(
                ShapeletActivationRow(
                    trajectory_id=trajectory.trajectory_id,
                    true_class=trajectory.true_class,
                    template_class=template_class,
                    start_index=start_index,
                    distance=distance,
                )
            )
            if distance < best_distance:
                best_distance = distance
                best_start_index = start_index
        log_scores[template_class] = -0.5 * best_distance / (0.38**2)
        best_start_index_by_template[template_class] = best_start_index
    weights = _normalize_log_scores(log_scores)
    predicted = max(weights, key=weights.get)
    return predicted, float(weights[predicted]), best_start_index_by_template[predicted], activation_rows


def analyze_shapelet_maneuver_motif_witness(*, seed: int = 709, trajectories_per_class: int = 12) -> ShapeletMotifWitnessResult:
    trajectories = tuple(
        _make_trajectory(
            class_name=class_name,
            example_index=example_index,
            seed=seed + class_index * 100 + example_index,
        )
        for class_index, class_name in enumerate(CLASS_NAMES)
        for example_index in range(trajectories_per_class)
    )
    prediction_rows: list[ShapeletPredictionRow] = []
    activation_rows: list[ShapeletActivationRow] = []
    for trajectory in trajectories:
        windowed_predicted_class, windowed_confidence = _windowed_predict(trajectory)
        shapelet_predicted_class, shapelet_confidence, best_match_start_index, row_activations = _shapelet_predict(trajectory)
        prediction_rows.append(
            ShapeletPredictionRow(
                trajectory_id=trajectory.trajectory_id,
                true_class=trajectory.true_class,
                windowed_predicted_class=windowed_predicted_class,
                windowed_confidence=windowed_confidence,
                shapelet_predicted_class=shapelet_predicted_class,
                shapelet_confidence=shapelet_confidence,
                shapelet_best_match_start_index=best_match_start_index,
                shapelet_alignment_error=abs(best_match_start_index - trajectory.motif_start_index),
            )
        )
        activation_rows.extend(row_activations)

    windowed_accuracy = sum(1.0 if row.windowed_predicted_class == row.true_class else 0.0 for row in prediction_rows) / len(prediction_rows)
    shapelet_accuracy = sum(1.0 if row.shapelet_predicted_class == row.true_class else 0.0 for row in prediction_rows) / len(prediction_rows)
    alignment_rate = sum(1.0 if row.shapelet_alignment_error <= 1 else 0.0 for row in prediction_rows) / len(prediction_rows)
    mean_windowed_confidence = sum(row.windowed_confidence for row in prediction_rows) / len(prediction_rows)
    mean_shapelet_confidence = sum(row.shapelet_confidence for row in prediction_rows) / len(prediction_rows)
    promotion_decision = (
        "promote_shapelet_for_localized_maneuver_motif"
        if shapelet_accuracy >= 0.85
        and shapelet_accuracy >= windowed_accuracy + 0.25
        and alignment_rate >= 0.80
        else "revise_shapelet_motif_witness"
    )
    metrics: dict[str, float | int | str] = {
        "study_id": "shapelet_maneuver_motif_v1",
        "seed": seed,
        "trajectory_count": len(trajectories),
        "windowed_accuracy": float(windowed_accuracy),
        "shapelet_accuracy": float(shapelet_accuracy),
        "shapelet_alignment_rate": float(alignment_rate),
        "mean_windowed_confidence": float(mean_windowed_confidence),
        "mean_shapelet_confidence": float(mean_shapelet_confidence),
        "promotion_decision": promotion_decision,
    }
    return ShapeletMotifWitnessResult(
        trajectories=trajectories,
        prediction_rows=tuple(prediction_rows),
        activation_rows=tuple(activation_rows),
        metrics=metrics,
    )


def _render_shapelet_examples(result: ShapeletMotifWitnessResult):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
    palette = {"burst_then_brake": "#2563eb", "brake_then_burst": "#dc2626"}
    for axis, class_name in zip(axes, CLASS_NAMES, strict=True):
        trajectory = next(trajectory for trajectory in result.trajectories if trajectory.true_class == class_name)
        axis.plot(trajectory.times, trajectory.true_position, color=palette[class_name], linewidth=2.0, label=f"{class_name} truth")
        axis.scatter(trajectory.times, trajectory.measurements, color=palette[class_name], s=20, alpha=0.75, label="measurements")
        start = trajectory.motif_start_index
        axis.axvspan(trajectory.times[start], trajectory.times[start + 4], color="#facc15", alpha=0.18)
        axis.set_title(class_name, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("position")
        axis.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    axes[-1].set_xlabel("time")
    fig.suptitle("Localized Maneuver Motif Witness", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _render_shapelet_distance_profiles(result: ShapeletMotifWitnessResult):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    palette = {"burst_then_brake": "#2563eb", "brake_then_burst": "#dc2626"}
    for axis, class_name in zip(axes, CLASS_NAMES, strict=True):
        trajectory = next(trajectory for trajectory in result.trajectories if trajectory.true_class == class_name)
        relevant = [row for row in result.activation_rows if row.trajectory_id == trajectory.trajectory_id]
        starts = sorted({row.start_index for row in relevant})
        for template_class in CLASS_NAMES:
            ys = [
                next(
                    row.distance
                    for row in relevant
                    if row.template_class == template_class and row.start_index == start_index
                )
                for start_index in starts
            ]
            axis.plot(starts, ys, marker="o", linewidth=2.0, color=palette[template_class], label=template_class)
        axis.axvline(trajectory.motif_start_index, color="#111827", linestyle="--", linewidth=1.2)
        axis.set_title(trajectory.true_class, loc="left", fontsize=11, fontweight="bold")
        axis.set_xlabel("window start")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("shapelet distance")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Template Distance by Window Start", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _render_shapelet_metric_bars(result: ShapeletMotifWitnessResult):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    metrics = result.metrics
    labels = ["windowed_accuracy", "shapelet_accuracy", "shapelet_alignment_rate"]
    values = [float(metrics[label]) for label in labels]
    colors = ("#9ca3af", "#2563eb", "#16a34a")
    ax.bar(range(len(labels)), values, color=colors, width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["windowed", "shapelet", "alignment"], rotation=12, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("value")
    ax.set_title("Shapelet Witness Metrics", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_shapelet_maneuver_motif_witness_artifacts(
    output_dir: str | Path,
    *,
    result: ShapeletMotifWitnessResult | None = None,
    seed: int = 709,
    trajectories_per_class: int = 12,
) -> ShapeletMotifWitnessArtifacts:
    witness = result or analyze_shapelet_maneuver_motif_witness(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
    )
    run_dir = Path(output_dir) / "shapelet_maneuver_motif_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = run_dir / "trajectory_summary.csv"
    prediction_path = run_dir / "prediction_summary.csv"
    activation_path = run_dir / "activation_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "shapelet_maneuver_motif_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    example_plot_path = plots_dir / "motif_examples.png"
    distance_plot_path = plots_dir / "distance_profiles.png"
    metric_plot_path = plots_dir / "metric_bars.png"

    write_csv(
        trajectory_path,
        [
            {
                "trajectory_id": row.trajectory_id,
                "true_class": row.true_class,
                "motif_start_index": row.motif_start_index,
                "times": "|".join(f"{value:.1f}" for value in row.times),
                "measurements": "|".join(f"{value:.4f}" for value in row.measurements),
            }
            for row in witness.trajectories
        ],
        ["trajectory_id", "true_class", "motif_start_index", "times", "measurements"],
    )
    write_csv(prediction_path, [asdict(row) for row in witness.prediction_rows], list(ShapeletPredictionRow.__dataclass_fields__.keys()))
    write_csv(activation_path, [asdict(row) for row in witness.activation_rows], list(ShapeletActivationRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(summary_path, [witness.metrics])
    write_csv(metrics_path, [witness.metrics], list(witness.metrics.keys()))

    report_lines = [
        "# Shapelet Maneuver Motif Witness",
        "",
        "- Study: `shapelet_maneuver_motif_v1`",
        "- Simpler baseline: global windowed summary",
        "- Candidate method: localized shapelet motif scan",
        "",
        "## Metrics",
        "",
        f"- windowed accuracy: `{float(witness.metrics['windowed_accuracy']):.3f}`",
        f"- shapelet accuracy: `{float(witness.metrics['shapelet_accuracy']):.3f}`",
        f"- shapelet alignment rate: `{float(witness.metrics['shapelet_alignment_rate']):.3f}`",
        f"- decision: `{witness.metrics['promotion_decision']}`",
        "",
        "## Claim Boundary",
        "",
        "This witness justifies a localized motif baseline for short maneuver signatures.",
        "It does not claim a general-purpose shapelet library or broader modern TSC coverage.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Previous method: `windowed`",
        "- Failure mode: localized maneuver signature with matched global summaries",
        "- Candidate method: `shapelet`",
        f"- Improvement: accuracy `{float(witness.metrics['windowed_accuracy']):.3f}` -> `{float(witness.metrics['shapelet_accuracy']):.3f}`",
        f"- Alignment: `{float(witness.metrics['shapelet_alignment_rate']):.3f}` within one step of truth",
        f"- Decision: `{witness.metrics['promotion_decision']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    example_plot_path.write_bytes(_figure_to_png(_render_shapelet_examples(witness)))
    distance_plot_path.write_bytes(_figure_to_png(_render_shapelet_distance_profiles(witness)))
    metric_plot_path.write_bytes(_figure_to_png(_render_shapelet_metric_bars(witness)))

    return ShapeletMotifWitnessArtifacts(
        run_dir=run_dir,
        trajectory_path=trajectory_path,
        prediction_path=prediction_path,
        activation_path=activation_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(example_plot_path, distance_plot_path, metric_plot_path),
    )


__all__ = [
    "ShapeletActivationRow",
    "ShapeletMotifWitnessArtifacts",
    "ShapeletMotifWitnessResult",
    "ShapeletPredictionRow",
    "ShapeletTrajectory",
    "analyze_shapelet_maneuver_motif_witness",
    "write_shapelet_maneuver_motif_witness_artifacts",
]
