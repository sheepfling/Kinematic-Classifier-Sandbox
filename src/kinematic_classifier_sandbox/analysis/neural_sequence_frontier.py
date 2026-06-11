from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    _kalman_predict,
    _rocket_proxy_predict,
    _windowed_predict,
    generate_shared_dynamics_dataset,
)
from kinematic_classifier_sandbox.analysis.common_dataset_comparison_contracts import (
    SharedDynamicsTrajectory,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import write_comparison_summary_csv
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt


CLASS_NAMES = ("constant_velocity", "constant_acceleration")
TCN_KERNELS = (
    (1, (-1.0, 1.0)),
    (2, (-1.0, 0.0, 1.0)),
    (4, (-1.0, 0.0, 0.0, 0.0, 1.0)),
)
INCEPTION_KERNELS = (
    (2, (1.0, -2.0, 1.0)),
    (3, (-1.0, 1.0, 1.0, -1.0)),
    (5, (-1.0, 0.5, 1.0, 0.0, -1.0, 0.5)),
)


@dataclass(frozen=True, slots=True)
class NeuralFrontierPredictionRow:
    trajectory_id: str
    scenario_name: str
    true_class: str
    split: str
    method_name: str
    predicted_class: str
    confidence: float


@dataclass(frozen=True, slots=True)
class NeuralFrontierMetricRow:
    method_name: str
    overall_accuracy: float
    test_accuracy: float
    short_noisy_accuracy: float
    endpoint_match_accuracy: float
    applicability_status: str
    claim_level: str


@dataclass(frozen=True, slots=True)
class NeuralSequenceFrontierResult:
    prediction_rows: tuple[NeuralFrontierPredictionRow, ...]
    metric_rows: tuple[NeuralFrontierMetricRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class NeuralSequenceFrontierArtifacts:
    run_dir: Path
    prediction_summary_path: Path
    metric_summary_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _normalize_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
    max_score = max(log_scores.values())
    weights = {label: math.exp(score - max_score) for label, score in log_scores.items()}
    total = max(sum(weights.values()), 1.0e-12)
    return {label: value / total for label, value in weights.items()}


def _series_centered(values: tuple[float, ...]) -> list[float]:
    mean_value = sum(values) / len(values)
    return [value - mean_value for value in values]


def _kernel_responses(values: tuple[float, ...], *, kernels: tuple[tuple[int, tuple[float, ...]], ...]) -> tuple[float, ...]:
    centered = _series_centered(values)
    features: list[float] = []
    for dilation, kernel in kernels:
        length = len(kernel)
        max_start = len(centered) - 1 - dilation * (length - 1)
        if max_start < 0:
            features.extend((0.0, 0.0, 0.0))
            continue
        responses: list[float] = []
        for start in range(max_start + 1):
            response = 0.0
            for offset, weight in enumerate(kernel):
                response += weight * centered[start + dilation * offset]
            responses.append(response)
        features.append(max(responses))
        features.append(sum(1.0 if value > 0.0 else 0.0 for value in responses) / max(len(responses), 1))
        features.append(sum(abs(value) for value in responses) / max(len(responses), 1))
    return tuple(features)


def _train_centroids(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    kernels: tuple[tuple[int, tuple[float, ...]], ...],
) -> dict[str, tuple[float, ...]]:
    by_class: dict[str, list[tuple[float, ...]]] = {class_name: [] for class_name in CLASS_NAMES}
    for trajectory in trajectories:
        by_class[trajectory.true_class].append(_kernel_responses(trajectory.measurements, kernels=kernels))
    centroids: dict[str, tuple[float, ...]] = {}
    for class_name, feature_rows in by_class.items():
        feature_dim = len(feature_rows[0])
        centroids[class_name] = tuple(
            sum(row[index] for row in feature_rows) / len(feature_rows)
            for index in range(feature_dim)
        )
    return centroids


def _predict_from_centroids(
    trajectory: SharedDynamicsTrajectory,
    *,
    kernels: tuple[tuple[int, tuple[float, ...]], ...],
    centroids: dict[str, tuple[float, ...]],
) -> tuple[str, float]:
    features = _kernel_responses(trajectory.measurements, kernels=kernels)
    log_scores = {}
    for class_name, centroid in centroids.items():
        squared_distance = sum((value - reference) ** 2 for value, reference in zip(features, centroid, strict=True))
        log_scores[class_name] = -0.5 * squared_distance
    weights = _normalize_log_scores(log_scores)
    predicted = max(weights, key=weights.get)
    return predicted, float(weights[predicted])


def _split_dataset(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
) -> tuple[tuple[SharedDynamicsTrajectory, ...], tuple[SharedDynamicsTrajectory, ...]]:
    train_rows: list[SharedDynamicsTrajectory] = []
    test_rows: list[SharedDynamicsTrajectory] = []
    for trajectory in trajectories:
        example_index = int(trajectory.trajectory_id.rsplit("_", 1)[-1])
        if example_index < 4:
            train_rows.append(trajectory)
        else:
            test_rows.append(trajectory)
    return tuple(train_rows), tuple(test_rows)


def _accuracy(rows: list[NeuralFrontierPredictionRow], *, scenario_name: str | None = None) -> float:
    selected = rows if scenario_name is None else [row for row in rows if row.scenario_name == scenario_name]
    return sum(1.0 if row.predicted_class == row.true_class else 0.0 for row in selected) / max(len(selected), 1)


def analyze_neural_sequence_vs_physics_frontier(*, seed: int = 907, trajectories_per_case: int = 8) -> NeuralSequenceFrontierResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    train_trajectories, test_trajectories = _split_dataset(trajectories)
    tcn_centroids = _train_centroids(train_trajectories, kernels=TCN_KERNELS)
    inception_centroids = _train_centroids(train_trajectories, kernels=INCEPTION_KERNELS)

    prediction_rows: list[NeuralFrontierPredictionRow] = []
    for trajectory in trajectories:
        split = "train" if trajectory in train_trajectories else "test"
        windowed_run = _windowed_predict(trajectory, robust=True)
        rocket_run = _rocket_proxy_predict(trajectory)
        kalman_run = _kalman_predict(trajectory)
        tcn_predicted, tcn_confidence = _predict_from_centroids(trajectory, kernels=TCN_KERNELS, centroids=tcn_centroids)
        inception_predicted, inception_confidence = _predict_from_centroids(trajectory, kernels=INCEPTION_KERNELS, centroids=inception_centroids)
        for method_name, predicted_class, confidence in (
            ("windowed_robust", windowed_run.final_predicted_class, windowed_run.final_confidence),
            ("rocket_proxy", rocket_run.final_predicted_class, rocket_run.final_confidence),
            ("kalman_bank", kalman_run.final_predicted_class, kalman_run.final_confidence),
            ("tcn_proxy", tcn_predicted, tcn_confidence),
            ("inception_proxy", inception_predicted, inception_confidence),
        ):
            prediction_rows.append(
                NeuralFrontierPredictionRow(
                    trajectory_id=trajectory.trajectory_id,
                    scenario_name=trajectory.scenario_name,
                    true_class=trajectory.true_class,
                    split=split,
                    method_name=method_name,
                    predicted_class=predicted_class,
                    confidence=float(confidence),
                )
            )

    metric_rows: list[NeuralFrontierMetricRow] = []
    for method_name, claim_level in (
        ("windowed_robust", "baseline"),
        ("rocket_proxy", "implemented_proxy"),
        ("kalman_bank", "baseline"),
        ("tcn_proxy", "implemented_proxy"),
        ("inception_proxy", "implemented_proxy"),
    ):
        method_rows = [row for row in prediction_rows if row.method_name == method_name]
        test_rows = [row for row in method_rows if row.split == "test"]
        metric_rows.append(
            NeuralFrontierMetricRow(
                method_name=method_name,
                overall_accuracy=_accuracy(method_rows),
                test_accuracy=_accuracy(test_rows),
                short_noisy_accuracy=_accuracy(test_rows, scenario_name="short_noisy"),
                endpoint_match_accuracy=_accuracy(test_rows, scenario_name="endpoint_match"),
                applicability_status="supported",
                claim_level=claim_level,
            )
        )

    row_map = {row.method_name: row for row in metric_rows}
    metrics: dict[str, float | int | str] = {
        "study_id": "neural_sequence_vs_physics_frontier_v1",
        "seed": seed,
        "trajectory_count": len(trajectories),
        "train_count": len(train_trajectories),
        "test_count": len(test_trajectories),
        "tcn_test_accuracy": row_map["tcn_proxy"].test_accuracy,
        "inception_test_accuracy": row_map["inception_proxy"].test_accuracy,
        "rocket_test_accuracy": row_map["rocket_proxy"].test_accuracy,
        "kalman_test_accuracy": row_map["kalman_bank"].test_accuracy,
        "promotion_decision": "hold_neural_sequence_at_proxy_stage",
    }
    return NeuralSequenceFrontierResult(
        prediction_rows=tuple(prediction_rows),
        metric_rows=tuple(metric_rows),
        metrics=metrics,
    )


def _render_frontier_bars(result: NeuralSequenceFrontierResult):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    labels = [row.method_name for row in result.metric_rows]
    values = [row.test_accuracy for row in result.metric_rows]
    colors = ["#9ca3af", "#16a34a", "#2563eb", "#7c3aed", "#dc2626"]
    ax.bar(range(len(labels)), values, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Neural Sequence Proxy Frontier", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_scenario_panel(result: NeuralSequenceFrontierResult):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    scenario_labels = ("endpoint_match", "short_noisy")
    method_names = [row.method_name for row in result.metric_rows]
    x = list(range(len(method_names)))
    width = 0.35
    endpoint = [row.endpoint_match_accuracy for row in result.metric_rows]
    noisy = [row.short_noisy_accuracy for row in result.metric_rows]
    ax.bar([value - width / 2 for value in x], endpoint, width=width, label=scenario_labels[0], color="#2563eb")
    ax.bar([value + width / 2 for value in x], noisy, width=width, label=scenario_labels[1], color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Scenario Slice: Neural Sequence vs Physics", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def write_neural_sequence_vs_physics_frontier_artifacts(
    output_dir: str | Path,
    *,
    result: NeuralSequenceFrontierResult | None = None,
    seed: int = 907,
    trajectories_per_case: int = 8,
) -> NeuralSequenceFrontierArtifacts:
    payload = result or analyze_neural_sequence_vs_physics_frontier(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
    )
    run_dir = Path(output_dir) / "neural_sequence_vs_physics_frontier_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_summary_path = run_dir / "prediction_summary.csv"
    metric_summary_path = run_dir / "metric_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "neural_sequence_vs_physics_frontier_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frontier_plot_path = plots_dir / "frontier_test_accuracy.png"
    scenario_plot_path = plots_dir / "scenario_slice_accuracy.png"

    write_csv(
        prediction_summary_path,
        [asdict(row) for row in payload.prediction_rows],
        list(NeuralFrontierPredictionRow.__dataclass_fields__.keys()),
    )
    write_csv(
        metric_summary_path,
        [asdict(row) for row in payload.metric_rows],
        list(NeuralFrontierMetricRow.__dataclass_fields__.keys()),
    )
    write_comparison_summary_csv(run_dir, [asdict(row) for row in payload.metric_rows], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Neural Sequence vs Physics Frontier",
        "",
        "- Study: `neural_sequence_vs_physics_frontier_v1`",
        "- Sequence methods: `tcn_proxy`, `inception_proxy`",
        "- Baselines: `windowed_robust`, `rocket_proxy`, `kalman_bank`",
        "",
        "## Claim Boundary",
        "",
        "This packet is a sequence-style proxy frontier, not a claim that real TCN or InceptionTime training has been completed.",
        "It exists to keep the lane explicit and benchmarkable without overclaiming deep-learning fidelity.",
        "",
        f"- decision: `{payload.metrics['promotion_decision']}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Previous methods: `gradient_boosted_features`, `minirocket_family`, `kalman_bank`",
        "- Candidate methods: `tcn`, `inceptiontime`",
        "- Implementation state: sequence-style proxy benchmark only",
        "- Decision: `hold at implemented until real training/fidelity arrives`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    frontier_plot_path.write_bytes(_figure_to_png(_render_frontier_bars(payload)))
    scenario_plot_path.write_bytes(_figure_to_png(_render_scenario_panel(payload)))

    return NeuralSequenceFrontierArtifacts(
        run_dir=run_dir,
        prediction_summary_path=prediction_summary_path,
        metric_summary_path=metric_summary_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(frontier_plot_path, scenario_plot_path),
    )


__all__ = [
    "NeuralFrontierMetricRow",
    "NeuralFrontierPredictionRow",
    "NeuralSequenceFrontierArtifacts",
    "NeuralSequenceFrontierResult",
    "analyze_neural_sequence_vs_physics_frontier",
    "write_neural_sequence_vs_physics_frontier_artifacts",
]
