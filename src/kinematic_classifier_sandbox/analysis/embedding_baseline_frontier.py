from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path

import numpy as np

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
EMBEDDING_DIMENSION = 3


@dataclass(frozen=True, slots=True)
class EmbeddingViewRow:
    trajectory_id: str
    scenario_name: str
    split: str
    view_name: str
    true_class: str
    crop_start: int
    crop_stop: int
    feature_norm: float


@dataclass(frozen=True, slots=True)
class EmbeddingRow:
    trajectory_id: str
    scenario_name: str
    split: str
    true_class: str
    embedding_method: str
    embedding_0: float
    embedding_1: float
    embedding_2: float
    embedding_norm: float


@dataclass(frozen=True, slots=True)
class EmbeddingFrontierPredictionRow:
    trajectory_id: str
    scenario_name: str
    true_class: str
    split: str
    method_name: str
    predicted_class: str
    confidence: float


@dataclass(frozen=True, slots=True)
class EmbeddingFrontierMetricRow:
    method_name: str
    overall_accuracy: float
    test_accuracy: float
    short_noisy_accuracy: float
    endpoint_match_accuracy: float
    mean_confidence: float
    claim_level: str


@dataclass(frozen=True, slots=True)
class EmbeddingBaselineFrontierResult:
    view_rows: tuple[EmbeddingViewRow, ...]
    embedding_rows: tuple[EmbeddingRow, ...]
    prediction_rows: tuple[EmbeddingFrontierPredictionRow, ...]
    metric_rows: tuple[EmbeddingFrontierMetricRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class EmbeddingBaselineFrontierArtifacts:
    run_dir: Path
    view_summary_path: Path
    embedding_summary_path: Path
    prediction_summary_path: Path
    metric_summary_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class Ts2VecProxyClassifier:
    fit: dict[str, np.ndarray]
    train_embeddings: tuple[tuple[str, np.ndarray], ...]
    centroids: dict[str, np.ndarray]
    embedding_dimension: int


def _trajectory_split(trajectory: SharedDynamicsTrajectory) -> str:
    return "train" if int(trajectory.trajectory_id.rsplit("_", 1)[-1]) < 4 else "test"


def _crop_bounds(length: int, *, view_name: str) -> tuple[int, int]:
    if length <= 3:
        return 0, length
    crop_length = min(length, max(3, length // 2 + 1))
    if view_name == "prefix":
        return 0, crop_length
    if view_name == "suffix":
        return length - crop_length, length
    center_start = max(0, (length - crop_length) // 2)
    return center_start, center_start + crop_length


def _slice_times_and_values(
    trajectory: SharedDynamicsTrajectory,
    *,
    view_name: str,
) -> tuple[int, int, tuple[float, ...], tuple[float, ...]]:
    crop_start, crop_stop = _crop_bounds(len(trajectory.measurements), view_name=view_name)
    return (
        crop_start,
        crop_stop,
        trajectory.times[crop_start:crop_stop],
        trajectory.measurements[crop_start:crop_stop],
    )


def _segment_indices(length: int, segments: int = 3) -> list[tuple[int, int]]:
    if length <= 0:
        return [(0, 0) for _ in range(segments)]
    indices: list[tuple[int, int]] = []
    for segment_index in range(segments):
        start = round(segment_index * length / segments)
        stop = round((segment_index + 1) * length / segments)
        if segment_index == segments - 1:
            stop = length
        if stop <= start:
            stop = min(length, start + 1)
        indices.append((start, stop))
    return indices


def _slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    centered_times = times - float(np.mean(times))
    centered_values = values - float(np.mean(values))
    denominator = float(np.dot(centered_times, centered_times))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.dot(centered_times, centered_values) / denominator)


def _feature_vector(times: tuple[float, ...], values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return (0.0,) * 24
    time_array = np.asarray(times, dtype=float)
    value_array = np.asarray(values, dtype=float)
    diffs = np.diff(value_array)
    second_diffs = np.diff(diffs) if len(diffs) >= 2 else np.asarray((), dtype=float)
    duration = float(time_array[-1] - time_array[0]) if len(time_array) > 1 else 0.0
    total_variation = float(np.sum(np.abs(diffs))) if len(diffs) else 0.0
    slope = _slope(time_array, value_array)
    monotonicity = float(np.mean(np.sign(diffs))) if len(diffs) else 0.0
    sign_changes = 0.0
    if len(diffs) >= 2:
        non_zero = [1 if value > 1.0e-9 else -1 if value < -1.0e-9 else 0 for value in diffs.tolist()]
        compact = [value for value in non_zero if value != 0]
        sign_changes = float(sum(1 for index in range(1, len(compact)) if compact[index] != compact[index - 1]))

    segment_features: list[float] = []
    for start, stop in _segment_indices(len(value_array), 3):
        segment_values = value_array[start:stop]
        segment_times = time_array[start:stop]
        if len(segment_values) == 0:
            segment_features.extend((0.0, 0.0, 0.0))
            continue
        segment_features.append(float(np.mean(segment_values)))
        segment_features.append(_slope(segment_times, segment_values))
        segment_features.append(float(np.max(segment_values) - np.min(segment_values)))

    return (
        float(len(value_array)),
        duration,
        float(np.mean(value_array)),
        float(np.std(value_array)),
        float(np.min(value_array)),
        float(np.max(value_array)),
        float(value_array[0]),
        float(value_array[-1]),
        slope,
        float(np.max(value_array) - np.min(value_array)),
        total_variation,
        float(np.mean(diffs)) if len(diffs) else 0.0,
        float(np.std(diffs)) if len(diffs) else 0.0,
        float(np.min(diffs)) if len(diffs) else 0.0,
        float(np.max(diffs)) if len(diffs) else 0.0,
        float(np.mean(second_diffs)) if len(second_diffs) else 0.0,
        float(np.std(second_diffs)) if len(second_diffs) else 0.0,
        monotonicity,
        sign_changes,
        *segment_features,
    )


def _normalize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1.0e-9, 1.0, std)
    normalized = (matrix - mean) / std
    return normalized, mean, std


def _inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.clip(eigenvalues, 1.0e-8, None)
    scale = np.diag(1.0 / np.sqrt(clipped))
    return eigenvectors @ scale @ eigenvectors.T


def _fit_cca(left: np.ndarray, right: np.ndarray, *, embedding_dimension: int) -> dict[str, np.ndarray]:
    left_norm, left_mean, left_std = _normalize_rows(left)
    right_norm, right_mean, right_std = _normalize_rows(right)
    sample_count = max(left_norm.shape[0] - 1, 1)
    cxx = (left_norm.T @ left_norm) / sample_count + np.eye(left_norm.shape[1]) * 1.0e-6
    cyy = (right_norm.T @ right_norm) / sample_count + np.eye(right_norm.shape[1]) * 1.0e-6
    cxy = (left_norm.T @ right_norm) / sample_count
    left_whitener = _inv_sqrt(cxx)
    right_whitener = _inv_sqrt(cyy)
    canonical_matrix = left_whitener @ cxy @ right_whitener
    u, singular_values, vt = np.linalg.svd(canonical_matrix, full_matrices=False)
    dim = min(embedding_dimension, u.shape[1], vt.shape[0])
    left_projection = left_whitener @ u[:, :dim]
    right_projection = right_whitener @ vt.T[:, :dim]
    return {
        "left_mean": left_mean,
        "left_std": left_std,
        "right_mean": right_mean,
        "right_std": right_std,
        "left_projection": left_projection,
        "right_projection": right_projection,
        "canonical_correlations": singular_values[:dim],
    }


def _project_row(features: np.ndarray, *, mean: np.ndarray, std: np.ndarray, projection: np.ndarray) -> np.ndarray:
    normalized = (features - mean) / std
    return normalized @ projection


def _embedding_from_views(
    trajectory: SharedDynamicsTrajectory,
    *,
    fit: dict[str, np.ndarray],
    embedding_method: str,
) -> tuple[np.ndarray, tuple[EmbeddingViewRow, ...]]:
    view_rows: list[EmbeddingViewRow] = []
    projected_views: list[np.ndarray] = []
    for view_name, which_projection in (("prefix", "left"), ("suffix", "right")):
        crop_start, crop_stop, times, values = _slice_times_and_values(trajectory, view_name=view_name)
        feature_vector = np.asarray(_feature_vector(times, values), dtype=float)
        projection = fit[f"{which_projection}_projection"]
        mean = fit[f"{which_projection}_mean"]
        std = fit[f"{which_projection}_std"]
        embedding = _project_row(feature_vector, mean=mean, std=std, projection=projection)
        projected_views.append(embedding)
        view_rows.append(
            EmbeddingViewRow(
                trajectory_id=trajectory.trajectory_id,
                scenario_name=trajectory.scenario_name,
                split=_trajectory_split(trajectory),
                view_name=view_name,
                true_class=trajectory.true_class,
                crop_start=crop_start,
                crop_stop=crop_stop,
                feature_norm=float(np.linalg.norm(feature_vector)),
            )
        )
    embedding = np.mean(np.vstack(projected_views), axis=0)
    return embedding, tuple(view_rows)


def _embedding_norm(embedding: np.ndarray) -> float:
    return float(np.linalg.norm(embedding))


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    pivot = max(logits.values())
    weights = {label: float(np.exp(value - pivot)) for label, value in logits.items()}
    total = max(sum(weights.values()), 1.0e-12)
    return {label: value / total for label, value in weights.items()}


def _nearest_centroid_predict(
    embedding: np.ndarray,
    *,
    centroids: dict[str, np.ndarray],
) -> tuple[str, float]:
    distances = {label: float(np.linalg.norm(embedding - centroid)) for label, centroid in centroids.items()}
    logits = {label: -distance * distance for label, distance in distances.items()}
    weights = _softmax(logits)
    predicted = max(weights, key=weights.get)
    return predicted, float(weights[predicted])


def _nearest_neighbor_predict(
    embedding: np.ndarray,
    *,
    train_embeddings: list[tuple[str, np.ndarray]],
) -> tuple[str, float]:
    distances = [
        (label, float(np.linalg.norm(embedding - other_embedding)))
        for label, other_embedding in train_embeddings
    ]
    distances.sort(key=lambda item: item[1])
    winner_label, winner_distance = distances[0]
    logits = {label: -distance * distance for label, distance in distances[: min(5, len(distances))]}
    weights = _softmax(logits)
    return winner_label, float(weights[winner_label] if winner_label in weights else np.exp(-winner_distance))


def fit_ts2vec_proxy_classifier(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    embedding_dimension: int = EMBEDDING_DIMENSION,
) -> Ts2VecProxyClassifier:
    train_trajectories = [trajectory for trajectory in trajectories if _trajectory_split(trajectory) == "train"]
    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []
    for trajectory in train_trajectories:
        _, _, left_times, left_values = _slice_times_and_values(trajectory, view_name="prefix")
        _, _, right_times, right_values = _slice_times_and_values(trajectory, view_name="suffix")
        left_rows.append(np.asarray(_feature_vector(left_times, left_values), dtype=float))
        right_rows.append(np.asarray(_feature_vector(right_times, right_values), dtype=float))
    fit = _fit_cca(np.vstack(left_rows), np.vstack(right_rows), embedding_dimension=embedding_dimension)

    train_embeddings: list[tuple[str, np.ndarray]] = []
    centroid_inputs: dict[str, list[np.ndarray]] = {class_name: [] for class_name in CLASS_NAMES}
    for trajectory in train_trajectories:
        embedding, _ = _embedding_from_views(
            trajectory,
            fit=fit,
            embedding_method="ts2vec_style_cca",
        )
        train_embeddings.append((trajectory.true_class, embedding))
        centroid_inputs[trajectory.true_class].append(embedding)
    centroids = {
        class_name: np.mean(np.vstack(rows), axis=0) if rows else np.zeros(embedding_dimension, dtype=float)
        for class_name, rows in centroid_inputs.items()
    }
    return Ts2VecProxyClassifier(
        fit=fit,
        train_embeddings=tuple(train_embeddings),
        centroids=centroids,
        embedding_dimension=embedding_dimension,
    )


def predict_ts2vec_proxy(
    trajectory: SharedDynamicsTrajectory,
    *,
    classifier: Ts2VecProxyClassifier,
    strategy: str = "nn",
) -> tuple[str, float, dict[str, float]]:
    embedding, _ = _embedding_from_views(
        trajectory,
        fit=classifier.fit,
        embedding_method="ts2vec_style_cca",
    )
    if strategy == "centroid":
        predicted_class, confidence = _nearest_centroid_predict(embedding, centroids=classifier.centroids)
    else:
        predicted_class, confidence = _nearest_neighbor_predict(
            embedding,
            train_embeddings=list(classifier.train_embeddings),
        )
    weights = {
        predicted_class: confidence,
        next(class_name for class_name in CLASS_NAMES if class_name != predicted_class): 1.0 - confidence,
    }
    return predicted_class, confidence, weights


def _accuracy(rows: list[EmbeddingFrontierPredictionRow], *, scenario_name: str | None = None) -> float:
    selected = rows if scenario_name is None else [row for row in rows if row.scenario_name == scenario_name]
    return sum(1.0 if row.predicted_class == row.true_class else 0.0 for row in selected) / max(len(selected), 1)


def analyze_embedding_baseline_frontier(
    *,
    seed: int = 913,
    trajectories_per_case: int = 8,
    embedding_dimension: int = EMBEDDING_DIMENSION,
) -> EmbeddingBaselineFrontierResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    train_trajectories = [trajectory for trajectory in trajectories if _trajectory_split(trajectory) == "train"]
    test_trajectories = [trajectory for trajectory in trajectories if _trajectory_split(trajectory) == "test"]
    classifier = fit_ts2vec_proxy_classifier(trajectories, embedding_dimension=embedding_dimension)
    fit = classifier.fit

    view_rows: list[EmbeddingViewRow] = []
    embedding_rows: list[EmbeddingRow] = []
    embedding_lookup: dict[str, np.ndarray] = {}

    for trajectory in trajectories:
        embedding, trajectory_view_rows = _embedding_from_views(
            trajectory,
            fit=fit,
            embedding_method="ts2vec_style_cca",
        )
        view_rows.extend(trajectory_view_rows)
        embedding_lookup[trajectory.trajectory_id] = embedding
        embedding_rows.append(
            EmbeddingRow(
                trajectory_id=trajectory.trajectory_id,
                scenario_name=trajectory.scenario_name,
                split=_trajectory_split(trajectory),
                true_class=trajectory.true_class,
                embedding_method="ts2vec_style_cca",
                embedding_0=float(embedding[0]) if len(embedding) > 0 else 0.0,
                embedding_1=float(embedding[1]) if len(embedding) > 1 else 0.0,
                embedding_2=float(embedding[2]) if len(embedding) > 2 else 0.0,
                embedding_norm=_embedding_norm(embedding),
            )
        )

    prediction_rows: list[EmbeddingFrontierPredictionRow] = []
    for trajectory in trajectories:
        split = _trajectory_split(trajectory)
        centroid_predicted, centroid_confidence, _ = predict_ts2vec_proxy(
            trajectory,
            classifier=classifier,
            strategy="centroid",
        )
        nn_predicted, nn_confidence, _ = predict_ts2vec_proxy(
            trajectory,
            classifier=classifier,
            strategy="nn",
        )
        windowed_run = _windowed_predict(trajectory, robust=True)
        rocket_run = _rocket_proxy_predict(trajectory)
        kalman_run = _kalman_predict(trajectory)
        for method_name, predicted_class, confidence in (
            ("windowed_robust", windowed_run.final_predicted_class, windowed_run.final_confidence),
            ("rocket_proxy", rocket_run.final_predicted_class, rocket_run.final_confidence),
            ("kalman_bank", kalman_run.final_predicted_class, kalman_run.final_confidence),
            ("ts2vec_centroid", centroid_predicted, centroid_confidence),
            ("ts2vec_nn", nn_predicted, nn_confidence),
        ):
            prediction_rows.append(
                EmbeddingFrontierPredictionRow(
                    trajectory_id=trajectory.trajectory_id,
                    scenario_name=trajectory.scenario_name,
                    true_class=trajectory.true_class,
                    split=split,
                    method_name=method_name,
                    predicted_class=predicted_class,
                    confidence=float(confidence),
                )
            )

    metric_rows: list[EmbeddingFrontierMetricRow] = []
    for method_name, claim_level in (
        ("windowed_robust", "baseline"),
        ("rocket_proxy", "implemented_proxy"),
        ("kalman_bank", "baseline"),
        ("ts2vec_centroid", "implemented_proxy"),
        ("ts2vec_nn", "implemented_proxy"),
    ):
        method_rows = [row for row in prediction_rows if row.method_name == method_name]
        test_rows = [row for row in method_rows if row.split == "test"]
        metric_rows.append(
            EmbeddingFrontierMetricRow(
                method_name=method_name,
                overall_accuracy=_accuracy(method_rows),
                test_accuracy=_accuracy(test_rows),
                short_noisy_accuracy=_accuracy(test_rows, scenario_name="short_noisy"),
                endpoint_match_accuracy=_accuracy(test_rows, scenario_name="endpoint_match"),
                mean_confidence=sum(row.confidence for row in test_rows) / max(len(test_rows), 1),
                claim_level=claim_level,
            )
        )

    row_map = {row.method_name: row for row in metric_rows}
    canonical_correlations = fit["canonical_correlations"]
    embedding_centroid_distance = float(np.linalg.norm(classifier.centroids[CLASS_NAMES[0]] - classifier.centroids[CLASS_NAMES[1]]))
    embedding_test_accuracy = row_map["ts2vec_centroid"].test_accuracy
    embedding_nn_test_accuracy = row_map["ts2vec_nn"].test_accuracy
    best_embedding_test_accuracy = max(embedding_test_accuracy, embedding_nn_test_accuracy)
    best_baseline_test_accuracy = max(
        row_map["windowed_robust"].test_accuracy,
        row_map["rocket_proxy"].test_accuracy,
        row_map["kalman_bank"].test_accuracy,
    )
    promotion_decision = (
        "promote_embedding_baseline_frontier"
        if best_embedding_test_accuracy >= best_baseline_test_accuracy
        and float(np.mean(canonical_correlations)) >= 0.70
        else "revise_embedding_baseline_frontier"
    )

    metrics: dict[str, float | int | str] = {
        "study_id": "embedding_baseline_frontier_v1",
        "seed": seed,
        "trajectory_count": len(trajectories),
        "train_count": len(train_trajectories),
        "test_count": len(test_trajectories),
        "embedding_dimension": embedding_dimension,
        "mean_canonical_correlation": float(np.mean(canonical_correlations)) if len(canonical_correlations) else 0.0,
        "first_canonical_correlation": float(canonical_correlations[0]) if len(canonical_correlations) else 0.0,
        "embedding_centroid_distance": embedding_centroid_distance,
        "ts2vec_centroid_test_accuracy": embedding_test_accuracy,
        "ts2vec_nn_test_accuracy": embedding_nn_test_accuracy,
        "windowed_test_accuracy": row_map["windowed_robust"].test_accuracy,
        "rocket_test_accuracy": row_map["rocket_proxy"].test_accuracy,
        "kalman_test_accuracy": row_map["kalman_bank"].test_accuracy,
        "promotion_decision": promotion_decision,
    }
    return EmbeddingBaselineFrontierResult(
        view_rows=tuple(view_rows),
        embedding_rows=tuple(embedding_rows),
        prediction_rows=tuple(prediction_rows),
        metric_rows=tuple(metric_rows),
        metrics=metrics,
    )


def _render_accuracy_bars(result: EmbeddingBaselineFrontierResult):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    labels = [row.method_name for row in result.metric_rows]
    values = [row.test_accuracy for row in result.metric_rows]
    colors = ["#9ca3af", "#2563eb", "#0f766e", "#7c3aed", "#dc2626"]
    ax.bar(range(len(labels)), values, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Embedding Baseline Frontier", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_embedding_projection(result: EmbeddingBaselineFrontierResult):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    palette = {CLASS_NAMES[0]: "#2563eb", CLASS_NAMES[1]: "#dc2626"}
    markers = {"train": "o", "test": "s"}
    for row in result.embedding_rows:
        ax.scatter(
            row.embedding_0,
            row.embedding_1,
            color=palette[row.true_class],
            marker=markers[row.split],
            s=42,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xlabel("embedding dim 1")
    ax.set_ylabel("embedding dim 2")
    ax.set_title("TS2Vec-Style Embedding Projection", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25)
    return fig


def _render_scenario_panel(result: EmbeddingBaselineFrontierResult):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = [row.method_name for row in result.metric_rows]
    x = list(range(len(labels)))
    width = 0.24
    short_noisy = [row.short_noisy_accuracy for row in result.metric_rows]
    endpoint = [row.endpoint_match_accuracy for row in result.metric_rows]
    overall = [row.overall_accuracy for row in result.metric_rows]
    ax.bar([value - width for value in x], overall, width=width, label="overall", color="#0f766e")
    ax.bar(x, endpoint, width=width, label="endpoint_match", color="#2563eb")
    ax.bar([value + width for value in x], short_noisy, width=width, label="short_noisy", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Scenario Slice: Embedding Frontier", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def write_embedding_baseline_frontier_artifacts(
    output_dir: str | Path,
    *,
    result: EmbeddingBaselineFrontierResult | None = None,
    seed: int = 913,
    trajectories_per_case: int = 8,
    embedding_dimension: int = EMBEDDING_DIMENSION,
) -> EmbeddingBaselineFrontierArtifacts:
    payload = result or analyze_embedding_baseline_frontier(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        embedding_dimension=embedding_dimension,
    )
    run_dir = Path(output_dir) / "embedding_baseline_frontier_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    view_summary_path = run_dir / "view_summary.csv"
    embedding_summary_path = run_dir / "embedding_summary.csv"
    prediction_summary_path = run_dir / "prediction_summary.csv"
    metric_summary_path = run_dir / "metric_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "embedding_baseline_frontier_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot_path = plots_dir / "accuracy_bars.png"
    projection_plot_path = plots_dir / "embedding_projection.png"
    scenario_plot_path = plots_dir / "scenario_slice.png"

    write_csv(view_summary_path, [asdict(row) for row in payload.view_rows], list(EmbeddingViewRow.__dataclass_fields__.keys()))
    write_csv(embedding_summary_path, [asdict(row) for row in payload.embedding_rows], list(EmbeddingRow.__dataclass_fields__.keys()))
    write_csv(prediction_summary_path, [asdict(row) for row in payload.prediction_rows], list(EmbeddingFrontierPredictionRow.__dataclass_fields__.keys()))
    write_csv(metric_summary_path, [asdict(row) for row in payload.metric_rows], list(EmbeddingFrontierMetricRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [payload.metrics], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# TS2Vec",
        "",
        "- Study: `embedding_baseline_frontier_v1`",
        "- Encoder: `ts2vec_style_cca`",
        "- Downstream heads: `ts2vec_centroid`, `ts2vec_nn`",
        "- Baselines: `windowed_robust`, `rocket_proxy`, `kalman_bank`",
        "",
        "## What It Proves",
        "",
        "This packet builds a first representation-learning witness on the shared 1D dynamics corpus.",
        "It uses paired trajectory views, a small CCA-style encoder, and downstream embedding classifiers so the lane is executable instead of only being a registry note.",
        "",
        "## Claim Boundary",
        "",
        "This is a TS2Vec-style proxy frontier, not a claim that the external TS2Vec library has been installed or benchmarked.",
        "It is enough to keep the embedding lane explicit and to test whether reusable trajectory embeddings buy anything over the current 1D baselines.",
        "",
        f"- mean canonical correlation: `{float(payload.metrics['mean_canonical_correlation']):.3f}`",
        f"- embedding centroid test accuracy: `{float(payload.metrics['ts2vec_centroid_test_accuracy']):.3f}`",
        f"- embedding NN test accuracy: `{float(payload.metrics['ts2vec_nn_test_accuracy']):.3f}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Previous methods: `windowed`, `rocket_proxy`, `kalman_bank`",
        "- Candidate method: `ts2vec`",
        "- Failure mode: handcrafted or single-pass baselines underfit reusable trajectory structure",
        f"- Improvement: NN test accuracy `{float(payload.metrics['windowed_test_accuracy']):.3f}` -> `{float(payload.metrics['ts2vec_nn_test_accuracy']):.3f}`",
        f"- Complexity: `CCA + centroid/NN head` over paired trajectory views",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    accuracy_plot_path.write_bytes(_figure_to_png(_render_accuracy_bars(payload)))
    projection_plot_path.write_bytes(_figure_to_png(_render_embedding_projection(payload)))
    scenario_plot_path.write_bytes(_figure_to_png(_render_scenario_panel(payload)))

    return EmbeddingBaselineFrontierArtifacts(
        run_dir=run_dir,
        view_summary_path=view_summary_path,
        embedding_summary_path=embedding_summary_path,
        prediction_summary_path=prediction_summary_path,
        metric_summary_path=metric_summary_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(accuracy_plot_path, projection_plot_path, scenario_plot_path),
    )


__all__ = [
    "EmbeddingBaselineFrontierArtifacts",
    "EmbeddingBaselineFrontierResult",
    "EmbeddingFrontierMetricRow",
    "EmbeddingFrontierPredictionRow",
    "EmbeddingRow",
    "EmbeddingViewRow",
    "Ts2VecProxyClassifier",
    "analyze_embedding_baseline_frontier",
    "fit_ts2vec_proxy_classifier",
    "predict_ts2vec_proxy",
    "write_embedding_baseline_frontier_artifacts",
]
