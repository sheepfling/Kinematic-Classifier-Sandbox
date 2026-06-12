from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    _kalman_predict,
    _rocket_proxy_predict,
    _windowed_predict,
    generate_shared_dynamics_dataset,
)
from kinematic_classifier_sandbox.analysis.common_dataset_comparison_contracts import (
    SharedDynamicsTrajectory,
)
from kinematic_classifier_sandbox.analysis.optional_external_backends import (
    Ts2VecExternalAdapter,
    fit_ts2vec_if_available,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png, plt

CLASS_NAMES = ("constant_velocity", "constant_acceleration")
EMBEDDING_DIMENSION = 3


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


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
class OnlineRouteRow:
    trajectory_id: str
    scenario_name: str
    split: str
    prefix_fraction: float
    prefix_stop: int
    true_class: str
    ts2vec_predicted_class: str
    ts2vec_confidence: float
    windowed_predicted_class: str
    windowed_confidence: float
    rocket_predicted_class: str
    rocket_confidence: float
    kalman_predicted_class: str
    kalman_confidence: float
    route_status: str


@dataclass(frozen=True, slots=True)
class EmbeddingBaselineFrontierResult:
    view_rows: tuple[EmbeddingViewRow, ...]
    embedding_rows: tuple[EmbeddingRow, ...]
    prediction_rows: tuple[EmbeddingFrontierPredictionRow, ...]
    online_route_rows: tuple[OnlineRouteRow, ...]
    metric_rows: tuple[EmbeddingFrontierMetricRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class EmbeddingBaselineFrontierArtifacts:
    run_dir: Path
    view_summary_path: Path
    embedding_summary_path: Path
    prediction_summary_path: Path
    metric_summary_path: Path
    online_route_summary_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class Ts2VecProxyClassifier:
    fit: dict[str, numpy.ndarray]
    train_embeddings: tuple[tuple[str, numpy.ndarray], ...]
    centroids: dict[str, numpy.ndarray]
    embedding_dimension: int
    backend_name: str = "local_proxy"
    external_adapter: Ts2VecExternalAdapter | None = None


Ts2VecBackendMode = Literal["auto", "proxy_only", "external_only"]


def _trajectory_split(trajectory: SharedDynamicsTrajectory) -> str:
    return "train" if int(trajectory.trajectory_id.rsplit("_", 1)[-1]) < 4 else "test"


def _prefix_trajectory(trajectory: SharedDynamicsTrajectory, prefix_stop: int) -> SharedDynamicsTrajectory:
    stop = max(2, min(prefix_stop, len(trajectory.times)))
    return SharedDynamicsTrajectory(
        trajectory_id=f"{trajectory.trajectory_id}__prefix_{stop}",
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        seed=trajectory.seed,
        times=trajectory.times[:stop],
        measurements=trajectory.measurements[:stop],
        true_position=trajectory.true_position[:stop],
        true_velocity=trajectory.true_velocity[:stop],
        true_acceleration=trajectory.true_acceleration[:stop],
        measurement_dim=trajectory.measurement_dim,
        coordinate_frame=trajectory.coordinate_frame,
    )


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


def _slope(times: numpy.ndarray, values: numpy.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    centered_times = times - float(numpy.mean(times))
    centered_values = values - float(numpy.mean(values))
    denominator = float(numpy.dot(centered_times, centered_times))
    if denominator <= 1.0e-12:
        return 0.0
    return float(numpy.dot(centered_times, centered_values) / denominator)


def _feature_vector(times: tuple[float, ...], values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return (0.0,) * 24
    time_array = numpy.asarray(times, dtype=float)
    value_array = numpy.asarray(values, dtype=float)
    diffs = numpy.diff(value_array)
    second_diffs = numpy.diff(diffs) if len(diffs) >= 2 else numpy.asarray((), dtype=float)
    duration = float(time_array[-1] - time_array[0]) if len(time_array) > 1 else 0.0
    total_variation = float(numpy.sum(numpy.abs(diffs))) if len(diffs) else 0.0
    slope = _slope(time_array, value_array)
    monotonicity = float(numpy.mean(numpy.sign(diffs))) if len(diffs) else 0.0
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
        segment_features.append(float(numpy.mean(segment_values)))
        segment_features.append(_slope(segment_times, segment_values))
        segment_features.append(float(numpy.max(segment_values) - numpy.min(segment_values)))

    return (
        float(len(value_array)),
        duration,
        float(numpy.mean(value_array)),
        float(numpy.std(value_array)),
        float(numpy.min(value_array)),
        float(numpy.max(value_array)),
        float(value_array[0]),
        float(value_array[-1]),
        slope,
        float(numpy.max(value_array) - numpy.min(value_array)),
        total_variation,
        float(numpy.mean(diffs)) if len(diffs) else 0.0,
        float(numpy.std(diffs)) if len(diffs) else 0.0,
        float(numpy.min(diffs)) if len(diffs) else 0.0,
        float(numpy.max(diffs)) if len(diffs) else 0.0,
        float(numpy.mean(second_diffs)) if len(second_diffs) else 0.0,
        float(numpy.std(second_diffs)) if len(second_diffs) else 0.0,
        monotonicity,
        sign_changes,
        *segment_features,
    )


def _normalize_rows(matrix: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = numpy.where(std < 1.0e-9, 1.0, std)
    normalized = (matrix - mean) / std
    return normalized, mean, std


def _inv_sqrt(matrix: numpy.ndarray) -> numpy.ndarray:
    eigenvalues, eigenvectors = numpy.linalg.eigh(matrix)
    clipped = numpy.clip(eigenvalues, 1.0e-8, None)
    scale = numpy.diag(1.0 / numpy.sqrt(clipped))
    return eigenvectors @ scale @ eigenvectors.T


def _fit_cca(left: numpy.ndarray, right: numpy.ndarray, *, embedding_dimension: int) -> dict[str, numpy.ndarray]:
    left_norm, left_mean, left_std = _normalize_rows(left)
    right_norm, right_mean, right_std = _normalize_rows(right)
    sample_count = max(left_norm.shape[0] - 1, 1)
    cxx = (left_norm.T @ left_norm) / sample_count + numpy.eye(left_norm.shape[1]) * 1.0e-6
    cyy = (right_norm.T @ right_norm) / sample_count + numpy.eye(right_norm.shape[1]) * 1.0e-6
    cxy = (left_norm.T @ right_norm) / sample_count
    left_whitener = _inv_sqrt(cxx)
    right_whitener = _inv_sqrt(cyy)
    canonical_matrix = left_whitener @ cxy @ right_whitener
    u, singular_values, vt = numpy.linalg.svd(canonical_matrix, full_matrices=False)
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


def _project_row(features: numpy.ndarray, *, mean: numpy.ndarray, std: numpy.ndarray, projection: numpy.ndarray) -> numpy.ndarray:
    normalized = (features - mean) / std
    return normalized @ projection


def _embedding_from_views(
    trajectory: SharedDynamicsTrajectory,
    *,
    fit: dict[str, numpy.ndarray],
    embedding_method: str,
) -> tuple[numpy.ndarray, tuple[EmbeddingViewRow, ...]]:
    view_rows: list[EmbeddingViewRow] = []
    projected_views: list[numpy.ndarray] = []
    for view_name, which_projection in (("prefix", "left"), ("suffix", "right")):
        crop_start, crop_stop, times, values = _slice_times_and_values(trajectory, view_name=view_name)
        feature_vector = numpy.asarray(_feature_vector(times, values), dtype=float)
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
                feature_norm=float(numpy.linalg.norm(feature_vector)),
            )
        )
    embedding = numpy.mean(numpy.vstack(projected_views), axis=0)
    return embedding, tuple(view_rows)


def _embedding_norm(embedding: numpy.ndarray) -> float:
    return float(numpy.linalg.norm(embedding))


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    pivot = max(logits.values())
    weights = {label: float(numpy.exp(value - pivot)) for label, value in logits.items()}
    total = max(sum(weights.values()), 1.0e-12)
    return {label: value / total for label, value in weights.items()}


def _nearest_centroid_predict(
    embedding: numpy.ndarray,
    *,
    centroids: dict[str, numpy.ndarray],
) -> tuple[str, float]:
    distances = {label: float(numpy.linalg.norm(embedding - centroid)) for label, centroid in centroids.items()}
    logits = {label: -distance * distance for label, distance in distances.items()}
    weights = _softmax(logits)
    predicted = max(weights, key=weights.get)
    return predicted, float(weights[predicted])


def _nearest_neighbor_predict(
    embedding: numpy.ndarray,
    *,
    train_embeddings: list[tuple[str, numpy.ndarray]],
) -> tuple[str, float]:
    distances = [
        (label, float(numpy.linalg.norm(embedding - other_embedding)))
        for label, other_embedding in train_embeddings
    ]
    distances.sort(key=lambda item: item[1])
    winner_label, winner_distance = distances[0]
    logits = {label: -distance * distance for label, distance in distances[: min(5, len(distances))]}
    weights = _softmax(logits)
    return winner_label, float(weights[winner_label] if winner_label in weights else numpy.exp(-winner_distance))


def fit_ts2vec_proxy_classifier(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    embedding_dimension: int = EMBEDDING_DIMENSION,
    backend_mode: Ts2VecBackendMode = "auto",
) -> Ts2VecProxyClassifier:
    external_adapter = None
    if backend_mode != "proxy_only":
        external_adapter = fit_ts2vec_if_available(trajectories, class_names=CLASS_NAMES)
        if backend_mode == "external_only" and external_adapter is None:
            raise ValueError("external_only backend_mode requested but optional ts2vec backend is unavailable")
    if external_adapter is not None:
        return Ts2VecProxyClassifier(
            fit={},
            train_embeddings=external_adapter.train_embeddings,
            centroids=external_adapter.centroids,
            embedding_dimension=int(next(iter(external_adapter.centroids.values())).shape[0]),
            backend_name=external_adapter.backend_name,
            external_adapter=external_adapter,
        )
    train_trajectories = [trajectory for trajectory in trajectories if _trajectory_split(trajectory) == "train"]
    left_rows: list[numpy.ndarray] = []
    right_rows: list[numpy.ndarray] = []
    for trajectory in train_trajectories:
        _, _, left_times, left_values = _slice_times_and_values(trajectory, view_name="prefix")
        _, _, right_times, right_values = _slice_times_and_values(trajectory, view_name="suffix")
        left_rows.append(numpy.asarray(_feature_vector(left_times, left_values), dtype=float))
        right_rows.append(numpy.asarray(_feature_vector(right_times, right_values), dtype=float))
    fit = _fit_cca(numpy.vstack(left_rows), numpy.vstack(right_rows), embedding_dimension=embedding_dimension)

    train_embeddings: list[tuple[str, numpy.ndarray]] = []
    centroid_inputs: dict[str, list[numpy.ndarray]] = {class_name: [] for class_name in CLASS_NAMES}
    for trajectory in train_trajectories:
        embedding, _ = _embedding_from_views(
            trajectory,
            fit=fit,
            embedding_method="ts2vec_style_cca",
        )
        train_embeddings.append((trajectory.true_class, embedding))
        centroid_inputs[trajectory.true_class].append(embedding)
    centroids = {
        class_name: numpy.mean(numpy.vstack(rows), axis=0) if rows else numpy.zeros(embedding_dimension, dtype=float)
        for class_name, rows in centroid_inputs.items()
    }
    return Ts2VecProxyClassifier(
        fit=fit,
        train_embeddings=tuple(train_embeddings),
        centroids=centroids,
        embedding_dimension=embedding_dimension,
        backend_name="local_proxy",
    )


def predict_ts2vec_proxy(
    trajectory: SharedDynamicsTrajectory,
    *,
    classifier: Ts2VecProxyClassifier,
    strategy: str = "nn",
) -> tuple[str, float, dict[str, float]]:
    if classifier.external_adapter is not None:
        embedding = classifier.external_adapter.encode(trajectory)
    else:
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
    online_route_rows: list[OnlineRouteRow] = []
    embedding_lookup: dict[str, numpy.ndarray] = {}

    for trajectory in trajectories:
        if classifier.external_adapter is not None:
            embedding = classifier.external_adapter.encode(trajectory)
            trajectory_view_rows = []
        else:
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
                embedding_method="ts2vec_external" if classifier.external_adapter is not None else "ts2vec_style_cca",
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

    online_checkpoints = (0.25, 0.5, 0.75, 1.0)
    for trajectory in trajectories:
        split = _trajectory_split(trajectory)
        for prefix_fraction in online_checkpoints:
            prefix_stop = max(2, min(len(trajectory.times), int(round(len(trajectory.times) * prefix_fraction))))
            prefix_trajectory = _prefix_trajectory(trajectory, prefix_stop)
            ts2vec_predicted, ts2vec_confidence, _ = predict_ts2vec_proxy(
                prefix_trajectory,
                classifier=classifier,
                strategy="nn",
            )
            windowed_run = _windowed_predict(prefix_trajectory, robust=True)
            rocket_run = _rocket_proxy_predict(prefix_trajectory)
            kalman_run = _kalman_predict(prefix_trajectory)
            route_status = (
                "ts2vec_route_preferred"
                if ts2vec_predicted == trajectory.true_class
                and ts2vec_confidence >= max(windowed_run.final_confidence, rocket_run.final_confidence, kalman_run.final_confidence)
                else "baseline_preferred"
            )
            online_route_rows.append(
                OnlineRouteRow(
                    trajectory_id=trajectory.trajectory_id,
                    scenario_name=trajectory.scenario_name,
                    split=split,
                    prefix_fraction=prefix_fraction,
                    prefix_stop=prefix_stop,
                    true_class=trajectory.true_class,
                    ts2vec_predicted_class=ts2vec_predicted,
                    ts2vec_confidence=ts2vec_confidence,
                    windowed_predicted_class=windowed_run.final_predicted_class,
                    windowed_confidence=windowed_run.final_confidence,
                    rocket_predicted_class=rocket_run.final_predicted_class,
                    rocket_confidence=rocket_run.final_confidence,
                    kalman_predicted_class=kalman_run.final_predicted_class,
                    kalman_confidence=kalman_run.final_confidence,
                    route_status=route_status,
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
    canonical_correlations = fit.get("canonical_correlations", numpy.asarray((1.0,), dtype=float))
    embedding_centroid_distance = float(numpy.linalg.norm(classifier.centroids[CLASS_NAMES[0]] - classifier.centroids[CLASS_NAMES[1]]))
    embedding_test_accuracy = row_map["ts2vec_centroid"].test_accuracy
    embedding_nn_test_accuracy = row_map["ts2vec_nn"].test_accuracy
    best_embedding_test_accuracy = max(embedding_test_accuracy, embedding_nn_test_accuracy)
    best_baseline_test_accuracy = max(
        row_map["windowed_robust"].test_accuracy,
        row_map["rocket_proxy"].test_accuracy,
        row_map["kalman_bank"].test_accuracy,
    )
    online_test_rows = [row for row in online_route_rows if row.split == "test"]
    online_final_rows = [row for row in online_test_rows if row.prefix_fraction == 1.0]
    online_route_final_accuracy = _mean([1.0 if row.ts2vec_predicted_class == row.true_class else 0.0 for row in online_final_rows])
    online_route_final_confidence = _mean([row.ts2vec_confidence for row in online_final_rows])
    online_route_win_rate = _mean(
        [
            1.0
            if row.ts2vec_confidence >= max(row.windowed_confidence, row.rocket_confidence, row.kalman_confidence)
            else 0.0
            for row in online_test_rows
        ]
    )
    promotion_decision = (
        "promote_embedding_baseline_frontier"
        if best_embedding_test_accuracy >= best_baseline_test_accuracy
        and float(numpy.mean(canonical_correlations)) >= 0.70
        and online_route_final_accuracy >= row_map["windowed_robust"].test_accuracy
        else "revise_embedding_baseline_frontier"
    )

    metrics: dict[str, float | int | str] = {
        "study_id": "embedding_baseline_frontier_v1",
        "seed": seed,
        "trajectory_count": len(trajectories),
        "train_count": len(train_trajectories),
        "test_count": len(test_trajectories),
        "embedding_dimension": embedding_dimension,
        "ts2vec_backend": classifier.backend_name,
        "mean_canonical_correlation": float(numpy.mean(canonical_correlations)) if len(canonical_correlations) else 0.0,
        "first_canonical_correlation": float(canonical_correlations[0]) if len(canonical_correlations) else 0.0,
        "embedding_centroid_distance": embedding_centroid_distance,
        "ts2vec_centroid_test_accuracy": embedding_test_accuracy,
        "ts2vec_nn_test_accuracy": embedding_nn_test_accuracy,
        "windowed_test_accuracy": row_map["windowed_robust"].test_accuracy,
        "rocket_test_accuracy": row_map["rocket_proxy"].test_accuracy,
        "kalman_test_accuracy": row_map["kalman_bank"].test_accuracy,
        "online_ts2vec_test_accuracy": online_route_final_accuracy,
        "online_ts2vec_mean_confidence": online_route_final_confidence,
        "online_route_win_rate": online_route_win_rate,
        "promotion_decision": promotion_decision,
    }
    return EmbeddingBaselineFrontierResult(
        view_rows=tuple(view_rows),
        embedding_rows=tuple(embedding_rows),
        prediction_rows=tuple(prediction_rows),
        online_route_rows=tuple(online_route_rows),
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


def _render_online_route_curve(result: EmbeddingBaselineFrontierResult):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    grouped: dict[str, list[OnlineRouteRow]] = {}
    for row in result.online_route_rows:
        if row.split != "test":
            continue
        grouped.setdefault(row.scenario_name, []).append(row)
    fractions = sorted({row.prefix_fraction for row in result.online_route_rows if row.split == "test"})
    if not fractions:
        fractions = [0.25, 0.5, 0.75, 1.0]
    ts2vec_means = []
    windowed_means = []
    rocket_means = []
    kalman_means = []
    for fraction in fractions:
        fraction_rows = [row for row in result.online_route_rows if row.split == "test" and row.prefix_fraction == fraction]
        ts2vec_means.append(_mean([1.0 if row.ts2vec_predicted_class == row.true_class else 0.0 for row in fraction_rows]))
        windowed_means.append(_mean([1.0 if row.windowed_predicted_class == row.true_class else 0.0 for row in fraction_rows]))
        rocket_means.append(_mean([1.0 if row.rocket_predicted_class == row.true_class else 0.0 for row in fraction_rows]))
        kalman_means.append(_mean([1.0 if row.kalman_predicted_class == row.true_class else 0.0 for row in fraction_rows]))
    ax.plot(fractions, ts2vec_means, marker="o", linewidth=2.0, label="ts2vec_online", color="#7c3aed")
    ax.plot(fractions, windowed_means, marker="o", linewidth=1.6, label="windowed_robust", color="#2563eb")
    ax.plot(fractions, rocket_means, marker="o", linewidth=1.6, label="rocket_proxy", color="#0f766e")
    ax.plot(fractions, kalman_means, marker="o", linewidth=1.6, label="kalman_bank", color="#dc2626")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("prefix fraction")
    ax.set_ylabel("mean online accuracy")
    ax.set_title("Online TS2Vec Route", loc="left", fontweight="bold")
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
    online_route_summary_path = run_dir / "online_route_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "embedding_baseline_frontier_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot_path = plots_dir / "accuracy_bars.png"
    projection_plot_path = plots_dir / "embedding_projection.png"
    scenario_plot_path = plots_dir / "scenario_slice.png"
    online_route_plot_path = plots_dir / "online_route_curve.png"

    write_csv(view_summary_path, [asdict(row) for row in payload.view_rows], list(EmbeddingViewRow.__dataclass_fields__.keys()))
    write_csv(embedding_summary_path, [asdict(row) for row in payload.embedding_rows], list(EmbeddingRow.__dataclass_fields__.keys()))
    write_csv(prediction_summary_path, [asdict(row) for row in payload.prediction_rows], list(EmbeddingFrontierPredictionRow.__dataclass_fields__.keys()))
    write_csv(metric_summary_path, [asdict(row) for row in payload.metric_rows], list(EmbeddingFrontierMetricRow.__dataclass_fields__.keys()))
    write_csv(online_route_summary_path, [asdict(row) for row in payload.online_route_rows], list(OnlineRouteRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [payload.metrics], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# TS2Vec",
        "",
        "- Study: `embedding_baseline_frontier_v1`",
        f"- Encoder backend: `{payload.metrics['ts2vec_backend']}`",
        "- Downstream heads: `ts2vec_centroid`, `ts2vec_nn`",
        "- Online route: `prefix-based ts2vec_nn` compared against `windowed_robust`, `rocket_proxy`, and `kalman_bank`",
        "- Baselines: `windowed_robust`, `rocket_proxy`, `kalman_bank`",
        "",
        "## What It Proves",
        "",
        "This packet builds a first representation-learning witness on the shared 1D dynamics corpus.",
        "It uses paired trajectory views, a small CCA-style encoder, downstream embedding classifiers, and a prefix-based online route so the lane is executable instead of only being a registry note.",
        "",
        "## Claim Boundary",
        "",
        "",
        f"- mean canonical correlation: `{float(payload.metrics['mean_canonical_correlation']):.3f}`",
        f"- embedding centroid test accuracy: `{float(payload.metrics['ts2vec_centroid_test_accuracy']):.3f}`",
        f"- embedding NN test accuracy: `{float(payload.metrics['ts2vec_nn_test_accuracy']):.3f}`",
        f"- online TS2Vec test accuracy: `{float(payload.metrics['online_ts2vec_test_accuracy']):.3f}`",
        f"- online route confidence: `{float(payload.metrics['online_ts2vec_mean_confidence']):.3f}`",
        f"- online route win rate: `{float(payload.metrics['online_route_win_rate']):.3f}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
    ]
    claim_boundary_lines = (
        [
            "This run used the optional external TS2Vec backend that is installed in the active environment.",
            "The claim stays bounded to this compact shared-corpus witness; it is not a broad parity or full-promotion claim for TS2Vec across Epic 2.",
        ]
        if payload.metrics["ts2vec_backend"] == "ts2vec_external"
        else [
            "This is a TS2Vec-style proxy frontier, not a claim that the external TS2Vec library has been installed or benchmarked.",
            "It is enough to keep the embedding lane explicit, prove an online prefix route, and test whether reusable trajectory embeddings buy anything over the current 1D baselines.",
        ]
    )
    report_lines[15:15] = claim_boundary_lines
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Previous methods: `windowed`, `rocket_proxy`, `kalman_bank`",
        "- Candidate method: `ts2vec`",
        "- Failure mode: handcrafted or single-pass baselines underfit reusable trajectory structure",
        f"- Improvement: NN test accuracy `{float(payload.metrics['windowed_test_accuracy']):.3f}` -> `{float(payload.metrics['ts2vec_nn_test_accuracy']):.3f}`",
        f"- Online route: prefix test accuracy `{float(payload.metrics['online_ts2vec_test_accuracy']):.3f}`",
        "- Complexity: `CCA + centroid/NN head` over paired trajectory views",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    accuracy_plot_path.write_bytes(_figure_to_png(_render_accuracy_bars(payload)))
    projection_plot_path.write_bytes(_figure_to_png(_render_embedding_projection(payload)))
    scenario_plot_path.write_bytes(_figure_to_png(_render_scenario_panel(payload)))
    online_route_plot_path.write_bytes(_figure_to_png(_render_online_route_curve(payload)))

    return EmbeddingBaselineFrontierArtifacts(
        run_dir=run_dir,
        view_summary_path=view_summary_path,
        embedding_summary_path=embedding_summary_path,
        prediction_summary_path=prediction_summary_path,
        metric_summary_path=metric_summary_path,
        online_route_summary_path=online_route_summary_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(accuracy_plot_path, projection_plot_path, scenario_plot_path, online_route_plot_path),
    )


__all__ = [
    "EmbeddingBaselineFrontierArtifacts",
    "EmbeddingBaselineFrontierResult",
    "EmbeddingFrontierMetricRow",
    "EmbeddingFrontierPredictionRow",
    "EmbeddingRow",
    "EmbeddingViewRow",
    "OnlineRouteRow",
    "Ts2VecProxyClassifier",
    "analyze_embedding_baseline_frontier",
    "fit_ts2vec_proxy_classifier",
    "predict_ts2vec_proxy",
    "write_embedding_baseline_frontier_artifacts",
]
