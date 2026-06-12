from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
import tempfile
from typing import Any
import warnings

import numpy

from .common_dataset_comparison_contracts import SharedDynamicsTrajectory

ArchivePanelVariant = str


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    method_family: str
    available: bool
    backend_name: str
    detail: str


@dataclass(frozen=True, slots=True)
class ArchiveClassifierAdapter:
    method_family: str
    backend_name: str
    model: Any
    class_names: tuple[str, ...]
    resample_length: int
    panel_variant: ArchivePanelVariant

    def predict(self, trajectory: SharedDynamicsTrajectory) -> tuple[str, float]:
        panel = _archive_compatible_panel(
            (trajectory,),
            resample_length=self.resample_length,
            panel_variant=self.panel_variant,
        )
        probabilities = _predict_probabilities(self.model, panel, self.class_names)
        predicted_index = int(numpy.argmax(probabilities[0]))
        return self.class_names[predicted_index], float(probabilities[0, predicted_index])

    def predict_many(
        self,
        trajectories: tuple[SharedDynamicsTrajectory, ...],
    ) -> tuple[tuple[str, float], ...]:
        panel = _archive_compatible_panel(
            trajectories,
            resample_length=self.resample_length,
            panel_variant=self.panel_variant,
        )
        probabilities = _predict_probabilities(self.model, panel, self.class_names)
        rows: list[tuple[str, float]] = []
        for row_index in range(probabilities.shape[0]):
            predicted_index = int(numpy.argmax(probabilities[row_index]))
            rows.append(
                (
                    self.class_names[predicted_index],
                    float(probabilities[row_index, predicted_index]),
                )
            )
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class ArchiveBackendFitOutcome:
    method_family: str
    availability: BackendAvailability
    attempted: bool
    succeeded: bool
    backend_name: str
    detail: str
    adapter: ArchiveClassifierAdapter | None


@dataclass(frozen=True, slots=True)
class Ts2VecExternalAdapter:
    backend_name: str
    model: Any
    class_names: tuple[str, ...]
    train_embeddings: tuple[tuple[str, numpy.ndarray], ...]
    centroids: dict[str, numpy.ndarray]
    resample_length: int

    def encode(self, trajectory: SharedDynamicsTrajectory) -> numpy.ndarray:
        panel = _trajectories_to_panel((trajectory,), resample_length=self.resample_length)
        return _encode_ts2vec(self.model, panel)[0]


def archive_backend_availability(method_family: str) -> BackendAvailability:
    if _load_archive_classifier_factory(method_family) is None:
        return BackendAvailability(
            method_family=method_family,
            available=False,
            backend_name="local_proxy",
            detail="Optional aeon/sktime archive classifier backend is not installed.",
        )
    return BackendAvailability(
        method_family=method_family,
        available=True,
        backend_name="optional_external_archive_backend",
        detail="Optional external archive classifier backend is available.",
    )


def fit_archive_classifier_if_available(
    method_family: str,
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    class_names: tuple[str, ...],
    resample_length: int = 16,
    panel_variant: ArchivePanelVariant = "normalized_position",
) -> ArchiveClassifierAdapter | None:
    outcome = fit_archive_classifier_with_outcome(
        method_family,
        trajectories,
        class_names=class_names,
        resample_length=resample_length,
        panel_variant=panel_variant,
    )
    return outcome.adapter


def fit_archive_classifier_with_outcome(
    method_family: str,
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    class_names: tuple[str, ...],
    resample_length: int = 16,
    panel_variant: ArchivePanelVariant = "normalized_position",
) -> ArchiveBackendFitOutcome:
    factory = _load_archive_classifier_factory(method_family)
    availability = archive_backend_availability(method_family)
    if factory is None:
        return ArchiveBackendFitOutcome(
            method_family=method_family,
            availability=availability,
            attempted=False,
            succeeded=False,
            backend_name="local_proxy",
            detail=availability.detail,
            adapter=None,
        )
    model, backend_name = factory()
    panel = _archive_compatible_panel(
        trajectories,
        resample_length=resample_length,
        panel_variant=panel_variant,
    )
    labels = numpy.asarray([trajectory.true_class for trajectory in trajectories], dtype=object)
    try:
        model.fit(panel, labels)
    except Exception as exc:
        return ArchiveBackendFitOutcome(
            method_family=method_family,
            availability=availability,
            attempted=True,
            succeeded=False,
            backend_name=backend_name,
            detail=f"external_fit_failed:{type(exc).__name__}",
            adapter=None,
        )
    adapter = ArchiveClassifierAdapter(
        method_family=method_family,
        backend_name=backend_name,
        model=model,
        class_names=class_names,
        resample_length=resample_length,
        panel_variant=panel_variant,
    )
    return ArchiveBackendFitOutcome(
        method_family=method_family,
        availability=availability,
        attempted=True,
        succeeded=True,
        backend_name=backend_name,
        detail="external_fit_succeeded",
        adapter=adapter,
    )


def ts2vec_backend_availability() -> BackendAvailability:
    if _load_ts2vec_class() is None:
        return BackendAvailability(
            method_family="ts2vec",
            available=False,
            backend_name="local_proxy",
            detail="Optional ts2vec package is not installed.",
        )
    return BackendAvailability(
        method_family="ts2vec",
        available=True,
        backend_name="ts2vec_external",
        detail="Optional ts2vec package is available.",
    )


def fit_ts2vec_if_available(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    class_names: tuple[str, ...],
    resample_length: int = 24,
) -> Ts2VecExternalAdapter | None:
    ts2vec_class = _load_ts2vec_class()
    if ts2vec_class is None:
        return None
    train_rows = [trajectory for trajectory in trajectories if int(trajectory.trajectory_id.rsplit("_", 1)[-1]) < 4]
    if not train_rows:
        return None
    panel = _trajectories_to_panel(tuple(train_rows), resample_length=resample_length)
    series = numpy.transpose(panel, (0, 2, 1))
    model = ts2vec_class(input_dims=series.shape[-1], output_dims=min(64, resample_length), device="cpu")
    fit_kwargs = {"verbose": False}
    try:
        model.fit(series, **fit_kwargs)
    except TypeError:
        model.fit(series)
    embeddings = _encode_ts2vec(model, panel)
    train_embeddings = tuple(
        (trajectory.true_class, embedding.astype(float, copy=True))
        for trajectory, embedding in zip(train_rows, embeddings, strict=True)
    )
    centroid_inputs: dict[str, list[numpy.ndarray]] = {class_name: [] for class_name in class_names}
    for label, embedding in train_embeddings:
        centroid_inputs[label].append(embedding)
    centroids = {
        class_name: numpy.mean(numpy.vstack(rows), axis=0) if rows else numpy.zeros(embeddings.shape[1], dtype=float)
        for class_name, rows in centroid_inputs.items()
    }
    return Ts2VecExternalAdapter(
        backend_name="ts2vec_external",
        model=model,
        class_names=class_names,
        train_embeddings=train_embeddings,
        centroids=centroids,
        resample_length=resample_length,
    )


def _load_archive_classifier_factory(method_family: str):
    _ensure_optional_backend_environment()
    factories = {
        "minirocket_family": _load_minirocket_factory,
        "drcif_interval_forests": _load_drcif_factory,
        "dictionary_tde_family": _load_dictionary_factory,
        "hive_cote": _load_hive_cote_factory,
    }
    loader = factories.get(method_family)
    return loader() if loader is not None else None


def _ensure_optional_backend_environment() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    if not os.environ.get("NUMBA_CACHE_DIR"):
        cache_dir = Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-numba-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    warnings.filterwarnings(
        "ignore",
        message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.",
        category=FutureWarning,
        module="sklearn.utils.deprecation",
    )
    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in log",
        category=RuntimeWarning,
        module="sktime.transformations._catch22_numba",
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in scalar subtract",
        category=RuntimeWarning,
        module="sktime.transformations._catch22_numba",
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in scalar multiply",
        category=RuntimeWarning,
        module="sktime.transformations._catch22_numba",
    )


def _load_minirocket_factory():
    return _first_working_factory(
        (
            (
                "aeon.classification.convolution_based",
                (
                    ("MiniRocketClassifier", {"n_kernels": 512, "random_state": 0}),
                    ("MultiRocketClassifier", {"n_kernels": 512, "random_state": 0}),
                    ("MultiRocketHydraClassifier", {"n_kernels": 512, "random_state": 0}),
                    ("HydraClassifier", {"n_kernels": 512, "random_state": 0}),
                ),
            ),
        )
    )


def _load_drcif_factory():
    return _first_working_factory(
        (
            (
                "sktime.classification.interval_based",
                (
                    (
                        "DrCIF",
                        {
                            "n_estimators": 4,
                            "att_subsample_size": 2,
                            "contract_max_n_estimators": 4,
                            "n_jobs": 1,
                            "random_state": 0,
                        },
                    ),
                ),
            ),
            (
                "aeon.classification.interval_based",
                (
                    (
                        "DrCIFClassifier",
                        {
                            "n_estimators": 4,
                            "att_subsample_size": 2,
                            "contract_max_n_estimators": 4,
                            "n_jobs": 1,
                            "random_state": 0,
                        },
                    ),
                ),
            ),
        )
    )


def _load_dictionary_factory():
    return _first_working_factory(
        (
            (
                "sktime.classification.dictionary_based",
                (
                    ("WEASEL", {"window_inc": 1, "support_probabilities": True, "n_jobs": 1, "random_state": 0}),
                    (
                        "TemporalDictionaryEnsemble",
                        {
                            "n_parameter_samples": 24,
                            "max_ensemble_size": 8,
                            "min_window": 4,
                            "randomly_selected_params": 12,
                            "n_jobs": 1,
                            "random_state": 0,
                        },
                    ),
                    ("BOSSEnsemble", {"min_window": 4, "max_ensemble_size": 16, "n_jobs": 1}),
                ),
            ),
            (
                "aeon.classification.dictionary_based",
                (
                    ("WEASEL", {"window_inc": 1, "support_probabilities": True, "n_jobs": 1, "random_state": 0}),
                    ("WEASEL_V2", {"window_inc": 1, "support_probabilities": True, "n_jobs": 1, "random_state": 0}),
                    (
                        "TemporalDictionaryEnsemble",
                        {
                            "n_parameter_samples": 24,
                            "max_ensemble_size": 8,
                            "min_window": 4,
                            "randomly_selected_params": 12,
                            "n_jobs": 1,
                            "random_state": 0,
                        },
                    ),
                    ("BOSSEnsemble", {"min_window": 4, "max_ensemble_size": 16, "n_jobs": 1}),
                ),
            ),
        )
    )


def _load_hive_cote_factory():
    return _first_working_factory(
        (
            (
                "sktime.classification.hybrid",
                (
                    (
                        "HIVECOTEV2",
                        {
                            "time_limit_in_minutes": 0.1,
                            "save_component_probas": False,
                            "verbose": 0,
                            "n_jobs": 1,
                            "random_state": 0,
                            "drcif_params": {
                                "n_estimators": 2,
                                "att_subsample_size": 2,
                                "contract_max_n_estimators": 2,
                            },
                            "arsenal_params": {
                                "num_kernels": 32,
                            },
                            "tde_params": {
                                "n_parameter_samples": 4,
                                "max_ensemble_size": 1,
                                "randomly_selected_params": 2,
                                "min_window": 4,
                            },
                        },
                    ),
                ),
            ),
        )
    )


def _load_ts2vec_class():
    for module_name, class_name in (
        ("ts2vec", "TS2Vec"),
        ("ts2vec.ts2vec", "TS2Vec"),
    ):
        try:
            module = import_module(module_name)
        except Exception:
            continue
        class_object = getattr(module, class_name, None)
        if class_object is not None:
            return class_object
    return None


def _first_working_factory(candidates: tuple[tuple[str, tuple[tuple[str, dict[str, object]], ...]], ...]):
    for module_name, class_specs in candidates:
        try:
            module = import_module(module_name)
        except Exception:
            continue
        for class_name, kwargs in class_specs:
            classifier_class = getattr(module, class_name, None)
            if classifier_class is None:
                continue
            return lambda classifier_class=classifier_class, class_name=class_name, kwargs=kwargs: (
                classifier_class(**kwargs),
                f"{module_name}:{class_name}",
            )
    return None


def _resample_trajectory(trajectory: SharedDynamicsTrajectory, *, resample_length: int) -> numpy.ndarray:
    times = numpy.asarray(trajectory.times, dtype=float)
    values = numpy.asarray(trajectory.measurements, dtype=float)
    if len(values) == 1:
        return numpy.repeat(values, resample_length)
    base_times = times - times[0]
    duration = max(float(base_times[-1]), 1.0e-6)
    target_times = numpy.linspace(0.0, duration, resample_length)
    return numpy.interp(target_times, base_times, values)


def _trajectories_to_panel(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    resample_length: int,
) -> numpy.ndarray:
    rows = [_resample_trajectory(trajectory, resample_length=resample_length) for trajectory in trajectories]
    return numpy.asarray(rows, dtype=float)[:, numpy.newaxis, :]


def _archive_compatible_panel(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    resample_length: int,
    panel_variant: ArchivePanelVariant = "normalized_position",
) -> numpy.ndarray:
    effective_length = max(resample_length, 32)
    if panel_variant == "raw_position":
        return _trajectories_to_panel(trajectories, resample_length=effective_length)
    if panel_variant == "normalized_position":
        base_panel = _trajectories_to_panel(trajectories, resample_length=effective_length)
        return _normalize_panel_channels(base_panel)
    if panel_variant == "normalized_position_velocity":
        return _archive_position_velocity_panel(trajectories, resample_length=effective_length)
    if panel_variant == "normalized_position_velocity_acceleration":
        return _archive_position_velocity_acceleration_panel(trajectories, resample_length=effective_length)
    raise ValueError(f"unsupported archive panel variant: {panel_variant}")


def _normalize_channel(values: numpy.ndarray) -> numpy.ndarray:
    centered = values - float(values.mean())
    scale = float(values.std())
    if scale < 1.0e-6:
        scale = 1.0
    return centered / scale


def _normalize_panel_channels(panel: numpy.ndarray) -> numpy.ndarray:
    normalized = panel.astype(float, copy=True)
    for row_index in range(normalized.shape[0]):
        for channel_index in range(normalized.shape[1]):
            normalized[row_index, channel_index, :] = _normalize_channel(normalized[row_index, channel_index, :])
    return normalized


def _archive_position_velocity_panel(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    resample_length: int,
) -> numpy.ndarray:
    position_panel = _trajectories_to_panel(trajectories, resample_length=resample_length)
    rows: list[numpy.ndarray] = []
    for row_index in range(position_panel.shape[0]):
        position = position_panel[row_index, 0, :]
        velocity = numpy.gradient(position)
        rows.append(numpy.stack((_normalize_channel(position), _normalize_channel(velocity)), axis=0))
    return numpy.asarray(rows, dtype=float)


def _archive_position_velocity_acceleration_panel(
    trajectories: tuple[SharedDynamicsTrajectory, ...],
    *,
    resample_length: int,
) -> numpy.ndarray:
    position_panel = _trajectories_to_panel(trajectories, resample_length=resample_length)
    rows: list[numpy.ndarray] = []
    for row_index in range(position_panel.shape[0]):
        position = position_panel[row_index, 0, :]
        velocity = numpy.gradient(position)
        acceleration = numpy.gradient(velocity)
        rows.append(
            numpy.stack(
                (
                    _normalize_channel(position),
                    _normalize_channel(velocity),
                    _normalize_channel(acceleration),
                ),
                axis=0,
            )
        )
    return numpy.asarray(rows, dtype=float)


def _predict_probabilities(model: Any, panel: numpy.ndarray, class_names: tuple[str, ...]) -> numpy.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = numpy.asarray(model.predict_proba(panel), dtype=float)
        if probabilities.ndim == 2:
            model_classes = getattr(model, "classes_", None)
            if model_classes is None:
                return probabilities
            label_to_index = {str(label): index for index, label in enumerate(model_classes)}
            reordered = numpy.zeros((probabilities.shape[0], len(class_names)), dtype=float)
            for target_index, class_name in enumerate(class_names):
                source_index = label_to_index.get(str(class_name))
                if source_index is not None:
                    reordered[:, target_index] = probabilities[:, source_index]
            row_sums = reordered.sum(axis=1, keepdims=True)
            valid_rows = row_sums[:, 0] > 0.0
            if numpy.any(valid_rows):
                reordered[valid_rows] = reordered[valid_rows] / row_sums[valid_rows]
            return reordered
    predictions = model.predict(panel)
    probabilities = numpy.zeros((len(predictions), len(class_names)), dtype=float)
    label_to_index = {label: index for index, label in enumerate(class_names)}
    for row_index, label in enumerate(predictions):
        probabilities[row_index, label_to_index[str(label)]] = 1.0
    return probabilities


def _encode_ts2vec(model: Any, panel: numpy.ndarray) -> numpy.ndarray:
    series = numpy.transpose(panel, (0, 2, 1))
    embeddings = model.encode(series)
    encoded = numpy.asarray(embeddings, dtype=float)
    if encoded.ndim == 3:
        encoded = encoded.mean(axis=1)
    return encoded


__all__ = [
    "ArchiveClassifierAdapter",
    "ArchiveBackendFitOutcome",
    "BackendAvailability",
    "Ts2VecExternalAdapter",
    "archive_backend_availability",
    "fit_archive_classifier_if_available",
    "fit_archive_classifier_with_outcome",
    "fit_ts2vec_if_available",
    "ts2vec_backend_availability",
]
