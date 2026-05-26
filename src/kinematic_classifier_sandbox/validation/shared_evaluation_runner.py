from __future__ import annotations

from typing import Any

from .shared_evaluation_contracts import SharedClassifierRun, SharedTrajectoryClassifier


def evaluate_shared_classifier_registry(
    trajectories: tuple[Any, ...],
    classifiers: tuple[SharedTrajectoryClassifier, ...],
    *,
    prior: dict[str, float] | None = None,
) -> tuple[SharedClassifierRun, ...]:
    runs: list[SharedClassifierRun] = []
    for trajectory in trajectories:
        for classifier in classifiers:
            runs.append(classifier.predict_trajectory(trajectory, prior=prior))
    return tuple(runs)


def sensor_regime_summary_rows(runs: tuple[SharedClassifierRun, ...]) -> list[dict[str, object]]:
    regimes = sorted({run.sensor_regime_id for run in runs})
    rows: list[dict[str, object]] = []
    for sensor_regime_id in regimes:
        regime_runs = [run for run in runs if run.sensor_regime_id == sensor_regime_id]
        accuracy = sum(1.0 if run.final_predicted_class == run.true_class else 0.0 for run in regime_runs) / max(len(regime_runs), 1)
        confidence = sum(run.final_confidence for run in regime_runs) / max(len(regime_runs), 1)
        measurement_dims = sorted({run.measurement_dim for run in regime_runs})
        coordinate_frames = sorted({run.coordinate_frame for run in regime_runs})
        rows.append(
            {
                "sensor_regime_id": sensor_regime_id,
                "num_predictions": len(regime_runs),
                "mean_accuracy": accuracy,
                "mean_confidence": confidence,
                "measurement_dims": " ".join(str(value) for value in measurement_dims),
                "coordinate_frames": " ".join(coordinate_frames),
                "methods": " ".join(sorted({run.method_name for run in regime_runs})),
            }
        )
    return rows


__all__ = ["evaluate_shared_classifier_registry", "sensor_regime_summary_rows"]
