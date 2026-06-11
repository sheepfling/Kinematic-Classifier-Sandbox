from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObservationSurfaceMetadata:
    surface_id: str
    display_name: str
    observation_surface_type: str
    corrupts: tuple[str, ...]
    classifier_allowed_fields: tuple[str, ...]
    hidden_fields: tuple[str, ...]
    status: str
    best_use: str
    lift_to_3d: str


def default_observation_surface_catalog() -> tuple[ObservationSurfaceMetadata, ...]:
    return (
        ObservationSurfaceMetadata(
            surface_id="clean_observation",
            display_name="Clean observation",
            observation_surface_type="identity_sampler",
            corrupts=(),
            classifier_allowed_fields=("time", "observed_position"),
            hidden_fields=("truth_position", "sensor_backend_id"),
            status="implemented_for_current_generators",
            best_use="Baseline dynamics-only evidence checks.",
            lift_to_3d="Identity observation over vector position.",
        ),
        ObservationSurfaceMetadata(
            surface_id="gaussian_position_noise",
            display_name="Gaussian position noise",
            observation_surface_type="additive_noise",
            corrupts=("position",),
            classifier_allowed_fields=("time", "observed_position"),
            hidden_fields=("noise_seed", "measurement_std", "truth_position"),
            status="implemented_for_current_generators",
            best_use="Noise sensitivity, posterior ambiguity, and calibration checks.",
            lift_to_3d="Vector Gaussian or axis-specific covariance.",
        ),
        ObservationSurfaceMetadata(
            surface_id="dropout_sampler",
            display_name="Dropout sampler",
            observation_surface_type="missingness",
            corrupts=("sample_presence",),
            classifier_allowed_fields=("time", "observed_position"),
            hidden_fields=("dropout_mask", "dropout_probability", "truth_position"),
            status="planned",
            best_use="Tracklet gaps and partial-history robustness.",
            lift_to_3d="Shared or sensor-specific dropout masks.",
        ),
        ObservationSurfaceMetadata(
            surface_id="quantized_position",
            display_name="Quantized position",
            observation_surface_type="quantization",
            corrupts=("position",),
            classifier_allowed_fields=("time", "observed_position"),
            hidden_fields=("quantization_step", "truth_position"),
            status="planned",
            best_use="Low-resolution sensor and discretization stress.",
            lift_to_3d="Per-axis or range-bearing quantization.",
        ),
        ObservationSurfaceMetadata(
            surface_id="outlier_injection",
            display_name="Outlier injection",
            observation_surface_type="sparse_corruption",
            corrupts=("position", "sample_quality"),
            classifier_allowed_fields=("time", "observed_position"),
            hidden_fields=("outlier_indices", "outlier_magnitude", "truth_position"),
            status="implemented_for_current_generators",
            best_use="Robust extrema and innovation-spike checks.",
            lift_to_3d="Vector outliers, clutter, or bearing/range jumps.",
        ),
        ObservationSurfaceMetadata(
            surface_id="low_rate_sampler",
            display_name="Low-rate sampler",
            observation_surface_type="temporal_sampling",
            corrupts=("sample_rate",),
            classifier_allowed_fields=("time", "observed_position"),
            hidden_fields=("dropped_times", "truth_position"),
            status="planned",
            best_use="Time-window versus sample-window diagnostics.",
            lift_to_3d="Shared temporal sampler across vector observations.",
        ),
    )


def observation_surface_rows() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "surface_id": surface.surface_id,
            "display_name": surface.display_name,
            "observation_surface_type": surface.observation_surface_type,
            "corrupts": ",".join(surface.corrupts),
            "classifier_allowed_fields": ",".join(surface.classifier_allowed_fields),
            "hidden_fields": ",".join(surface.hidden_fields),
            "status": surface.status,
            "best_use": surface.best_use,
            "lift_to_3d": surface.lift_to_3d,
        }
        for surface in default_observation_surface_catalog()
    )
