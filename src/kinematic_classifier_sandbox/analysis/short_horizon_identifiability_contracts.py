from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ShortHorizonTimeRow:
    time: float
    constant_velocity_position: float
    constant_acceleration_position: float
    absolute_gap: float
    normalized_gap_at_nominal_noise: float


@dataclass(frozen=True, slots=True)
class ShortHorizonNoiseRow:
    measurement_sigma: float
    mean_normalized_gap: float
    max_normalized_gap: float
    final_step_normalized_gap: float


@dataclass(frozen=True, slots=True)
class ShortHorizonDurationThresholdRow:
    measurement_sigma: float
    first_time_at_1sigma: float | None
    first_time_at_2sigma: float | None


@dataclass(frozen=True, slots=True)
class ShortHorizonIdentifiabilityResult:
    nominal_measurement_sigma: float
    times: tuple[ShortHorizonTimeRow, ...]
    noise_sweep: tuple[ShortHorizonNoiseRow, ...]
    duration_thresholds: tuple[ShortHorizonDurationThresholdRow, ...]


@dataclass(frozen=True, slots=True)
class ShortHorizonIdentifiabilityArtifacts:
    run_dir: Path
    report_path: Path
    time_series_path: Path
    noise_sweep_path: Path
    duration_thresholds_path: Path
    time_plot_png_path: Path
    noise_plot_png_path: Path
    duration_plot_png_path: Path
