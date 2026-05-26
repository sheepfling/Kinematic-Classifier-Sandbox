from __future__ import annotations

from dataclasses import dataclass

from ..scenarios import get_scenario_dynamics, get_scenario_times


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


def _position_at_time(class_name: str, time: float, scenario_name: str) -> float:
    velocity0, acceleration = get_scenario_dynamics(scenario_name, class_name)
    return velocity0 * time + 0.5 * acceleration * time * time


def analyze_short_horizon_identifiability() -> ShortHorizonIdentifiabilityResult:
    scenario_name = "short_noisy"
    times = get_scenario_times(scenario_name)
    nominal_sigma = 0.28

    time_rows: list[ShortHorizonTimeRow] = []
    raw_gaps: list[float] = []
    for time in times:
        cv_position = _position_at_time("constant_velocity", time, scenario_name)
        ca_position = _position_at_time("constant_acceleration", time, scenario_name)
        absolute_gap = abs(ca_position - cv_position)
        raw_gaps.append(absolute_gap)
        time_rows.append(
            ShortHorizonTimeRow(
                time=time,
                constant_velocity_position=cv_position,
                constant_acceleration_position=ca_position,
                absolute_gap=absolute_gap,
                normalized_gap_at_nominal_noise=absolute_gap / nominal_sigma,
            )
        )

    sigma_values = (0.10, 0.16, 0.22, 0.28, 0.34, 0.40, 0.50)
    noise_sweep_rows: list[ShortHorizonNoiseRow] = []
    for sigma in sigma_values:
        normalized = [gap / sigma for gap in raw_gaps]
        noise_sweep_rows.append(
            ShortHorizonNoiseRow(
                measurement_sigma=sigma,
                mean_normalized_gap=sum(normalized) / len(normalized),
                max_normalized_gap=max(normalized),
                final_step_normalized_gap=normalized[-1],
            )
        )
    duration_times = tuple(0.5 * step for step in range(13))
    duration_gaps = [
        abs(
            _position_at_time("constant_acceleration", time, scenario_name)
            - _position_at_time("constant_velocity", time, scenario_name)
        )
        for time in duration_times
    ]
    duration_threshold_rows: list[ShortHorizonDurationThresholdRow] = []
    for sigma in sigma_values:
        normalized = [gap / sigma for gap in duration_gaps]
        first_1sigma = next((time for time, value in zip(duration_times, normalized) if value >= 1.0), None)
        first_2sigma = next((time for time, value in zip(duration_times, normalized) if value >= 2.0), None)
        duration_threshold_rows.append(
            ShortHorizonDurationThresholdRow(
                measurement_sigma=sigma,
                first_time_at_1sigma=first_1sigma,
                first_time_at_2sigma=first_2sigma,
            )
        )

    return ShortHorizonIdentifiabilityResult(
        nominal_measurement_sigma=nominal_sigma,
        times=tuple(time_rows),
        noise_sweep=tuple(noise_sweep_rows),
        duration_thresholds=tuple(duration_threshold_rows),
    )


from .short_horizon_identifiability_rendering import (  # noqa: E402
    ShortHorizonIdentifiabilityArtifacts,
    render_short_horizon_identifiability_report,
    write_short_horizon_identifiability_artifacts,
)

__all__ = [
    "ShortHorizonDurationThresholdRow",
    "ShortHorizonIdentifiabilityArtifacts",
    "ShortHorizonIdentifiabilityResult",
    "ShortHorizonNoiseRow",
    "ShortHorizonTimeRow",
    "analyze_short_horizon_identifiability",
    "render_short_horizon_identifiability_report",
    "write_short_horizon_identifiability_artifacts",
]
