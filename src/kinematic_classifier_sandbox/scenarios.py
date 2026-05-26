from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScenarioProfile:
    scenario_id: str
    scenario_family: str
    scenario_tier: str
    times: tuple[float, ...]
    measurement_sigma: float
    default_horizon: float


SCENARIO_PROFILES: dict[str, ScenarioProfile] = {
    "easy": ScenarioProfile("easy", "nominal", "easy_v1", tuple(float(step) for step in range(10)), 0.10, 9.0),
    "irregular": ScenarioProfile("irregular", "irregular_sampling", "realistic_v1", (0.0, 0.7, 1.6, 2.8, 4.1, 5.0, 6.6, 7.4, 8.9, 10.0), 0.10, 10.0),
    "endpoint_match": ScenarioProfile("endpoint_match", "boundary", "boundary_v1", (0.0, 0.6, 1.4, 2.2, 3.7, 5.0), 0.16, 5.0),
    "short": ScenarioProfile("short", "short_horizon", "boundary_v1", (0.0, 0.5, 1.0, 1.5), 0.10, 1.5),
    "short_noisy": ScenarioProfile("short_noisy", "noise_stress", "stress_v1", (0.0, 0.5, 1.0, 1.5), 0.28, 1.5),
    "outlier": ScenarioProfile("outlier", "outlier_stress", "adversarial_v1", tuple(float(step) for step in range(8)), 0.10, 7.0),
}


SCENARIO_IDS: tuple[str, ...] = tuple(SCENARIO_PROFILES)
SCENARIO_TIMES: dict[str, tuple[float, ...]] = {
    scenario_id: profile.times for scenario_id, profile in SCENARIO_PROFILES.items()
}
SCENARIO_MEASUREMENT_SIGMA: dict[str, float] = {
    scenario_id: profile.measurement_sigma for scenario_id, profile in SCENARIO_PROFILES.items()
}


def list_scenario_ids() -> tuple[str, ...]:
    return SCENARIO_IDS


def get_scenario_profile(scenario_id: str) -> ScenarioProfile:
    return SCENARIO_PROFILES[scenario_id]


def get_scenario_family(scenario_id: str) -> str:
    return SCENARIO_PROFILES[scenario_id].scenario_family


def get_scenario_tier(scenario_id: str) -> str:
    return SCENARIO_PROFILES[scenario_id].scenario_tier


def get_scenario_times(scenario_id: str) -> tuple[float, ...]:
    return SCENARIO_PROFILES[scenario_id].times


def get_scenario_measurement_sigma(scenario_id: str) -> float:
    return SCENARIO_PROFILES[scenario_id].measurement_sigma


def get_scenario_default_horizon(scenario_id: str) -> float:
    return SCENARIO_PROFILES[scenario_id].default_horizon


def get_scenario_dynamics(scenario_name: str, true_class: str) -> tuple[float, float]:
    if scenario_name == "endpoint_match":
        if true_class == "constant_velocity":
            return 1.10, 0.0
        return 0.35, 0.30
    velocity0 = 0.8
    acceleration = 0.0 if true_class == "constant_velocity" else 0.28
    return velocity0, acceleration
