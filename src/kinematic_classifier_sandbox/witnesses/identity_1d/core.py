from __future__ import annotations

import io
import random
from dataclasses import dataclass
from math import erf, exp, log, sqrt
from pathlib import Path
from typing import NamedTuple

from ...analysis.classification_benchmark import (
    summarize_classification_features,
    summarize_classification_outcomes,
)
from ...markdown_builder import MarkdownDocument
from ...utils.io import _write_csv_rows
from ...utils.math import _clamp, _entropy, _gaussian_logpdf, _logsumexp
from ...utils.plotting import plt
from ..study_surface import NamedArtifactWriter, OneDWitnessSurface, write_one_d_surface_artifacts


class IdentityArtifactPaths(NamedTuple):
    summary_path: Path
    plot_path: Path
    trace_path: Path


class ObservedDynamicsProfile(NamedTuple):
    mean_delta: float
    mean_abs_delta: float
    flip_rate: float


class RepresentativeIdentityPairs(NamedTuple):
    pairs: tuple[tuple[SpeedScenario, IdentityClassificationRun], ...]

FEATURE_NAMES = (
    "bike_envelope",
    "horse_envelope",
    "car_envelope",
    "near_horse_limit",
    "over_horse_limit",
    "volatile_trace",
    "surging_trace",
    "persistent_push",
)

CLASS_FEATURE_ANCHORS = {
    "bike": ("bike_envelope", "surging_trace"),
    "horse": ("horse_envelope", "near_horse_limit"),
    "car": ("car_envelope", "over_horse_limit", "persistent_push"),
}

CLASS_VALIDITY_MARGIN_MPH = {
    "bike": 3.0,
    "horse": 3.0,
    "car": 0.0,
}


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _gaussian_interval_probability(mean: float, variance: float, upper_limit: float) -> float:
    safe_variance = max(variance, 1e-9)
    sigma = sqrt(safe_variance)
    return max(0.0, min(1.0, _normal_cdf((upper_limit - mean) / sigma)))


def _normalize_priors(class_specs: tuple["IdentityClassSpec", ...]) -> dict[str, float]:
    prior_total = sum(spec.prior_weight for spec in class_specs)
    return {spec.name: spec.prior_weight / prior_total for spec in class_specs}


def _class_color_map(class_names: tuple[str, ...]) -> dict[str, str]:
    palette = (
        "#b45309",
        "#1d4ed8",
        "#15803d",
        "#9a3412",
        "#6d28d9",
        "#0f766e",
        "#be185d",
        "#4338ca",
    )
    return {name: palette[index % len(palette)] for index, name in enumerate(class_names)}


def _mean_absolute_delta(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(abs(values[index] - values[index - 1]) for index in range(1, len(values))) / (len(values) - 1)


def _speed_differences(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(values[index] - values[index - 1] for index in range(1, len(values)))


def _difference_flip_rate(differences: tuple[float, ...]) -> float:
    if len(differences) < 2:
        return 0.0
    flips = sum(1 for index in range(1, len(differences)) if differences[index - 1] * differences[index] < -0.5)
    return flips / (len(differences) - 1)


def _observed_dynamics_profile(speeds_mph: tuple[float, ...]) -> ObservedDynamicsProfile:
    differences = _speed_differences(speeds_mph)
    if not differences:
        return ObservedDynamicsProfile(mean_delta=0.0, mean_abs_delta=0.0, flip_rate=0.0)
    mean_delta = sum(differences) / len(differences)
    mean_abs_delta = sum(abs(delta) for delta in differences) / len(differences)
    flip_rate = _difference_flip_rate(differences)
    return ObservedDynamicsProfile(mean_delta=mean_delta, mean_abs_delta=mean_abs_delta, flip_rate=flip_rate)


def _cadence_like_score(speeds_mph: tuple[float, ...]) -> float:
    if len(speeds_mph) < 4:
        return 0.0
    differences = _speed_differences(speeds_mph)
    if len(differences) < 3:
        return 0.0
    flip_rate = _difference_flip_rate(differences)
    mean_speed = sum(speeds_mph) / len(speeds_mph)
    mean_abs_delta = sum(abs(delta) for delta in differences) / len(differences)
    return (
        _clamp((flip_rate - 0.45) / 0.30, 0.0, 1.0)
        * _clamp((28.0 - mean_speed) / 10.0, 0.0, 1.0)
        * _clamp((mean_abs_delta - 1.8) / 1.6, 0.0, 1.0)
    )


def _feature_probability_profile(speeds_mph: tuple[float, ...]) -> dict[str, float]:
    if not speeds_mph:
        return {feature_name: 0.0 for feature_name in FEATURE_NAMES}
    peak_speed = max(speeds_mph)
    mean_speed = sum(speeds_mph) / len(speeds_mph)
    trace_range = peak_speed - min(speeds_mph)
    mean_delta = _mean_absolute_delta(speeds_mph)
    dynamics_profile = _observed_dynamics_profile(speeds_mph)
    signed_delta = dynamics_profile.mean_delta
    mean_abs_step_delta = dynamics_profile.mean_abs_delta
    flip_rate = dynamics_profile.flip_rate
    cadence_like = _cadence_like_score(speeds_mph)
    bike_envelope = _clamp((24.0 - peak_speed) / 6.0, 0.0, 1.0) * _clamp((19.0 - mean_speed) / 5.0, 0.0, 1.0)
    horse_envelope = _clamp((mean_speed - 16.0) / 10.0, 0.0, 1.0) * _clamp((41.0 - peak_speed) / 18.0, 0.0, 1.0)
    car_envelope = max(
        _clamp((mean_speed - 33.0) / 18.0, 0.0, 1.0),
        _clamp((peak_speed - 40.0) / 14.0, 0.0, 1.0),
    )
    near_horse_limit = _clamp(1.0 - abs(peak_speed - 38.5) / 5.0, 0.0, 1.0)
    over_horse_limit = _clamp((peak_speed - 40.0) / 10.0, 0.0, 1.0)
    volatile_trace = max(
        _clamp((trace_range - 7.0) / 12.0, 0.0, 1.0),
        _clamp((mean_delta - 2.0) / 4.0, 0.0, 1.0),
    )
    surging_trace = max(
        cadence_like,
        _clamp((mean_abs_step_delta - 2.8) / 1.4, 0.0, 1.0) * _clamp((26.0 - mean_speed) / 8.0, 0.0, 1.0),
    )
    persistent_push = max(
        _clamp((signed_delta - 0.30) / 0.90, 0.0, 1.0),
        _clamp((mean_speed - 40.0) / 12.0, 0.0, 1.0),
    )
    return {
        "bike_envelope": bike_envelope,
        "horse_envelope": horse_envelope,
        "car_envelope": car_envelope,
        "near_horse_limit": near_horse_limit,
        "over_horse_limit": over_horse_limit,
        "volatile_trace": volatile_trace,
        "surging_trace": surging_trace,
        "persistent_push": persistent_push,
    }


def _extract_identity_features(speeds_mph: tuple[float, ...]) -> tuple[str, ...]:
    probabilities = _feature_probability_profile(speeds_mph)
    return tuple(
        feature_name
        for feature_name in FEATURE_NAMES
        if probabilities[feature_name] >= _feature_threshold(feature_name)
    )


def _feature_threshold(feature_name: str) -> float:
    if feature_name == "horse_envelope":
        return 0.35
    if feature_name in {"surging_trace", "persistent_push"}:
        return 0.45
    if feature_name == "volatile_trace":
        return 0.50
    return 0.55


def _identity_dynamics_log_likelihood(
    spec: "IdentityClassSpec",
    observed_history: tuple[float, ...],
) -> float:
    recent_window = observed_history[-min(6, len(observed_history)) :]
    if len(recent_window) < 3:
        return 0.0
    dynamics_profile = _observed_dynamics_profile(recent_window)
    mean_delta = dynamics_profile.mean_delta
    mean_abs_delta = dynamics_profile.mean_abs_delta
    flip_rate = dynamics_profile.flip_rate
    cadence_like = _cadence_like_score(recent_window)
    if spec.name == "bike":
        return (
            0.18 * _gaussian_logpdf(mean_abs_delta, 2.45, 0.95 * 0.95)
            + 0.10 * _gaussian_logpdf(flip_rate, 0.58, 0.18 * 0.18)
            + 0.12 * _gaussian_logpdf(mean_delta, 0.05, 0.60 * 0.60)
            + 0.16 * log(max(0.15 + cadence_like, 1e-9))
        )
    if spec.name == "horse":
        return (
            0.20 * _gaussian_logpdf(mean_abs_delta, 2.75, 1.05 * 1.05)
            + 0.12 * _gaussian_logpdf(flip_rate, 0.52, 0.18 * 0.18)
            + 0.12 * _gaussian_logpdf(mean_delta, 0.00, 0.70 * 0.70)
            + 0.08 * log(max(1.05 - cadence_like, 1e-9))
        )
    if spec.name == "car":
        return (
            0.22 * _gaussian_logpdf(mean_abs_delta, 2.85, 1.10 * 1.10)
            + 0.12 * _gaussian_logpdf(flip_rate, 0.45, 0.20 * 0.20)
            + 0.12 * _gaussian_logpdf(mean_delta, 0.28, 0.80 * 0.80)
            + 0.10 * log(max(1.08 - cadence_like, 1e-9))
        )
    return 0.0


def _identity_mode_log_likelihood(
    spec: "IdentityClassSpec",
    observed_history: tuple[float, ...],
) -> float:
    if not observed_history:
        return 0.0

    recent_window = observed_history[-min(5, len(observed_history)) :]
    recent_mean = sum(recent_window) / len(recent_window)
    recent_peak = max(recent_window)
    recent_floor = min(recent_window)
    recent_range = recent_peak - recent_floor
    last_speed = recent_window[-1]

    if spec.name == "bike":
        modes = [
            0.55 * _gaussian_logpdf(recent_mean, 13.5, 3.4 * 3.4) + 0.45 * _gaussian_logpdf(recent_range, 2.2, 1.8 * 1.8),
            0.55 * _gaussian_logpdf(recent_mean, 18.5, 3.2 * 3.2) + 0.45 * _gaussian_logpdf(recent_range, 3.8, 2.2 * 2.2),
            0.45 * _gaussian_logpdf(last_speed, 20.5, 3.0 * 3.0) + 0.55 * _gaussian_logpdf(recent_range, 5.5, 2.6 * 2.6),
            0.50 * _gaussian_logpdf(recent_mean, 22.8, 2.8 * 2.8)
            + 0.25 * _gaussian_logpdf(recent_peak, 25.5, 2.8 * 2.8)
            + 0.25 * _gaussian_logpdf(recent_range, 4.5, 2.0 * 2.0),
        ]
        return 0.45 + 0.34 * (_logsumexp(modes) - log(len(modes)))

    if spec.name == "horse":
        modes = [
            0.60 * _gaussian_logpdf(recent_mean, 24.5, 4.2 * 4.2) + 0.40 * _gaussian_logpdf(recent_range, 3.2, 2.2 * 2.2),
            0.60 * _gaussian_logpdf(recent_mean, 36.5, 3.8 * 3.8) + 0.40 * _gaussian_logpdf(recent_peak, 39.0, 3.0 * 3.0),
            0.50 * _gaussian_logpdf(recent_mean, 31.0, 4.5 * 4.5) + 0.50 * _gaussian_logpdf(recent_range, 7.0, 3.0 * 3.0),
        ]
        return 0.35 + 0.30 * (_logsumexp(modes) - log(len(modes)))

    if spec.name == "car":
        modes = [
            0.60 * _gaussian_logpdf(recent_mean, 50.0, 6.0 * 6.0) + 0.40 * _gaussian_logpdf(recent_range, 4.5, 3.0 * 3.0),
            0.55 * _gaussian_logpdf(recent_mean, 42.0, 5.5 * 5.5) + 0.45 * _gaussian_logpdf(recent_peak, 45.0, 3.5 * 3.5),
            0.55 * _gaussian_logpdf(recent_mean, 65.0, 7.0 * 7.0) + 0.45 * _gaussian_logpdf(recent_peak, 71.0, 4.2 * 4.2),
        ]
        return 0.25 + 0.28 * (_logsumexp(modes) - log(len(modes)))

    return 0.0


@dataclass(frozen=True, slots=True)
class IdentityClassSpec:
    name: str
    max_speed_mph: float
    cruise_speed_mph: float
    speed_sigma_mph: float
    prior_weight: float


@dataclass(frozen=True, slots=True)
class SpeedScenario:
    name: str
    family_name: str
    expected_class: str
    seed: int
    speeds_true_mph: tuple[float, ...]
    speeds_obs_mph: tuple[float, ...]
    obs_sigma_mph: float
    true_features: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IdentityPosteriorStep:
    observed_speed_mph: float
    updated_class_weights: dict[str, float]
    log_likelihood_terms: dict[str, dict[str, float]]
    feature_probabilities: dict[str, float]
    posterior_entropy: float
    map_class: str


@dataclass(frozen=True, slots=True)
class IdentityClassificationRun:
    scenario_name: str
    family_name: str
    expected_class: str
    true_features: tuple[str, ...]
    detected_features: tuple[str, ...]
    aggregated_feature_probabilities: dict[str, float]
    steps: tuple[IdentityPosteriorStep, ...]
    final_weights: dict[str, float]
    final_map_class: str
    aggregate_map_class: str
    transient_map_class: str
    terminal_map_class: str


@dataclass(frozen=True, slots=True)
class IdentityBenchmarkSummary:
    total_runs: int
    overall_accuracy: float
    transient_accuracy: float
    terminal_accuracy: float
    per_class_accuracy: dict[str, float]
    confusion_counts: dict[str, dict[str, int]]
    transient_confusion_counts: dict[str, dict[str, int]]
    terminal_confusion_counts: dict[str, dict[str, int]]
    scenario_confusion_counts: dict[str, dict[str, int]]
    class_feature_detection_counts: dict[str, dict[str, int]]
    feature_confusion_counts: dict[str, dict[str, int]]
    true_feature_predicted_class_counts: dict[str, dict[str, int]]
    detected_feature_predicted_class_counts: dict[str, dict[str, int]]
    entropy_mean_by_step: tuple[float, ...]
    mean_feature_probability_by_step: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class IdentityBenchmarkResult:
    class_specs: tuple[IdentityClassSpec, ...]
    scenarios: tuple[SpeedScenario, ...]
    runs: tuple[IdentityClassificationRun, ...]
    summary: IdentityBenchmarkSummary


def default_identity_class_specs() -> tuple[IdentityClassSpec, ...]:
    return (
        IdentityClassSpec("horse", max_speed_mph=40.0, cruise_speed_mph=26.0, speed_sigma_mph=8.0, prior_weight=0.4),
        IdentityClassSpec("car", max_speed_mph=80.0, cruise_speed_mph=58.0, speed_sigma_mph=18.0, prior_weight=0.4),
        IdentityClassSpec("bike", max_speed_mph=22.0, cruise_speed_mph=17.0, speed_sigma_mph=4.0, prior_weight=0.2),
    )


def make_identity_scenario(
    *,
    name: str,
    expected_class: str,
    speeds_true_mph: tuple[float, ...] | list[float],
    obs_sigma_mph: float,
    seed: int = 0,
    speeds_obs_mph: tuple[float, ...] | list[float] | None = None,
) -> SpeedScenario:
    true_speeds = tuple(float(speed) for speed in speeds_true_mph)
    if speeds_obs_mph is None:
        rng = random.Random(seed)
        observed_speeds = tuple(speed + rng.gauss(0.0, obs_sigma_mph) for speed in true_speeds)
    else:
        observed_speeds = tuple(float(speed) for speed in speeds_obs_mph)

    if len(true_speeds) != len(observed_speeds):
        raise ValueError("True and observed speed traces must have the same length.")

    return SpeedScenario(
        name=name,
        family_name=name,
        expected_class=expected_class,
        seed=seed,
        speeds_true_mph=true_speeds,
        speeds_obs_mph=observed_speeds,
        obs_sigma_mph=obs_sigma_mph,
        true_features=_extract_identity_features(true_speeds),
    )


def _make_horse_cruise_speeds(steps: int, rng: random.Random) -> tuple[float, ...]:
    speed = rng.uniform(18.0, 26.0)
    speeds: list[float] = []
    for _ in range(steps):
        speed = max(8.0, min(34.0, speed + rng.gauss(0.0, 1.8)))
        speeds.append(speed)
    return tuple(speeds)


def _make_horse_near_limit_speeds(steps: int, rng: random.Random) -> tuple[float, ...]:
    speed = rng.uniform(34.0, 37.0)
    speeds: list[float] = []
    for _ in range(steps):
        speed = max(30.0, min(39.4, speed + rng.gauss(0.15, 1.1)))
        speeds.append(speed)
    return tuple(speeds)


def _make_horse_burst_speeds(steps: int, rng: random.Random) -> tuple[float, ...]:
    speed = rng.uniform(22.0, 29.0)
    speeds: list[float] = []
    direction = 1.0
    for step in range(steps):
        if step > 0 and step % 4 == 0:
            direction *= -1.0
        speed = max(16.0, min(39.0, speed + direction * 1.7 + rng.gauss(0.0, 1.3)))
        speeds.append(speed)
    return tuple(speeds)


def _make_car_border_dance_speeds(steps: int, rng: random.Random) -> tuple[float, ...]:
    base = 39.0
    speeds: list[float] = []
    direction = 1.0
    for step in range(steps):
        if step > 0 and step % 3 == 0:
            direction *= -1.0
        base = max(35.0, min(47.0, base + direction * 2.2 + rng.gauss(0.0, 0.9)))
        speeds.append(base)
    return tuple(speeds)


def _make_car_cruise_speeds(steps: int, rng: random.Random) -> tuple[float, ...]:
    speed = rng.uniform(44.0, 56.0)
    speeds: list[float] = []
    for _ in range(steps):
        speed = max(36.0, min(68.0, speed + rng.gauss(0.1, 2.4)))
        speeds.append(speed)
    return tuple(speeds)


def _make_car_sprint_speeds(steps: int, rng: random.Random) -> tuple[float, ...]:
    speed = rng.uniform(50.0, 62.0)
    speeds: list[float] = []
    for _ in range(steps):
        speed = max(40.0, min(76.0, speed + rng.gauss(0.3, 3.0)))
        speeds.append(speed)
    return tuple(speeds)


def _make_bike_cruise_speeds(steps: int, rng: random.Random) -> tuple[float, ...]:
    speed = rng.uniform(10.0, 15.0)
    speeds: list[float] = []
    for _ in range(steps):
        speed = max(7.0, min(21.0, speed + rng.gauss(0.0, 1.2)))
        speeds.append(speed)
    return tuple(speeds)


def _make_bike_fast_pack_speeds(steps: int, rng: random.Random) -> tuple[float, ...]:
    speed = rng.uniform(16.0, 19.0)
    speeds: list[float] = []
    for _ in range(steps):
        speed = max(12.0, min(22.0, speed + rng.gauss(0.1, 1.0)))
        speeds.append(speed)
    return tuple(speeds)


def _make_bike_surge_speeds(steps: int, rng: random.Random) -> tuple[float, ...]:
    speed = rng.uniform(11.0, 15.0)
    speeds: list[float] = []
    direction = 1.0
    for step in range(steps):
        if step > 0 and step % 5 == 0:
            direction *= -1.0
        speed = max(8.0, min(21.5, speed + direction * 1.6 + rng.gauss(0.0, 0.9)))
        speeds.append(speed)
    return tuple(speeds)


def _make_bike_horse_border_speeds(steps: int, rng: random.Random) -> tuple[float, ...]:
    speed = rng.uniform(18.0, 20.5)
    speeds: list[float] = []
    direction = 1.0
    for step in range(steps):
        if step > 0 and step % 4 == 0:
            direction *= -1.0
        speed = max(14.0, min(27.0, speed + direction * 1.2 + rng.gauss(0.0, 1.4)))
        speeds.append(speed)
    return tuple(speeds)


def _make_horse_car_border_speeds(steps: int, rng: random.Random) -> tuple[float, ...]:
    speed = rng.uniform(34.0, 38.0)
    speeds: list[float] = []
    direction = 1.0
    for step in range(steps):
        if step > 0 and step % 3 == 0:
            direction *= -1.0
        speed = max(28.0, min(48.0, speed + direction * 1.8 + rng.gauss(0.0, 1.8)))
        speeds.append(speed)
    return tuple(speeds)


def _make_horse_car_push_speeds(steps: int, rng: random.Random) -> tuple[float, ...]:
    speed = rng.uniform(37.0, 41.0)
    speeds: list[float] = []
    for _ in range(steps):
        speed = max(31.0, min(53.0, speed + rng.gauss(0.2, 2.0)))
        speeds.append(speed)
    return tuple(speeds)


def generate_identity_scenarios(
    *,
    steps: int = 20,
    seed: int = 7,
    obs_sigma_mph: float = 2.0,
    tracks_per_family: int = 1,
) -> tuple[SpeedScenario, ...]:
    rng = random.Random(seed)
    scenario_defs = (
        ("horse_cruise", "horse", _make_horse_cruise_speeds),
        ("horse_near_limit", "horse", _make_horse_near_limit_speeds),
        ("horse_burst", "horse", _make_horse_burst_speeds),
        ("horse_car_border", "horse", _make_horse_car_border_speeds),
        ("car_cruise", "car", _make_car_cruise_speeds),
        ("border_dance", "car", _make_car_border_dance_speeds),
        ("car_sprint", "car", _make_car_sprint_speeds),
        ("car_push", "car", _make_horse_car_push_speeds),
        ("bike_cruise", "bike", _make_bike_cruise_speeds),
        ("bike_fast_pack", "bike", _make_bike_fast_pack_speeds),
        ("bike_surge", "bike", _make_bike_surge_speeds),
        ("bike_horse_border", "bike", _make_bike_horse_border_speeds),
    )
    scenarios: list[SpeedScenario] = []
    for family_index, (family_name, expected_class, speed_builder) in enumerate(scenario_defs):
        for track_index in range(tracks_per_family):
            scenario_rng = random.Random(rng.randrange(1 << 30) + family_index * 10_000 + track_index)
            true_speeds = speed_builder(steps, scenario_rng)
            scenario_name = family_name if tracks_per_family == 1 else f"{family_name}__mc{track_index:02d}"
            scenario = make_identity_scenario(
                name=scenario_name,
                expected_class=expected_class,
                seed=seed + family_index * 100 + track_index,
                speeds_true_mph=true_speeds,
                obs_sigma_mph=obs_sigma_mph,
                speeds_obs_mph=tuple(speed + scenario_rng.gauss(0.0, obs_sigma_mph) for speed in true_speeds),
            )
            scenarios.append(
                SpeedScenario(
                    name=scenario.name,
                    family_name=family_name,
                    expected_class=scenario.expected_class,
                    seed=scenario.seed,
                    speeds_true_mph=scenario.speeds_true_mph,
                    speeds_obs_mph=scenario.speeds_obs_mph,
                    obs_sigma_mph=scenario.obs_sigma_mph,
                    true_features=scenario.true_features,
                )
            )
    return tuple(scenarios)


def default_identity_hand_authored_scenarios() -> tuple[SpeedScenario, ...]:
    return (
        make_identity_scenario(
            name="manual_horse_cruise",
            expected_class="horse",
            seed=101,
            obs_sigma_mph=1.0,
            speeds_true_mph=(20.0, 22.5, 24.0, 25.5, 24.5, 23.0, 26.0, 27.0),
            speeds_obs_mph=(20.2, 21.9, 24.4, 25.1, 24.0, 23.3, 25.7, 27.5),
        ),
        make_identity_scenario(
            name="manual_near_horse_limit",
            expected_class="horse",
            seed=102,
            obs_sigma_mph=1.0,
            speeds_true_mph=(35.5, 36.8, 37.1, 38.0, 38.6, 39.1, 38.8, 39.3),
            speeds_obs_mph=(35.9, 36.4, 37.6, 37.9, 38.5, 39.2, 38.6, 39.1),
        ),
        make_identity_scenario(
            name="manual_border_dance",
            expected_class="car",
            seed=103,
            obs_sigma_mph=1.0,
            speeds_true_mph=(38.5, 39.2, 40.7, 41.5, 39.6, 42.1, 38.9, 43.0),
            speeds_obs_mph=(38.8, 39.8, 40.4, 41.9, 39.4, 42.4, 39.1, 42.6),
        ),
        make_identity_scenario(
            name="manual_car_cruise",
            expected_class="car",
            seed=104,
            obs_sigma_mph=1.0,
            speeds_true_mph=(48.0, 50.0, 52.5, 55.0, 56.0, 54.5, 57.0, 59.0),
            speeds_obs_mph=(48.4, 49.6, 52.2, 55.3, 55.8, 54.7, 56.6, 58.5),
        ),
        make_identity_scenario(
            name="manual_bike_cruise",
            expected_class="bike",
            seed=105,
            obs_sigma_mph=1.0,
            speeds_true_mph=(11.5, 12.8, 14.2, 15.0, 16.1, 15.4, 14.8, 13.9),
            speeds_obs_mph=(11.9, 12.4, 14.0, 15.4, 15.7, 15.6, 14.5, 14.1),
        ),
        make_identity_scenario(
            name="manual_bike_fast_pack",
            expected_class="bike",
            seed=106,
            obs_sigma_mph=1.0,
            speeds_true_mph=(17.2, 18.0, 18.8, 19.1, 20.0, 19.5, 18.7, 18.1),
            speeds_obs_mph=(17.5, 17.7, 19.0, 18.8, 20.2, 19.2, 18.9, 18.0),
        ),
    )


def run_identity_classifier(
    scenario: SpeedScenario,
    class_specs: tuple[IdentityClassSpec, ...],
) -> IdentityClassificationRun:
    weights = _normalize_priors(class_specs)
    steps: list[IdentityPosteriorStep] = []
    obs_variance = scenario.obs_sigma_mph * scenario.obs_sigma_mph
    observed_history: list[float] = []

    for observed_speed in scenario.speeds_obs_mph:
        observed_history.append(observed_speed)
        log_updates: dict[str, float] = {}
        log_terms: dict[str, dict[str, float]] = {}
        for spec in class_specs:
            log_shape = _gaussian_logpdf(
                observed_speed,
                mean=spec.cruise_speed_mph,
                variance=spec.speed_sigma_mph * spec.speed_sigma_mph + obs_variance,
            )
            validity_prob = _gaussian_interval_probability(
                mean=observed_speed,
                variance=obs_variance,
                upper_limit=spec.max_speed_mph + CLASS_VALIDITY_MARGIN_MPH.get(spec.name, 0.0),
            )
            overspeed_penalty = log(max(validity_prob, 1e-9))
            peak_speed = max(observed_history)
            mean_speed = sum(observed_history) / len(observed_history)
            history_var = max(4.0, obs_variance + 0.35 * max(0.0, peak_speed - mean_speed))
            history_shape = _gaussian_logpdf(mean_speed, spec.cruise_speed_mph, spec.speed_sigma_mph * spec.speed_sigma_mph + history_var)
            mode_shape = _identity_mode_log_likelihood(spec, tuple(observed_history))
            dynamics_shape = _identity_dynamics_log_likelihood(spec, tuple(observed_history))
            log_total = log_shape + 1.4 * overspeed_penalty + 0.45 * history_shape + mode_shape + dynamics_shape
            log_updates[spec.name] = log(max(weights[spec.name], 1e-9)) + log_total
            log_terms[spec.name] = {
                "speed_shape": log_shape,
                "speed_validity": overspeed_penalty,
                "history_shape": 0.45 * history_shape,
                "mode_shape": mode_shape,
                "dynamics_shape": dynamics_shape,
                "total": log_total,
            }

        log_norm = _logsumexp(list(log_updates.values()))
        weights = {name: exp(log_updates[name] - log_norm) for name in log_updates}
        feature_probabilities = _feature_probability_profile(tuple(observed_history))
        steps.append(
            IdentityPosteriorStep(
                observed_speed_mph=observed_speed,
                updated_class_weights=dict(weights),
                log_likelihood_terms=log_terms,
                feature_probabilities=feature_probabilities,
                posterior_entropy=_entropy(weights),
                map_class=max(weights, key=weights.get),
            )
        )

    final_weights = dict(weights)
    aggregated_feature_probabilities = {
        feature_name: sum(step.feature_probabilities[feature_name] for step in steps) / max(len(steps), 1)
        for feature_name in FEATURE_NAMES
    }
    detected_features = tuple(
        feature_name
        for feature_name in FEATURE_NAMES
        if aggregated_feature_probabilities[feature_name] >= _feature_threshold(feature_name)
    )
    if steps:
        aggregate_weights = {
            class_name: sum(step.updated_class_weights[class_name] for step in steps) / len(steps)
            for class_name in final_weights
        }
        transient_end = max(1, len(steps) // 3)
        terminal_start = max(0, len(steps) - max(1, len(steps) // 3))
        transient_weights = {
            class_name: sum(step.updated_class_weights[class_name] for step in steps[:transient_end]) / transient_end
            for class_name in final_weights
        }
        terminal_slice = steps[terminal_start:]
        terminal_weights = {
            class_name: sum(step.updated_class_weights[class_name] for step in terminal_slice) / len(terminal_slice)
            for class_name in final_weights
        }
    else:
        aggregate_weights = dict(final_weights)
        transient_weights = dict(final_weights)
        terminal_weights = dict(final_weights)
    return IdentityClassificationRun(
        scenario_name=scenario.name,
        family_name=scenario.family_name,
        expected_class=scenario.expected_class,
        true_features=scenario.true_features,
        detected_features=detected_features,
        aggregated_feature_probabilities=aggregated_feature_probabilities,
        steps=tuple(steps),
        final_weights=final_weights,
        final_map_class=max(final_weights, key=final_weights.get),
        aggregate_map_class=max(aggregate_weights, key=aggregate_weights.get),
        transient_map_class=max(transient_weights, key=transient_weights.get),
        terminal_map_class=max(terminal_weights, key=terminal_weights.get),
    )


def summarize_identity_runs(runs: tuple[IdentityClassificationRun, ...]) -> IdentityBenchmarkSummary:
    outcome_summary = summarize_classification_outcomes(
        runs,
        true_class_fn=lambda run: run.expected_class,
        aggregate_pred_fn=lambda run: run.aggregate_map_class,
        transient_pred_fn=lambda run: run.transient_map_class,
        terminal_pred_fn=lambda run: run.terminal_map_class,
        scenario_group_fn=lambda run: run.family_name,
    )
    feature_summary = summarize_classification_features(
        runs,
        feature_names=FEATURE_NAMES,
        true_class_fn=lambda run: run.expected_class,
        aggregate_pred_fn=lambda run: run.aggregate_map_class,
        true_features_fn=lambda run: run.true_features,
        detected_features_fn=lambda run: run.detected_features,
        step_iter_fn=lambda run: run.steps,
        step_entropy_fn=lambda step: step.posterior_entropy,
        step_feature_probability_fn=lambda step: step.feature_probabilities,
    )
    return IdentityBenchmarkSummary(
        total_runs=outcome_summary.total_runs,
        overall_accuracy=outcome_summary.overall_accuracy,
        transient_accuracy=outcome_summary.transient_accuracy,
        terminal_accuracy=outcome_summary.terminal_accuracy,
        per_class_accuracy=outcome_summary.per_class_accuracy,
        confusion_counts=outcome_summary.confusion_counts,
        transient_confusion_counts=outcome_summary.transient_confusion_counts,
        terminal_confusion_counts=outcome_summary.terminal_confusion_counts,
        scenario_confusion_counts=outcome_summary.scenario_confusion_counts,
        class_feature_detection_counts=feature_summary.class_feature_detection_counts,
        feature_confusion_counts=feature_summary.feature_confusion_counts,
        true_feature_predicted_class_counts=feature_summary.true_feature_predicted_class_counts,
        detected_feature_predicted_class_counts=feature_summary.detected_feature_predicted_class_counts,
        entropy_mean_by_step=feature_summary.entropy_mean_by_step,
        mean_feature_probability_by_step=feature_summary.mean_feature_probability_by_step,
    )


def run_identity_benchmark(
    *,
    steps: int = 20,
    seed: int = 7,
    obs_sigma_mph: float = 2.0,
    tracks_per_family: int = 6,
    class_specs: tuple[IdentityClassSpec, ...] | None = None,
    scenarios: tuple[SpeedScenario, ...] | None = None,
) -> IdentityBenchmarkResult:
    specs = class_specs or default_identity_class_specs()
    benchmark_scenarios = scenarios or generate_identity_scenarios(
        steps=steps,
        seed=seed,
        obs_sigma_mph=obs_sigma_mph,
        tracks_per_family=tracks_per_family,
    )
    runs = tuple(run_identity_classifier(scenario, specs) for scenario in benchmark_scenarios)
    return IdentityBenchmarkResult(
        class_specs=specs,
        scenarios=benchmark_scenarios,
        runs=runs,
        summary=summarize_identity_runs(runs),
    )


def render_identity_benchmark_markdown(result: IdentityBenchmarkResult) -> str:
    class_names = tuple(spec.name for spec in result.class_specs)
    family_counts: dict[str, int] = {}
    for scenario in result.scenarios:
        family_counts[scenario.family_name] = family_counts.get(scenario.family_name, 0) + 1
    report = MarkdownDocument("1D Identity Speed Benchmark")
    report.paragraph(
        "This benchmark treats class as a static identity and updates posterior mass over "
        "`horse`, `car`, and `bike` as new speed measurements arrive."
    )
    report.paragraph(
        f"Overall accuracy: {result.summary.overall_accuracy:.3f}\n"
        f"Transient accuracy: {result.summary.transient_accuracy:.3f}\n"
        f"Terminal accuracy: {result.summary.terminal_accuracy:.3f}"
    )
    report.heading("Scenario Family Counts", level=2)
    report.bullet_list([f"{family_name}: {family_counts[family_name]} runs" for family_name in sorted(family_counts)])

    report.heading("Per-Class Accuracy", level=2)
    report.bullet_list([f"{class_name}: {result.summary.per_class_accuracy[class_name]:.3f}" for class_name in sorted(result.summary.per_class_accuracy)])

    report.heading("Class Confusion Counts", level=2)
    report.bullet_list(
        [
            f"{true_class}: " + ", ".join(
                f"{predicted_class}={result.summary.confusion_counts[true_class][predicted_class]}"
                for predicted_class in sorted(result.summary.confusion_counts[true_class])
            )
            for true_class in sorted(result.summary.confusion_counts)
        ]
    )
    report.heading("Scenario Confusion Counts", level=2)
    report.bullet_list(
        [
            f"{scenario_name}: " + ", ".join(
                f"{predicted_class}={result.summary.scenario_confusion_counts[scenario_name][predicted_class]}"
                for predicted_class in sorted(result.summary.scenario_confusion_counts[scenario_name])
            )
            for scenario_name in sorted(result.summary.scenario_confusion_counts)
        ]
    )
    report.heading("Feature Semantics", level=2)
    report.bullet_list(
        [f"{class_name}: expected anchors -> {', '.join(CLASS_FEATURE_ANCHORS[class_name])}" for class_name in sorted(CLASS_FEATURE_ANCHORS)]
    )
    report.heading("Class vs Detected Feature Counts", level=2)
    report.bullet_list(
        [
            f"{class_name}: "
            + ", ".join(
                f"{feature_name}={result.summary.class_feature_detection_counts[class_name][feature_name]}"
                for feature_name in FEATURE_NAMES
            )
            for class_name in sorted(result.summary.class_feature_detection_counts)
        ]
    )
    report.heading("Feature Confusion Counts", level=2)
    report.bullet_list(
        [
            (
                f"{feature_name}: tp={result.summary.feature_confusion_counts[feature_name]['tp']}, "
                f"fp={result.summary.feature_confusion_counts[feature_name]['fp']}, "
                f"fn={result.summary.feature_confusion_counts[feature_name]['fn']}, "
                f"tn={result.summary.feature_confusion_counts[feature_name]['tn']}"
            )
            for feature_name in FEATURE_NAMES
        ]
    )
    report.heading("True Feature vs Predicted Class Matrix", level=2)
    report.bullet_list(
        [
            f"{feature_name}: "
            + ", ".join(
                f"{predicted_class}={result.summary.true_feature_predicted_class_counts[feature_name][predicted_class]}"
                for predicted_class in sorted(result.summary.true_feature_predicted_class_counts[feature_name])
            )
            for feature_name in FEATURE_NAMES
        ]
    )
    report.heading("Detected Feature vs Predicted Class Matrix", level=2)
    report.bullet_list(
        [
            f"{feature_name}: "
            + ", ".join(
                f"{predicted_class}={result.summary.detected_feature_predicted_class_counts[feature_name][predicted_class]}"
                for predicted_class in sorted(result.summary.detected_feature_predicted_class_counts[feature_name])
            )
            for feature_name in FEATURE_NAMES
        ]
    )
    report.heading("Mean Posterior Entropy by Step", level=2)
    report.paragraph(
        ", ".join(f"{step_index}:{value:.3f}" for step_index, value in enumerate(result.summary.entropy_mean_by_step))
    )
    report.heading("Scenario Details", level=2)
    for scenario, run in zip(result.scenarios, result.runs):
        final_probs = ", ".join(f"{name}={run.final_weights[name]:.3f}" for name in class_names)
        report.heading(f"{scenario.name}", level=3)
        report.bullet_list(
            [
                f"Scenario family: `{scenario.family_name}`",
                f"Expected class: `{scenario.expected_class}`",
                f"Aggregate MAP class: `{run.aggregate_map_class}`",
                f"Final MAP class: `{run.final_map_class}`",
                f"True features: {', '.join(run.true_features) if run.true_features else 'none'}",
                f"Detected features: {', '.join(run.detected_features) if run.detected_features else 'none'}",
                f"Final probabilities: {final_probs}",
                f"Peak observed speed: {max(scenario.speeds_obs_mph):.2f} mph",
            ]
        )
    return report.text()


def _representative_identity_pairs(result: IdentityBenchmarkResult) -> RepresentativeIdentityPairs:
    pairs = list(zip(result.scenarios, result.runs))
    by_family: dict[str, list[tuple[SpeedScenario, IdentityClassificationRun]]] = {}
    for scenario, run in pairs:
        by_family.setdefault(scenario.family_name, []).append((scenario, run))
    representatives: list[tuple[SpeedScenario, IdentityClassificationRun]] = []
    for family_name in sorted(by_family):
        family_pairs = by_family[family_name]
        misclassified = [pair for pair in family_pairs if pair[1].aggregate_map_class != pair[1].expected_class]
        representatives.append(misclassified[0] if misclassified else family_pairs[0])
    return RepresentativeIdentityPairs(pairs=tuple(representatives))


def _build_identity_figure(result: IdentityBenchmarkResult):
    representative_pairs = _representative_identity_pairs(result).pairs
    fig, axes = plt.subplots(len(representative_pairs), 2, figsize=(12, 3.8 * len(representative_pairs)), sharex=False)
    if len(representative_pairs) == 1:
        axes_rows = [axes]
    else:
        axes_rows = list(axes)

    class_names = tuple(spec.name for spec in result.class_specs)
    colors = _class_color_map(class_names)
    max_speed_by_class = {spec.name: spec.max_speed_mph for spec in result.class_specs}

    for row_axes, (scenario, run) in zip(axes_rows, representative_pairs):
        speed_ax, posterior_ax = row_axes
        steps = list(range(len(scenario.speeds_obs_mph)))
        speed_ax.plot(steps, scenario.speeds_obs_mph, color="#111827", linewidth=2.0, label="observed speed")
        for class_name in class_names:
            speed_ax.axhline(
                max_speed_by_class[class_name],
                color=colors[class_name],
                linestyle="--",
                linewidth=1.2,
                label=f"{class_name} max",
            )
        speed_ax.set_title(f"{scenario.name} speed trace", loc="left", fontsize=12, fontweight="bold")
        speed_ax.set_ylabel("mph")
        speed_ax.grid(True, alpha=0.25)
        speed_ax.legend(loc="upper left", frameon=False, ncol=2)

        for class_name in class_names:
            posterior_ax.plot(
                steps,
                [step.updated_class_weights[class_name] for step in run.steps],
                color=colors[class_name],
                linewidth=2.2,
                label=class_name,
            )
        posterior_ax.set_ylim(0.0, 1.0)
        posterior_ax.set_title(f"{scenario.name} posterior", loc="left", fontsize=12, fontweight="bold")
        posterior_ax.set_ylabel("probability")
        posterior_ax.grid(True, alpha=0.25)
        posterior_ax.legend(loc="upper right", frameon=False)

        speed_ax.set_xlabel("step")
        posterior_ax.set_xlabel("step")

    fig.suptitle("Identity Posterior Evolution", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def render_identity_benchmark_png_bytes(result: IdentityBenchmarkResult) -> bytes:
    fig = _build_identity_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _build_identity_feature_confusion_figure(result: IdentityBenchmarkResult):
    summary = result.summary
    class_names = sorted(summary.confusion_counts)
    feature_names = list(FEATURE_NAMES)
    true_matrix = [
        [summary.true_feature_predicted_class_counts[feature_name][class_name] for class_name in class_names]
        for feature_name in feature_names
    ]
    detected_matrix = [
        [summary.detected_feature_predicted_class_counts[feature_name][class_name] for class_name in class_names]
        for feature_name in feature_names
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.9 * len(feature_names) + 3.5))
    true_ax, detected_ax = axes

    for ax, matrix, title in (
        (true_ax, true_matrix, "True Feature vs Predicted Class"),
        (detected_ax, detected_matrix, "Detected Feature vs Predicted Class"),
    ):
        image = ax.imshow(matrix, aspect="auto", cmap="Blues")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(feature_names)), feature_names)
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                text_color = "#ffffff" if value > max(1, max(max(r) for r in matrix) * 0.45) else "#111827"
                ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=8, color=text_color)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Identity Feature-Class Confusion Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def render_identity_feature_confusion_png_bytes(result: IdentityBenchmarkResult) -> bytes:
    fig = _build_identity_feature_confusion_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_identity_benchmark_trace_csv(result: IdentityBenchmarkResult, output_dir: str | Path) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "identity_1d_benchmark_traces.csv"
    class_names = tuple(spec.name for spec in result.class_specs)
    header = [
        "scenario_name",
        "scenario_family",
        "expected_class",
        "aggregate_map_class",
        "final_map_class",
        "transient_map_class",
        "terminal_map_class",
        "step",
        "observed_speed_mph",
        "posterior_entropy",
        "true_features",
        "detected_features",
        *class_names,
        *FEATURE_NAMES,
    ]
    rows: list[tuple[str, ...]] = []
    for run in result.runs:
        for step_idx, step in enumerate(run.steps):
            rows.append(
                (
                    run.scenario_name,
                    run.family_name,
                    run.expected_class,
                    run.aggregate_map_class,
                    run.final_map_class,
                    run.transient_map_class,
                    run.terminal_map_class,
                    str(step_idx),
                    f"{step.observed_speed_mph:.6f}",
                    f"{step.posterior_entropy:.6f}",
                    "|".join(run.true_features),
                    "|".join(run.detected_features),
                    *(f"{step.updated_class_weights[class_name]:.6f}" for class_name in class_names),
                    *(f"{step.feature_probabilities[feature_name]:.6f}" for feature_name in FEATURE_NAMES),
                )
            )
    _write_csv_rows(output_path, rows, header)
    return output_path


def write_identity_benchmark_artifacts(
    output_dir: str | Path,
    *,
    steps: int = 20,
    seed: int = 7,
    obs_sigma_mph: float = 2.0,
    result: IdentityBenchmarkResult | None = None,
) -> IdentityArtifactPaths:
    benchmark_result = result or run_identity_benchmark(steps=steps, seed=seed, obs_sigma_mph=obs_sigma_mph)
    artifacts = write_one_d_surface_artifacts(
        identity_witness_surface(),
        output_dir,
        result=benchmark_result,
        summary_filename="identity_1d_benchmark_summary.md",
        plot_filename="identity_1d_benchmark_posteriors.png",
        write_extra=False,
        nest_under_study_id=False,
    )
    assert artifacts.summary_path is not None
    assert artifacts.plot_path is not None
    assert artifacts.trace_path is not None
    return IdentityArtifactPaths(
        summary_path=artifacts.summary_path,
        plot_path=artifacts.plot_path,
        trace_path=artifacts.trace_path,
    )


def write_identity_feature_confusion_artifacts(
    output_dir: str | Path,
    *,
    steps: int = 20,
    seed: int = 7,
    obs_sigma_mph: float = 2.0,
    result: IdentityBenchmarkResult | None = None,
) -> Path:
    benchmark_result = result or run_identity_benchmark(steps=steps, seed=seed, obs_sigma_mph=obs_sigma_mph)
    artifacts = write_one_d_surface_artifacts(
        identity_witness_surface(),
        output_dir,
        result=benchmark_result,
        summary_filename="identity_1d_benchmark_summary.md",
        plot_filename="identity_1d_benchmark_posteriors.png",
        write_summary=False,
        write_plot=False,
        write_trace=False,
        write_extra=True,
        nest_under_study_id=False,
    )
    return artifacts.extra_paths["identity_1d_feature_confusion.png"]


def identity_witness_surface() -> OneDWitnessSurface[IdentityBenchmarkResult, IdentityClassSpec]:
    surface_ref: dict[str, OneDWitnessSurface[IdentityBenchmarkResult, IdentityClassSpec]] = {}

    def _write_feature_confusion_artifact(result: IdentityBenchmarkResult, output_path: Path) -> Path:
        output_path.write_bytes(render_identity_feature_confusion_png_bytes(result))
        return output_path

    def _write_artifacts(output_dir: str | Path):
        return write_one_d_surface_artifacts(
            surface_ref["surface"],
            output_dir,
            summary_filename="identity_1d_benchmark_summary.md",
            plot_filename="identity_1d_benchmark_posteriors.png",
            nest_under_study_id=False,
        )

    def _describe_artifacts(artifacts) -> tuple[str, ...]:
        entries = [artifacts.summary_path, artifacts.plot_path, artifacts.trace_path, *artifacts.extra_paths.values()]
        return tuple(str(path) for path in entries if path is not None)

    surface = OneDWitnessSurface(
        study_id="identity_1d",
        class_specs=default_identity_class_specs(),
        feature_names=FEATURE_NAMES,
        run=run_identity_benchmark,
        write_artifacts=_write_artifacts,
        describe_artifacts=_describe_artifacts,
        render_markdown=render_identity_benchmark_markdown,
        render_png_bytes=render_identity_benchmark_png_bytes,
        write_trace_csv=write_identity_benchmark_trace_csv,
        extra_artifact_writers=(
            NamedArtifactWriter(
                filename="identity_1d_feature_confusion.png",
                write=_write_feature_confusion_artifact,
            ),
        ),
        metadata={
            "study_kind": "1d_witness",
            "problem_family": "identity_1d",
        },
    )
    surface_ref["surface"] = surface
    return surface
