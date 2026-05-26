from __future__ import annotations

import random

from ..trajectory_generator import (
    KinematicSeries,
    TrajectoryArtifact,
    _class_by_name,
    _generate_states,
    _inject_measurement_noise,
    _make_manual_trajectory,
)


def generate_short_horizon_scenarios(*, seed: int = 7) -> tuple[TrajectoryArtifact, ...]:
    rng = random.Random(seed + 10_000)
    scenarios: list[TrajectoryArtifact] = []
    specs = (
        (
            "short_horizon_constant_velocity",
            "constant_velocity",
            (0.0, 0.45, 0.95, 1.55, 2.25),
            {"position": -0.5, "velocity": 1.2},
        ),
        (
            "short_horizon_constant_acceleration",
            "constant_acceleration",
            (0.0, 0.40, 0.85, 1.35, 1.90),
            {"position": 0.2, "velocity": 0.45, "acceleration": 0.55},
        ),
        (
            "short_horizon_braking",
            "braking",
            (0.0, 0.35, 0.75, 1.10, 1.45),
            {"position": 0.0, "velocity": 1.8, "deceleration": -0.9},
        ),
        (
            "short_horizon_maneuver",
            "maneuver",
            (0.0, 0.40, 0.80, 1.20, 1.60),
            {"position": -0.2, "velocity": 0.65, "accel_early": 0.7, "accel_late": -0.85, "switch_fraction": 0.58},
        ),
    )
    for index, (scenario_id, class_name, times, params) in enumerate(specs):
        class_definition = _class_by_name(class_name)
        positions_true, velocities_true, accelerations_true = _generate_states(class_definition, times, params)
        measurement_std = 0.06
        noisy_measurements = _inject_measurement_noise(
            rng,
            positions_true,
            measurement_std,
            outlier_probability=0.0,
        )
        scenarios.append(
            _make_manual_trajectory(
                trajectory_id=scenario_id,
                true_class=class_name,
                tier="short_horizon_v1",
                scenario_family="short_horizon",
                measurements=noisy_measurements.measurements,
                times=times,
                true_position=positions_true,
                true_velocity=velocities_true,
                true_acceleration=accelerations_true,
                measurement_std=measurement_std,
                outlier_indices=noisy_measurements.outlier_indices,
                seed=seed + index,
                generator_parameters={"coverage_target": "short_horizon"},
            )
        )
    return tuple(scenarios)


def generate_perturbation_sweep_scenarios(*, seed: int = 7) -> tuple[TrajectoryArtifact, ...]:
    times_regular = tuple(0.45 * index for index in range(8))
    times_irregular = (0.0, 0.30, 0.86, 1.22, 1.95, 2.36, 3.11, 3.48)
    base_class = _class_by_name("constant_velocity")
    base_params = {"position": -1.0, "velocity": 1.05}
    positions_true_regular, velocities_true_regular, accelerations_true_regular = _generate_states(base_class, times_regular, base_params)
    positions_true_irregular, velocities_true_irregular, accelerations_true_irregular = _generate_states(base_class, times_irregular, base_params)
    specs = (
        ("noise_sweep_low", times_regular, positions_true_regular, velocities_true_regular, accelerations_true_regular, 0.03, 0.0),
        ("noise_sweep_medium", times_regular, positions_true_regular, velocities_true_regular, accelerations_true_regular, 0.10, 0.0),
        ("noise_sweep_high", times_regular, positions_true_regular, velocities_true_regular, accelerations_true_regular, 0.20, 0.0),
        ("outlier_sweep_none", times_regular, positions_true_regular, velocities_true_regular, accelerations_true_regular, 0.07, 0.0),
        ("outlier_sweep_heavy", times_regular, positions_true_regular, velocities_true_regular, accelerations_true_regular, 0.07, 0.20),
        ("irregular_dt_sweep_regular", times_regular, positions_true_regular, velocities_true_regular, accelerations_true_regular, 0.07, 0.0),
        ("irregular_dt_sweep_irregular", times_irregular, positions_true_irregular, velocities_true_irregular, accelerations_true_irregular, 0.07, 0.0),
    )
    scenarios: list[TrajectoryArtifact] = []
    for index, (scenario_id, times, positions_true, velocities_true, accelerations_true, measurement_std, outlier_probability) in enumerate(specs):
        rng = random.Random(seed + 20_000 + index)
        noisy_measurements = _inject_measurement_noise(
            rng,
            positions_true,
            measurement_std,
            outlier_probability=outlier_probability,
        )
        scenarios.append(
            _make_manual_trajectory(
                trajectory_id=scenario_id,
                true_class="constant_velocity",
                tier="perturbation_sweeps_v1",
                scenario_family="perturbation_sweep",
                measurements=noisy_measurements.measurements,
                times=times,
                true_position=positions_true,
                true_velocity=velocities_true,
                true_acceleration=accelerations_true,
                measurement_std=measurement_std,
                outlier_indices=noisy_measurements.outlier_indices,
                seed=seed + 100 + index,
                generator_parameters={
                    "coverage_target": "sweep",
                    "outlier_probability": outlier_probability,
                    "time_regime": "irregular" if "irregular" in scenario_id else "regular",
                },
            )
        )
    return tuple(scenarios)


def _switching_segment_trajectory(
    *,
    initial_position: float,
    initial_velocity: float,
    times: tuple[float, ...],
    accelerations: tuple[float, ...],
) -> KinematicSeries:
    positions = [initial_position]
    velocities = [initial_velocity]
    applied_accelerations = [accelerations[0] if accelerations else 0.0]
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        accel = accelerations[index - 1]
        next_position = positions[-1] + velocities[-1] * dt + 0.5 * accel * dt * dt
        next_velocity = velocities[-1] + accel * dt
        positions.append(next_position)
        velocities.append(next_velocity)
        applied_accelerations.append(accel)
    return KinematicSeries(tuple(positions), tuple(velocities), tuple(applied_accelerations))


def generate_switching_scenarios(*, seed: int = 7) -> tuple[TrajectoryArtifact, ...]:
    rng = random.Random(seed + 30_000)
    scenarios: list[TrajectoryArtifact] = []

    times_a = tuple(0.5 * index for index in range(10))
    pos_a = [0.0]
    vel_a = [0.0]
    accel_series_a = [0.0]
    for index in range(1, len(times_a)):
        dt = times_a[index] - times_a[index - 1]
        if times_a[index - 1] < 2.0:
            accel = 0.0
            next_velocity = 0.0
            next_position = pos_a[-1]
        else:
            accel = 0.0
            next_velocity = 1.15
            next_position = pos_a[-1] + next_velocity * dt
        pos_a.append(next_position)
        vel_a.append(next_velocity)
        accel_series_a.append(accel)
    switching_specs = [
        (
            "stationary_then_moving",
            times_a,
            tuple(pos_a),
            tuple(vel_a),
            tuple(accel_series_a),
            0.05,
            {"segment_modes": ["stationary", "constant_velocity"], "switch_time": 2.0},
        ),
        (
            "constant_velocity_then_braking",
            tuple(0.45 * index for index in range(11)),
            *_switching_segment_trajectory(
                initial_position=-1.0,
                initial_velocity=1.6,
                times=tuple(0.45 * index for index in range(11)),
                accelerations=tuple(0.0 if index < 5 else -0.85 for index in range(10)),
            ),
            0.06,
            {"segment_modes": ["constant_velocity", "braking"], "switch_time": 2.25},
        ),
        (
            "constant_velocity_then_maneuver",
            tuple(0.40 * index for index in range(12)),
            *_switching_segment_trajectory(
                initial_position=0.5,
                initial_velocity=1.0,
                times=tuple(0.40 * index for index in range(12)),
                accelerations=tuple(0.0 if index < 5 else (0.65 if index < 8 else -0.75) for index in range(11)),
            ),
            0.06,
            {"segment_modes": ["constant_velocity", "maneuver"], "switch_time": 2.0},
        ),
    ]
    for index, (scenario_id, times, positions_true, velocities_true, accelerations_true, measurement_std, extra_params) in enumerate(switching_specs):
        noisy_measurements = _inject_measurement_noise(
            rng,
            positions_true,
            measurement_std,
            outlier_probability=0.0,
        )
        scenarios.append(
            _make_manual_trajectory(
                trajectory_id=scenario_id,
                true_class="switching",
                tier="switching_scenarios_v1",
                scenario_family="switching",
                measurements=noisy_measurements.measurements,
                times=times,
                true_position=positions_true,
                true_velocity=velocities_true,
                true_acceleration=accelerations_true,
                measurement_std=measurement_std,
                outlier_indices=noisy_measurements.outlier_indices,
                seed=seed + 200 + index,
                generator_parameters={"coverage_target": "switching", **extra_params},
            )
        )
    return tuple(scenarios)
