from __future__ import annotations

import random
from math import sin

from ..corpus.adequacy_audit_utils import load_class_pair_manifest
from ..scenarios import get_scenario_measurement_sigma, get_scenario_times
from .contracts import ExecutablePairSpec, ExecutableTrajectory
from ..trajectory_series import KinematicSeries


def build_pair_specs(
    *,
    declared_class_pairs: tuple[tuple[str, str], ...],
    class_pair_manifest_path,
) -> tuple[ExecutablePairSpec, ...]:
    manifest_entries = {
        f"{left}_vs_{right}": entry
        for entry in load_class_pair_manifest(class_pair_manifest_path)
        for left, right in [tuple(entry["pair"])]
    }
    specs: list[ExecutablePairSpec] = []
    for left, right in declared_class_pairs:
        pair_id = f"{left}_vs_{right}"
        entry = manifest_entries[pair_id]
        class_a, class_b = tuple(entry["pair"])
        specs.append(
            ExecutablePairSpec(
                pair_id=pair_id,
                class_a=str(class_a),
                class_b=str(class_b),
                expected_difficulty=str(entry.get("expected_difficulty", "unknown")),
            )
        )
    return tuple(specs)


def _pair_state(
    pair_id: str,
    class_name: str,
    scenario_id: str,
    times: tuple[float, ...],
) -> KinematicSeries:
    if pair_id == "stationary_vs_constant_velocity":
        if class_name == "stationary":
            position = tuple(0.0 for _ in times)
            velocity = tuple(0.0 for _ in times)
            acceleration = tuple(0.0 for _ in times)
            return KinematicSeries(position, velocity, acceleration)
        speed = 0.85 if scenario_id != "endpoint_match" else 0.30
        position = tuple(speed * time for time in times)
        velocity = tuple(speed for _ in times)
        acceleration = tuple(0.0 for _ in times)
        return KinematicSeries(position, velocity, acceleration)

    if pair_id == "constant_velocity_vs_constant_acceleration":
        if class_name == "constant_velocity":
            speed = 0.80 if scenario_id != "endpoint_match" else 1.10
            position = tuple(speed * time for time in times)
            velocity = tuple(speed for _ in times)
            acceleration = tuple(0.0 for _ in times)
            return KinematicSeries(position, velocity, acceleration)
        speed0 = 0.80 if scenario_id != "endpoint_match" else 0.35
        accel = 0.24 if scenario_id != "endpoint_match" else 0.30
        position = tuple(speed0 * time + 0.5 * accel * time * time for time in times)
        velocity = tuple(speed0 + accel * time for time in times)
        acceleration = tuple(accel for _ in times)
        return KinematicSeries(position, velocity, acceleration)

    if pair_id == "constant_velocity_vs_braking":
        if class_name == "constant_velocity":
            speed = 0.82 if scenario_id != "endpoint_match" else 0.18
            position = tuple(speed * time for time in times)
            velocity = tuple(speed for _ in times)
            acceleration = tuple(0.0 for _ in times)
            return KinematicSeries(position, velocity, acceleration)
        speed0 = 0.90 if scenario_id != "endpoint_match" else 0.16
        accel = -0.02 if scenario_id != "endpoint_match" else -0.016
        position = tuple(speed0 * time + 0.5 * accel * time * time for time in times)
        velocity = tuple(speed0 + accel * time for time in times)
        acceleration = tuple(accel for _ in times)
        return KinematicSeries(position, velocity, acceleration)

    if pair_id == "maneuver_vs_bounded_acceleration":
        if class_name == "bounded_acceleration":
            velocity = [0.38 if scenario_id != "endpoint_match" else 0.31]
            position = [0.0]
            accelerations = [0.10]
            for previous_time, current_time in zip(times, times[1:]):
                dt = current_time - previous_time
                accel = max(-0.12, min(0.12, 0.09 + 0.03 * sin(0.8 * previous_time)))
                accelerations.append(accel)
                next_velocity = velocity[-1] + accel * dt
                position.append(position[-1] + velocity[-1] * dt + 0.5 * accel * dt * dt)
                velocity.append(next_velocity)
            return KinematicSeries(tuple(position), tuple(velocity), tuple(accelerations))
        velocity = [0.36 if scenario_id != "endpoint_match" else 0.28]
        position = [0.0]
        accelerations = [0.05]
        for previous_time, current_time in zip(times, times[1:]):
            dt = current_time - previous_time
            accel = 0.14 * sin(2.35 * previous_time) + 0.07 * sin(0.65 * previous_time + 0.3)
            if scenario_id == "short_noisy":
                accel *= 1.2
            accelerations.append(accel)
            next_velocity = velocity[-1] + accel * dt
            position.append(position[-1] + velocity[-1] * dt + 0.5 * accel * dt * dt)
            velocity.append(next_velocity)
        return KinematicSeries(tuple(position), tuple(velocity), tuple(accelerations))

    if pair_id == "constant_acceleration_vs_maneuver":
        if class_name == "constant_acceleration":
            speed0 = 0.44 if scenario_id != "endpoint_match" else 0.26
            accel = 0.11 if scenario_id != "endpoint_match" else 0.08
            position = tuple(speed0 * time + 0.5 * accel * time * time for time in times)
            velocity = tuple(speed0 + accel * time for time in times)
            acceleration = tuple(accel for _ in times)
            return KinematicSeries(position, velocity, acceleration)
        velocity = [0.42 if scenario_id != "endpoint_match" else 0.24]
        position = [0.0]
        accelerations = [0.09]
        for previous_time, current_time in zip(times, times[1:]):
            dt = current_time - previous_time
            accel = 0.12 + 0.10 * sin(1.65 * previous_time) - 0.08 * sin(3.20 * previous_time + 0.2)
            if scenario_id in {"short", "short_noisy"}:
                accel += 0.03 * sin(5.0 * previous_time + 0.5)
            accelerations.append(accel)
            next_velocity = velocity[-1] + accel * dt
            position.append(position[-1] + velocity[-1] * dt + 0.5 * accel * dt * dt)
            velocity.append(next_velocity)
        return KinematicSeries(tuple(position), tuple(velocity), tuple(accelerations))

    raise KeyError(f"unsupported pair id: {pair_id}")


def _make_pair_trajectory(
    *,
    pair_spec: ExecutablePairSpec,
    class_name: str,
    scenario_id: str,
    seed: int,
    example_index: int,
) -> ExecutableTrajectory:
    rng = random.Random(seed)
    times = get_scenario_times(scenario_id)
    true_position, true_velocity, true_acceleration = _pair_state(pair_spec.pair_id, class_name, scenario_id, times)
    sigma = get_scenario_measurement_sigma(scenario_id)
    measurements = tuple(value + rng.gauss(0.0, sigma) for value in true_position)
    if scenario_id == "outlier":
        midpoint = len(measurements) // 2
        rebound = min(midpoint + 1, len(measurements) - 1)
        glitch = 1.65
        measurements = tuple(
            value
            + (-glitch if index == midpoint else 0.0)
            + (glitch if index == rebound else 0.0)
            for index, value in enumerate(measurements)
        )
    return ExecutableTrajectory(
        trajectory_id=f"{pair_spec.pair_id}_{scenario_id}_{class_name}_{example_index}",
        class_pair_id=pair_spec.pair_id,
        class_a=pair_spec.class_a,
        class_b=pair_spec.class_b,
        true_class=class_name,
        scenario_id=scenario_id,
        seed=seed,
        times=times,
        measurements=measurements,
        true_position=true_position,
        true_velocity=true_velocity,
        true_acceleration=true_acceleration,
    )


def build_reference_trajectory(
    pair_spec: ExecutablePairSpec,
    class_name: str,
    scenario_id: str,
    times: tuple[float, ...],
) -> ExecutableTrajectory:
    true_position, true_velocity, true_acceleration = _pair_state(pair_spec.pair_id, class_name, scenario_id, times)
    return ExecutableTrajectory(
        trajectory_id=f"reference_{pair_spec.pair_id}_{scenario_id}_{class_name}",
        class_pair_id=pair_spec.pair_id,
        class_a=pair_spec.class_a,
        class_b=pair_spec.class_b,
        true_class=class_name,
        scenario_id=scenario_id,
        seed=0,
        times=times,
        measurements=true_position,
        true_position=true_position,
        true_velocity=true_velocity,
        true_acceleration=true_acceleration,
    )


def _generate_pair_dataset_for_scenarios(
    pair_specs: tuple[ExecutablePairSpec, ...],
    seed: int,
    trajectories_per_case: int,
    scenario_ids: tuple[str, ...],
) -> tuple[ExecutableTrajectory, ...]:
    trajectories: list[ExecutableTrajectory] = []
    for pair_index, pair_spec in enumerate(pair_specs):
        classes = (pair_spec.class_a, pair_spec.class_b)
        for scenario_index, scenario_id in enumerate(scenario_ids):
            for class_index, class_name in enumerate(classes):
                for example_index in range(trajectories_per_case):
                    trajectories.append(
                        _make_pair_trajectory(
                            pair_spec=pair_spec,
                            class_name=class_name,
                            scenario_id=scenario_id,
                            seed=seed + pair_index * 1000 + scenario_index * 100 + class_index * 20 + example_index,
                            example_index=example_index,
                        )
                    )
    return tuple(trajectories)


def generate_pair_dataset(
    pair_specs: tuple[ExecutablePairSpec, ...],
    seed: int,
    trajectories_per_case: int,
) -> tuple[ExecutableTrajectory, ...]:
    return _generate_pair_dataset_for_scenarios(
        pair_specs,
        seed,
        trajectories_per_case,
        ("easy", "irregular", "endpoint_match", "short", "short_noisy", "outlier"),
    )


def generate_boundary_pair_dataset(
    pair_specs: tuple[ExecutablePairSpec, ...],
    seed: int,
    trajectories_per_case: int,
) -> tuple[ExecutableTrajectory, ...]:
    return _generate_pair_dataset_for_scenarios(
        pair_specs,
        seed,
        trajectories_per_case,
        ("endpoint_match", "short", "short_noisy", "outlier"),
    )
