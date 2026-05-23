from __future__ import annotations

from dataclasses import dataclass, asdict
from math import pi, sin, cos
import csv
import io
import json
import os
from pathlib import Path
import random

from .contracts import TrajectoryArtifact, validate_trajectory_artifact


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


@dataclass(frozen=True, slots=True)
class TrajectoryClassDefinition:
    name: str
    kind: str
    description: str
    parameter_ranges: dict[str, tuple[float, float]]
    nominal_steps: tuple[int, int]
    dt_range: tuple[float, float]
    measurement_std_range: tuple[float, float]
    outlier_probability: float
    dropout_probability: float
    irregular_sampling_strength: float


@dataclass(frozen=True, slots=True)
class DatasetTierDefinition:
    name: str
    description: str
    trajectories_per_class: int
    steps_range: tuple[int, int]
    dt_range: tuple[float, float]
    measurement_std_range: tuple[float, float]
    outlier_probability: float
    dropout_probability: float
    irregular_sampling_strength: float
    parameter_mode: str


@dataclass(frozen=True, slots=True)
class GeneratedTrajectoryDataset:
    tier: str
    seed: int
    class_definitions: tuple[TrajectoryClassDefinition, ...]
    tier_definition: DatasetTierDefinition
    trajectories: tuple[TrajectoryArtifact, ...]


@dataclass(frozen=True, slots=True)
class TrajectoryGeneratorArtifacts:
    run_dir: Path
    report_path: Path
    class_definitions_path: Path
    config_path: Path
    dataset_manifest_paths: dict[str, Path]
    generated_trajectories_paths: dict[str, Path]
    true_states_paths: dict[str, Path]
    plot_svg_path: Path
    plot_png_path: Path


def default_trajectory_class_definitions() -> tuple[TrajectoryClassDefinition, ...]:
    return (
        TrajectoryClassDefinition(
            name="stationary",
            kind="stationary",
            description="Position remains nearly constant.",
            parameter_ranges={"position": (-8.0, 8.0)},
            nominal_steps=(20, 28),
            dt_range=(0.35, 0.9),
            measurement_std_range=(0.01, 0.05),
            outlier_probability=0.0,
            dropout_probability=0.0,
            irregular_sampling_strength=0.0,
        ),
        TrajectoryClassDefinition(
            name="constant_velocity",
            kind="constant_velocity",
            description="Straight-line motion at near-constant velocity.",
            parameter_ranges={"position": (-8.0, 8.0), "velocity": (-3.0, 3.0)},
            nominal_steps=(20, 30),
            dt_range=(0.30, 0.85),
            measurement_std_range=(0.02, 0.06),
            outlier_probability=0.0,
            dropout_probability=0.0,
            irregular_sampling_strength=0.10,
        ),
        TrajectoryClassDefinition(
            name="constant_acceleration",
            kind="constant_acceleration",
            description="Quadratic motion with fixed acceleration.",
            parameter_ranges={"position": (-8.0, 8.0), "velocity": (-2.5, 2.5), "acceleration": (-1.4, 1.4)},
            nominal_steps=(20, 32),
            dt_range=(0.25, 0.8),
            measurement_std_range=(0.02, 0.08),
            outlier_probability=0.0,
            dropout_probability=0.0,
            irregular_sampling_strength=0.12,
        ),
        TrajectoryClassDefinition(
            name="braking",
            kind="braking",
            description="Positive motion slows to a stop under negative acceleration.",
            parameter_ranges={"position": (-8.0, 8.0), "velocity": (0.8, 4.0), "deceleration": (-2.4, -0.2)},
            nominal_steps=(18, 28),
            dt_range=(0.25, 0.75),
            measurement_std_range=(0.02, 0.06),
            outlier_probability=0.0,
            dropout_probability=0.0,
            irregular_sampling_strength=0.10,
        ),
        TrajectoryClassDefinition(
            name="maneuver",
            kind="maneuver",
            description="Acceleration changes sign during the track.",
            parameter_ranges={
                "position": (-8.0, 8.0),
                "velocity": (-2.0, 2.0),
                "accel_early": (-1.5, 1.5),
                "accel_late": (-1.5, 1.5),
                "switch_fraction": (0.25, 0.75),
            },
            nominal_steps=(22, 34),
            dt_range=(0.25, 0.75),
            measurement_std_range=(0.03, 0.08),
            outlier_probability=0.0,
            dropout_probability=0.0,
            irregular_sampling_strength=0.15,
        ),
        TrajectoryClassDefinition(
            name="oscillatory",
            kind="oscillatory",
            description="Sinusoidal motion with repeated direction changes.",
            parameter_ranges={
                "position": (-8.0, 8.0),
                "amplitude": (0.5, 4.5),
                "frequency": (0.15, 1.2),
                "phase": (0.0, 2.0 * pi),
            },
            nominal_steps=(24, 36),
            dt_range=(0.20, 0.70),
            measurement_std_range=(0.03, 0.10),
            outlier_probability=0.0,
            dropout_probability=0.0,
            irregular_sampling_strength=0.20,
        ),
        TrajectoryClassDefinition(
            name="bounded_acceleration",
            kind="bounded_acceleration",
            description="Acceleration is driven by a clipped oscillatory control law.",
            parameter_ranges={
                "position": (-8.0, 8.0),
                "velocity": (-2.5, 2.5),
                "accel_bias": (-0.5, 0.5),
                "accel_limit": (0.5, 1.6),
                "amplitude": (0.2, 1.8),
                "frequency": (0.2, 1.0),
                "phase": (0.0, 2.0 * pi),
            },
            nominal_steps=(22, 34),
            dt_range=(0.20, 0.75),
            measurement_std_range=(0.03, 0.10),
            outlier_probability=0.0,
            dropout_probability=0.0,
            irregular_sampling_strength=0.20,
        ),
    )


def default_dataset_tiers() -> tuple[DatasetTierDefinition, ...]:
    return (
        DatasetTierDefinition(
            name="easy_v1",
            description="Separated trajectories with low noise and regular timing.",
            trajectories_per_class=3,
            steps_range=(24, 30),
            dt_range=(0.35, 0.75),
            measurement_std_range=(0.01, 0.04),
            outlier_probability=0.0,
            dropout_probability=0.0,
            irregular_sampling_strength=0.0,
            parameter_mode="center",
        ),
        DatasetTierDefinition(
            name="boundary_v1",
            description="Trajectories placed near class boundaries with moderate noise.",
            trajectories_per_class=3,
            steps_range=(20, 28),
            dt_range=(0.25, 0.75),
            measurement_std_range=(0.03, 0.08),
            outlier_probability=0.01,
            dropout_probability=0.0,
            irregular_sampling_strength=0.12,
            parameter_mode="boundary",
        ),
        DatasetTierDefinition(
            name="adversarial_v1",
            description="Shorter, noisier tracks chosen to stress confusable class pairs.",
            trajectories_per_class=3,
            steps_range=(12, 22),
            dt_range=(0.18, 0.65),
            measurement_std_range=(0.05, 0.14),
            outlier_probability=0.04,
            dropout_probability=0.0,
            irregular_sampling_strength=0.25,
            parameter_mode="adversarial",
        ),
        DatasetTierDefinition(
            name="stress_v1",
            description="Very short and noisy tracks with large timing irregularity.",
            trajectories_per_class=3,
            steps_range=(6, 14),
            dt_range=(0.12, 0.90),
            measurement_std_range=(0.08, 0.22),
            outlier_probability=0.08,
            dropout_probability=0.0,
            irregular_sampling_strength=0.45,
            parameter_mode="stress",
        ),
        DatasetTierDefinition(
            name="realistic_v1",
            description="Mixed-difficulty tracks with moderate noise and timing irregularity.",
            trajectories_per_class=3,
            steps_range=(18, 32),
            dt_range=(0.20, 0.85),
            measurement_std_range=(0.02, 0.10),
            outlier_probability=0.02,
            dropout_probability=0.0,
            irregular_sampling_strength=0.18,
            parameter_mode="realistic",
        ),
    )


def _tier_by_name(name: str) -> DatasetTierDefinition:
    for tier in default_dataset_tiers():
        if tier.name == name:
            return tier
    raise KeyError(f"unknown dataset tier: {name}")


def _class_by_name(name: str) -> TrajectoryClassDefinition:
    for class_definition in default_trajectory_class_definitions():
        if class_definition.name == name:
            return class_definition
    raise KeyError(f"unknown trajectory class: {name}")


def _sample_from_range(rng: random.Random, bounds: tuple[float, float], mode: str) -> float:
    lower, upper = bounds
    if lower == upper:
        return lower
    if mode == "center":
        center = 0.5 * (lower + upper)
        span = 0.25 * (upper - lower)
        return _clamp(rng.uniform(center - span, center + span), lower, upper)
    if mode == "boundary":
        edge = lower if rng.random() < 0.5 else upper
        span = 0.12 * (upper - lower)
        if edge == lower:
            return _clamp(rng.uniform(lower, lower + span), lower, upper)
        return _clamp(rng.uniform(upper - span, upper), lower, upper)
    if mode == "adversarial":
        if rng.random() < 0.5:
            return _clamp(rng.uniform(lower, lower + 0.18 * (upper - lower)), lower, upper)
        return _clamp(rng.uniform(upper - 0.18 * (upper - lower), upper), lower, upper)
    if mode == "stress":
        return rng.uniform(lower, upper)
    return rng.uniform(lower, upper)


def _sample_steps_and_dt(
    rng: random.Random,
    class_definition: TrajectoryClassDefinition,
    tier_definition: DatasetTierDefinition,
) -> tuple[int, float, float]:
    min_steps = max(class_definition.nominal_steps[0], tier_definition.steps_range[0])
    max_steps = min(class_definition.nominal_steps[1], tier_definition.steps_range[1])
    if min_steps > max_steps:
        min_steps, max_steps = tier_definition.steps_range
    steps = rng.randint(min_steps, max_steps)
    dt = _sample_from_range(rng, tier_definition.dt_range, tier_definition.parameter_mode)
    measurement_std = _sample_from_range(rng, tier_definition.measurement_std_range, tier_definition.parameter_mode)
    return steps, dt, measurement_std


def _sample_parameters(
    rng: random.Random,
    class_definition: TrajectoryClassDefinition,
    tier_definition: DatasetTierDefinition,
) -> dict[str, float]:
    params: dict[str, float] = {}
    mode = tier_definition.parameter_mode
    for name, bounds in class_definition.parameter_ranges.items():
        params[name] = _sample_from_range(rng, bounds, mode)
    if class_definition.kind == "braking":
        params["deceleration"] = -abs(params["deceleration"])
    if class_definition.kind == "maneuver":
        params["switch_fraction"] = _clamp(params["switch_fraction"], 0.2, 0.8)
    if class_definition.kind == "bounded_acceleration":
        params["accel_limit"] = max(0.2, params["accel_limit"])
    if tier_definition.parameter_mode == "adversarial":
        if class_definition.kind in {"constant_velocity", "constant_acceleration"}:
            params["velocity"] = _clamp(params.get("velocity", 0.0), -0.55, 0.55)
        if class_definition.kind == "braking":
            params["velocity"] = max(0.55, abs(params["velocity"]))
            params["deceleration"] = -_clamp(abs(params["deceleration"]), 0.15, 0.65)
        if class_definition.kind == "oscillatory":
            params["frequency"] = _clamp(params["frequency"], 0.12, 0.35)
            params["amplitude"] = _clamp(params["amplitude"], 0.4, 1.2)
        if class_definition.kind == "bounded_acceleration":
            params["accel_bias"] = _clamp(params["accel_bias"], -0.15, 0.15)
            params["accel_limit"] = _clamp(params["accel_limit"], 0.35, 0.9)
    if tier_definition.parameter_mode == "stress":
        if class_definition.kind in {"stationary", "constant_velocity"}:
            params["velocity"] = _clamp(params.get("velocity", 0.0), -0.35, 0.35)
        if class_definition.kind == "constant_acceleration":
            params["acceleration"] = _clamp(params.get("acceleration", 0.0), -0.35, 0.35)
        if class_definition.kind == "maneuver":
            params["switch_fraction"] = _clamp(params["switch_fraction"], 0.35, 0.65)
    return params


def _generate_times(rng: random.Random, steps: int, dt: float, irregular_sampling_strength: float) -> tuple[float, ...]:
    times = [0.0]
    current = 0.0
    for _ in range(1, steps):
        jitter = 1.0 + irregular_sampling_strength * rng.uniform(-1.0, 1.0)
        current += max(0.03, dt * jitter)
        times.append(current)
    return tuple(times)


def _evaluate_stationary(times: tuple[float, ...], params: dict[str, float]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    p0 = params["position"]
    positions = tuple(p0 for _ in times)
    velocities = tuple(0.0 for _ in times)
    accelerations = tuple(0.0 for _ in times)
    return positions, velocities, accelerations


def _evaluate_constant_velocity(times: tuple[float, ...], params: dict[str, float]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    p0 = params["position"]
    v0 = params["velocity"]
    positions = tuple(p0 + v0 * time for time in times)
    velocities = tuple(v0 for _ in times)
    accelerations = tuple(0.0 for _ in times)
    return positions, velocities, accelerations


def _evaluate_constant_acceleration(times: tuple[float, ...], params: dict[str, float]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    p0 = params["position"]
    v0 = params["velocity"]
    a0 = params["acceleration"]
    positions = tuple(p0 + v0 * time + 0.5 * a0 * time * time for time in times)
    velocities = tuple(v0 + a0 * time for time in times)
    accelerations = tuple(a0 for _ in times)
    return positions, velocities, accelerations


def _evaluate_braking(times: tuple[float, ...], params: dict[str, float]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    p0 = params["position"]
    v0 = params["velocity"]
    decel = params["deceleration"]
    stop_time = max(0.0, -v0 / decel) if decel < 0.0 else 0.0
    stop_position = p0 + v0 * stop_time + 0.5 * decel * stop_time * stop_time
    positions: list[float] = []
    velocities: list[float] = []
    accelerations: list[float] = []
    for time in times:
        if time <= stop_time:
            positions.append(p0 + v0 * time + 0.5 * decel * time * time)
            velocities.append(v0 + decel * time)
            accelerations.append(decel)
        else:
            positions.append(stop_position)
            velocities.append(0.0)
            accelerations.append(0.0)
    return tuple(positions), tuple(velocities), tuple(accelerations)


def _evaluate_maneuver(times: tuple[float, ...], params: dict[str, float]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    p0 = params["position"]
    v0 = params["velocity"]
    a1 = params["accel_early"]
    a2 = params["accel_late"]
    switch_time = params["switch_fraction"] * times[-1] if times else 0.0
    switch_position = p0 + v0 * switch_time + 0.5 * a1 * switch_time * switch_time
    switch_velocity = v0 + a1 * switch_time
    positions: list[float] = []
    velocities: list[float] = []
    accelerations: list[float] = []
    for time in times:
        if time <= switch_time:
            positions.append(p0 + v0 * time + 0.5 * a1 * time * time)
            velocities.append(v0 + a1 * time)
            accelerations.append(a1)
        else:
            delta = time - switch_time
            positions.append(switch_position + switch_velocity * delta + 0.5 * a2 * delta * delta)
            velocities.append(switch_velocity + a2 * delta)
            accelerations.append(a2)
    return tuple(positions), tuple(velocities), tuple(accelerations)


def _evaluate_oscillatory(times: tuple[float, ...], params: dict[str, float]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    p0 = params["position"]
    amplitude = params["amplitude"]
    frequency = params["frequency"]
    phase = params["phase"]
    positions = tuple(p0 + amplitude * sin(frequency * time + phase) for time in times)
    velocities = tuple(amplitude * frequency * cos(frequency * time + phase) for time in times)
    accelerations = tuple(-amplitude * frequency * frequency * sin(frequency * time + phase) for time in times)
    return positions, velocities, accelerations


def _evaluate_bounded_acceleration(times: tuple[float, ...], params: dict[str, float]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    p0 = params["position"]
    v0 = params["velocity"]
    accel_bias = params["accel_bias"]
    accel_limit = params["accel_limit"]
    amplitude = params["amplitude"]
    frequency = params["frequency"]
    phase = params["phase"]
    positions = [p0]
    velocities = [v0]
    accelerations = [0.0]
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        current_time = times[index - 1]
        accel = accel_bias + amplitude * sin(frequency * current_time + phase)
        accel = _clamp(accel, -accel_limit, accel_limit)
        next_velocity = velocities[-1] + accel * dt
        next_position = positions[-1] + velocities[-1] * dt + 0.5 * accel * dt * dt
        positions.append(next_position)
        velocities.append(next_velocity)
        accelerations.append(accel)
    return tuple(positions), tuple(velocities), tuple(accelerations)


def _generate_states(
    class_definition: TrajectoryClassDefinition,
    times: tuple[float, ...],
    params: dict[str, float],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if class_definition.kind == "stationary":
        return _evaluate_stationary(times, params)
    if class_definition.kind == "constant_velocity":
        return _evaluate_constant_velocity(times, params)
    if class_definition.kind == "constant_acceleration":
        return _evaluate_constant_acceleration(times, params)
    if class_definition.kind == "braking":
        return _evaluate_braking(times, params)
    if class_definition.kind == "maneuver":
        return _evaluate_maneuver(times, params)
    if class_definition.kind == "oscillatory":
        return _evaluate_oscillatory(times, params)
    if class_definition.kind == "bounded_acceleration":
        return _evaluate_bounded_acceleration(times, params)
    raise ValueError(f"unsupported trajectory kind: {class_definition.kind}")


def _inject_measurement_noise(
    rng: random.Random,
    positions_true: tuple[float, ...],
    measurement_std: float,
    outlier_probability: float,
) -> tuple[tuple[float, ...], list[int]]:
    measurements: list[float] = []
    outlier_indices: list[int] = []
    for index, position in enumerate(positions_true):
        measurement = position + rng.gauss(0.0, measurement_std)
        if rng.random() < outlier_probability:
            measurement += rng.choice((-1.0, 1.0)) * rng.uniform(4.0, 7.5) * measurement_std
            outlier_indices.append(index)
        measurements.append(measurement)
    return tuple(measurements), outlier_indices


def _make_trajectory(
    *,
    class_definition: TrajectoryClassDefinition,
    tier_definition: DatasetTierDefinition,
    steps: int,
    dt: float,
    measurement_std: float,
    seed: int,
    scenario_id: str,
    params: dict[str, float],
    times: tuple[float, ...],
    positions_true: tuple[float, ...],
    velocities_true: tuple[float, ...],
    accelerations_true: tuple[float, ...],
    measurements: tuple[float, ...],
    outlier_indices: list[int],
) -> TrajectoryArtifact:
    generator_parameters = {
        "tier": tier_definition.name,
        "tier_mode": tier_definition.parameter_mode,
        "class_kind": class_definition.kind,
        "steps": steps,
        "dt": dt,
        "measurement_std": measurement_std,
        "outlier_indices": outlier_indices,
        **params,
    }
    return TrajectoryArtifact(
        trajectory_id=scenario_id,
        true_class=class_definition.name,
        scenario_id=scenario_id,
        seed=seed,
        times=times,
        measurements=measurements,
        measurement_std=measurement_std,
        true_position=positions_true,
        true_velocity=velocities_true,
        true_acceleration=accelerations_true,
        generator_parameters=generator_parameters,
    )


def generate_trajectory_dataset(
    tier_name: str,
    *,
    seed: int = 7,
    trajectories_per_class: int | None = None,
) -> GeneratedTrajectoryDataset:
    tier_definition = _tier_by_name(tier_name)
    class_definitions = default_trajectory_class_definitions()
    rng = random.Random(seed)
    trajectories: list[TrajectoryArtifact] = []
    for class_index, class_definition in enumerate(class_definitions):
        count = trajectories_per_class if trajectories_per_class is not None else tier_definition.trajectories_per_class
        for trajectory_index in range(count):
            trajectory_seed = rng.randrange(1 << 30) + class_index * 10_000 + trajectory_index
            local_rng = random.Random(trajectory_seed)
            steps, dt, measurement_std = _sample_steps_and_dt(local_rng, class_definition, tier_definition)
            params = _sample_parameters(local_rng, class_definition, tier_definition)
            times = _generate_times(
                local_rng,
                steps,
                dt,
                tier_definition.irregular_sampling_strength + class_definition.irregular_sampling_strength,
            )
            positions_true, velocities_true, accelerations_true = _generate_states(class_definition, times, params)
            measurements, outlier_indices = _inject_measurement_noise(
                local_rng,
                positions_true,
                measurement_std,
                tier_definition.outlier_probability + class_definition.outlier_probability,
            )
            scenario_id = f"{tier_definition.name}_{class_definition.name}_{trajectory_index}"
            trajectory = _make_trajectory(
                class_definition=class_definition,
                tier_definition=tier_definition,
                steps=steps,
                dt=dt,
                measurement_std=measurement_std,
                seed=trajectory_seed,
                scenario_id=scenario_id,
                params=params,
                times=times,
                positions_true=positions_true,
                velocities_true=velocities_true,
                accelerations_true=accelerations_true,
                measurements=measurements,
                outlier_indices=outlier_indices,
            )
            trajectories.append(trajectory)
    return GeneratedTrajectoryDataset(
        tier=tier_definition.name,
        seed=seed,
        class_definitions=class_definitions,
        tier_definition=tier_definition,
        trajectories=tuple(trajectories),
    )


def generate_trajectory_datasets(
    *,
    seed: int = 7,
    trajectories_per_class: int | None = None,
) -> tuple[GeneratedTrajectoryDataset, ...]:
    return tuple(
        generate_trajectory_dataset(
            tier_definition.name,
            seed=seed + index * 101,
            trajectories_per_class=trajectories_per_class,
        )
        for index, tier_definition in enumerate(default_dataset_tiers())
    )


def _trajectory_rows(dataset: GeneratedTrajectoryDataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trajectory in dataset.trajectories:
        for index, time in enumerate(trajectory.times):
            rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "tier": dataset.tier,
                    "scenario_id": trajectory.scenario_id,
                    "true_class": trajectory.true_class,
                    "seed": trajectory.seed,
                    "step": index,
                    "time": time,
                    "measurement": trajectory.measurements[index],
                    "true_position": trajectory.true_position[index] if trajectory.true_position else "",
                    "true_velocity": trajectory.true_velocity[index] if trajectory.true_velocity else "",
                    "true_acceleration": trajectory.true_acceleration[index] if trajectory.true_acceleration else "",
                }
            )
    return rows


def _true_state_rows(dataset: GeneratedTrajectoryDataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trajectory in dataset.trajectories:
        for index, time in enumerate(trajectory.times):
            rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "tier": dataset.tier,
                    "scenario_id": trajectory.scenario_id,
                    "true_class": trajectory.true_class,
                    "seed": trajectory.seed,
                    "step": index,
                    "time": time,
                    "true_position": trajectory.true_position[index] if trajectory.true_position else "",
                    "true_velocity": trajectory.true_velocity[index] if trajectory.true_velocity else "",
                    "true_acceleration": trajectory.true_acceleration[index] if trajectory.true_acceleration else "",
                }
            )
    return rows


def _dataset_manifest(dataset: GeneratedTrajectoryDataset) -> dict[str, object]:
    class_counts: dict[str, int] = {}
    scenario_counts: dict[str, int] = {}
    step_counts: list[int] = []
    dt_values: list[float] = []
    measurement_values: list[float] = []
    for trajectory in dataset.trajectories:
        class_counts[trajectory.true_class] = class_counts.get(trajectory.true_class, 0) + 1
        scenario_counts[trajectory.scenario_id] = scenario_counts.get(trajectory.scenario_id, 0) + 1
        step_counts.append(len(trajectory.times))
        if len(trajectory.times) >= 2:
            dt_values.extend(
                trajectory.times[index] - trajectory.times[index - 1]
                for index in range(1, len(trajectory.times))
            )
        if trajectory.measurement_std is not None:
            measurement_values.append(trajectory.measurement_std)
    return {
        "tier": dataset.tier,
        "seed": dataset.seed,
        "generator_version": "trajectory_generator_v1",
        "trajectory_count": len(dataset.trajectories),
        "class_counts": class_counts,
        "scenario_counts": scenario_counts,
        "steps": {
            "min": min(step_counts) if step_counts else 0,
            "max": max(step_counts) if step_counts else 0,
            "mean": _mean(step_counts) if step_counts else 0.0,
        },
        "dt": {
            "min": min(dt_values) if dt_values else 0.0,
            "max": max(dt_values) if dt_values else 0.0,
            "mean": _mean(dt_values) if dt_values else 0.0,
        },
        "measurement_std": {
            "min": min(measurement_values) if measurement_values else 0.0,
            "max": max(measurement_values) if measurement_values else 0.0,
            "mean": _mean(measurement_values) if measurement_values else 0.0,
        },
        "notes": [
            "Trajectories are one-dimensional and deterministic under seed control.",
            "Outlier and timing irregularity are included for adversarial and stress tiers.",
            "Measurement dropouts are deferred until the contracts support explicit missing-data masks.",
        ],
    }


def _render_dataset_plot(dataset: GeneratedTrajectoryDataset):
    plt = _prepare_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    class_names = [class_definition.name for class_definition in dataset.class_definitions]
    colors = {
        name: color
        for name, color in zip(class_names, ("#2563eb", "#16a34a", "#7c3aed", "#d97706", "#db2777", "#0f766e", "#dc2626"))
    }
    for class_name in class_names:
        class_trajectories = [trajectory for trajectory in dataset.trajectories if trajectory.true_class == class_name][:2]
        for trajectory in class_trajectories:
            axes[0].plot(trajectory.times, trajectory.measurements, color=colors[class_name], alpha=0.7, linewidth=1.5)
            axes[1].plot(trajectory.times, trajectory.true_position, color=colors[class_name], alpha=0.7, linewidth=1.5)
    axes[0].set_title(f"{dataset.tier} measurements", loc="left", fontweight="bold")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("measurement")
    axes[0].grid(True, alpha=0.2)
    axes[1].set_title(f"{dataset.tier} true position", loc="left", fontweight="bold")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("position")
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def render_trajectory_generator_report(datasets: tuple[GeneratedTrajectoryDataset, ...]) -> str:
    lines = [
        "# Trajectory Generator v1",
        "",
        "This generator defines explicit 1D class models and produces tiered synthetic datasets for the roadmap baseline.",
        "",
        "## Class Definitions",
        "",
    ]
    for class_definition in default_trajectory_class_definitions():
        lines.extend(
            [
                f"### {class_definition.name}",
                "",
                f"- kind: `{class_definition.kind}`",
                f"- description: {class_definition.description}",
                f"- nominal steps: {class_definition.nominal_steps[0]}..{class_definition.nominal_steps[1]}",
                f"- dt range: {class_definition.dt_range[0]:.2f}..{class_definition.dt_range[1]:.2f}",
                f"- measurement std range: {class_definition.measurement_std_range[0]:.2f}..{class_definition.measurement_std_range[1]:.2f}",
                "",
            ]
        )
    lines.extend(["## Dataset Tiers", ""])
    for dataset in datasets:
        manifest = _dataset_manifest(dataset)
        lines.extend(
            [
                f"### {dataset.tier}",
                "",
                f"- trajectories: {manifest['trajectory_count']}",
                f"- steps: {manifest['steps']['min']}..{manifest['steps']['max']}",
                f"- dt mean: {manifest['dt']['mean']:.3f}",
                f"- measurement std mean: {manifest['measurement_std']['mean']:.3f}",
                f"- parameter mode: `{dataset.tier_definition.parameter_mode}`",
                "",
                "| class | count |",
                "| --- | ---: |",
            ]
        )
        for class_name, count in sorted(manifest["class_counts"].items()):
            lines.append(f"| {class_name} | {count} |")
        lines.append("")
    lines.extend(
        [
            "## Validation Notes",
            "",
            "- Trajectories are validated with the shared trajectory artifact contract.",
            "- The easy tier is cleanly separated; boundary and adversarial tiers intentionally stress overlap.",
            "- Stress tier emphasizes short sequences, irregular timing, and larger measurement noise.",
        ]
    )
    return "\n".join(lines)


def write_trajectory_generator_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int | None = None,
) -> TrajectoryGeneratorArtifacts:
    datasets = generate_trajectory_datasets(seed=seed, trajectories_per_class=trajectories_per_class)
    output_root = Path(output_dir)
    run_dir = output_root / "trajectory_generator_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    class_definitions_path = run_dir / "class_definitions.json"
    config_path = run_dir / "trajectory_generator_config.yaml"
    report_path = run_dir / "trajectory_generator_report.md"
    plot_svg_path = run_dir / "trajectory_generator_overview.svg"
    plot_png_path = run_dir / "trajectory_generator_overview.png"

    class_definitions_path.write_text(
        json.dumps([asdict(class_definition) for class_definition in default_trajectory_class_definitions()], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: trajectory_generator_v1",
                f"  seed: {seed}",
                f"  trajectories_per_class: {trajectories_per_class if trajectories_per_class is not None else 'default'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report_path.write_text(render_trajectory_generator_report(datasets), encoding="utf-8")
    plot_svg_path.write_text(
        _render_figure_svg(_render_dataset_plot(datasets[0])),
        encoding="utf-8",
    )
    plot_png_path.write_bytes(
        _render_figure_png(_render_dataset_plot(datasets[0]))
    )

    dataset_manifest_paths: dict[str, Path] = {}
    generated_paths: dict[str, Path] = {}
    true_state_paths: dict[str, Path] = {}
    for dataset in datasets:
        manifest_path = run_dir / f"{dataset.tier}_dataset_manifest.json"
        trajectories_path = run_dir / f"{dataset.tier}_generated_trajectories.csv"
        true_states_path = run_dir / f"{dataset.tier}_true_states.csv"
        dataset_manifest_paths[dataset.tier] = manifest_path
        generated_paths[dataset.tier] = trajectories_path
        true_state_paths[dataset.tier] = true_states_path
        manifest_path.write_text(
            json.dumps(_dataset_manifest(dataset), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_csv(
            trajectories_path,
            _trajectory_rows(dataset),
            [
                "trajectory_id",
                "tier",
                "scenario_id",
                "true_class",
                "seed",
                "step",
                "time",
                "measurement",
                "true_position",
                "true_velocity",
                "true_acceleration",
            ],
        )
        _write_csv(
            true_states_path,
            _true_state_rows(dataset),
            [
                "trajectory_id",
                "tier",
                "scenario_id",
                "true_class",
                "seed",
                "step",
                "time",
                "true_position",
                "true_velocity",
                "true_acceleration",
            ],
        )
        for trajectory in dataset.trajectories:
            errors = validate_trajectory_artifact(trajectory)
            if errors:
                raise ValueError(f"invalid generated trajectory {trajectory.trajectory_id}: {errors}")

    return TrajectoryGeneratorArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        class_definitions_path=class_definitions_path,
        config_path=config_path,
        dataset_manifest_paths=dataset_manifest_paths,
        generated_trajectories_paths=generated_paths,
        true_states_paths=true_state_paths,
        plot_svg_path=plot_svg_path,
        plot_png_path=plot_png_path,
    )


def _render_figure_svg(fig) -> str:
    plt = _prepare_matplotlib()
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _render_figure_png(fig) -> bytes:
    plt = _prepare_matplotlib()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)
