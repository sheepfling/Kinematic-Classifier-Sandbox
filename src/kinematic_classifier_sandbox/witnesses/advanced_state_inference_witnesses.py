from __future__ import annotations

from dataclasses import dataclass

from ..trajectory_generator import generate_switching_scenarios


@dataclass(frozen=True, slots=True)
class SwitchingWitness:
    trajectory_id: str
    scenario_name: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_position: tuple[float, ...]
    true_velocity: tuple[float, ...]
    true_acceleration: tuple[float, ...]
    true_modes: tuple[str, ...]


def generate_advanced_state_inference_witnesses(*, seed: int = 7, replicas: int = 6) -> tuple[SwitchingWitness, ...]:
    witnesses: list[SwitchingWitness] = []
    for replica in range(replicas):
        for artifact in generate_switching_scenarios(seed=seed + replica * 31):
            params = artifact.generator_parameters
            segment_modes = list(params["segment_modes"])
            switch_time = float(params["switch_time"])
            true_modes = tuple(segment_modes[0] if time < switch_time else segment_modes[1] for time in artifact.times)
            witnesses.append(
                SwitchingWitness(
                    trajectory_id=f"{artifact.trajectory_id}_{replica}",
                    scenario_name=artifact.scenario_id,
                    seed=artifact.seed,
                    times=tuple(float(time) for time in artifact.times),
                    measurements=tuple(float(value) for value in artifact.measurements),
                    true_position=tuple(float(value) for value in artifact.true_position),
                    true_velocity=tuple(float(value) for value in artifact.true_velocity),
                    true_acceleration=tuple(float(value) for value in artifact.true_acceleration),
                    true_modes=true_modes,
                )
            )
    return tuple(witnesses)
