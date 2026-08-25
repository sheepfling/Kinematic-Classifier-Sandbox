from __future__ import annotations

from pathlib import Path
from typing import Callable

from .adapters import ExecutablePairSpec, ExecutableTrajectory
from .analysis import analyze_common_experiment as _analyze_common_experiment
from .analysis import analyze_common_trajectory_corpus as _analyze_common_trajectory_corpus
from .contracts import CommonExperimentResult
from .persistence import write_common_experiment_artifacts as _write_common_experiment_artifacts
from .reporting import render_common_experiment_report as _render_common_experiment_report


def analyze_common_experiment(
    *,
    config_path: str | Path | None = None,
    seed: int = 7,
    trajectories_per_case: int = 8,
) -> CommonExperimentResult:
    return _analyze_common_experiment(
        config_path=config_path,
        seed=seed,
        trajectories_per_case=trajectories_per_case,
    )


def analyze_common_trajectory_corpus(
    *,
    pair_specs: tuple[ExecutablePairSpec, ...],
    trajectories: tuple[ExecutableTrajectory, ...],
    config_path: str | Path | None = None,
    seed: int = 7,
    trajectories_per_case: int | None = None,
    include_comparison: bool = True,
    reference_builder: Callable[..., ExecutableTrajectory] | None = None,
    measurement_sigma: Callable[[str], float] | None = None,
) -> CommonExperimentResult:
    return _analyze_common_trajectory_corpus(
        pair_specs=pair_specs,
        trajectories=trajectories,
        config_path=config_path,
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        include_comparison=include_comparison,
        reference_builder=reference_builder,
        measurement_sigma=measurement_sigma,
    )


def render_common_experiment_report(result: CommonExperimentResult) -> str:
    return _render_common_experiment_report(result)


def write_common_experiment_artifacts(
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    seed: int | None = None,
    trajectories_per_case: int = 8,
    result: CommonExperimentResult | None = None,
):
    analysis = result or _analyze_common_experiment(
        config_path=config_path,
        seed=7 if seed is None else seed,
        trajectories_per_case=trajectories_per_case,
    )
    return _write_common_experiment_artifacts(output_dir, analysis=analysis)
