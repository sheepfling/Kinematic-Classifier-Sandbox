from __future__ import annotations

from pathlib import Path

from .analysis import analyze_common_experiment as _analyze_common_experiment
from .analysis import analyze_common_trajectory_corpus as _analyze_common_trajectory_corpus
from .config import CommonExperimentConfig
from .contracts import CommonExperimentResult
from .persistence import write_common_experiment_artifacts as _write_common_experiment_artifacts
from .reporting import render_common_experiment_report as _render_common_experiment_report
from .adapters import ExecutablePairSpec, ExecutableTrajectory


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
) -> CommonExperimentResult:
    return _analyze_common_trajectory_corpus(
        pair_specs=pair_specs,
        trajectories=trajectories,
        config_path=config_path,
        seed=seed,
        trajectories_per_case=trajectories_per_case,
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
    return _write_common_experiment_artifacts(
        output_dir,
        config_path=config_path,
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        result=result,
    )
