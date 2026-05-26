from __future__ import annotations

from pathlib import Path

from .contracts import CommonExperimentArtifacts, CommonExperimentResult
from .persistence import write_common_experiment_artifacts as _write_common_experiment_artifacts
from .runner import analyze_common_experiment


def write_common_experiment_artifacts(
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    seed: int | None = None,
    trajectories_per_case: int = 8,
    result: CommonExperimentResult | None = None,
) -> CommonExperimentArtifacts:
    analysis = result or analyze_common_experiment(
        config_path=config_path,
        seed=7 if seed is None else seed,
        trajectories_per_case=trajectories_per_case,
    )
    return _write_common_experiment_artifacts(
        output_dir,
        analysis=analysis,
    )
