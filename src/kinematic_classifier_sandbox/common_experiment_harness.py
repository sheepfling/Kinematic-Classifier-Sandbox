from __future__ import annotations

"""Compatibility surface over the canonical ``common_experiment`` package."""

from .common_experiment.adapters import ExecutablePairSpec, ExecutableTrajectory
from .common_experiment.analysis import (
    analyze_common_experiment,
    analyze_common_trajectory_corpus,
)
from .common_experiment.artifact_io import write_common_experiment_artifacts
from .common_experiment.config import (
    BOUNDARY_EXPERIMENT_DIR,
    CLASSIFIER_MANIFEST_PATH,
    CLASS_PAIR_PATH,
    CONFIG_PATH,
    EXPERIMENT_DIR,
    FEATURE_SET_PATH,
    ROOT,
    list_common_studies,
    load_common_experiment_config,
    resolve_common_study_adapter,
)
from .common_experiment.contracts import (
    CommonExperimentArtifacts,
    CommonExperimentConfig,
    CommonExperimentResult,
    CommonExperimentSummary,
    CommonStudyAdapter,
)
from .common_experiment.reporting import render_common_experiment_report

__all__ = [
    "BOUNDARY_EXPERIMENT_DIR",
    "CLASSIFIER_MANIFEST_PATH",
    "CLASS_PAIR_PATH",
    "CONFIG_PATH",
    "EXPERIMENT_DIR",
    "FEATURE_SET_PATH",
    "ROOT",
    "CommonExperimentArtifacts",
    "CommonExperimentConfig",
    "CommonExperimentResult",
    "CommonExperimentSummary",
    "CommonStudyAdapter",
    "ExecutablePairSpec",
    "ExecutableTrajectory",
    "analyze_common_experiment",
    "analyze_common_trajectory_corpus",
    "list_common_studies",
    "load_common_experiment_config",
    "render_common_experiment_report",
    "resolve_common_study_adapter",
    "write_common_experiment_artifacts",
]
