from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..trajectory_generator import TrajectoryArtifact


@dataclass(frozen=True, slots=True)
class CorpusGymTarget:
    target_id: str
    target_type: str
    description: str
    class_name: str | None = None
    class_pair: tuple[str, str] | None = None
    feature_constraints: dict[str, dict[str, float]] | None = None
    target_tier: str | None = None
    target_failure_mode: str | None = None
    target_prior_sensitivity: str | None = None
    target_switching_pattern: str | None = None


@dataclass(frozen=True, slots=True)
class CorpusGymAction:
    seed: int
    tier_name: str
    duration_scale: float = 1.0
    measurement_scale: float = 1.0
    irregularity_scale: float = 1.0
    outlier_scale: float = 1.0
    step_scale: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CorpusGymReward:
    class_validity: float
    feature_excitation: float
    coverage_gain: float
    boundary_closeness: float
    classifier_stress: float
    prior_sensitivity: float
    leakage_penalty: float
    physical_invalidity_penalty: float
    total_utility: float


@dataclass(frozen=True, slots=True)
class CorpusGymEpisode:
    target: CorpusGymTarget
    action: CorpusGymAction
    trajectory: TrajectoryArtifact
    diagnostics: dict[str, object]
    reward: CorpusGymReward


@dataclass(frozen=True, slots=True)
class CorpusGymContractResult:
    environment_contract: dict[str, object]
    example_targets: tuple[dict[str, object], ...]
    example_episode: CorpusGymEpisode
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CorpusGymArtifacts:
    run_dir: Path
    environment_contract_path: Path
    example_targets_path: Path
    report_path: Path
    numeric_walkthrough_path: Path
