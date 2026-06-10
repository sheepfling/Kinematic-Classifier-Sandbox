from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..gym_types import CorpusGymAction, CorpusGymTarget


@dataclass(frozen=True, slots=True)
class TrajectoryExplorationObjective:
    objective_id: str
    mode: str
    geometry_target: str
    description: str
    target: CorpusGymTarget
    reward_weights: dict[str, float]
    thresholds: dict[str, float]
    evaluation_budget: int
    backend_constraints: dict[str, object] = field(default_factory=dict)
    classifier_family: str | None = None


@dataclass(frozen=True, slots=True)
class TrajectoryExplorationProposal:
    proposal_id: str
    backend_id: str
    iteration: int
    candidate_index: int
    action: CorpusGymAction
    parent_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrajectoryExplorationEvaluation:
    proposal_id: str
    backend_id: str
    objective_id: str
    iteration: int
    candidate_index: int
    target_id: str
    trajectory_id: str
    true_class: str
    total_utility: float
    class_validity: float
    feature_excitation: float
    coverage_gain: float
    boundary_closeness: float
    classifier_stress: float
    prior_sensitivity: float
    leakage_penalty: float
    physical_invalidity_penalty: float
    feature_cell_coverage_gain: float
    class_pair_overlap_reduction: float
    pairwise_auc_gain: float
    pca_margin_gain: float
    confusion_witness_score: float
    feature_dependency_stress: float
    prior_flip_witness_score: float
    geometry_score: float
    selected: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "backend_id": self.backend_id,
            "objective_id": self.objective_id,
            "iteration": self.iteration,
            "candidate_index": self.candidate_index,
            "target_id": self.target_id,
            "trajectory_id": self.trajectory_id,
            "true_class": self.true_class,
            "total_utility": self.total_utility,
            "class_validity": self.class_validity,
            "feature_excitation": self.feature_excitation,
            "coverage_gain": self.coverage_gain,
            "boundary_closeness": self.boundary_closeness,
            "classifier_stress": self.classifier_stress,
            "prior_sensitivity": self.prior_sensitivity,
            "leakage_penalty": self.leakage_penalty,
            "physical_invalidity_penalty": self.physical_invalidity_penalty,
            "feature_cell_coverage_gain": self.feature_cell_coverage_gain,
            "class_pair_overlap_reduction": self.class_pair_overlap_reduction,
            "pairwise_auc_gain": self.pairwise_auc_gain,
            "pca_margin_gain": self.pca_margin_gain,
            "confusion_witness_score": self.confusion_witness_score,
            "feature_dependency_stress": self.feature_dependency_stress,
            "prior_flip_witness_score": self.prior_flip_witness_score,
            "geometry_score": self.geometry_score,
            "selected": self.selected,
            **self.diagnostics,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryExplorationResult:
    backend_id: str
    objective: TrajectoryExplorationObjective
    candidate_rows: tuple[dict[str, object], ...]
    selected_rows: tuple[dict[str, object], ...]
    coverage_rows: tuple[dict[str, object], ...]
    frontier_rows: tuple[dict[str, object], ...]
    objective_summary: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class TrajectoryExplorationBenchmarkResult:
    contract_payload: dict[str, object]
    objective_rows: tuple[dict[str, object], ...]
    evaluation_rows: tuple[dict[str, object], ...]
    metrics_rows: tuple[dict[str, object], ...]
    coverage_gain_rows: tuple[dict[str, object], ...]
    excitation_gain_rows: tuple[dict[str, object], ...]
    overlap_reduction_rows: tuple[dict[str, object], ...]
    failure_witness_gain_rows: tuple[dict[str, object], ...]
    budget_efficiency_rows: tuple[dict[str, object], ...]
    backend_status_rows: tuple[dict[str, object], ...]
    rl_vs_blackbox_rows: tuple[dict[str, object], ...]
    comparison_report_markdown: str
    rl_decision_report_markdown: str


@dataclass(frozen=True, slots=True)
class TrajectoryExplorationArtifacts:
    contract_dir: Path
    benchmarks_dir: Path
    rl_dir: Path
    blackbox_dir: Path
    backend_contract_path: Path
    objective_schema_path: Path
    evaluation_schema_path: Path
    comparison_report_path: Path
    metrics_by_backend_path: Path
    coverage_gain_by_backend_path: Path
    excitation_gain_by_backend_path: Path
    overlap_reduction_by_backend_path: Path
    failure_witness_gain_by_backend_path: Path
    budget_efficiency_path: Path
    rl_decision_report_path: Path
    rl_vs_blackbox_path: Path
    optimizer_trace_path: Path
    elite_frontier_path: Path
    objective_progress_path: Path


@dataclass(frozen=True, slots=True)
class TrajectoryExplorationBenchmarkArtifacts:
    run_dir: Path
    summary_path: Path
    report_path: Path


class TrajectoryExplorationBackend(Protocol):
    backend_id: str

    def initialize(self, objective: TrajectoryExplorationObjective, seed: int) -> None: ...

    def propose_batch(self, batch_size: int) -> tuple[TrajectoryExplorationProposal, ...]: ...

    def observe(self, evaluations: tuple[TrajectoryExplorationEvaluation, ...]) -> None: ...

    def state_summary(self) -> dict[str, object]: ...

    def diagnostics(self) -> dict[str, object]: ...
