from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CorpusPolicyTuningArtifacts:
    run_dir: Path
    report_path: Path
    recommended_policy_path: Path
    sweep_results_path: Path
    ablation_results_path: Path
    stability_path: Path
    numeric_walkthrough_path: Path


@dataclass(frozen=True, slots=True)
class PolicyEvaluationRow:
    policy_id: str
    objective_id: str
    selected_candidate_count: int
    candidate_count: int
    selected_set: str
    ranked_ids: str
    mean_total_utility: float
    validity: float
    feature_excitation: float
    boundary_coverage: float
    classifier_stress: float
    provenance_completeness: float
    leakage: float
    triviality: float
    degeneracy: float
    adequacy_score: float
    downstream_proxy: float
    policy_score: float
    weights: dict[str, float]
    policy: dict[str, Any]

    def as_row_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "objective_id": self.objective_id,
            "selected_candidate_count": self.selected_candidate_count,
            "candidate_count": self.candidate_count,
            "selected_set": self.selected_set,
            "ranked_ids": self.ranked_ids,
            "mean_total_utility": self.mean_total_utility,
            "validity": self.validity,
            "feature_excitation": self.feature_excitation,
            "boundary_coverage": self.boundary_coverage,
            "classifier_stress": self.classifier_stress,
            "provenance_completeness": self.provenance_completeness,
            "leakage": self.leakage,
            "triviality": self.triviality,
            "degeneracy": self.degeneracy,
            "adequacy_score": self.adequacy_score,
            "downstream_proxy": self.downstream_proxy,
            "policy_score": self.policy_score,
            **{f"weight_{key}": value for key, value in self.weights.items()},
            "policy": self.policy,
        }
