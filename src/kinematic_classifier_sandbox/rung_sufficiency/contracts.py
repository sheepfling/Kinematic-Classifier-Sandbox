from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RungCapabilitySpec:
    rung_id: str
    rank: int
    adds_capability: str
    main_failure_addressed: str
    next_rung_id: str | None
    complexity_cost: float
    required_artifacts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RungSufficiencyThresholds:
    min_corpus_score: float = 0.65
    min_feature_score: float = 0.65
    min_oracle_score: float = 0.85
    min_oracle_gap_for_algorithm_failure: float = 0.08
    max_prior_flip_fraction: float = 0.12
    max_confident_error_rate: float = 0.10
    min_improvement_to_promote: float = 0.05
    max_runtime_cost_ratio: float = 10.0
    max_overlap_for_learnable: float = 0.90
    min_confusability_for_feature_limited: float = 0.80
    min_pairwise_auc_for_learnable: float = 0.70
    min_posterior_margin_for_learnable: float = 0.05


@dataclass(frozen=True, slots=True)
class RungThresholdConfig:
    default_profile: RungSufficiencyThresholds
    rung_profiles: tuple[tuple[str, RungSufficiencyThresholds], ...]

    def profile_for(self, rung_id: str) -> RungSufficiencyThresholds:
        for profile_rung_id, profile in self.rung_profiles:
            if profile_rung_id == rung_id:
                return profile
        return self.default_profile


@dataclass(frozen=True, slots=True)
class RungSufficiencyArtifacts:
    run_dir: Path
    config_path: Path
    threshold_profile_path: Path
    capability_matrix_path: Path
    corpus_precondition_path: Path
    oracle_gap_path: Path
    learnability_surface_path: Path
    posterior_quality_path: Path
    failure_mode_path: Path
    promotion_matrix_path: Path
    report_path: Path
    score_vs_oracle_plot_path: Path
    oracle_gap_plot_path: Path
    failure_mode_heatmap_path: Path
    promotion_decision_plot_path: Path
    posterior_quality_plot_path: Path


@dataclass(frozen=True, slots=True)
class LadderWitnessSuiteArtifacts:
    run_dir: Path
    config_path: Path
    schema_path: Path
    manifest_path: Path
    claim_matrix_path: Path
    index_path: Path
