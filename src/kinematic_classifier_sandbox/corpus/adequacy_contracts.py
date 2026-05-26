from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CorpusAdequacyThresholds:
    min_feature_moderate_fraction_green: float = 0.45
    min_feature_moderate_fraction_yellow: float = 0.25
    min_feature_strong_count_green: int = 10
    min_feature_strong_count_yellow: int = 5
    min_feature_tier_count_green: int = 3
    min_feature_tier_count_yellow: int = 2
    min_feature_class_count_green: int = 2
    min_feature_class_count_yellow: int = 2
    min_pair_examples_per_required_tier: int = 2
    max_covariate_spread_ratio_green: float = 0.85
    max_covariate_spread_ratio_yellow: float = 1.15
    max_covariate_pairwise_auc_green: float = 0.70
    max_covariate_pairwise_auc_yellow: float = 0.83
    easy_pair_auc_threshold: float = 0.95
    hard_pair_auc_floor: float = 0.65
    hard_pair_auc_ceiling: float = 0.95
    hard_pair_overlap_floor: float = 0.12
    duplicate_distance_threshold: float = 0.05
    physical_acceleration_limit: float = 2.6
    green_q_corpus: float = 0.80
    yellow_q_corpus: float = 0.65
    green_leakage_max: float = 0.20
    yellow_leakage_max: float = 0.35
    green_triviality_max: float = 0.20
    yellow_triviality_max: float = 0.35
    green_validity_min: float = 0.90
    yellow_validity_min: float = 0.75


@dataclass(frozen=True, slots=True)
class CorpusAdequacyScorecard:
    class_balance: float
    tier_balance: float
    covariate_balance: float
    feature_excitation: float
    pair_boundary_coverage: float
    class_validity: float
    leakage_penalty: float
    triviality_penalty: float
    degeneracy_penalty: float
    q_corpus: float


@dataclass(frozen=True, slots=True)
class CorpusAdequacySummary:
    overall_status: str
    overall_pass: bool
    feature_status: str
    class_pair_status: str
    class_balance_status: str
    covariate_status: str
    total_trajectories: int
    total_classes: int
    total_feature_sets: int
    total_manifest_pairs: int
    red_count: int
    yellow_count: int
    recommendation_count: int
    q_corpus: float
    leakage_penalty: float
    triviality_penalty: float
    class_validity_score: float
    degeneracy_penalty: float


@dataclass(frozen=True, slots=True)
class CorpusAdequacyResult:
    feature_analysis: object
    feature_set_rows: tuple[dict[str, object], ...]
    class_pair_rows: tuple[dict[str, object], ...]
    class_balance_rows: tuple[dict[str, object], ...]
    covariate_rows: tuple[dict[str, object], ...]
    validity_rows: tuple[dict[str, object], ...]
    degeneracy_rows: tuple[dict[str, object], ...]
    scorecard_rows: tuple[dict[str, object], ...]
    recommendations: tuple[str, ...]
    summary: CorpusAdequacySummary
    thresholds: CorpusAdequacyThresholds
    scorecard: CorpusAdequacyScorecard


@dataclass(frozen=True, slots=True)
class CorpusAdequacyArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    feature_set_coverage_path: Path
    class_pair_coverage_path: Path
    class_balance_path: Path
    covariate_leakage_path: Path
    scorecard_path: Path
    validity_audit_path: Path
    degeneracy_report_path: Path
    pair_status_heatmap_path: Path
    covariate_leakage_plot_path: Path
