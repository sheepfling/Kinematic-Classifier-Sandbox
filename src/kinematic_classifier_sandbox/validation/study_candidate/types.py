from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any


class _RowMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> Any:
        return asdict(self)[key]

    def __iter__(self) -> Iterator[str]:
        return iter(asdict(self))

    def __len__(self) -> int:
        return len(asdict(self))


@dataclass(frozen=True, slots=True)
class StudyCandidateRow(_RowMapping):
    study_id: str
    hypothesis: str
    corpus_spec: dict[str, object]
    feature_set_spec: dict[str, object]
    class_set_spec: dict[str, object]
    classifier_spec: dict[str, object]
    prior_spec: dict[str, object]
    filter_spec: dict[str, object]
    visualization_spec: dict[str, object]
    expected_failure_modes: list[str]
    decision_policy: dict[str, object]


@dataclass(frozen=True, slots=True)
class StudyCandidateStaticScoreRow(_RowMapping):
    study_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    classifier_family: str
    prior_id: str
    corpus_id: str
    compatible: bool
    feature_class_compatibility_score: float
    expected_separability_score: float
    feature_dependency_risk: float
    cumulative_double_counting_risk: float
    prior_sensitivity_risk: float
    corpus_coverage_score: float
    classifier_assumption_fit: float
    three_d_transferability_score: float
    implementation_readiness_score: float
    static_score: float
    expected_difficulty: str
    policy_id: str


@dataclass(frozen=True, slots=True)
class StudyCandidateMonteCarloScoreRow(_RowMapping):
    study_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    prior_id: str
    accuracy: float | None
    oracle_accuracy: float | None
    oracle_gap: float | None
    uniform_accuracy: float | None
    strong_bias_accuracy: float | None
    prior_flip_fraction: float | None
    monte_carlo_score: float | None
    decision: str
    policy_id: str


@dataclass(frozen=True, slots=True)
class StudyCandidateFeatureEvidenceRow(_RowMapping):
    feature_name: str
    feature_group: str
    history_behavior: str
    evidence_role: str
    double_counting_risk: str
    noise_sensitivity: str
    duration_sensitivity: str
    sample_count_sensitivity: str
    three_d_transfer_status: str
    best_class_pairs: str
    worst_class_pairs: str


@dataclass(frozen=True, slots=True)
class StudyCandidatePriorSensitivityExplanationRow(_RowMapping):
    study_id: str
    class_pair: str
    feature_set: str
    classifier: str
    baseline_prior: str
    flip_fraction: float
    median_log_prior_shift_to_flip: float
    most_prior_sensitive_scenario: str
    interpretation: str
