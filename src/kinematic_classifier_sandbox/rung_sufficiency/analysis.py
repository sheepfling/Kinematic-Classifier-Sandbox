from __future__ import annotations

import io
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..utils.runtime import repo_root
from typing import Any, Iterable

import yaml
from numpy import zeros

from kinematic_classifier_sandbox.utils.io import _write_text, write_csv

from ..common_experiment.runner import analyze_common_experiment
from ..corpus.adequacy_audit import analyze_corpus_adequacy
from ..inference.advanced_state_inference import analyze_advanced_state_inference
from ..inference.transition_matrix_accumulator import run_transition_benchmark
from ..markdown_builder import MarkdownDocument
from ..utils.math import _entropy, _mean
from ..utils.plotting import plt
from ..validation.validation_ladder import analyze_validation_ladder
from .capability_matrix import (
    canonicalize_rung_id,
    capability_lookup,
    capability_rows,
    next_rung_id,
)
from .contracts import (
    LadderWitnessSuiteArtifacts,
    RungSufficiencyArtifacts,
    RungSufficiencyThresholds,
    RungThresholdConfig,
)

ROOT = repo_root()
DEFAULT_LADDER_WITNESS_SUITE_CONFIG_PATH = ROOT / "experiments" / "ladder_witness_suite" / "ladder_witness_suite.yaml"
LADDER_WITNESS_SUITE_RUN_DIR_NAME = "ladder_witness_suite_v1"


@dataclass(frozen=True, slots=True)
class BinaryPredictionRow:
    class_a: str
    class_b: str
    true_class: str
    predicted_class: str
    confidence: float
    posterior_class_a: float
    posterior_class_b: float


@dataclass(frozen=True, slots=True)
class RungThresholdRow:
    rung_id: str
    min_corpus_score: float
    min_feature_score: float
    min_oracle_score: float
    min_oracle_gap_for_algorithm_failure: float
    max_prior_flip_fraction: float
    max_confident_error_rate: float
    min_improvement_to_promote: float
    max_runtime_cost_ratio: float
    max_overlap_for_learnable: float
    min_confusability_for_feature_limited: float
    min_pairwise_auc_for_learnable: float
    min_posterior_margin_for_learnable: float


@dataclass(frozen=True, slots=True)
class SwitchingCorpusRow:
    study_id: str
    corpus_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    corpus_status: str
    class_validity_status: str
    feature_excitation_status: str
    leakage_status: str
    boundary_coverage_status: str
    identifiability_status: str
    confusability_score: float
    overlap_estimate: float
    can_evaluate_classifier: bool
    blocking_reason: str


@dataclass(frozen=True, slots=True)
class SwitchingOracleRow:
    study_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    oracle_accuracy: float
    best_oracle_accuracy_for_pair: float
    current_accuracy: float
    oracle_gap: float
    mean_posterior_margin: float
    overlap_estimate: float
    learnability_status: str
    learnable: bool


@dataclass(frozen=True, slots=True)
class SwitchingLearnabilityRow:
    study_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    corpus_status: str
    feature_excitation_status: str
    class_validity_status: str
    oracle_accuracy: float
    best_oracle_accuracy_for_pair: float
    current_accuracy: float
    oracle_gap: float
    mean_posterior_margin: float
    pairwise_auc: float
    overlap_estimate: float
    confusability_score: float
    oracle_status: str
    pairwise_status: str
    posterior_margin_status: str
    overlap_status: str
    learnability_status: str
    learnable: bool
    oracle_threshold: float
    pairwise_auc_threshold: float
    posterior_margin_threshold: float
    overlap_threshold: float
    threshold_rung_id: str


@dataclass(frozen=True, slots=True)
class SwitchingPosteriorRow:
    study_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    current_accuracy: float
    balanced_accuracy: float
    nll: float
    brier: float
    ece: float
    mean_entropy: float
    mean_confidence: float
    confident_error_rate: float
    prior_flip_fraction: float
    posterior_quality_score: float
    posterior_quality_status: str


@dataclass(frozen=True, slots=True)
class SwitchingFailureRow:
    study_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    current_rung_id: str
    candidate_next_rung_id: str
    failure_mode: str
    failure_rationale: str


@dataclass(frozen=True, slots=True)
class SwitchingPromotionRow:
    study_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    current_rung_id: str
    candidate_next_rung_id: str
    current_accuracy: float
    oracle_accuracy: float
    oracle_gap: float
    measured_next_accuracy: float | str
    measured_improvement: float | str
    runtime_cost_ratio: float
    decision: str
    rationale: str


@dataclass(frozen=True, slots=True)
class SpecialCorpusRow:
    study_id: str
    corpus_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    corpus_status: str
    class_validity_status: str
    feature_excitation_status: str
    leakage_status: str
    boundary_coverage_status: str
    identifiability_status: str
    confusability_score: float
    overlap_estimate: float
    can_evaluate_classifier: bool
    blocking_reason: str


@dataclass(frozen=True, slots=True)
class SpecialOracleRow:
    study_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    oracle_accuracy: float
    best_oracle_accuracy_for_pair: float
    current_accuracy: float
    oracle_gap: float
    mean_posterior_margin: float
    overlap_estimate: float
    learnability_status: str
    learnable: bool


@dataclass(frozen=True, slots=True)
class SpecialLearnabilityRow:
    study_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    corpus_status: str
    feature_excitation_status: str
    class_validity_status: str
    oracle_accuracy: float
    best_oracle_accuracy_for_pair: float
    current_accuracy: float
    oracle_gap: float
    mean_posterior_margin: float
    pairwise_auc: float
    overlap_estimate: float
    confusability_score: float
    oracle_status: str
    pairwise_status: str
    posterior_margin_status: str
    overlap_status: str
    learnability_status: str
    learnable: bool
    oracle_threshold: float
    pairwise_auc_threshold: float
    posterior_margin_threshold: float
    overlap_threshold: float
    threshold_rung_id: str


@dataclass(frozen=True, slots=True)
class SpecialPosteriorRow:
    study_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    current_accuracy: float
    balanced_accuracy: float
    nll: float
    brier: float
    ece: float
    mean_entropy: float
    mean_confidence: float
    confident_error_rate: float
    prior_flip_fraction: float
    posterior_quality_score: float
    posterior_quality_status: str


@dataclass(frozen=True, slots=True)
class SpecialFailureRow:
    study_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    current_rung_id: str
    candidate_next_rung_id: str
    failure_mode: str
    failure_rationale: str


@dataclass(frozen=True, slots=True)
class SpecialPromotionRow:
    study_id: str
    class_pair_id: str
    feature_set_id: str
    classifier_id: str
    current_rung_id: str
    candidate_next_rung_id: str
    current_accuracy: float
    oracle_accuracy: float
    oracle_gap: float
    measured_next_accuracy: float
    measured_improvement: float
    runtime_cost_ratio: float
    decision: str
    rationale: str


def _binary_prediction_row(row: dict[str, object]) -> BinaryPredictionRow:
    return BinaryPredictionRow(
        class_a=str(row["class_a"]),
        class_b=str(row["class_b"]),
        true_class=str(row["true_class"]),
        predicted_class=str(row["predicted_class"]),
        confidence=float(row["confidence"]),
        posterior_class_a=float(row["posterior_class_a"]),
        posterior_class_b=float(row["posterior_class_b"]),
    )


def _ece(rows: list[BinaryPredictionRow], bins: int = 10) -> float:
    if not rows:
        return 0.0
    bin_totals = [0 for _ in range(bins)]
    bin_confidence = [0.0 for _ in range(bins)]
    bin_accuracy = [0.0 for _ in range(bins)]
    for row in rows:
        confidence = row.confidence
        predicted = row.predicted_class
        true_class = row.true_class
        bin_index = min(bins - 1, int(confidence * bins))
        bin_totals[bin_index] += 1
        bin_confidence[bin_index] += confidence
        bin_accuracy[bin_index] += 1.0 if predicted == true_class else 0.0
    total = len(rows)
    error = 0.0
    for index in range(bins):
        if not bin_totals[index]:
            continue
        mean_confidence = bin_confidence[index] / bin_totals[index]
        mean_accuracy = bin_accuracy[index] / bin_totals[index]
        error += (bin_totals[index] / total) * abs(mean_confidence - mean_accuracy)
    return error


def _binary_prediction_metrics(rows: list[BinaryPredictionRow]) -> dict[str, float]:
    if not rows:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "nll": 0.0,
            "brier": 0.0,
            "ece": 0.0,
            "mean_entropy": 0.0,
            "confident_error_rate": 0.0,
            "mean_confidence": 0.0,
        }

    hits = 0
    confidence_values = []
    true_prob_values = []
    pred_by_class: dict[str, list[int]] = {}
    total_by_class: dict[str, int] = {}
    confident_error_count = 0
    log_loss = 0.0
    brier = 0.0
    entropy = 0.0
    for row in rows:
        class_a = row.class_a
        true_class = row.true_class
        predicted_class = row.predicted_class
        confidence = row.confidence
        posterior_true = row.posterior_class_a if true_class == class_a else row.posterior_class_b
        posterior_false = row.posterior_class_b if true_class == class_a else row.posterior_class_a
        if predicted_class == true_class:
            hits += 1
        if confidence >= 0.80 and predicted_class != true_class:
            confident_error_count += 1
        confidence_values.append(confidence)
        true_prob_values.append(posterior_true)
        pred_by_class.setdefault(true_class, []).append(1 if predicted_class == true_class else 0)
        total_by_class[true_class] = total_by_class.get(true_class, 0) + 1
        log_loss -= math.log(max(posterior_true, 1.0e-12))
        brier += (1.0 - posterior_true) ** 2 + (0.0 - posterior_false) ** 2
        entropy += _entropy([posterior_true, posterior_false])

    recalls = []
    for class_name, values in pred_by_class.items():
        recalls.append(sum(values) / max(total_by_class.get(class_name, 1), 1))

    return {
        "accuracy": hits / len(rows),
        "balanced_accuracy": _mean(recalls),
        "nll": log_loss / len(rows),
        "brier": brier / len(rows),
        "ece": _ece(rows),
        "mean_entropy": entropy / len(rows),
        "confident_error_rate": confident_error_count / len(rows),
        "mean_confidence": _mean(confidence_values),
        "mean_true_probability": _mean(true_prob_values),
    }


def _status_from_score(score: float, *, pass_threshold: float, partial_threshold: float) -> str:
    if score >= pass_threshold:
        return "pass"
    if score >= partial_threshold:
        return "warn"
    return "fail"


def _canonical_pair_id(class_a: str, class_b: str) -> str:
    left, right = sorted((class_a, class_b))
    return f"{left}_vs_{right}"


def _threshold_profile(
    base: RungSufficiencyThresholds,
    *,
    min_corpus_score: float | None = None,
    min_feature_score: float | None = None,
    min_oracle_score: float | None = None,
    min_oracle_gap_for_algorithm_failure: float | None = None,
    max_prior_flip_fraction: float | None = None,
    max_confident_error_rate: float | None = None,
    min_improvement_to_promote: float | None = None,
    max_runtime_cost_ratio: float | None = None,
    max_overlap_for_learnable: float | None = None,
    min_confusability_for_feature_limited: float | None = None,
    min_pairwise_auc_for_learnable: float | None = None,
    min_posterior_margin_for_learnable: float | None = None,
) -> RungSufficiencyThresholds:
    return replace(
        base,
        min_corpus_score=base.min_corpus_score if min_corpus_score is None else min_corpus_score,
        min_feature_score=base.min_feature_score if min_feature_score is None else min_feature_score,
        min_oracle_score=base.min_oracle_score if min_oracle_score is None else min_oracle_score,
        min_oracle_gap_for_algorithm_failure=base.min_oracle_gap_for_algorithm_failure if min_oracle_gap_for_algorithm_failure is None else min_oracle_gap_for_algorithm_failure,
        max_prior_flip_fraction=base.max_prior_flip_fraction if max_prior_flip_fraction is None else max_prior_flip_fraction,
        max_confident_error_rate=base.max_confident_error_rate if max_confident_error_rate is None else max_confident_error_rate,
        min_improvement_to_promote=base.min_improvement_to_promote if min_improvement_to_promote is None else min_improvement_to_promote,
        max_runtime_cost_ratio=base.max_runtime_cost_ratio if max_runtime_cost_ratio is None else max_runtime_cost_ratio,
        max_overlap_for_learnable=base.max_overlap_for_learnable if max_overlap_for_learnable is None else max_overlap_for_learnable,
        min_confusability_for_feature_limited=base.min_confusability_for_feature_limited if min_confusability_for_feature_limited is None else min_confusability_for_feature_limited,
        min_pairwise_auc_for_learnable=base.min_pairwise_auc_for_learnable if min_pairwise_auc_for_learnable is None else min_pairwise_auc_for_learnable,
        min_posterior_margin_for_learnable=base.min_posterior_margin_for_learnable if min_posterior_margin_for_learnable is None else min_posterior_margin_for_learnable,
    )


def default_rung_threshold_config() -> RungThresholdConfig:
    base = RungSufficiencyThresholds()
    return RungThresholdConfig(
        default_profile=base,
        rung_profiles=(
            ("pointwise", _threshold_profile(base, min_oracle_score=0.82, min_oracle_gap_for_algorithm_failure=0.06, min_improvement_to_promote=0.04, max_prior_flip_fraction=0.14, max_confident_error_rate=0.12, max_overlap_for_learnable=0.92, min_pairwise_auc_for_learnable=0.68, min_posterior_margin_for_learnable=0.03)),
            ("windowed", _threshold_profile(base, min_oracle_score=0.83, min_oracle_gap_for_algorithm_failure=0.06, min_improvement_to_promote=0.05, max_prior_flip_fraction=0.13, max_confident_error_rate=0.11, max_overlap_for_learnable=0.91, min_pairwise_auc_for_learnable=0.69, min_posterior_margin_for_learnable=0.04)),
            ("sequential_bayes", _threshold_profile(base, min_oracle_score=0.84, min_oracle_gap_for_algorithm_failure=0.07, min_improvement_to_promote=0.05, max_prior_flip_fraction=0.12, max_confident_error_rate=0.10, max_overlap_for_learnable=0.90, min_pairwise_auc_for_learnable=0.70, min_posterior_margin_for_learnable=0.05)),
            ("kalman_bank", _threshold_profile(base, min_oracle_score=0.85, min_oracle_gap_for_algorithm_failure=0.08, min_improvement_to_promote=0.05, max_prior_flip_fraction=0.12, max_confident_error_rate=0.10, max_overlap_for_learnable=0.90, min_pairwise_auc_for_learnable=0.72, min_posterior_margin_for_learnable=0.05)),
            ("transition_matrix", _threshold_profile(base, min_oracle_score=0.86, min_oracle_gap_for_algorithm_failure=0.08, min_improvement_to_promote=0.05, max_prior_flip_fraction=0.11, max_confident_error_rate=0.10, max_overlap_for_learnable=0.89, min_pairwise_auc_for_learnable=0.74, min_posterior_margin_for_learnable=0.05)),
            ("imm", _threshold_profile(base, min_oracle_score=0.88, min_oracle_gap_for_algorithm_failure=0.09, min_improvement_to_promote=0.06, max_prior_flip_fraction=0.10, max_confident_error_rate=0.09, max_overlap_for_learnable=0.88, min_pairwise_auc_for_learnable=0.76, min_posterior_margin_for_learnable=0.06)),
            ("particle_filter", _threshold_profile(base, min_oracle_score=0.90, min_oracle_gap_for_algorithm_failure=0.10, min_improvement_to_promote=0.07, max_prior_flip_fraction=0.10, max_confident_error_rate=0.08, max_overlap_for_learnable=0.86, min_pairwise_auc_for_learnable=0.78, min_posterior_margin_for_learnable=0.06)),
            ("rbpf", _threshold_profile(base, min_oracle_score=0.92, min_oracle_gap_for_algorithm_failure=0.10, min_improvement_to_promote=0.07, max_prior_flip_fraction=0.10, max_confident_error_rate=0.08, max_overlap_for_learnable=0.85, min_pairwise_auc_for_learnable=0.80, min_posterior_margin_for_learnable=0.07)),
        ),
    )


def rung_threshold_rows(threshold_config: RungThresholdConfig) -> tuple[dict[str, object], ...]:
    rows: list[RungThresholdRow] = []
    for rung_id, profile in threshold_config.rung_profiles:
        rows.append(
            RungThresholdRow(
                rung_id=rung_id,
                min_corpus_score=profile.min_corpus_score,
                min_feature_score=profile.min_feature_score,
                min_oracle_score=profile.min_oracle_score,
                min_oracle_gap_for_algorithm_failure=profile.min_oracle_gap_for_algorithm_failure,
                max_prior_flip_fraction=profile.max_prior_flip_fraction,
                max_confident_error_rate=profile.max_confident_error_rate,
                min_improvement_to_promote=profile.min_improvement_to_promote,
                max_runtime_cost_ratio=profile.max_runtime_cost_ratio,
                max_overlap_for_learnable=profile.max_overlap_for_learnable,
                min_confusability_for_feature_limited=profile.min_confusability_for_feature_limited,
                min_pairwise_auc_for_learnable=profile.min_pairwise_auc_for_learnable,
                min_posterior_margin_for_learnable=profile.min_posterior_margin_for_learnable,
            )
        )
    return tuple(asdict(row) for row in rows)


def _best_lookup(rows: Iterable[dict[str, object]], *, class_pair_id: str, feature_set_id: str, classifier_family: str) -> dict[str, object] | None:
    selected = [row for row in rows if str(row["class_pair_id"]) == class_pair_id and str(row["feature_set_id"]) == feature_set_id]
    if classifier_family == "windowed":
        selected = [row for row in selected if str(row["classifier_id"]).startswith("windowed_")]
    else:
        selected = [row for row in selected if canonicalize_rung_id(str(row["classifier_id"])) == classifier_family]
    if not selected:
        return None
    return max(selected, key=lambda row: (float(row["classifier_accuracy"]), float(row["oracle_accuracy"])))


@dataclass(frozen=True, slots=True)
class RungSufficiencyResult:
    capability_rows: tuple[dict[str, object], ...]
    corpus_precondition_rows: tuple[dict[str, object], ...]
    oracle_gap_rows: tuple[dict[str, object], ...]
    learnability_surface_rows: tuple[dict[str, object], ...]
    posterior_quality_rows: tuple[dict[str, object], ...]
    failure_mode_rows: tuple[dict[str, object], ...]
    promotion_rows: tuple[dict[str, object], ...]
    summary_rows: tuple[dict[str, object], ...]
    report_markdown: str


def _corpus_precondition_row(
    *,
    validation_row: dict[str, object],
    feature_status: str,
    corpus_summary,
    pair_lookup: dict[str, dict[str, object]],
    ident_lookup: dict[tuple[str, str], dict[str, object]],
    thresholds: RungSufficiencyThresholds,
) -> dict[str, object]:
    class_pair_id = str(validation_row["class_pair_id"])
    feature_set_id = str(validation_row["feature_set_id"])
    pair_row = pair_lookup.get(class_pair_id)
    ident_row = ident_lookup.get((class_pair_id, feature_set_id))
    feature_gate_status = feature_status
    class_validity_status = "pass" if corpus_summary.class_validity_score >= thresholds.min_corpus_score else "warn"
    leakage_status = "pass" if corpus_summary.leakage_penalty <= 0.20 else ("warn" if corpus_summary.leakage_penalty <= 0.70 else "fail")
    boundary_status = str(pair_row["status"]) if pair_row is not None else "defer"
    confusability = float(ident_row["confusability_score"]) if ident_row is not None else 0.0
    overlap = float(ident_row["overlap_estimate"]) if ident_row is not None else 1.0
    ident_status = str(ident_row["identifiability_status"]) if ident_row is not None else "unknown"

    blocking_reason: list[str] = []
    if feature_gate_status == "fail":
        blocking_reason.append("feature separability fails")
    if boundary_status == "fail":
        blocking_reason.append("pair boundary coverage fails")
    if leakage_status == "fail":
        blocking_reason.append("leakage is too high")
    if class_validity_status == "fail":
        blocking_reason.append("class validity is too low")

    can_evaluate = not blocking_reason
    return {
        "study_id": str(validation_row["study_id"]),
        "corpus_id": "selected_generated_corpus_v1",
        "class_pair_id": class_pair_id,
        "feature_set_id": feature_set_id,
        "classifier_id": str(validation_row["classifier_id"]),
        "corpus_status": str(corpus_summary.overall_status),
        "class_validity_status": class_validity_status,
        "feature_excitation_status": feature_gate_status,
        "leakage_status": leakage_status,
        "boundary_coverage_status": boundary_status,
        "identifiability_status": ident_status,
        "confusability_score": confusability,
        "overlap_estimate": overlap,
        "can_evaluate_classifier": can_evaluate,
        "blocking_reason": " | ".join(blocking_reason) if blocking_reason else "",
    }


def _oracle_gap_row(
    *,
    validation_row: dict[str, object],
    common_oracle_lookup: dict[tuple[str, str], dict[str, object]],
    ident_lookup: dict[tuple[str, str], dict[str, object]],
    thresholds: RungSufficiencyThresholds,
) -> dict[str, object]:
    class_pair_id = str(validation_row["class_pair_id"])
    feature_set_id = str(validation_row["feature_set_id"])
    current_accuracy = float(validation_row["classifier_accuracy"])
    oracle_accuracy = float(validation_row["oracle_accuracy"])
    oracle_gap = max(0.0, oracle_accuracy - current_accuracy)
    common_oracle = common_oracle_lookup.get((class_pair_id, feature_set_id))
    ident_row = ident_lookup.get((class_pair_id, feature_set_id))
    best_oracle = float(common_oracle["best_oracle_accuracy_for_pair"]) if common_oracle is not None else oracle_accuracy
    mean_margin = float(common_oracle["mean_posterior_margin"]) if common_oracle is not None else 0.0
    overlap = float(ident_row["overlap_estimate"]) if ident_row is not None else 1.0
    pairwise_auc = float(common_oracle["best_oracle_accuracy_for_pair"]) if common_oracle is not None else oracle_accuracy
    learnable = (
        oracle_accuracy >= thresholds.min_oracle_score
        and overlap <= thresholds.max_overlap_for_learnable
        and pairwise_auc >= thresholds.min_pairwise_auc_for_learnable
        and float(common_oracle["mean_posterior_margin"]) >= thresholds.min_posterior_margin_for_learnable
    )
    if overlap > thresholds.max_overlap_for_learnable or pairwise_auc < thresholds.min_pairwise_auc_for_learnable:
        learnability_status = "feature_limited"
    elif oracle_accuracy < thresholds.min_oracle_score:
        learnability_status = "oracle_limited"
    elif oracle_gap < thresholds.min_oracle_gap_for_algorithm_failure:
        learnability_status = "close_to_limit"
    else:
        learnability_status = "algorithm_limited"
    return {
        "study_id": str(validation_row["study_id"]),
        "class_pair_id": class_pair_id,
        "feature_set_id": feature_set_id,
        "classifier_id": str(validation_row["classifier_id"]),
        "oracle_accuracy": oracle_accuracy,
        "best_oracle_accuracy_for_pair": best_oracle,
        "current_accuracy": current_accuracy,
        "oracle_gap": oracle_gap,
        "mean_posterior_margin": mean_margin,
        "pairwise_auc_proxy": pairwise_auc,
        "overlap_estimate": overlap,
        "learnability_status": learnability_status,
        "learnable": learnable,
    }


def _learnability_surface_row(
    *,
    validation_row: dict[str, object],
    oracle_row: dict[str, object],
    posterior_row: dict[str, object],
    corpus_row: dict[str, object],
    common_oracle_lookup: dict[tuple[str, str], dict[str, object]],
    ident_lookup: dict[tuple[str, str], dict[str, object]],
    thresholds: RungSufficiencyThresholds,
) -> dict[str, object]:
    class_pair_id = str(validation_row["class_pair_id"])
    feature_set_id = str(validation_row["feature_set_id"])
    common_oracle = common_oracle_lookup.get((class_pair_id, feature_set_id))
    ident_row = ident_lookup.get((class_pair_id, feature_set_id))
    pairwise_auc = float(common_oracle["best_oracle_accuracy_for_pair"]) if common_oracle is not None else float(oracle_row["best_oracle_accuracy_for_pair"])
    mean_margin = float(common_oracle["mean_posterior_margin"]) if common_oracle is not None else float(oracle_row["mean_posterior_margin"])
    overlap = float(ident_row["overlap_estimate"]) if ident_row is not None else float(oracle_row["overlap_estimate"])
    confusability = float(ident_row["confusability_score"]) if ident_row is not None else 0.0
    pairwise_status = "pass" if pairwise_auc >= thresholds.min_pairwise_auc_for_learnable else ("warn" if pairwise_auc >= 0.60 else "fail")
    margin_status = "pass" if mean_margin >= thresholds.min_posterior_margin_for_learnable else ("warn" if mean_margin >= 0.02 else "fail")
    overlap_status = "pass" if overlap <= thresholds.max_overlap_for_learnable else ("warn" if overlap <= 0.95 else "fail")
    oracle_status = "pass" if float(oracle_row["oracle_accuracy"]) >= thresholds.min_oracle_score else ("warn" if float(oracle_row["oracle_accuracy"]) >= 0.75 else "fail")
    learnability_status = str(oracle_row["learnability_status"])
    return {
        "study_id": str(validation_row["study_id"]),
        "class_pair_id": class_pair_id,
        "feature_set_id": feature_set_id,
        "classifier_id": str(validation_row["classifier_id"]),
        "corpus_status": str(corpus_row["corpus_status"]),
        "feature_excitation_status": str(corpus_row["feature_excitation_status"]),
        "class_validity_status": str(corpus_row["class_validity_status"]),
        "oracle_accuracy": float(oracle_row["oracle_accuracy"]),
        "best_oracle_accuracy_for_pair": float(oracle_row["best_oracle_accuracy_for_pair"]),
        "current_accuracy": float(oracle_row["current_accuracy"]),
        "oracle_gap": float(oracle_row["oracle_gap"]),
        "mean_posterior_margin": mean_margin,
        "pairwise_auc": pairwise_auc,
        "overlap_estimate": overlap,
        "confusability_score": confusability,
        "oracle_status": oracle_status,
        "pairwise_status": pairwise_status,
        "posterior_margin_status": margin_status,
        "overlap_status": overlap_status,
        "learnability_status": learnability_status,
        "learnable": bool(oracle_row["learnable"]),
        "oracle_threshold": thresholds.min_oracle_score,
        "pairwise_auc_threshold": thresholds.min_pairwise_auc_for_learnable,
        "posterior_margin_threshold": thresholds.min_posterior_margin_for_learnable,
        "overlap_threshold": thresholds.max_overlap_for_learnable,
    }


def _posterior_quality_row(
    *,
    validation_row: dict[str, object],
    prediction_rows: list[BinaryPredictionRow],
    thresholds: RungSufficiencyThresholds,
) -> dict[str, object]:
    metrics = _binary_prediction_metrics(prediction_rows)
    prior_flip_fraction = max(0.0, 1.0 - float(validation_row["prior_sensitivity_score"]))
    quality_score = float(validation_row["posterior_quality_score"])
    status = _status_from_score(
        quality_score,
        pass_threshold=0.80,
        partial_threshold=0.60,
    )
    if metrics["confident_error_rate"] > thresholds.max_confident_error_rate:
        status = "fail"
    if prior_flip_fraction > thresholds.max_prior_flip_fraction:
        status = "fail"
    return {
        "study_id": str(validation_row["study_id"]),
        "class_pair_id": str(validation_row["class_pair_id"]),
        "feature_set_id": str(validation_row["feature_set_id"]),
        "classifier_id": str(validation_row["classifier_id"]),
        "current_accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "nll": metrics["nll"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        "mean_entropy": metrics["mean_entropy"],
        "mean_confidence": metrics["mean_confidence"],
        "confident_error_rate": metrics["confident_error_rate"],
        "prior_flip_fraction": prior_flip_fraction,
        "posterior_quality_score": quality_score,
        "posterior_quality_status": status,
    }


def _failure_mode_row(
    *,
    corpus_row: dict[str, object],
    oracle_row: dict[str, object],
    posterior_row: dict[str, object],
    validation_row: dict[str, object],
    current_family: str,
    candidate_family: str | None,
    measured_improvement: float | None,
    thresholds: RungSufficiencyThresholds,
) -> dict[str, object]:
    if not bool(corpus_row["can_evaluate_classifier"]):
        failure_mode = "corpus_limited"
        rationale = corpus_row["blocking_reason"] or "corpus precondition failed"
    elif corpus_row["feature_excitation_status"] == "fail":
        failure_mode = "feature_limited"
        rationale = "feature excitation is too weak"
    elif posterior_row["prior_flip_fraction"] > thresholds.max_prior_flip_fraction:
        failure_mode = "prior_limited"
        rationale = "prior sensitivity remains too high"
    elif posterior_row["posterior_quality_status"] == "fail" and oracle_row["learnable"]:
        failure_mode = "calibration_limited"
        rationale = "posterior quality is weak even though the task is learnable"
    elif oracle_row["learnability_status"] == "feature_limited":
        failure_mode = "feature_limited"
        rationale = "oracle or overlap evidence suggests feature-level limitation"
    elif measured_improvement is None:
        failure_mode = "insufficient_data"
        rationale = "no measured next-rung comparison is available"
    elif current_family == "pointwise" and measured_improvement > 0.0:
        failure_mode = "pointwise_memory_failure"
        rationale = "a windowed candidate improves on the current instantaneous rung"
    elif current_family == "windowed" and measured_improvement > 0.0:
        failure_mode = "history_accumulation_failure"
        rationale = "recursive posterior memory improves the windowed rung"
    elif current_family == "sequential_bayes" and measured_improvement > 0.0:
        failure_mode = "dynamics_model_failure"
        rationale = "state-space prediction improves the recursive Bayes rung"
    elif current_family == "kalman_bank" and candidate_family == "transition_matrix" and measured_improvement > 0.0:
        failure_mode = "switching_state_failure"
        rationale = "mode persistence and switching prior improve the Kalman bank"
    elif current_family == "transition_matrix" and candidate_family == "imm" and measured_improvement > 0.0:
        failure_mode = "switching_state_failure"
        rationale = "state mixing across modes improves the transition matrix rung"
    else:
        failure_mode = "model_limited"
        rationale = "current rung is useful but still underuses the evidence surface"
    return {
        "study_id": str(validation_row["study_id"]),
        "class_pair_id": str(validation_row["class_pair_id"]),
        "feature_set_id": str(validation_row["feature_set_id"]),
        "classifier_id": str(validation_row["classifier_id"]),
        "current_rung_id": current_family,
        "candidate_next_rung_id": candidate_family or "",
        "failure_mode": failure_mode,
        "failure_rationale": rationale,
    }


def _promotion_row(
    *,
    validation_row: dict[str, object],
    corpus_row: dict[str, object],
    oracle_row: dict[str, object],
    posterior_row: dict[str, object],
    failure_row: dict[str, object],
    next_row: dict[str, object] | None,
    thresholds: RungSufficiencyThresholds,
) -> dict[str, object]:
    candidate_family = str(failure_row["candidate_next_rung_id"]) if failure_row["candidate_next_rung_id"] else ""
    current_family = str(failure_row["current_rung_id"])
    current_accuracy = float(validation_row["classifier_accuracy"])
    next_accuracy = float(next_row["classifier_accuracy"]) if next_row is not None else None
    measured_improvement = None if next_accuracy is None else next_accuracy - current_accuracy
    capability_match = failure_row["failure_mode"] in {
        "pointwise_memory_failure",
        "history_accumulation_failure",
        "dynamics_model_failure",
        "switching_state_failure",
    }
    if not bool(corpus_row["can_evaluate_classifier"]):
        decision = "revise_corpus"
        rationale = corpus_row["blocking_reason"] or "corpus gate failed"
    elif corpus_row["feature_excitation_status"] == "fail":
        decision = "revise_features"
        rationale = "feature excitation gate failed"
    elif posterior_row["prior_flip_fraction"] > thresholds.max_prior_flip_fraction:
        decision = "revise_prior"
        rationale = "prior sensitivity is too high"
    elif oracle_row["learnability_status"] == "feature_limited":
        decision = "feature_limited"
        rationale = "oracle and overlap evidence indicate feature limitation"
    elif not candidate_family:
        decision = "stay"
        rationale = "current rung is sufficient and no next rung is configured"
    elif not capability_match:
        decision = "reject_escalation"
        rationale = f"{candidate_family} does not match the diagnosed failure mode"
    elif measured_improvement is None:
        decision = "defer_advanced"
        rationale = "next rung capability matches, but no measured comparison exists yet"
    elif measured_improvement >= thresholds.min_improvement_to_promote:
        decision = "promote"
        rationale = f"next rung improves accuracy by {measured_improvement:.3f}"
    elif oracle_row["oracle_gap"] < thresholds.min_oracle_gap_for_algorithm_failure:
        decision = "stay"
        rationale = "current rung is near the oracle limit"
    else:
        decision = "defer_advanced"
        rationale = f"measured improvement {measured_improvement:.3f} does not clear threshold"
    runtime_cost_ratio = float(capability_lookup()[candidate_family].complexity_cost / capability_lookup()[current_family].complexity_cost) if candidate_family in capability_lookup() and current_family in capability_lookup() else 1.0
    return {
        "study_id": str(validation_row["study_id"]),
        "class_pair_id": str(validation_row["class_pair_id"]),
        "feature_set_id": str(validation_row["feature_set_id"]),
        "classifier_id": str(validation_row["classifier_id"]),
        "current_rung_id": current_family,
        "candidate_next_rung_id": candidate_family,
        "current_accuracy": current_accuracy,
        "oracle_accuracy": float(validation_row["oracle_accuracy"]),
        "oracle_gap": float(oracle_row["oracle_gap"]),
        "measured_next_accuracy": "" if next_accuracy is None else next_accuracy,
        "measured_improvement": "" if measured_improvement is None else measured_improvement,
        "runtime_cost_ratio": runtime_cost_ratio,
        "decision": decision,
        "rationale": rationale,
    }


def _special_switching_rows(
    *,
    transition_result,
    imm_result,
    threshold_config: RungThresholdConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    corpus_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    learnability_rows: list[dict[str, object]] = []
    posterior_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    promotion_rows: list[dict[str, object]] = []

    kalman_thresholds = threshold_config.profile_for("kalman_bank")
    transition_thresholds = threshold_config.profile_for("transition_matrix")
    transition_improvement = float(transition_result.summary.transition_post_switch_accuracy - transition_result.summary.kalman_post_switch_accuracy)
    imm_improvement = float(imm_result.summary.imm_post_switch_accuracy - imm_result.summary.transition_post_switch_accuracy)

    corpus_rows.append(
        {
            "study_id": "switching_witness_transition_vs_kalman",
            "corpus_id": "switching_witness_bundle",
            "class_pair_id": "switching_modes",
            "feature_set_id": "model_residuals",
            "classifier_id": "kalman_bank",
            "corpus_status": "pass",
            "class_validity_status": "pass",
            "feature_excitation_status": "pass",
            "leakage_status": "pass",
            "boundary_coverage_status": "pass",
            "identifiability_status": "switching",
            "confusability_score": 1.0 - transition_result.summary.transition_post_switch_accuracy,
            "overlap_estimate": 1.0 - transition_result.summary.transition_post_switch_accuracy,
            "can_evaluate_classifier": True,
            "blocking_reason": "",
        }
    )
    oracle_rows.append(
        {
            "study_id": "switching_witness_transition_vs_kalman",
            "class_pair_id": "switching_modes",
            "feature_set_id": "model_residuals",
            "classifier_id": "kalman_bank",
            "oracle_accuracy": float(max(transition_result.summary.transition_accuracy, transition_result.summary.kalman_accuracy)),
            "best_oracle_accuracy_for_pair": float(max(transition_result.summary.transition_accuracy, transition_result.summary.kalman_accuracy)),
            "current_accuracy": float(transition_result.summary.kalman_accuracy),
            "oracle_gap": float(max(0.0, transition_result.summary.transition_accuracy - transition_result.summary.kalman_accuracy)),
            "mean_posterior_margin": 0.0,
            "overlap_estimate": 0.0,
            "learnability_status": "algorithm_limited" if transition_improvement > 0.0 else "close_to_limit",
            "learnable": True,
        }
    )
    learnability_rows.append(
        {
            "study_id": "switching_witness_transition_vs_kalman",
            "class_pair_id": "switching_modes",
            "feature_set_id": "model_residuals",
            "classifier_id": "kalman_bank",
            "corpus_status": "pass",
            "feature_excitation_status": "pass",
            "class_validity_status": "pass",
            "oracle_accuracy": float(max(transition_result.summary.transition_accuracy, transition_result.summary.kalman_accuracy)),
            "best_oracle_accuracy_for_pair": float(max(transition_result.summary.transition_accuracy, transition_result.summary.kalman_accuracy)),
            "current_accuracy": float(transition_result.summary.kalman_accuracy),
            "oracle_gap": float(max(0.0, transition_result.summary.transition_accuracy - transition_result.summary.kalman_accuracy)),
            "mean_posterior_margin": 0.0,
            "pairwise_auc": float(transition_result.summary.transition_accuracy),
            "overlap_estimate": float(max(0.0, 1.0 - transition_result.summary.transition_accuracy)),
            "confusability_score": float(max(0.0, 1.0 - transition_result.summary.transition_accuracy)),
            "oracle_status": "pass" if transition_result.summary.transition_accuracy >= kalman_thresholds.min_oracle_score else "warn",
            "pairwise_status": "pass" if transition_result.summary.transition_accuracy >= kalman_thresholds.min_pairwise_auc_for_learnable else "warn",
            "posterior_margin_status": "fail",
            "overlap_status": "pass",
            "learnability_status": "algorithm_limited" if transition_improvement > 0.0 else "close_to_limit",
            "learnable": True,
            "oracle_threshold": kalman_thresholds.min_oracle_score,
            "pairwise_auc_threshold": kalman_thresholds.min_pairwise_auc_for_learnable,
            "posterior_margin_threshold": kalman_thresholds.min_posterior_margin_for_learnable,
            "overlap_threshold": kalman_thresholds.max_overlap_for_learnable,
            "threshold_rung_id": "kalman_bank",
        }
    )
    posterior_rows.append(
        {
            "study_id": "switching_witness_transition_vs_kalman",
            "class_pair_id": "switching_modes",
            "feature_set_id": "model_residuals",
            "classifier_id": "kalman_bank",
            "current_accuracy": float(transition_result.summary.kalman_accuracy),
            "balanced_accuracy": float(transition_result.summary.kalman_accuracy),
            "nll": 0.0,
            "brier": 0.0,
            "ece": 0.0,
            "mean_entropy": 0.0,
            "mean_confidence": 0.0,
            "confident_error_rate": 0.0,
            "prior_flip_fraction": 0.0,
            "posterior_quality_score": float(transition_result.summary.kalman_accuracy),
            "posterior_quality_status": "pass",
        }
    )
    failure_rows.append(
        {
            "study_id": "switching_witness_transition_vs_kalman",
            "class_pair_id": "switching_modes",
            "feature_set_id": "model_residuals",
            "classifier_id": "kalman_bank",
            "current_rung_id": "kalman_bank",
            "candidate_next_rung_id": "transition_matrix",
            "failure_mode": "switching_state_failure",
            "failure_rationale": "transition-matrix post-switch accuracy exceeds the Kalman bank on switching witnesses",
        }
    )
    promotion_rows.append(
        {
            "study_id": "switching_witness_transition_vs_kalman",
            "class_pair_id": "switching_modes",
            "feature_set_id": "model_residuals",
            "classifier_id": "kalman_bank",
            "current_rung_id": "kalman_bank",
            "candidate_next_rung_id": "transition_matrix",
            "current_accuracy": float(transition_result.summary.kalman_accuracy),
            "oracle_accuracy": float(max(transition_result.summary.transition_accuracy, transition_result.summary.kalman_accuracy)),
            "oracle_gap": float(max(0.0, transition_result.summary.transition_accuracy - transition_result.summary.kalman_accuracy)),
            "measured_next_accuracy": float(transition_result.summary.transition_accuracy),
            "measured_improvement": transition_improvement,
            "runtime_cost_ratio": 1.10,
            "decision": "promote" if transition_improvement >= kalman_thresholds.min_improvement_to_promote else "defer_advanced",
            "rationale": "transition matrix improves post-switch accuracy on the switching witness" if transition_improvement >= kalman_thresholds.min_improvement_to_promote else "transition matrix improvement is positive but below the promotion threshold",
        }
    )

    corpus_rows.append(
        {
            "study_id": "switching_witness_imm_vs_transition",
            "corpus_id": "switching_witness_bundle",
            "class_pair_id": "switching_modes",
            "feature_set_id": "model_residuals",
            "classifier_id": "transition_matrix",
            "corpus_status": "pass",
            "class_validity_status": "pass",
            "feature_excitation_status": "pass",
            "leakage_status": "pass",
            "boundary_coverage_status": "pass",
            "identifiability_status": "switching",
            "confusability_score": 1.0 - imm_result.summary.imm_post_switch_accuracy,
            "overlap_estimate": 1.0 - imm_result.summary.imm_post_switch_accuracy,
            "can_evaluate_classifier": True,
            "blocking_reason": "",
        }
    )
    oracle_rows.append(
        {
            "study_id": "switching_witness_imm_vs_transition",
            "class_pair_id": "switching_modes",
            "feature_set_id": "model_residuals",
            "classifier_id": "transition_matrix",
            "oracle_accuracy": float(max(imm_result.summary.imm_accuracy, imm_result.summary.transition_post_switch_accuracy)),
            "best_oracle_accuracy_for_pair": float(max(imm_result.summary.imm_accuracy, imm_result.summary.transition_post_switch_accuracy)),
            "current_accuracy": float(imm_result.summary.transition_post_switch_accuracy),
            "oracle_gap": float(max(0.0, imm_result.summary.imm_accuracy - imm_result.summary.transition_post_switch_accuracy)),
            "mean_posterior_margin": 0.0,
            "overlap_estimate": 0.0,
            "learnability_status": "algorithm_limited" if imm_improvement > 0.0 else "close_to_limit",
            "learnable": True,
        }
    )
    learnability_rows.append(
        {
            "study_id": "switching_witness_imm_vs_transition",
            "class_pair_id": "switching_modes",
            "feature_set_id": "model_residuals",
            "classifier_id": "transition_matrix",
            "corpus_status": "pass",
            "feature_excitation_status": "pass",
            "class_validity_status": "pass",
            "oracle_accuracy": float(max(imm_result.summary.imm_accuracy, imm_result.summary.transition_post_switch_accuracy)),
            "best_oracle_accuracy_for_pair": float(max(imm_result.summary.imm_accuracy, imm_result.summary.transition_post_switch_accuracy)),
            "current_accuracy": float(imm_result.summary.transition_post_switch_accuracy),
            "oracle_gap": float(max(0.0, imm_result.summary.imm_accuracy - imm_result.summary.transition_post_switch_accuracy)),
            "mean_posterior_margin": 0.0,
            "pairwise_auc": float(imm_result.summary.imm_accuracy),
            "overlap_estimate": float(max(0.0, 1.0 - imm_result.summary.imm_accuracy)),
            "confusability_score": float(max(0.0, 1.0 - imm_result.summary.imm_accuracy)),
            "oracle_status": "pass" if imm_result.summary.imm_accuracy >= transition_thresholds.min_oracle_score else "warn",
            "pairwise_status": "pass" if imm_result.summary.imm_accuracy >= transition_thresholds.min_pairwise_auc_for_learnable else "warn",
            "posterior_margin_status": "fail",
            "overlap_status": "pass",
            "learnability_status": "algorithm_limited" if imm_improvement > 0.0 else "close_to_limit",
            "learnable": True,
            "oracle_threshold": transition_thresholds.min_oracle_score,
            "pairwise_auc_threshold": transition_thresholds.min_pairwise_auc_for_learnable,
            "posterior_margin_threshold": transition_thresholds.min_posterior_margin_for_learnable,
            "overlap_threshold": transition_thresholds.max_overlap_for_learnable,
            "threshold_rung_id": "transition_matrix",
        }
    )
    posterior_rows.append(
        {
            "study_id": "switching_witness_imm_vs_transition",
            "class_pair_id": "switching_modes",
            "feature_set_id": "model_residuals",
            "classifier_id": "transition_matrix",
            "current_accuracy": float(imm_result.summary.transition_post_switch_accuracy),
            "balanced_accuracy": float(imm_result.summary.transition_post_switch_accuracy),
            "nll": 0.0,
            "brier": 0.0,
            "ece": 0.0,
            "mean_entropy": 0.0,
            "mean_confidence": 0.0,
            "confident_error_rate": 0.0,
            "prior_flip_fraction": 0.0,
            "posterior_quality_score": float(imm_result.summary.transition_post_switch_accuracy),
            "posterior_quality_status": "pass",
        }
    )
    failure_rows.append(
        {
            "study_id": "switching_witness_imm_vs_transition",
            "class_pair_id": "switching_modes",
            "feature_set_id": "model_residuals",
            "classifier_id": "transition_matrix",
            "current_rung_id": "transition_matrix",
            "candidate_next_rung_id": "imm",
            "failure_mode": "switching_state_failure",
            "failure_rationale": "IMM improves the post-switch evidence surface beyond the transition-matrix rung",
        }
    )
    promotion_rows.append(
        {
            "study_id": "switching_witness_imm_vs_transition",
            "class_pair_id": "switching_modes",
            "feature_set_id": "model_residuals",
            "classifier_id": "transition_matrix",
            "current_rung_id": "transition_matrix",
            "candidate_next_rung_id": "imm",
            "current_accuracy": float(imm_result.summary.transition_post_switch_accuracy),
            "oracle_accuracy": float(max(imm_result.summary.imm_accuracy, imm_result.summary.transition_post_switch_accuracy)),
            "oracle_gap": float(max(0.0, imm_result.summary.imm_accuracy - imm_result.summary.transition_post_switch_accuracy)),
            "measured_next_accuracy": float(imm_result.summary.imm_post_switch_accuracy),
            "measured_improvement": imm_improvement,
            "runtime_cost_ratio": 1.30,
            "decision": "promote" if imm_improvement >= transition_thresholds.min_improvement_to_promote else "defer_advanced",
            "rationale": "IMM improves post-switch accuracy on the switching witness" if imm_improvement >= transition_thresholds.min_improvement_to_promote else "IMM improvement is positive but below the promotion threshold",
        }
    )

    return corpus_rows, oracle_rows, learnability_rows, posterior_rows, failure_rows, promotion_rows


def analyze_rung_sufficiency(
    *,
    seed: int = 7,
    trajectories_per_case: int = 6,
    threshold_config: RungThresholdConfig | None = None,
) -> RungSufficiencyResult:
    threshold_config = threshold_config or default_rung_threshold_config()
    validation = analyze_validation_ladder(seed=seed, trajectories_per_case=trajectories_per_case)
    corpus = analyze_corpus_adequacy(seed=seed, trajectories_per_class=5)
    common = analyze_common_experiment(seed=seed, trajectories_per_case=trajectories_per_case)
    transition = run_transition_benchmark(seed=seed, replicas=8)
    imm = analyze_advanced_state_inference(seed=seed, replicas=6)

    validation_rows = list(validation.decision_rows)
    score_rows = list(validation.score_rows)
    feature_status_lookup = {
        str(row["study_id"]): str(row["status"])
        for row in score_rows
        if str(row["level_name"]) == "feature_separability"
    }
    corpus_pair_lookup = {_canonical_pair_id(str(row["class_a"]), str(row["class_b"])): dict(row) for row in corpus.class_pair_rows}
    ident_lookup = {
        (str(row["class_pair_id"]), str(row["feature_set_id"])): dict(row)
        for row in common.identifiability_rows
    }
    common_oracle_lookup = {
        (str(row["class_pair_id"]), str(row["feature_set_id"])): dict(row)
        for row in common.oracle_rows
    }
    pair_prediction_lookup: dict[tuple[str, str, str], list[BinaryPredictionRow]] = {}
    for row in common.pair_prediction_rows:
        pair_prediction_lookup.setdefault((str(row["classifier_id"]), str(row["class_pair_id"]), str(row["feature_set_id"])), []).append(_binary_prediction_row(dict(row)))

    capability_rows_out = list(capability_rows())
    corpus_precondition_rows: list[dict[str, object]] = []
    oracle_gap_rows: list[dict[str, object]] = []
    learnability_surface_rows: list[dict[str, object]] = []
    posterior_quality_rows: list[dict[str, object]] = []
    failure_mode_rows: list[dict[str, object]] = []
    promotion_rows: list[dict[str, object]] = []

    for row in validation_rows:
        current_family = canonicalize_rung_id(str(row["classifier_id"]))
        candidate_family = next_rung_id(current_family)
        current_thresholds = threshold_config.profile_for(current_family)
        corpus_row = _corpus_precondition_row(
            validation_row=row,
            feature_status=feature_status_lookup.get(str(row["study_id"]), "defer"),
            corpus_summary=corpus.summary,
            pair_lookup=corpus_pair_lookup,
            ident_lookup=ident_lookup,
            thresholds=current_thresholds,
        )
        oracle_row = _oracle_gap_row(
            validation_row=row,
            common_oracle_lookup=common_oracle_lookup,
            ident_lookup=ident_lookup,
            thresholds=current_thresholds,
        )
        prediction_rows = pair_prediction_lookup.get((str(row["classifier_id"]), str(row["class_pair_id"]), str(row["feature_set_id"])), [])
        posterior_row = _posterior_quality_row(
            validation_row=row,
            prediction_rows=prediction_rows,
            thresholds=current_thresholds,
        )
        learnability_row = _learnability_surface_row(
            validation_row=row,
            oracle_row=oracle_row,
            posterior_row=posterior_row,
            corpus_row=corpus_row,
            common_oracle_lookup=common_oracle_lookup,
            ident_lookup=ident_lookup,
            thresholds=current_thresholds,
        )
        next_row = None
        if candidate_family is not None:
            next_row = _best_lookup(
                validation_rows,
                class_pair_id=str(row["class_pair_id"]),
                feature_set_id=str(row["feature_set_id"]),
                classifier_family=candidate_family,
            )
        measured_improvement = None
        if next_row is not None:
            measured_improvement = float(next_row["classifier_accuracy"]) - float(row["classifier_accuracy"])
        if current_family == "kalman_bank" and candidate_family == "transition_matrix":
            measured_improvement = float(transition.summary.transition_post_switch_accuracy - transition.summary.kalman_post_switch_accuracy)
            next_row = {"classifier_accuracy": transition.summary.transition_accuracy}
        if current_family == "transition_matrix" and candidate_family == "imm":
            measured_improvement = float(imm.summary.imm_post_switch_accuracy - imm.summary.transition_post_switch_accuracy)
            next_row = {"classifier_accuracy": imm.summary.imm_accuracy}
        failure_row = _failure_mode_row(
            corpus_row=corpus_row,
            oracle_row=oracle_row,
            posterior_row=posterior_row,
            validation_row=row,
            current_family=current_family,
            candidate_family=candidate_family,
            measured_improvement=measured_improvement,
            thresholds=current_thresholds,
        )
        promotion_row = _promotion_row(
            validation_row=row,
            corpus_row=corpus_row,
            oracle_row=oracle_row,
            posterior_row=posterior_row,
            failure_row=failure_row,
            next_row=next_row,
            thresholds=current_thresholds,
        )

        corpus_precondition_rows.append(corpus_row)
        oracle_gap_rows.append(oracle_row)
        learnability_surface_rows.append(learnability_row)
        posterior_quality_rows.append(posterior_row)
        failure_mode_rows.append(failure_row)
        promotion_rows.append(promotion_row)

    special_corpus_rows, special_oracle_rows, special_learnability_rows, special_posterior_rows, special_failure_rows, special_promotion_rows = _special_switching_rows(
        transition_result=transition,
        imm_result=imm,
        threshold_config=threshold_config,
    )
    corpus_precondition_rows.extend(special_corpus_rows)
    oracle_gap_rows.extend(special_oracle_rows)
    learnability_surface_rows.extend(special_learnability_rows)
    posterior_quality_rows.extend(special_posterior_rows)
    failure_mode_rows.extend(special_failure_rows)
    promotion_rows.extend(special_promotion_rows)

    summary_rows = [
        {
            "rung_id": spec.rung_id,
            "rank": spec.rank,
            "samples": sum(1 for row in promotion_rows if str(row["current_rung_id"]) == spec.rung_id),
            "promote_count": sum(1 for row in promotion_rows if str(row["current_rung_id"]) == spec.rung_id and str(row["decision"]) == "promote"),
            "defer_count": sum(1 for row in promotion_rows if str(row["current_rung_id"]) == spec.rung_id and str(row["decision"]) == "defer_advanced"),
            "stay_count": sum(1 for row in promotion_rows if str(row["current_rung_id"]) == spec.rung_id and str(row["decision"]) == "stay"),
            "revise_count": sum(1 for row in promotion_rows if str(row["current_rung_id"]) == spec.rung_id and str(row["decision"]).startswith("revise")),
            "reject_count": sum(1 for row in promotion_rows if str(row["current_rung_id"]) == spec.rung_id and str(row["decision"]) == "reject_escalation"),
        }
        for spec in capability_lookup().values()
    ]

    report_markdown = _report(
        validation_rows=validation_rows,
        corpus=corpus,
        transition=transition,
        imm=imm,
        oracle_gap_rows=oracle_gap_rows,
        learnability_surface_rows=learnability_surface_rows,
        posterior_quality_rows=posterior_quality_rows,
        failure_mode_rows=failure_mode_rows,
        promotion_rows=promotion_rows,
        summary_rows=summary_rows,
        threshold_config=threshold_config,
    )

    return RungSufficiencyResult(
        capability_rows=tuple(capability_rows_out),
        corpus_precondition_rows=tuple(corpus_precondition_rows),
        oracle_gap_rows=tuple(oracle_gap_rows),
        posterior_quality_rows=tuple(posterior_quality_rows),
        failure_mode_rows=tuple(failure_mode_rows),
        learnability_surface_rows=tuple(learnability_surface_rows),
        promotion_rows=tuple(promotion_rows),
        summary_rows=tuple(summary_rows),
        report_markdown=report_markdown,
    )


def _plot_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _render_plots(output_dir: Path, result: RungSufficiencyResult) -> dict[str, Path]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    score_vs_oracle_path = plots_dir / "rung_score_vs_oracle.png"
    oracle_gap_path = plots_dir / "oracle_gap_by_class_pair.png"
    failure_heatmap_path = plots_dir / "failure_mode_heatmap.png"
    promotion_matrix_path = plots_dir / "promotion_decision_matrix.png"
    posterior_quality_path = plots_dir / "posterior_quality_by_rung.png"

    # Score vs oracle.
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    decision_colors = {"promote": "#16a34a", "stay": "#2563eb", "defer_advanced": "#d97706", "revise_corpus": "#dc2626", "revise_features": "#f97316", "revise_prior": "#7c3aed", "feature_limited": "#b91c1c", "corpus_limited": "#991b1b", "reject_escalation": "#7f1d1d"}
    for row in result.promotion_rows:
        if "current_accuracy" not in row:
            continue
        ax.scatter(float(row["oracle_accuracy"]), float(row["current_accuracy"]), color=decision_colors.get(str(row["decision"]), "#6b7280"), alpha=0.75, s=24)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1.0)
    ax.set_xlabel("Oracle accuracy")
    ax.set_ylabel("Current rung accuracy")
    ax.set_title("Rung score vs oracle")
    ax.grid(True, alpha=0.25)
    fig.savefig(score_vs_oracle_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Oracle gap by class pair.
    fig, ax = plt.subplots(figsize=(10.0, max(4.5, 0.35 * len(result.oracle_gap_rows))))
    grouped: dict[str, list[float]] = {}
    for row in result.oracle_gap_rows:
        grouped.setdefault(str(row["class_pair_id"]), []).append(float(row["oracle_gap"]))
    labels = sorted(grouped)
    gaps = [_mean(grouped[label]) for label in labels]
    ax.barh(labels, gaps, color="#2563eb")
    ax.set_xlabel("Mean oracle gap")
    ax.set_title("Oracle gap by class pair")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(oracle_gap_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Failure mode heatmap.
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    current_rungs = [spec.rung_id for spec in capability_lookup().values()]
    failure_modes = sorted({str(row["failure_mode"]) for row in result.failure_mode_rows})
    matrix = zeros((len(failure_modes), len(current_rungs)), dtype=float)
    for row in result.failure_mode_rows:
        x = current_rungs.index(str(row["current_rung_id"]))
        y = failure_modes.index(str(row["failure_mode"]))
        matrix[y, x] += 1.0
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(current_rungs)))
    ax.set_xticklabels(current_rungs, rotation=30, ha="right")
    ax.set_yticks(range(len(failure_modes)))
    ax.set_yticklabels(failure_modes)
    ax.set_title("Failure mode heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="count")
    fig.savefig(failure_heatmap_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Promotion decision matrix.
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    candidate_rungs = [spec.rung_id for spec in capability_lookup().values()]
    decision_rank = {"promote": 3, "stay": 2, "defer_advanced": 1, "revise_corpus": 0, "revise_features": 0, "revise_prior": 0, "feature_limited": 0, "corpus_limited": 0, "reject_escalation": 0}
    matrix = zeros((len(candidate_rungs), len(current_rungs)), dtype=float)
    for row in result.promotion_rows:
        if str(row["candidate_next_rung_id"]) not in candidate_rungs:
            continue
        x = current_rungs.index(str(row["current_rung_id"]))
        y = candidate_rungs.index(str(row["candidate_next_rung_id"]))
        matrix[y, x] = max(matrix[y, x], float(decision_rank.get(str(row["decision"]), 0)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=3.0)
    ax.set_xticks(range(len(current_rungs)))
    ax.set_xticklabels(current_rungs, rotation=30, ha="right")
    ax.set_yticks(range(len(candidate_rungs)))
    ax.set_yticklabels(candidate_rungs)
    ax.set_title("Promotion decision matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="decision rank")
    fig.savefig(promotion_matrix_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Posterior quality by rung.
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    quality_by_rung: dict[str, list[float]] = {}
    for row in result.posterior_quality_rows:
        quality_by_rung.setdefault(str(row["classifier_id"]), []).append(float(row["posterior_quality_score"]))
    labels = sorted(quality_by_rung)
    scores = [_mean(quality_by_rung[label]) for label in labels]
    ax.bar(labels, scores, color="#0f766e")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Posterior quality score")
    ax.set_title("Posterior quality by rung")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(posterior_quality_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "score_vs_oracle": score_vs_oracle_path,
        "oracle_gap": oracle_gap_path,
        "failure_heatmap": failure_heatmap_path,
        "promotion_matrix": promotion_matrix_path,
        "posterior_quality": posterior_quality_path,
    }


def _report(
    *,
    validation_rows: list[dict[str, object]],
    corpus,
    transition,
    imm,
    oracle_gap_rows: list[dict[str, object]],
    learnability_surface_rows: list[dict[str, object]],
    posterior_quality_rows: list[dict[str, object]],
    failure_mode_rows: list[dict[str, object]],
    promotion_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    threshold_config: RungThresholdConfig,
) -> str:
    decision_counts: dict[str, int] = {}
    for row in promotion_rows:
        decision_counts[str(row["decision"])] = decision_counts.get(str(row["decision"]), 0) + 1
    corpus_gate_passes = sum(1 for row in oracle_gap_rows if row["learnable"])
    top_failures = sorted(
        ((mode, sum(1 for row in failure_mode_rows if str(row["failure_mode"]) == mode)) for mode in sorted({str(row["failure_mode"]) for row in failure_mode_rows})),
        key=lambda item: item[1],
        reverse=True,
    )
    learnability_counts: dict[str, int] = {}
    for row in learnability_surface_rows:
        learnability_counts[str(row["learnability_status"])] = learnability_counts.get(str(row["learnability_status"]), 0) + 1
    promote_examples = [row for row in promotion_rows if str(row["decision"]) == "promote"][:5]
    stay_examples = [row for row in promotion_rows if str(row["decision"]) == "stay"][:5]
    threshold_rows = list(rung_threshold_rows(threshold_config))
    
    doc = MarkdownDocument("Rung Sufficiency And Escalation Evaluator")
    doc.paragraph(
        "This artifact decides whether the current rung is sufficient for the evidence available in a corpus + feature + class + prior study, or whether escalation is justified."
    )
    
    doc.heading("Summary", level=2)
    doc.bullet_list(
        [
            f"Validation studies analyzed: `{len(validation_rows)}`",
            f"Learnable cases: `{corpus_gate_passes}`",
            f"Promotion decisions: `{decision_counts.get('promote', 0)}`",
            f"Stay decisions: `{decision_counts.get('stay', 0)}`",
            f"Revise corpus decisions: `{decision_counts.get('revise_corpus', 0)}`",
            f"Revise feature decisions: `{decision_counts.get('revise_features', 0)}`",
            f"Revise prior decisions: `{decision_counts.get('revise_prior', 0)}`",
            f"Defer decisions: `{decision_counts.get('defer_advanced', 0)}`",
            f"Reject decisions: `{decision_counts.get('reject_escalation', 0)}`",
            f"Learnable rows: `{sum(1 for row in learnability_surface_rows if bool(row['learnable']))}`",
        ]
    )

    doc.heading("Threshold Profiles", level=2)
    doc.bullet_list(
        [
            f"`{row['rung_id']}`: oracle>={row['min_oracle_score']}, gap>={row['min_oracle_gap_for_algorithm_failure']}, improvement>={row['min_improvement_to_promote']}, pairwise_auc>={row['min_pairwise_auc_for_learnable']}, margin>={row['min_posterior_margin_for_learnable']}"
            for row in threshold_rows
        ]
    )

    doc.heading("Corpus Gate", level=2)
    doc.bullet_list(
        [
            f"Corpus status: `{corpus.summary.overall_status}`",
            f"Feature status: `{corpus.summary.feature_status}`",
        ]
    )

    doc.heading("Learnability Surfaces", level=2)
    doc.bullet_list(
        [
            f"`{status}`: `{count}`"
            for status, count in sorted(learnability_counts.items(), key=lambda item: item[0])
        ]
    )

    doc.heading("Switching Evidence", level=2)
    doc.bullet_list(
        [
            f"Transition vs Kalman post-switch accuracy: `{transition.summary.transition_post_switch_accuracy:.3f}` vs `{transition.summary.kalman_post_switch_accuracy:.3f}`",
            f"IMM vs transition post-switch accuracy: `{imm.summary.imm_post_switch_accuracy:.3f}` vs `{imm.summary.transition_post_switch_accuracy:.3f}`",
            f"IMM mean state RMSE: `{imm.summary.mean_state_rmse:.3f}`",
        ]
    )

    doc.heading("Top Failure Modes", level=2)
    doc.bullet_list([f"`{mode}`: `{count}`" for mode, count in top_failures[:8]])

    doc.heading("Example Promotions", level=2)
    doc.bullet_list(
        [
            f"`{row['study_id']}` -> `{row['decision']}` ({row['current_rung_id']} -> {row['candidate_next_rung_id']}, improvement={row['measured_improvement']})"
            for row in promote_examples
        ]
    )

    doc.heading("Example Stay Decisions", level=2)
    doc.bullet_list(
        [
            f"`{row['study_id']}` -> `{row['decision']}` ({row['current_rung_id']} near oracle gap {row['oracle_gap']:.3f})"
            for row in stay_examples
        ]
    )

    doc.heading("Interpretation", level=2)
    doc.bullet_list(
        [
            "Corpus or feature failures are reported before algorithm escalation is considered.",
            "Oracle gap, pairwise AUC, overlap, confusability, and posterior margin are reported explicitly so the evaluator can separate learnability limits from model limits.",
            "IMM is only treated as justified when the switching witness shows a measurable post-switch improvement over the transition-matrix rung.",
            "PF and RBPF remain deferred until a nonlinear or latent-structure witness is added with measurable evidence.",
        ]
    )

    return doc.text()


def write_rung_sufficiency_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_case: int = 6,
    threshold_config: RungThresholdConfig | None = None,
    result: RungSufficiencyResult | None = None,
) -> RungSufficiencyArtifacts:
    threshold_config = threshold_config or default_rung_threshold_config()
    analysis = result or analyze_rung_sufficiency(seed=seed, trajectories_per_case=trajectories_per_case, threshold_config=threshold_config)
    run_dir = Path(output_dir) / "rung_sufficiency"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "rung_sufficiency_config.yaml"
    threshold_profile_path = run_dir / "rung_threshold_profiles.csv"
    capability_matrix_path = run_dir / "rung_capability_matrix.csv"
    corpus_precondition_path = run_dir / "corpus_precondition_report.csv"
    oracle_gap_path = run_dir / "oracle_gap_report.csv"
    learnability_surface_path = run_dir / "learnability_surface_report.csv"
    posterior_quality_path = run_dir / "posterior_quality_by_rung.csv"
    failure_mode_path = run_dir / "failure_mode_diagnosis.csv"
    promotion_matrix_path = run_dir / "rung_promotion_matrix.csv"
    report_path = run_dir / "rung_sufficiency_report.md"

    threshold_rows = list(rung_threshold_rows(threshold_config))
    write_csv(threshold_profile_path, threshold_rows, list(threshold_rows[0].keys()))
    write_csv(capability_matrix_path, list(analysis.capability_rows), list(analysis.capability_rows[0].keys()))
    write_csv(corpus_precondition_path, list(analysis.corpus_precondition_rows), list(analysis.corpus_precondition_rows[0].keys()))
    write_csv(oracle_gap_path, list(analysis.oracle_gap_rows), list(analysis.oracle_gap_rows[0].keys()))
    write_csv(learnability_surface_path, list(analysis.learnability_surface_rows), list(analysis.learnability_surface_rows[0].keys()))
    write_csv(posterior_quality_path, list(analysis.posterior_quality_rows), list(analysis.posterior_quality_rows[0].keys()))
    write_csv(failure_mode_path, list(analysis.failure_mode_rows), list(analysis.failure_mode_rows[0].keys()))
    write_csv(promotion_matrix_path, list(analysis.promotion_rows), list(analysis.promotion_rows[0].keys()))
    _write_text(report_path, analysis.report_markdown)
    _write_text(
        config_path,
        "\n".join(
            [
                "threshold_config:",
                "  default_profile:",
                f"    min_corpus_score: {threshold_config.default_profile.min_corpus_score}",
                f"    min_feature_score: {threshold_config.default_profile.min_feature_score}",
                f"    min_oracle_score: {threshold_config.default_profile.min_oracle_score}",
                f"    min_oracle_gap_for_algorithm_failure: {threshold_config.default_profile.min_oracle_gap_for_algorithm_failure}",
                f"    max_prior_flip_fraction: {threshold_config.default_profile.max_prior_flip_fraction}",
                f"    max_confident_error_rate: {threshold_config.default_profile.max_confident_error_rate}",
                f"    min_improvement_to_promote: {threshold_config.default_profile.min_improvement_to_promote}",
                f"    max_runtime_cost_ratio: {threshold_config.default_profile.max_runtime_cost_ratio}",
                f"    max_overlap_for_learnable: {threshold_config.default_profile.max_overlap_for_learnable}",
                f"    min_confusability_for_feature_limited: {threshold_config.default_profile.min_confusability_for_feature_limited}",
                f"    min_pairwise_auc_for_learnable: {threshold_config.default_profile.min_pairwise_auc_for_learnable}",
                f"    min_posterior_margin_for_learnable: {threshold_config.default_profile.min_posterior_margin_for_learnable}",
                "  rung_profiles:",
            ]
            + [
                line
                for rung_id, profile in threshold_config.rung_profiles
                for line in (
                    f"    - rung_id: {rung_id}",
                    f"      min_corpus_score: {profile.min_corpus_score}",
                    f"      min_feature_score: {profile.min_feature_score}",
                    f"      min_oracle_score: {profile.min_oracle_score}",
                    f"      min_oracle_gap_for_algorithm_failure: {profile.min_oracle_gap_for_algorithm_failure}",
                    f"      max_prior_flip_fraction: {profile.max_prior_flip_fraction}",
                    f"      max_confident_error_rate: {profile.max_confident_error_rate}",
                    f"      min_improvement_to_promote: {profile.min_improvement_to_promote}",
                    f"      max_runtime_cost_ratio: {profile.max_runtime_cost_ratio}",
                    f"      max_overlap_for_learnable: {profile.max_overlap_for_learnable}",
                    f"      min_confusability_for_feature_limited: {profile.min_confusability_for_feature_limited}",
                    f"      min_pairwise_auc_for_learnable: {profile.min_pairwise_auc_for_learnable}",
                    f"      min_posterior_margin_for_learnable: {profile.min_posterior_margin_for_learnable}",
                )
            ]
        )
        + "\n",
    )

    plot_paths = _render_plots(run_dir, analysis)
    return RungSufficiencyArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        threshold_profile_path=threshold_profile_path,
        capability_matrix_path=capability_matrix_path,
        corpus_precondition_path=corpus_precondition_path,
        oracle_gap_path=oracle_gap_path,
        learnability_surface_path=learnability_surface_path,
        posterior_quality_path=posterior_quality_path,
        failure_mode_path=failure_mode_path,
        promotion_matrix_path=promotion_matrix_path,
        report_path=report_path,
        score_vs_oracle_plot_path=plot_paths["score_vs_oracle"],
        oracle_gap_plot_path=plot_paths["oracle_gap"],
        failure_mode_heatmap_path=plot_paths["failure_heatmap"],
        promotion_decision_plot_path=plot_paths["promotion_matrix"],
        posterior_quality_plot_path=plot_paths["posterior_quality"],
    )


def _ladder_witness_schema() -> dict[str, Any]:
    witness_objective_schema = {
        "type": "object",
        "required": ["difficulty", "evidence_strength", "temporal_consistency"],
        "properties": {
            "difficulty": {"type": "string"},
            "evidence_strength": {"type": "string"},
            "temporal_consistency": {"type": "string"},
        },
        "additionalProperties": True,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "LadderWitnessSuiteConfig",
        "type": "object",
        "required": ["suite_id", "description", "witnesses"],
        "properties": {
            "suite_id": {"type": "string"},
            "description": {"type": "string"},
            "witnesses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "witness_id",
                        "rung_under_test",
                        "claim_type",
                        "classes",
                        "feature_sets",
                        "methods",
                        "priors",
                        "corpus_objective",
                        "expected_result",
                        "success_criteria",
                        "visual_story",
                    ],
                    "properties": {
                        "witness_id": {"type": "string"},
                        "rung_under_test": {"type": "string"},
                        "claim_type": {"type": "string"},
                        "comparison_method": {"type": "string"},
                        "classes": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "feature_sets": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "methods": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "priors": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "corpus_objective": witness_objective_schema,
                        "expected_result": {"type": "object", "additionalProperties": {"type": "string"}},
                        "success_criteria": {"type": "object", "additionalProperties": True},
                        "visual_story": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    },
                    "additionalProperties": True,
                },
            },
        },
        "additionalProperties": True,
    }


def _normalize_witness_entry(entry: dict[str, Any]) -> dict[str, Any]:
    required = (
        "witness_id",
        "rung_under_test",
        "claim_type",
        "classes",
        "feature_sets",
        "methods",
        "priors",
        "corpus_objective",
        "expected_result",
        "success_criteria",
        "visual_story",
    )
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(f"witness is missing required keys: {', '.join(sorted(missing))}")
    corpus_objective = dict(entry["corpus_objective"])
    for key in ("difficulty", "evidence_strength", "temporal_consistency"):
        if key not in corpus_objective:
            raise ValueError(f"witness {entry['witness_id']} missing corpus objective key: {key}")
    methods = tuple(str(value) for value in entry["methods"])
    expected_result = {str(key): str(value) for key, value in dict(entry["expected_result"]).items()}
    comparison_method = entry.get("comparison_method")
    return {
        "witness_id": str(entry["witness_id"]),
        "rung_under_test": str(entry["rung_under_test"]),
        "claim_type": str(entry["claim_type"]),
        "comparison_method": str(comparison_method) if comparison_method is not None else "",
        "classes": tuple(str(value) for value in entry["classes"]),
        "feature_sets": tuple(str(value) for value in entry["feature_sets"]),
        "methods": methods,
        "priors": tuple(str(value) for value in entry["priors"]),
        "corpus_objective": {
            "difficulty": str(corpus_objective["difficulty"]),
            "evidence_strength": str(corpus_objective["evidence_strength"]),
            "temporal_consistency": str(corpus_objective["temporal_consistency"]),
        },
        "expected_result": expected_result,
        "success_criteria": dict(entry["success_criteria"]),
        "visual_story": tuple(str(value) for value in entry["visual_story"]),
    }


def load_ladder_witness_suite_config(path: str | Path = DEFAULT_LADDER_WITNESS_SUITE_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    witnesses = tuple(_normalize_witness_entry(dict(entry)) for entry in payload.get("witnesses", ()))
    if not witnesses:
        raise ValueError("ladder witness suite config must define at least one witness")
    suite_id = str(payload.get("suite_id", "ladder_witness_suite_v1"))
    description = str(payload.get("description", "Ladder witness corpus suite"))
    return {
        "suite_id": suite_id,
        "description": description,
        "config_path": config_path,
        "witnesses": witnesses,
    }


def _witness_claim_matrix_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for witness in config["witnesses"]:
        expected_result = dict(witness["expected_result"])
        sufficient_methods = [method for method, status in expected_result.items() if status == "sufficient"]
        insufficient_methods = [method for method, status in expected_result.items() if status == "insufficient"]
        comparison_method = witness["comparison_method"] or (sufficient_methods[0] if sufficient_methods else (witness["methods"][1] if len(witness["methods"]) > 1 else ""))
        rows.append(
            {
                "witness_id": witness["witness_id"],
                "rung_under_test": witness["rung_under_test"],
                "claim_type": witness["claim_type"],
                "comparison_method": comparison_method,
                "class_count": len(witness["classes"]),
                "feature_set_count": len(witness["feature_sets"]),
                "method_count": len(witness["methods"]),
                "prior_count": len(witness["priors"]),
                "difficulty": witness["corpus_objective"]["difficulty"],
                "evidence_strength": witness["corpus_objective"]["evidence_strength"],
                "temporal_consistency": witness["corpus_objective"]["temporal_consistency"],
                "sufficient_methods": ";".join(sufficient_methods),
                "insufficient_methods": ";".join(insufficient_methods),
                "visual_story_count": len(witness["visual_story"]),
                "success_criteria_keys": ";".join(sorted(witness["success_criteria"].keys())),
            }
        )
    return rows


def _witness_suite_index_markdown(config: dict[str, Any], matrix_rows: list[dict[str, Any]]) -> str:
    claim_counts = {"sufficiency": 0, "insufficiency": 0}
    rung_counts: dict[str, int] = {}
    for witness in config["witnesses"]:
        claim_counts[witness["claim_type"]] = claim_counts.get(witness["claim_type"], 0) + 1
        rung_counts[witness["rung_under_test"]] = rung_counts.get(witness["rung_under_test"], 0) + 1
    
    doc = MarkdownDocument("Ladder Witness Corpus Suite")
    doc.bullet_list(
        [
            f"suite id: `{config['suite_id']}`",
            f"witnesses declared: `{len(config['witnesses'])}`",
            f"sufficiency claims: `{claim_counts.get('sufficiency', 0)}`",
            f"insufficiency claims: `{claim_counts.get('insufficiency', 0)}`",
        ]
    )

    doc.heading("Witness Matrix", level=2)
    doc.table(
        ["Witness", "Rung", "Claim", "Comparison", "Difficulty", "Evidence"],
        [
            (
                f"`{row['witness_id']}`",
                f"`{row['rung_under_test']}`",
                f"`{row['claim_type']}`",
                f"`{row['comparison_method']}`",
                f"`{row['difficulty']}`",
                f"`{row['evidence_strength']}`",
            )
            for row in matrix_rows
        ]
    )

    doc.heading("Rung Coverage", level=2)
    doc.table(
        ["Rung", "Witness Count"],
        [
            (f"`{rung}`", f"`{count}`")
            for rung, count in sorted(rung_counts.items())
        ]
    )

    doc.heading("Notes", level=2)
    doc.bullet_list(
        [
            "Witness corpora are controlled 1D proof cases, not benchmark vanity sets.",
            "Each witness declares the methods it compares, the claim it is proving, and the evidence story the later milestones will render into plots and tables.",
        ]
    )
    return doc.text()


def write_ladder_witness_suite_artifacts(
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> LadderWitnessSuiteArtifacts:
    run_dir = Path(output_dir) / LADDER_WITNESS_SUITE_RUN_DIR_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("datasets", "plots", "reports"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    resolved_config = config or load_ladder_witness_suite_config(config_path or DEFAULT_LADDER_WITNESS_SUITE_CONFIG_PATH)
    normalized_config = {
        "suite_id": resolved_config["suite_id"],
        "description": resolved_config["description"],
        "source_config_path": str(resolved_config["config_path"]),
        "witness_count": len(resolved_config["witnesses"]),
        "witnesses": [
            {
                "witness_id": witness["witness_id"],
                "rung_under_test": witness["rung_under_test"],
                "claim_type": witness["claim_type"],
                "comparison_method": witness["comparison_method"],
                "classes": list(witness["classes"]),
                "feature_sets": list(witness["feature_sets"]),
                "methods": list(witness["methods"]),
                "priors": list(witness["priors"]),
                "corpus_objective": dict(witness["corpus_objective"]),
                "expected_result": dict(witness["expected_result"]),
                "success_criteria": witness["success_criteria"],
                "visual_story": list(witness["visual_story"]),
            }
            for witness in resolved_config["witnesses"]
        ],
    }
    schema_path = run_dir / "witness_schema.json"
    manifest_path = run_dir / "witness_manifest.json"
    claim_matrix_path = run_dir / "rung_claim_matrix.csv"
    index_path = run_dir / "index.md"
    config_copy_path = run_dir / "witness_suite_config.yaml"

    schema_path.write_text(json.dumps(_ladder_witness_schema(), indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(json.dumps(normalized_config, indent=2, sort_keys=True), encoding="utf-8")
    claim_rows = _witness_claim_matrix_rows(resolved_config)
    write_csv(claim_matrix_path, claim_rows, list(claim_rows[0].keys()))
    index_path.write_text(_witness_suite_index_markdown(resolved_config, claim_rows), encoding="utf-8")
    config_copy_path.write_text(yaml.safe_dump(normalized_config, sort_keys=False), encoding="utf-8")

    return LadderWitnessSuiteArtifacts(
        run_dir=run_dir,
        config_path=config_copy_path,
        schema_path=schema_path,
        manifest_path=manifest_path,
        claim_matrix_path=claim_matrix_path,
        index_path=index_path,
    )
