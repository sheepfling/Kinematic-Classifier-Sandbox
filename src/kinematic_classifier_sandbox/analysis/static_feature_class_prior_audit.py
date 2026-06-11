from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import log, sqrt
from statistics import mean

import numpy as np

from kinematic_classifier_sandbox.utils.math import _normalize_log_scores
from kinematic_classifier_sandbox.utils.stats import histogram_overlap, js_divergence

from .static_feature_class_prior_audit_contracts import (
    StaticAuditFeatureSchemaEntry,
    StaticAuditSample,
    StaticFeatureClassPriorAuditResult,
)
from .static_feature_class_prior_audit_reporting import (
    render_static_feature_class_prior_audit_report,
)


def _safe_log(value: float) -> float:
    return log(max(value, 1e-12))


def _pairwise_auc(scores_a: list[float], scores_b: list[float]) -> float:
    if not scores_a or not scores_b:
        return 0.5
    wins = 0.0
    total = 0.0
    for score_a in scores_a:
        for score_b in scores_b:
            if score_a > score_b:
                wins += 1.0
            elif score_a == score_b:
                wins += 0.5
            total += 1.0
    auc = wins / max(total, 1.0)
    return max(auc, 1.0 - auc)


def _rank(values: list[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0 for _ in values]
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        rank_value = (cursor + end - 1) / 2.0
        for _, original_index in ordered[cursor:end]:
            ranks[original_index] = rank_value
        cursor = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denom = sqrt(left_ss * right_ss)
    return numerator / denom if denom > 0.0 else 0.0


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_rank(left), _rank(right))


def _discretize(values: list[float], bins: int = 8) -> list[int]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0 for _ in values]
    width = (hi - lo) / bins
    return [min(int((value - lo) / width), bins - 1) for value in values]


def _mutual_information(discrete_left: list[object], discrete_right: list[object]) -> float:
    if not discrete_left or len(discrete_left) != len(discrete_right):
        return 0.0
    total = len(discrete_left)
    left_counts = Counter(discrete_left)
    right_counts = Counter(discrete_right)
    joint_counts = Counter(zip(discrete_left, discrete_right))
    value = 0.0
    for (left, right), count in joint_counts.items():
        pxy = count / total
        px = left_counts[left] / total
        py = right_counts[right] / total
        value += pxy * log(pxy / max(px * py, 1e-12))
    return value


def _joint_mutual_information(feature_a: list[float], feature_b: list[float], labels: list[str]) -> float:
    bins_a = _discretize(feature_a)
    bins_b = _discretize(feature_b)
    return _mutual_information(list(zip(bins_a, bins_b)), labels)


def _cohens_d(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2 or len(values_b) < 2:
        return 0.0
    mean_a = mean(values_a)
    mean_b = mean(values_b)
    var_a = sum((value - mean_a) ** 2 for value in values_a) / (len(values_a) - 1)
    var_b = sum((value - mean_b) ** 2 for value in values_b) / (len(values_b) - 1)
    pooled = sqrt(max((var_a + var_b) / 2.0, 1e-12))
    return abs(mean_a - mean_b) / pooled


def _class_matrix(
    samples: tuple[StaticAuditSample, ...],
    feature_names: tuple[str, ...],
    class_name: str,
) -> np.ndarray:
    rows = [
        [float(sample.feature_values.get(feature_name, 0.0)) for feature_name in feature_names]
        for sample in samples
        if sample.true_class == class_name
    ]
    return np.asarray(rows, dtype=float)


def _gaussian_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return np.zeros((0,), dtype=float), np.zeros((0, 0), dtype=float)
    mean_vector = matrix.mean(axis=0)
    if matrix.shape[0] < 2:
        covariance = np.eye(matrix.shape[1], dtype=float) * 1e-6
    else:
        covariance = np.cov(matrix, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.asarray([[float(covariance)]], dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    covariance += np.eye(covariance.shape[0], dtype=float) * 1e-6
    return mean_vector, covariance


def _mahalanobis_distance(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    if matrix_a.size == 0 or matrix_b.size == 0:
        return 0.0
    mean_a, cov_a = _gaussian_stats(matrix_a)
    mean_b, cov_b = _gaussian_stats(matrix_b)
    pooled = 0.5 * (cov_a + cov_b)
    diff = mean_a - mean_b
    inv_cov = np.linalg.pinv(pooled)
    value = float(diff.T @ inv_cov @ diff)
    return sqrt(max(value, 0.0))


def _log_gaussian_pdf(vector: np.ndarray, mean_vector: np.ndarray, covariance: np.ndarray) -> float:
    if covariance.size == 0:
        return 0.0
    dimension = covariance.shape[0]
    covariance = covariance + np.eye(dimension, dtype=float) * 1e-6
    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0:
        log_det = float(np.log(np.linalg.det(covariance + np.eye(dimension) * 1e-3)))
    diff = vector - mean_vector
    quad = float(diff.T @ np.linalg.pinv(covariance) @ diff)
    return -0.5 * (dimension * log(2.0 * np.pi) + log_det + quad)


def _feature_values(samples: tuple[StaticAuditSample, ...], feature_name: str) -> list[float]:
    return [float(sample.feature_values.get(feature_name, 0.0)) for sample in samples]


def _status_from_pair(pairwise_auc: float, overlap: float, mahalanobis: float) -> str:
    if pairwise_auc >= 0.90 and overlap <= 0.30 and mahalanobis >= 1.50:
        return "easy"
    if pairwise_auc < 0.70 or overlap >= 0.60 or mahalanobis < 0.75:
        return "hard"
    return "medium"


def _normalize_priors(class_names: tuple[str, ...], priors: dict[str, float] | None) -> dict[str, float]:
    if not priors:
        weight = 1.0 / max(len(class_names), 1)
        return {class_name: weight for class_name in class_names}
    cleaned = {class_name: max(float(priors.get(class_name, 0.0)), 0.0) for class_name in class_names}
    total = sum(cleaned.values())
    if total <= 0.0:
        weight = 1.0 / max(len(class_names), 1)
        return {class_name: weight for class_name in class_names}
    return {class_name: value / total for class_name, value in cleaned.items()}


def analyze_static_feature_class_prior_audit(
    samples: list[StaticAuditSample] | tuple[StaticAuditSample, ...],
    *,
    priors: dict[str, float] | None = None,
    feature_schema: list[StaticAuditFeatureSchemaEntry] | tuple[StaticAuditFeatureSchemaEntry, ...] = (),
    feature_names: tuple[str, ...] | None = None,
    study_name: str = "static_feature_class_prior_audit",
) -> StaticFeatureClassPriorAuditResult:
    sample_tuple = tuple(samples)
    if not sample_tuple:
        raise ValueError("static audit requires at least one class-labeled feature sample")

    class_names = tuple(sorted({sample.true_class for sample in sample_tuple}))
    if len(class_names) < 2:
        raise ValueError("static audit requires at least two classes")

    if feature_names is None:
        discovered = sorted({name for sample in sample_tuple for name in sample.feature_values})
        feature_names = tuple(discovered)
    if not feature_names:
        raise ValueError("static audit requires at least one feature")

    normalized_priors = _normalize_priors(class_names, priors)
    labels = [sample.true_class for sample in sample_tuple]
    class_matrices = {
        class_name: _class_matrix(sample_tuple, feature_names, class_name) for class_name in class_names
    }
    class_stats = {class_name: _gaussian_stats(matrix) for class_name, matrix in class_matrices.items()}

    class_pair_rows: list[dict[str, object]] = []
    for class_a, class_b in combinations(class_names, 2):
        matrix_a = class_matrices[class_a]
        matrix_b = class_matrices[class_b]
        mean_a = matrix_a.mean(axis=0)
        mean_b = matrix_b.mean(axis=0)
        diff = mean_a - mean_b
        projection = diff if float(np.linalg.norm(diff)) > 1e-12 else np.ones(len(feature_names))
        scores_a = [float(row @ projection) for row in matrix_a]
        scores_b = [float(row @ projection) for row in matrix_b]
        auc = _pairwise_auc(scores_a, scores_b)
        overlap = mean(
            histogram_overlap(
                [float(row[index]) for row in matrix_a],
                [float(row[index]) for row in matrix_b],
            )
            for index, _feature_name in enumerate(feature_names)
        )
        js = mean(
            js_divergence(
                [float(row[index]) for row in matrix_a],
                [float(row[index]) for row in matrix_b],
            )
            for index, _feature_name in enumerate(feature_names)
        )
        variance_sum = float(np.var(matrix_a, axis=0).sum() + np.var(matrix_b, axis=0).sum())
        fisher_ratio = float(np.square(diff).sum() / max(variance_sum, 1e-12))
        mahalanobis = _mahalanobis_distance(matrix_a, matrix_b)
        status = _status_from_pair(auc, overlap, mahalanobis)
        class_pair_rows.append(
            {
                "class_a": class_a,
                "class_b": class_b,
                "pairwise_auc": auc,
                "mahalanobis_distance": mahalanobis,
                "jensen_shannon": js,
                "overlap_coefficient": overlap,
                "fisher_ratio": fisher_ratio,
                "status": status,
            }
        )

    feature_relevance_rows: list[dict[str, object]] = []
    for feature_name in feature_names:
        values = _feature_values(sample_tuple, feature_name)
        mi = _mutual_information(_discretize(values), labels)
        pair_aucs = []
        effects = []
        overlaps = []
        for class_a, class_b in combinations(class_names, 2):
            values_a = [
                float(sample.feature_values.get(feature_name, 0.0))
                for sample in sample_tuple
                if sample.true_class == class_a
            ]
            values_b = [
                float(sample.feature_values.get(feature_name, 0.0))
                for sample in sample_tuple
                if sample.true_class == class_b
            ]
            pair_aucs.append(_pairwise_auc(values_a, values_b))
            effects.append(_cohens_d(values_a, values_b))
            overlaps.append(histogram_overlap(values_a, values_b))
        missing = sum(1 for sample in sample_tuple if feature_name not in sample.feature_values)
        recommended_status = "keep"
        if mi < 0.02 and max(pair_aucs, default=0.5) < 0.65:
            recommended_status = "weak"
        elif max(overlaps, default=1.0) > 0.70:
            recommended_status = "pair_limited"
        feature_relevance_rows.append(
            {
                "feature": feature_name,
                "mi_with_class": mi,
                "max_pairwise_auc": max(pair_aucs, default=0.5),
                "mean_effect_size": mean(effects) if effects else 0.0,
                "worst_pair_overlap": max(overlaps, default=0.0),
                "missing_rate": missing / len(sample_tuple),
                "recommended_status": recommended_status,
            }
        )

    feature_redundancy_rows: list[dict[str, object]] = []
    for feature_a, feature_b in combinations(feature_names, 2):
        values_a = _feature_values(sample_tuple, feature_a)
        values_b = _feature_values(sample_tuple, feature_b)
        spearman = _spearman(values_a, values_b)
        mi = _mutual_information(_discretize(values_a), _discretize(values_b))
        status = "high_redundancy" if abs(spearman) >= 0.90 else "ok"
        feature_redundancy_rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "spearman_corr": spearman,
                "mutual_information": mi,
                "status": status,
            }
        )

    feature_mi = {
        str(row["feature"]): float(row["mi_with_class"]) for row in feature_relevance_rows
    }
    feature_synergy_rows: list[dict[str, object]] = []
    for feature_a, feature_b in combinations(feature_names, 2):
        values_a = _feature_values(sample_tuple, feature_a)
        values_b = _feature_values(sample_tuple, feature_b)
        joint_mi = _joint_mutual_information(values_a, values_b, labels)
        best_single = max(feature_mi[feature_a], feature_mi[feature_b])
        pair_gain = joint_mi - best_single
        status = "synergy_candidate" if pair_gain >= 0.05 and best_single < joint_mi else "ordinary"
        feature_synergy_rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "joint_mutual_information": joint_mi,
                "best_single_feature_mi": best_single,
                "pair_gain": pair_gain,
                "conditional_gain_proxy": pair_gain,
                "status": status,
            }
        )

    prior_pathology_rows: list[dict[str, object]] = []
    for class_a, class_b in combinations(class_names, 2):
        mean_a, cov_a = class_stats[class_a]
        mean_b, cov_b = class_stats[class_b]
        vectors = [
            np.asarray([float(sample.feature_values.get(feature, 0.0)) for feature in feature_names])
            for sample in sample_tuple
            if sample.true_class in {class_a, class_b}
        ]
        log_lrs = [
            _log_gaussian_pdf(vector, mean_a, cov_a) - _log_gaussian_pdf(vector, mean_b, cov_b)
            for vector in vectors
        ]
        prior_odds_log = _safe_log(normalized_priors[class_a]) - _safe_log(normalized_priors[class_b])
        threshold = -prior_odds_log
        observed_min = min(log_lrs) if log_lrs else 0.0
        observed_max = max(log_lrs) if log_lrs else 0.0
        flip_possible = observed_min <= threshold <= observed_max
        collapse_count = 0
        for vector in vectors:
            posterior = _normalize_log_scores(
                {
                    class_a: _log_gaussian_pdf(vector, mean_a, cov_a)
                    + _safe_log(normalized_priors[class_a]),
                    class_b: _log_gaussian_pdf(vector, mean_b, cov_b)
                    + _safe_log(normalized_priors[class_b]),
                }
            )
            if max(posterior.values()) >= 0.95:
                collapse_count += 1
        collapse_rate = collapse_count / max(len(vectors), 1)
        evidence_margin = max(abs(observed_min - threshold), abs(observed_max - threshold))
        pathology_flag = "pass"
        if not flip_possible and abs(prior_odds_log) >= 1.5:
            pathology_flag = "prior_domination"
        elif collapse_rate >= 0.85:
            pathology_flag = "posterior_collapse_risk"
        prior_pathology_rows.append(
            {
                "class_a": class_a,
                "class_b": class_b,
                "prior_odds_log": prior_odds_log,
                "observed_log_lr_min": observed_min,
                "observed_log_lr_max": observed_max,
                "flip_threshold_log_lr": threshold,
                "flip_possible": flip_possible,
                "evidence_margin": evidence_margin,
                "posterior_collapse_rate": collapse_rate,
                "pathology_flag": pathology_flag,
            }
        )

    coverage_rows: list[dict[str, object]] = []
    class_counts = Counter(labels)
    for class_name in class_names:
        class_samples = tuple(sample for sample in sample_tuple if sample.true_class == class_name)
        for feature_name in feature_names:
            values = _feature_values(class_samples, feature_name)
            occupied_bins = len(set(_discretize(values, bins=5)))
            coverage_rows.append(
                {
                    "class_name": class_name,
                    "feature": feature_name,
                    "sample_count": class_counts[class_name],
                    "occupied_bins": occupied_bins,
                    "empty_bin_rate": 1.0 - occupied_bins / 5.0,
                    "min_value": min(values) if values else 0.0,
                    "max_value": max(values) if values else 0.0,
                    "status": "low_count" if class_counts[class_name] < 3 else "pass",
                }
            )

    schema_by_name = {entry.name: entry for entry in feature_schema}
    leakage_rows: list[dict[str, object]] = []
    for feature_name in feature_names:
        entry = schema_by_name.get(feature_name, StaticAuditFeatureSchemaEntry(name=feature_name))
        tags = set(entry.provenance_tags)
        has_future_dependency = "future" in tags or "future_dependency" in tags
        has_metadata_leakage = "metadata" in tags or "generator_metadata" in tags
        status = "pass"
        if entry.label_rule_overlap or has_future_dependency or has_metadata_leakage:
            status = "blocker"
        elif not entry.online_available:
            status = "warning"
        leakage_rows.append(
            {
                "feature": feature_name,
                "provenance_tags": "|".join(entry.provenance_tags),
                "online_available": entry.online_available,
                "label_rule_overlap_flag": entry.label_rule_overlap,
                "future_dependency_flag": has_future_dependency,
                "metadata_leakage_flag": has_metadata_leakage,
                "status": status,
            }
        )

    hard_pairs = [row for row in class_pair_rows if row["status"] == "hard"]
    weak_features = [row for row in feature_relevance_rows if row["recommended_status"] == "weak"]
    redundant_pairs = [row for row in feature_redundancy_rows if row["status"] == "high_redundancy"]
    synergy_candidates = [row for row in feature_synergy_rows if row["status"] == "synergy_candidate"]
    prior_blockers = [row for row in prior_pathology_rows if row["pathology_flag"] == "prior_domination"]
    leakage_blockers = [row for row in leakage_rows if row["status"] == "blocker"]
    low_count_rows = [row for row in coverage_rows if row["status"] == "low_count"]

    blockers: list[str] = []
    warnings: list[str] = []
    next_work: list[str] = []
    status = "promote_to_corpus_explorer"
    adequacy_label = "sufficient_for_corpus_search"

    if leakage_blockers:
        status = "reject"
        adequacy_label = "insufficient_due_to_leakage_risk"
        blockers.append("leakage blocker in static feature provenance")
        next_work.append("remove label-rule, future-dependent, or generator-metadata features")
    elif hard_pairs:
        status = "revise_class_set"
        adequacy_label = "insufficient_due_to_class_overlap"
        blockers.append("one or more class pairs are hard under the proposed feature set")
        next_work.append("tighten class boundaries or add targeted separating features")
    elif len(weak_features) == len(feature_relevance_rows):
        status = "revise_feature_set"
        adequacy_label = "insufficient_due_to_feature_blindness"
        blockers.append("all proposed features have weak class relevance")
        next_work.append("add class-relevant kinematic or residual features")
    elif prior_blockers:
        status = "revise_prior"
        adequacy_label = "insufficient_due_to_prior_domination"
        blockers.append("prior odds dominate at least one class-pair likelihood range")
        next_work.append("sweep priors or require stronger pairwise evidence")

    if redundant_pairs:
        warnings.append("high feature redundancy cluster detected")
        next_work.append("cluster redundant features before classifier escalation")
    if synergy_candidates:
        warnings.append("weak individual features may carry joint class evidence")
        next_work.append("preserve synergy candidates for pairwise feature tests")
    if low_count_rows:
        warnings.append("one or more class-feature cells have low sample count")
        next_work.append("expand controlled witness coverage before broad corpus search")
    if not warnings and not blockers:
        next_work.append("promote to corpus explorer with the audited feature/class/prior regime")

    hardest_pair = min(class_pair_rows, key=lambda row: float(row["pairwise_auc"]))
    weakest_feature = min(feature_relevance_rows, key=lambda row: float(row["mi_with_class"]))
    worst_prior = max(prior_pathology_rows, key=lambda row: abs(float(row["prior_odds_log"])))
    decision_card_rows = (
        {
            "lane": "class separability",
            "score": min(float(row["pairwise_auc"]) for row in class_pair_rows),
            "hardest_pair_or_feature": f"{hardest_pair['class_a']} vs {hardest_pair['class_b']}",
            "status": "warning" if hard_pairs else "pass",
            "next_action": "revise boundary" if hard_pairs else "promote",
        },
        {
            "lane": "feature relevance",
            "score": max(float(row["mi_with_class"]) for row in feature_relevance_rows),
            "hardest_pair_or_feature": str(weakest_feature["feature"]),
            "status": "warning" if weak_features else "pass",
            "next_action": "add features" if weak_features else "keep",
        },
        {
            "lane": "feature redundancy",
            "score": len(redundant_pairs),
            "hardest_pair_or_feature": str(redundant_pairs[0]["feature_a"]) if redundant_pairs else "none",
            "status": "warning" if redundant_pairs else "pass",
            "next_action": "cluster" if redundant_pairs else "promote",
        },
        {
            "lane": "feature synergy",
            "score": len(synergy_candidates),
            "hardest_pair_or_feature": str(synergy_candidates[0]["feature_a"]) if synergy_candidates else "none",
            "status": "candidate" if synergy_candidates else "pass",
            "next_action": "test pair" if synergy_candidates else "promote",
        },
        {
            "lane": "prior pathology",
            "score": abs(float(worst_prior["prior_odds_log"])),
            "hardest_pair_or_feature": f"{worst_prior['class_a']} vs {worst_prior['class_b']}",
            "status": "warning" if prior_blockers else "pass",
            "next_action": "sweep prior" if prior_blockers else "promote",
        },
        {
            "lane": "coverage feasibility",
            "score": min(class_counts.values()),
            "hardest_pair_or_feature": "minimum class count",
            "status": "warning" if low_count_rows else "pass",
            "next_action": "expand witness cells" if low_count_rows else "promote",
        },
        {
            "lane": "leakage risk",
            "score": len(leakage_blockers),
            "hardest_pair_or_feature": str(leakage_blockers[0]["feature"]) if leakage_blockers else "none",
            "status": "blocker" if leakage_blockers else "pass",
            "next_action": "remove feature" if leakage_blockers else "promote",
        },
        {
            "lane": "decisionability",
            "score": 0 if blockers else 1,
            "hardest_pair_or_feature": adequacy_label,
            "status": status,
            "next_action": next_work[0] if next_work else "promote",
        },
    )

    return StaticFeatureClassPriorAuditResult(
        study_name=study_name,
        feature_names=feature_names,
        class_names=class_names,
        priors=normalized_priors,
        class_pair_rows=tuple(class_pair_rows),
        feature_relevance_rows=tuple(feature_relevance_rows),
        feature_redundancy_rows=tuple(feature_redundancy_rows),
        feature_synergy_rows=tuple(feature_synergy_rows),
        prior_pathology_rows=tuple(prior_pathology_rows),
        coverage_rows=tuple(coverage_rows),
        leakage_rows=tuple(leakage_rows),
        decision_card_rows=decision_card_rows,
        static_decision={
            "status": status,
            "adequacy_label": adequacy_label,
            "blockers": tuple(blockers),
            "warnings": tuple(warnings),
            "next_work": tuple(dict.fromkeys(next_work)),
        },
    )

__all__ = [
    "StaticAuditFeatureSchemaEntry",
    "StaticAuditSample",
    "StaticFeatureClassPriorAuditResult",
    "analyze_static_feature_class_prior_audit",
    "render_static_feature_class_prior_audit_report",
]
