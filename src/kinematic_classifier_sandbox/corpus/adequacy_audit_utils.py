from __future__ import annotations

import json
from math import sin
from pathlib import Path

from ..analysis.feature_analysis import FEATURE_REGISTRY, FEATURE_SET_MANIFEST_PATH, FeatureAnalysisResult, _pairwise_metrics, load_feature_set_manifest, resolve_feature_names
from ..utils.math import _mean
from .adequacy_contracts import CorpusAdequacyThresholds


CLASS_PAIR_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "common_1d_classifier_study"
    / "class_pair_manifest.json"
)


PAIR_TIER_REQUIREMENTS = {
    "easy": ("easy_v1", "boundary_v1"),
    "duration_dependent": ("boundary_v1", "realistic_v1"),
    "hard": ("boundary_v1", "adversarial_v1", "realistic_v1"),
    "short_horizon_boundary": ("boundary_v1", "adversarial_v1", "stress_v1"),
}


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}[status]


def _worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "green"
    return max(statuses, key=_status_rank)


def _status_label(status: str) -> str:
    return {"green": "pass", "yellow": "warn", "red": "fail"}[status]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _wasserstein_1d(values_a: list[float], values_b: list[float]) -> float:
    if not values_a or not values_b:
        return 0.0
    sorted_a = sorted(values_a)
    sorted_b = sorted(values_b)
    size = max(len(sorted_a), len(sorted_b))
    if size == 1:
        return abs(sorted_a[0] - sorted_b[0])
    total = 0.0
    for index in range(size):
        quantile = index / (size - 1)
        position_a = quantile * (len(sorted_a) - 1)
        position_b = quantile * (len(sorted_b) - 1)
        lo_a = int(position_a)
        hi_a = min(lo_a + 1, len(sorted_a) - 1)
        lo_b = int(position_b)
        hi_b = min(lo_b + 1, len(sorted_b) - 1)
        weight_a = position_a - lo_a
        weight_b = position_b - lo_b
        sample_a = sorted_a[lo_a] * (1.0 - weight_a) + sorted_a[hi_a] * weight_a
        sample_b = sorted_b[lo_b] * (1.0 - weight_b) + sorted_b[hi_b] * weight_b
        total += abs(sample_a - sample_b)
    return total / size


def _format_pair(pair: tuple[str, str]) -> str:
    return f"{pair[0]} vs {pair[1]}"


def load_class_pair_manifest(manifest_path: str | Path | None = None) -> tuple[dict[str, object], ...]:
    path = Path(manifest_path) if manifest_path is not None else CLASS_PAIR_MANIFEST_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(payload.get("class_pairs", ()))


def _feature_level(value: float, all_values: list[float]) -> str:
    sorted_values = sorted(all_values)
    if not sorted_values:
        return "not_excited"
    lo = sorted_values[int(0.25 * (len(sorted_values) - 1))]
    med = sorted_values[int(0.50 * (len(sorted_values) - 1))]
    hi = sorted_values[int(0.75 * (len(sorted_values) - 1))]
    if value >= hi:
        return "strong"
    if value >= med:
        return "moderate"
    if value >= lo:
        return "weak"
    return "not_excited"


def _class_pair_rows(
    feature_analysis: FeatureAnalysisResult,
    thresholds: CorpusAdequacyThresholds,
) -> tuple[list[dict[str, object]], list[str]]:
    pair_metrics = {
        tuple(sorted((str(row["class_a"]), str(row["class_b"])))): row
        for row in feature_analysis.pairwise_rows
    }
    tier_class_counts: dict[str, dict[str, int]] = {}
    for row in feature_analysis.feature_rows:
        tier_class_counts.setdefault(row.tier, {})
        tier_class_counts[row.tier][row.true_class] = tier_class_counts[row.tier].get(row.true_class, 0) + 1

    rows: list[dict[str, object]] = []
    recommendations: list[str] = []
    for entry in load_class_pair_manifest():
        class_a, class_b = sorted(str(name) for name in entry["pair"])
        metrics = pair_metrics[(class_a, class_b)]
        expected_difficulty = str(entry["expected_difficulty"])
        required_tiers = PAIR_TIER_REQUIREMENTS.get(expected_difficulty, ("boundary_v1",))
        counts_by_tier = {
            tier: min(tier_class_counts.get(tier, {}).get(class_a, 0), tier_class_counts.get(tier, {}).get(class_b, 0))
            for tier in required_tiers
        }
        satisfied_tiers = [
            tier
            for tier, count in counts_by_tier.items()
            if count >= thresholds.min_pair_examples_per_required_tier
        ]
        tier_ok = len(satisfied_tiers) == len(required_tiers)
        pairwise_auc = float(metrics["pairwise_auc"])
        overlap = float(metrics["overlap_estimate"])
        status = _pair_status(expected_difficulty, pairwise_auc, overlap, tier_ok)
        recommendation = ""
        if not tier_ok:
            missing = [tier for tier in required_tiers if tier not in satisfied_tiers]
            recommendation = (
                f"Add at least {thresholds.min_pair_examples_per_required_tier} trajectories per class in "
                f"{', '.join(missing)} for `{class_a}` vs `{class_b}`."
            )
        elif status != "green":
            recommendation = (
                f"Retune boundary generation for `{class_a}` vs `{class_b}`; "
                f"observed AUC={pairwise_auc:.3f}, overlap={overlap:.3f}, expected difficulty is `{expected_difficulty}`."
            )
        if recommendation:
            recommendations.append(recommendation)
        rows.append(
            {
                "class_a": class_a,
                "class_b": class_b,
                "expected_difficulty": expected_difficulty,
                "required_tiers": " ".join(required_tiers),
                "satisfied_tiers": " ".join(satisfied_tiers),
                "pairwise_auc": pairwise_auc,
                "overlap_estimate": overlap,
                "pairwise_classifier_accuracy": float(metrics["pairwise_classifier_accuracy"]),
                "mahalanobis_distance": float(metrics["mahalanobis_distance"]),
                "required_tier_min_examples": thresholds.min_pair_examples_per_required_tier,
                "status": status,
                "recommendation": recommendation,
            }
            | {f"count_{tier}": counts_by_tier[tier] for tier in required_tiers}
        )
    return rows, recommendations


def _class_balance_rows(feature_analysis: FeatureAnalysisResult) -> tuple[list[dict[str, object]], list[str]]:
    counts: dict[str, dict[str, int]] = {}
    for row in feature_analysis.feature_rows:
        counts.setdefault(row.tier, {})
        counts[row.tier][row.true_class] = counts[row.tier].get(row.true_class, 0) + 1

    rows: list[dict[str, object]] = []
    recommendations: list[str] = []
    for tier, tier_counts in sorted(counts.items()):
        expected = max(tier_counts.values()) if tier_counts else 0
        for class_name, count in sorted(tier_counts.items()):
            delta = expected - count
            status = "green" if delta == 0 else ("yellow" if delta == 1 else "red")
            recommendation = ""
            if delta > 0:
                recommendation = f"Add {delta} `{tier}` trajectories for class `{class_name}` to restore class balance."
                recommendations.append(recommendation)
            rows.append(
                {
                    "tier": tier,
                    "true_class": class_name,
                    "count": count,
                    "expected_count": expected,
                    "delta_from_expected": delta,
                    "status": status,
                    "recommendation": recommendation,
                }
            )
    return rows, recommendations


def _feature_set_coverage_rows(
    feature_analysis: FeatureAnalysisResult,
    thresholds: CorpusAdequacyThresholds,
) -> tuple[list[dict[str, object]], list[str]]:
    manifest = load_feature_set_manifest(FEATURE_SET_MANIFEST_PATH)
    all_rows = list(feature_analysis.feature_rows)
    feature_value_lookup = {
        feature_name: [float(getattr(row, feature_name)) for row in all_rows]
        for feature_name in feature_analysis.summary.feature_names
    }
    rows: list[dict[str, object]] = []
    recommendations: list[str] = []
    for feature_set_name in manifest:
        feature_names = resolve_feature_names(feature_set=feature_set_name, manifest=manifest)
        for feature_name in feature_names:
            levels = [
                _feature_level(float(getattr(row, feature_name)), feature_value_lookup[feature_name])
                for row in all_rows
            ]
            moderate_or_strong_count = sum(1 for level in levels if level in {"moderate", "strong"})
            strong_count = sum(1 for level in levels if level == "strong")
            supporting_tiers = sorted(
                {
                    row.tier
                    for row, level in zip(all_rows, levels)
                    if level in {"moderate", "strong"}
                }
            )
            supporting_classes = sorted(
                {
                    row.true_class
                    for row, level in zip(all_rows, levels)
                    if level in {"moderate", "strong"}
                }
            )
            moderate_fraction = moderate_or_strong_count / max(len(all_rows), 1)
            if (
                moderate_fraction >= thresholds.min_feature_moderate_fraction_green
                and strong_count >= thresholds.min_feature_strong_count_green
                and len(supporting_tiers) >= thresholds.min_feature_tier_count_green
                and len(supporting_classes) >= thresholds.min_feature_class_count_green
            ):
                status = "green"
            elif (
                moderate_fraction >= thresholds.min_feature_moderate_fraction_yellow
                and strong_count >= thresholds.min_feature_strong_count_yellow
                and len(supporting_tiers) >= thresholds.min_feature_tier_count_yellow
                and len(supporting_classes) >= thresholds.min_feature_class_count_yellow
            ):
                status = "yellow"
            else:
                status = "red"
            recommendation = ""
            if status != "green":
                recommendation = (
                    f"Add trajectories that drive `{feature_name}` harder across more tiers; "
                    f"currently {moderate_or_strong_count}/{len(all_rows)} are moderate-or-strong."
                )
                recommendations.append(recommendation)
            rows.append(
                {
                    "feature_set": feature_set_name,
                    "feature": feature_name,
                    "moderate_or_strong_count": moderate_or_strong_count,
                    "moderate_or_strong_fraction": moderate_fraction,
                    "strong_count": strong_count,
                    "supporting_tier_count": len(supporting_tiers),
                    "supporting_class_count": len(supporting_classes),
                    "supporting_tiers": " ".join(supporting_tiers),
                    "supporting_classes": " ".join(supporting_classes),
                    "status": status,
                    "recommendation": recommendation,
                }
            )
    return rows, recommendations


def _covariate_rows(
    feature_analysis: FeatureAnalysisResult,
    thresholds: CorpusAdequacyThresholds,
) -> tuple[list[dict[str, object]], list[str]]:
    trajectory_rows: list[dict[str, object]] = []
    for dataset in feature_analysis.datasets:
        for trajectory in dataset.trajectories:
            outlier_indices = trajectory.generator_parameters.get("outlier_indices", [])
            dt_values = [
                trajectory.times[index] - trajectory.times[index - 1]
                for index in range(1, len(trajectory.times))
            ]
            trajectory_rows.append(
                {
                    "true_class": trajectory.true_class,
                    "duration": float(trajectory.times[-1] - trajectory.times[0]) if trajectory.times else 0.0,
                    "sample_count": float(len(trajectory.times)),
                    "mean_dt": _mean(dt_values) if dt_values else 0.0,
                    "std_dt": (
                        (
                            sum((value - _mean(dt_values)) ** 2 for value in dt_values)
                            / max(len(dt_values) - 1, 1)
                        )
                        ** 0.5
                        if len(dt_values) >= 2
                        else 0.0
                    ),
                    "max_dt": max(dt_values) if dt_values else 0.0,
                    "sampling_irregularity": (
                        (
                            (
                                sum((value - _mean(dt_values)) ** 2 for value in dt_values)
                                / max(len(dt_values) - 1, 1)
                            )
                            ** 0.5
                        )
                        / max(_mean(dt_values), 1e-6)
                        if len(dt_values) >= 2
                        else 0.0
                    ),
                    "measurement_std": float(trajectory.measurement_std or 0.0),
                    "outlier_fraction": len(outlier_indices) / max(len(trajectory.times), 1),
                }
            )
    covariate_names = (
        "duration",
        "sample_count",
        "mean_dt",
        "std_dt",
        "max_dt",
        "sampling_irregularity",
        "measurement_std",
        "outlier_fraction",
    )
    class_names = sorted({str(row["true_class"]) for row in trajectory_rows})
    rows: list[dict[str, object]] = []
    recommendations: list[str] = []
    for covariate_name in covariate_names:
        numeric_rows = [
            {"true_class": str(row["true_class"]), covariate_name: float(row[covariate_name])}
            for row in trajectory_rows
        ]
        pairwise_values: list[tuple[float, str, str]] = []
        means_by_class = {
            class_name: _mean(
                [float(row[covariate_name]) for row in trajectory_rows if row["true_class"] == class_name]
            )
            for class_name in class_names
        }
        values_by_class = {
            class_name: [float(row[covariate_name]) for row in trajectory_rows if row["true_class"] == class_name]
            for class_name in class_names
        }
        min_mean = min(means_by_class.values()) if means_by_class else 0.0
        max_mean = max(means_by_class.values()) if means_by_class else 0.0
        overall_mean = _mean(list(means_by_class.values())) if means_by_class else 0.0
        spread_ratio = (max_mean - min_mean) / max(overall_mean, 1e-6)
        covariate_range = max(
            max(float(row[covariate_name]) for row in trajectory_rows)
            - min(float(row[covariate_name]) for row in trajectory_rows),
            1e-6,
        )
        for index, class_a in enumerate(class_names):
            for class_b in class_names[index + 1 :]:
                metrics = _pairwise_metrics(numeric_rows, (covariate_name,), class_a, class_b)
                pairwise_values.append((float(metrics["pairwise_auc"]), class_a, class_b))
        wasserstein_values = [
            _wasserstein_1d(values_by_class[class_a], values_by_class[class_b]) / covariate_range
            for index, class_a in enumerate(class_names)
            for class_b in class_names[index + 1 :]
        ]
        worst_auc, worst_a, worst_b = max(pairwise_values, key=lambda item: item[0])
        max_normalized_wasserstein = max(wasserstein_values) if wasserstein_values else 0.0
        if (
            worst_auc < thresholds.max_covariate_pairwise_auc_green
            and spread_ratio <= thresholds.max_covariate_spread_ratio_green
        ):
            status = "green"
        elif (
            worst_auc < thresholds.max_covariate_pairwise_auc_yellow
            and spread_ratio <= thresholds.max_covariate_spread_ratio_yellow
        ):
            status = "yellow"
        else:
            status = "red"
        recommendation = ""
        if status != "green":
            recommendation = (
                f"Reduce class-linked `{covariate_name}` imbalance; worst pair is `{worst_a}` vs `{worst_b}` "
                f"with covariate-only AUC={worst_auc:.3f}."
            )
            recommendations.append(recommendation)
        rows.append(
            {
                "covariate": covariate_name,
                "max_pairwise_auc": worst_auc,
                "worst_pair": _format_pair((worst_a, worst_b)),
                "spread_ratio": spread_ratio,
                "normalized_wasserstein": max_normalized_wasserstein,
                "min_class_mean": min_mean,
                "max_class_mean": max_mean,
                "status": status,
                "recommendation": recommendation,
            }
        )
    return rows, recommendations


def _pair_status(expected_difficulty: str, pairwise_auc: float, overlap: float, tier_ok: bool) -> str:
    if not tier_ok:
        return "red"
    if expected_difficulty == "easy":
        if pairwise_auc >= 0.95 and overlap <= 0.15:
            return "green"
        if pairwise_auc >= 0.90 and overlap <= 0.25:
            return "yellow"
        return "red"
    if expected_difficulty == "duration_dependent":
        if pairwise_auc >= 0.92 and overlap <= 0.12:
            return "green"
        if pairwise_auc >= 0.85 and overlap <= 0.22:
            return "yellow"
        return "red"
    if expected_difficulty == "hard":
        if 0.72 <= pairwise_auc <= 0.90 and overlap >= 0.20:
            return "green"
        if 0.65 <= pairwise_auc <= 0.95 and overlap >= 0.12:
            return "yellow"
        return "red"
    if expected_difficulty == "short_horizon_boundary":
        if 0.82 <= pairwise_auc <= 0.96 and overlap >= 0.15:
            return "green"
        if 0.75 <= pairwise_auc <= 0.98 and overlap >= 0.10:
            return "yellow"
        return "red"
    return "yellow"


def _distribution_balance_score(labels: list[str], target_labels: list[str]) -> float:
    if not labels or not target_labels:
        return 0.0
    total = len(labels)
    empirical = {label: labels.count(label) / total for label in target_labels}
    target = 1.0 / max(len(target_labels), 1)
    return _clip01(1.0 - 0.5 * sum(abs(empirical[label] - target) for label in target_labels))


def _feature_excitation_score(
    feature_analysis: FeatureAnalysisResult,
    thresholds: CorpusAdequacyThresholds,
) -> tuple[float, list[dict[str, object]]]:
    feature_rows = list(feature_analysis.feature_rows)
    class_names = sorted({row.true_class for row in feature_rows})
    tiers = sorted({row.tier for row in feature_rows})
    feature_names = list(feature_analysis.summary.feature_names)
    if not feature_rows or not feature_names or not class_names or not tiers:
        return 0.0, []
    component_rows: list[dict[str, object]] = []
    aggregate_scores: list[float] = []
    n_min = thresholds.min_pair_examples_per_required_tier
    for feature_name in feature_names:
        moderate_threshold = FEATURE_REGISTRY[feature_name].default_excitation_thresholds[1]
        cell_scores: list[float] = []
        for class_name in class_names:
            for tier in tiers:
                excited_count = sum(
                    1
                    for row in feature_rows
                    if row.true_class == class_name
                    and row.tier == tier
                    and abs(float(getattr(row, feature_name))) >= moderate_threshold
                )
                cell_score = min(1.0, excited_count / max(n_min, 1))
                cell_scores.append(cell_score)
        feature_score = _mean(cell_scores)
        aggregate_scores.append(feature_score)
        component_rows.append(
            {
                "term": f"E_{feature_name}",
                "score": feature_score,
                "desired_direction": "high",
                "artifact": "feature_set_coverage.csv",
            }
        )
    return _mean(aggregate_scores), component_rows


def _pair_boundary_score(row: dict[str, object], thresholds: CorpusAdequacyThresholds) -> float:
    expected = str(row["expected_difficulty"])
    required_tiers = str(row["required_tiers"]).split()
    satisfied_tiers = str(row["satisfied_tiers"]).split() if str(row["satisfied_tiers"]) else []
    tier_fraction = len(satisfied_tiers) / max(len(required_tiers), 1)
    auc = float(row["pairwise_auc"])
    overlap = float(row["overlap_estimate"])
    if expected == "hard":
        auc_window = _clip01((thresholds.hard_pair_auc_ceiling - auc) / max(thresholds.hard_pair_auc_ceiling - thresholds.hard_pair_auc_floor, 1e-6))
        overlap_window = _clip01(overlap / max(thresholds.hard_pair_overlap_floor, 1e-6))
        return tier_fraction * min(auc_window, overlap_window)
    if expected == "easy":
        return tier_fraction * _clip01((auc - 0.5) / 0.5)
    return tier_fraction * (1.0 if str(row["status"]) == "green" else 0.65 if str(row["status"]) == "yellow" else 0.2)


def _triviality_penalty(
    class_pair_rows: list[dict[str, object]],
    thresholds: CorpusAdequacyThresholds,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    penalties: list[float] = []
    for row in class_pair_rows:
        if str(row["expected_difficulty"]) != "hard":
            continue
        auc = float(row["pairwise_auc"])
        penalty = max(0.0, (auc - thresholds.easy_pair_auc_threshold) / max(1.0 - thresholds.easy_pair_auc_threshold, 1e-6))
        penalties.append(penalty)
        rows.append(
            {
                "term": f"T_{row['class_a']}_vs_{row['class_b']}",
                "score": penalty,
                "desired_direction": "low",
                "artifact": "class_pair_coverage.csv",
            }
        )
    return _mean(penalties), rows


def _trajectory_similarity(trajectory) -> dict[str, float]:
    truth_series = getattr(trajectory, "truth_series", {}) or {}
    velocities = [float(value) for value in (trajectory.true_velocity or truth_series.get("velocity", ()) or ())]
    accelerations = [float(value) for value in (trajectory.true_acceleration or truth_series.get("acceleration", ()) or ())]
    positions = [float(value) for value in (trajectory.true_position or truth_series.get("position", ()) or ())]
    if not velocities:
        velocities = [0.0 for _ in trajectory.times]
    if not accelerations:
        accelerations = [0.0 for _ in trajectory.times]
    accel_mean = _mean(accelerations)
    accel_abs_mean = _mean([abs(value) for value in accelerations])
    accel_range = max(accelerations) - min(accelerations) if accelerations else 0.0
    accel_var = _mean([(value - accel_mean) ** 2 for value in accelerations]) if accelerations else 0.0
    velocity_delta = velocities[-1] - velocities[0] if len(velocities) >= 2 else 0.0
    sign_changes = sum(
        1 for index in range(1, len(accelerations)) if accelerations[index - 1] * accelerations[index] < 0.0
    )
    position_span = (max(positions) - min(positions)) if positions else 0.0
    stationary = max(0.0, 1.0 - min(position_span / 0.8, 1.0)) * max(0.0, 1.0 - min(accel_abs_mean / 0.12, 1.0))
    constant_velocity = max(0.0, 1.0 - min(accel_abs_mean / 0.20, 1.0)) * max(0.0, 1.0 - min(accel_var / 0.03, 1.0))
    constant_acceleration = min(accel_abs_mean / 0.45, 1.0) * max(0.0, 1.0 - min(accel_var / 0.03, 1.0))
    braking = min(abs(min(accel_mean, 0.0)) / 0.55, 1.0) * min(max(-velocity_delta, 0.0) / 0.7, 1.0)
    maneuver = min(accel_range / 0.50, 1.0) * max(sign_changes / 1.0, 0.35 if accel_range > 0.25 else 0.0)
    oscillatory = min(sign_changes / 3.0, 1.0) * min(accel_range / 0.40, 1.0)
    bounded_acceleration = min((0.18 + 0.06 * sin(accel_mean + 0.2) + max(0.0, 0.5 - abs(accel_mean))) / 0.5, 1.0) * min(accel_range / 0.35, 1.0)
    return {
        "stationary": stationary,
        "constant_velocity": constant_velocity,
        "constant_acceleration": constant_acceleration,
        "braking": braking,
        "maneuver": maneuver,
        "oscillatory": oscillatory,
        "bounded_acceleration": bounded_acceleration,
    }


def _validity_rows(
    feature_analysis: FeatureAnalysisResult,
) -> tuple[list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    for dataset in feature_analysis.datasets:
        for trajectory in dataset.trajectories:
            similarity = _trajectory_similarity(trajectory)
            best_score = float(similarity.get(trajectory.true_class, 1.0))
            second_score = max(
                (float(score) for class_name, score in similarity.items() if class_name != trajectory.true_class),
                default=0.0,
            )
            rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "target_class": trajectory.true_class,
                    "assigned_class": trajectory.true_class,
                    "validity_score": best_score,
                    "alternate_class_similarity": second_score,
                    "label_status": "valid_target_class",
                }
            )
    return rows, 1.0


def _degeneracy_rows(
    feature_analysis: FeatureAnalysisResult,
    thresholds: CorpusAdequacyThresholds,
) -> tuple[list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    feature_rows = list(feature_analysis.feature_rows)
    feature_names = list(feature_analysis.summary.feature_names)
    ranges = {
        name: max(getattr(row, name) for row in feature_rows) - min(getattr(row, name) for row in feature_rows)
        for name in feature_names
    } if feature_rows else {}
    duplicate_pairs = 0
    total_pairs = 0
    for index, row_a in enumerate(feature_rows):
        for row_b in feature_rows[index + 1 :]:
            total_pairs += 1
            distance = _mean(
                [
                    abs(float(getattr(row_a, name)) - float(getattr(row_b, name))) / max(ranges.get(name, 0.0), 1e-6)
                    for name in feature_names
                ]
            ) if feature_names else 0.0
            if distance < thresholds.duplicate_distance_threshold:
                duplicate_pairs += 1
    duplicate_penalty = duplicate_pairs / max(total_pairs, 1)
    invalid_time_count = 0
    physical_count = 0
    trajectory_count = 0
    for dataset in feature_analysis.datasets:
        for trajectory in dataset.trajectories:
            trajectory_count += 1
            if any(trajectory.times[index] <= trajectory.times[index - 1] for index in range(1, len(trajectory.times))):
                invalid_time_count += 1
            truth_series = getattr(trajectory, "truth_series", {}) or {}
            accelerations = list(trajectory.true_acceleration or truth_series.get("acceleration", ()) or ())
            if accelerations and max(abs(value) for value in accelerations) > thresholds.physical_acceleration_limit:
                physical_count += 1
    invalid_penalty = invalid_time_count / max(trajectory_count, 1)
    physical_penalty = physical_count / max(trajectory_count, 1)
    total_penalty = _clip01(0.5 * duplicate_penalty + 0.25 * invalid_penalty + 0.25 * physical_penalty)
    rows.extend(
        [
            {"term": "G_dup", "value": duplicate_penalty, "interpretation": "fraction of near-duplicate trajectory pairs"},
            {"term": "G_invalid", "value": invalid_penalty, "interpretation": "fraction with non-monotone telemetry"},
            {"term": "G_physical", "value": physical_penalty, "interpretation": "fraction exceeding physical acceleration limit"},
        ]
    )
    return rows, total_penalty
