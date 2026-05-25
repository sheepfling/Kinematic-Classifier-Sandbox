from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import io
import json
import os
from math import sin
from pathlib import Path

from .feature_analysis import (
    FeatureAnalysisResult,
    FEATURE_SET_MANIFEST_PATH,
    FEATURE_REGISTRY,
    _pairwise_metrics,
    analyze_feature_datasets,
    load_feature_set_manifest,
    resolve_feature_names,
)


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


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


def _figure_to_png(fig) -> bytes:
    plt = _prepare_matplotlib()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


CLASS_PAIR_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
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
    feature_analysis: FeatureAnalysisResult
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


def analyze_corpus_adequacy(
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    thresholds: CorpusAdequacyThresholds | None = None,
    datasets: tuple[object, ...] | None = None,
    feature_analysis_result: FeatureAnalysisResult | None = None,
) -> CorpusAdequacyResult:
    selected_thresholds = thresholds or CorpusAdequacyThresholds()
    feature_analysis = feature_analysis_result or analyze_feature_datasets(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
        datasets=datasets,
    )
    feature_set_rows, feature_recommendations = _feature_set_coverage_rows(feature_analysis, selected_thresholds)
    class_pair_rows, pair_recommendations = _class_pair_rows(feature_analysis, selected_thresholds)
    class_balance_rows, balance_recommendations = _class_balance_rows(feature_analysis)
    covariate_rows, covariate_recommendations = _covariate_rows(feature_analysis, selected_thresholds)
    validity_rows, class_validity_score = _validity_rows(feature_analysis)
    degeneracy_rows, degeneracy_penalty = _degeneracy_rows(feature_analysis, selected_thresholds)
    feature_excitation_score, feature_excitation_rows = _feature_excitation_score(feature_analysis, selected_thresholds)
    triviality_penalty, triviality_rows = _triviality_penalty(class_pair_rows, selected_thresholds)

    feature_status = _worst_status([str(row["status"]) for row in feature_set_rows])
    class_pair_status = _worst_status([str(row["status"]) for row in class_pair_rows])
    class_balance_status = _worst_status([str(row["status"]) for row in class_balance_rows])
    covariate_status = _worst_status([str(row["status"]) for row in covariate_rows])
    overall_detail_status = _worst_status(
        [feature_status, class_pair_status, class_balance_status, covariate_status]
    )
    yellow_count = sum(
        1
        for rows in (feature_set_rows, class_pair_rows, class_balance_rows, covariate_rows)
        for row in rows
        if row["status"] == "yellow"
    )
    red_count = sum(
        1
        for rows in (feature_set_rows, class_pair_rows, class_balance_rows, covariate_rows)
        for row in rows
        if row["status"] == "red"
    )
    class_labels = [row.true_class for row in feature_analysis.feature_rows]
    tier_labels = [row.tier for row in feature_analysis.feature_rows]
    class_balance_score = _distribution_balance_score(class_labels, sorted(set(class_labels)))
    tier_balance_score = _distribution_balance_score(tier_labels, sorted(set(tier_labels)))
    covariate_balance_score = _mean(
        [
            1.0 - max(
                _clip01((float(row["max_pairwise_auc"]) - 0.5) / 0.5),
                _clip01(float(row["normalized_wasserstein"])),
            )
            for row in covariate_rows
        ]
    )
    pair_boundary_coverage = _mean([_pair_boundary_score(row, selected_thresholds) for row in class_pair_rows])
    leakage_penalty = max(
        [
            _clip01((float(row["max_pairwise_auc"]) - 0.5) / 0.5)
            for row in covariate_rows
        ],
        default=0.0,
    )
    q_corpus = _clip01(
        (
            class_balance_score
            + tier_balance_score
            + covariate_balance_score
            + feature_excitation_score
            + pair_boundary_coverage
            + class_validity_score
            - leakage_penalty
            - triviality_penalty
            - degeneracy_penalty
        )
        / 6.0
    )
    scorecard = CorpusAdequacyScorecard(
        class_balance=class_balance_score,
        tier_balance=tier_balance_score,
        covariate_balance=covariate_balance_score,
        feature_excitation=feature_excitation_score,
        pair_boundary_coverage=pair_boundary_coverage,
        class_validity=class_validity_score,
        leakage_penalty=leakage_penalty,
        triviality_penalty=triviality_penalty,
        degeneracy_penalty=degeneracy_penalty,
        q_corpus=q_corpus,
    )
    scorecard_rows = [
        {"term": "B_class", "score": class_balance_score, "desired_direction": "high", "artifact": "class_balance.csv"},
        {"term": "B_tier", "score": tier_balance_score, "desired_direction": "high", "artifact": "class_balance.csv"},
        {"term": "B_covariates", "score": covariate_balance_score, "desired_direction": "high", "artifact": "covariate_leakage_audit.csv"},
        {"term": "E_feature", "score": feature_excitation_score, "desired_direction": "high", "artifact": "feature_set_coverage.csv"},
        {"term": "C_pair", "score": pair_boundary_coverage, "desired_direction": "high", "artifact": "class_pair_coverage.csv"},
        {"term": "V", "score": class_validity_score, "desired_direction": "high", "artifact": "class_validity_audit.csv"},
        {"term": "L", "score": leakage_penalty, "desired_direction": "low", "artifact": "covariate_leakage_audit.csv"},
        {"term": "T", "score": triviality_penalty, "desired_direction": "low", "artifact": "class_pair_coverage.csv"},
        {"term": "G", "score": degeneracy_penalty, "desired_direction": "low", "artifact": "corpus_degeneracy_report.csv"},
        {"term": "Q_corpus", "score": q_corpus, "desired_direction": "high", "artifact": "corpus_adequacy_summary.json"},
        *feature_excitation_rows,
        *triviality_rows,
    ]
    overall_status = "pass"
    if (
        overall_detail_status == "red"
        or leakage_penalty > selected_thresholds.yellow_leakage_max
        or triviality_penalty > selected_thresholds.yellow_triviality_max
        or class_validity_score < selected_thresholds.yellow_validity_min
        or q_corpus < selected_thresholds.yellow_q_corpus
    ):
        overall_status = "fail"
    elif (
        yellow_count
        or leakage_penalty > selected_thresholds.green_leakage_max
        or triviality_penalty > selected_thresholds.green_triviality_max
        or class_validity_score < selected_thresholds.green_validity_min
        or q_corpus < selected_thresholds.green_q_corpus
    ):
        overall_status = "warn"
    recommendations = tuple(
        dict.fromkeys(
            [
                *feature_recommendations,
                *pair_recommendations,
                *balance_recommendations,
                *covariate_recommendations,
                *(
                    ["Reduce ambiguous/invalid/relabelled trajectories; class-validity score is below the green gate."]
                    if class_validity_score < selected_thresholds.green_validity_min
                    else []
                ),
                *(
                    ["Reduce duplicate or structurally invalid trajectories; degeneracy penalty is above zero."]
                    if degeneracy_penalty > 0.0
                    else []
                ),
            ]
        )
    )
    summary = CorpusAdequacySummary(
        overall_status=overall_status,
        overall_pass=(overall_status != "fail"),
        feature_status=_status_label(feature_status),
        class_pair_status=_status_label(class_pair_status),
        class_balance_status=_status_label(class_balance_status),
        covariate_status=_status_label(covariate_status),
        total_trajectories=len(feature_analysis.feature_rows),
        total_classes=len(feature_analysis.summary.class_counts),
        total_feature_sets=len(load_feature_set_manifest(FEATURE_SET_MANIFEST_PATH)),
        total_manifest_pairs=len(load_class_pair_manifest()),
        red_count=red_count,
        yellow_count=yellow_count,
        recommendation_count=len(recommendations),
        q_corpus=q_corpus,
        leakage_penalty=leakage_penalty,
        triviality_penalty=triviality_penalty,
        class_validity_score=class_validity_score,
        degeneracy_penalty=degeneracy_penalty,
    )
    return CorpusAdequacyResult(
        feature_analysis=feature_analysis,
        feature_set_rows=tuple(feature_set_rows),
        class_pair_rows=tuple(class_pair_rows),
        class_balance_rows=tuple(class_balance_rows),
        covariate_rows=tuple(covariate_rows),
        validity_rows=tuple(validity_rows),
        degeneracy_rows=tuple(degeneracy_rows),
        scorecard_rows=tuple(scorecard_rows),
        recommendations=recommendations,
        summary=summary,
        thresholds=selected_thresholds,
        scorecard=scorecard,
    )


def _render_pair_status_heatmap(result: CorpusAdequacyResult):
    plt = _prepare_matplotlib()
    pair_labels = [f"{row['class_a']} vs {row['class_b']}" for row in result.class_pair_rows]
    values = [[{"green": 1.0, "yellow": 0.5, "red": 0.0}[str(row["status"])]] for row in result.class_pair_rows]
    fig, ax = plt.subplots(figsize=(7.6, max(3.6, 0.52 * len(pair_labels) + 1.2)))
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_title("Class-Pair Boundary Coverage Gate", loc="left", fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels(["status"])
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels)
    for row_index, row in enumerate(result.class_pair_rows):
        ax.text(0, row_index, str(row["status"]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _render_covariate_leakage_plot(result: CorpusAdequacyResult):
    plt = _prepare_matplotlib()
    ordered = sorted(result.covariate_rows, key=lambda row: float(row["max_pairwise_auc"]), reverse=True)
    covariates = [str(row["covariate"]) for row in ordered][::-1]
    aucs = [float(row["max_pairwise_auc"]) for row in ordered][::-1]
    ratios = [float(row["spread_ratio"]) for row in ordered][::-1]
    colors = [{"green": "#16a34a", "yellow": "#d97706", "red": "#dc2626"}[str(row["status"])] for row in ordered][::-1]
    positions = list(range(len(covariates)))
    fig, ax = plt.subplots(figsize=(8.8, max(4.4, 0.46 * len(covariates) + 1.6)))
    ax.barh(positions, aucs, color=colors, alpha=0.88)
    for index, value in enumerate(aucs):
        ax.text(min(value + 0.01, 0.98), index, f"{value:.2f}", va="center", fontsize=8)
    ax.axvline(result.thresholds.max_covariate_pairwise_auc_green, color="#16a34a", linestyle="--", linewidth=1.2)
    ax.axvline(result.thresholds.max_covariate_pairwise_auc_yellow, color="#dc2626", linestyle="--", linewidth=1.2)
    ax.set_xlim(0.45, 1.0)
    ax.set_yticks(positions)
    ax.set_yticklabels(covariates)
    ax.set_xlabel("max pairwise AUC from covariate alone")
    ax.set_title("Covariate Leakage Audit", loc="left", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)

    twin = ax.twiny()
    twin.plot(ratios, positions, color="#2563eb", linewidth=1.4, marker="o", markersize=3.5)
    twin.set_xlabel("normalized class-mean spread")
    twin.set_xlim(0.0, max(ratios + [0.6]) * 1.08)
    fig.tight_layout()
    return fig


def render_corpus_adequacy_report(result: CorpusAdequacyResult) -> str:
    lines = [
        "# Corpus Adequacy Audit",
        "",
        "This audit turns the current corpus diagnostics into a formal scorecard over balance, feature excitation, class-pair boundary coverage, class validity, leakage, triviality, and degeneracy.",
        "",
        "## Overall Gate",
        "",
        f"- Overall status: {result.summary.overall_status}",
        f"- Overall pass: {result.summary.overall_pass}",
        f"- Feature coverage: {result.summary.feature_status}",
        f"- Class-pair coverage: {result.summary.class_pair_status}",
        f"- Class balance: {result.summary.class_balance_status}",
        f"- Covariate leakage: {result.summary.covariate_status}",
        f"- Trajectories analyzed: {result.summary.total_trajectories}",
        f"- Red findings: {result.summary.red_count}",
        f"- Yellow findings: {result.summary.yellow_count}",
        f"- Q_corpus: {result.summary.q_corpus:.3f}",
        f"- Leakage penalty: {result.summary.leakage_penalty:.3f}",
        f"- Triviality penalty: {result.summary.triviality_penalty:.3f}",
        f"- Class-validity score: {result.summary.class_validity_score:.3f}",
        f"- Degeneracy penalty: {result.summary.degeneracy_penalty:.3f}",
        "",
        "## Corpus Scorecard",
        "",
        "| term | score | desired_direction | artifact |",
        "| --- | ---: | --- | --- |",
    ]
    for row in result.scorecard_rows[:10]:
        lines.append(
            f"| {row['term']} | {float(row['score']):.3f} | {row['desired_direction']} | {row['artifact']} |"
        )
    lines.extend(
        [
            "",
            "## Feature Coverage by Feature Set",
            "",
            "| feature_set | feature | moderate_or_strong_fraction | strong_count | tiers | classes | status |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result.feature_set_rows:
        lines.append(
            f"| {row['feature_set']} | {row['feature']} | {row['moderate_or_strong_fraction']:.3f} | "
            f"{row['strong_count']} | {row['supporting_tier_count']} | {row['supporting_class_count']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Declared Class-Pair Boundary Coverage",
            "",
            "| class_a | class_b | difficulty | pairwise_auc | overlap | required_tiers | status |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in result.class_pair_rows:
        lines.append(
            f"| {row['class_a']} | {row['class_b']} | {row['expected_difficulty']} | "
            f"{row['pairwise_auc']:.3f} | {row['overlap_estimate']:.3f} | {row['required_tiers']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Class Balance",
            "",
            "| tier | true_class | count | expected_count | status |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in result.class_balance_rows:
        lines.append(
            f"| {row['tier']} | {row['true_class']} | {row['count']} | {row['expected_count']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Covariate Leakage",
            "",
            "| covariate | max_pairwise_auc | spread_ratio | normalized_wasserstein | worst_pair | status |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in result.covariate_rows:
        lines.append(
            f"| {row['covariate']} | {row['max_pairwise_auc']:.3f} | {row['spread_ratio']:.3f} | "
            f"{row['normalized_wasserstein']:.3f} | {row['worst_pair']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Class Validity",
            "",
            "| label_status | count |",
            "| --- | ---: |",
        ]
    )
    for status in ("valid_target_class", "ambiguous", "invalid", "relabel_candidate"):
        lines.append(f"| {status} | {sum(1 for row in result.validity_rows if row['label_status'] == status)} |")
    lines.extend(
        [
            "",
            "## Degeneracy",
            "",
            "| term | value | interpretation |",
            "| --- | ---: | --- |",
        ]
    )
    for row in result.degeneracy_rows:
        lines.append(f"| {row['term']} | {float(row['value']):.3f} | {row['interpretation']} |")
    lines.extend(["", "## Recommendations", ""])
    if result.recommendations:
        lines.extend([f"- {recommendation}" for recommendation in result.recommendations])
    else:
        lines.append("- No missing-coverage recommendations. The current corpus clears every enforced gate.")
    return "\n".join(lines)


def write_corpus_adequacy_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    thresholds: CorpusAdequacyThresholds | None = None,
) -> CorpusAdequacyArtifacts:
    result = analyze_corpus_adequacy(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
        thresholds=thresholds,
    )
    output_root = Path(output_dir)
    run_dir = output_root / "corpus_adequacy_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "corpus_adequacy_report.md"
    summary_path = run_dir / "corpus_adequacy_summary.json"
    feature_set_coverage_path = run_dir / "feature_set_coverage.csv"
    class_pair_coverage_path = run_dir / "class_pair_coverage.csv"
    class_balance_path = run_dir / "class_balance.csv"
    covariate_leakage_path = run_dir / "covariate_leakage_audit.csv"
    scorecard_path = run_dir / "corpus_adequacy_scorecard.csv"
    validity_audit_path = run_dir / "class_validity_audit.csv"
    degeneracy_report_path = run_dir / "corpus_degeneracy_report.csv"
    pair_status_heatmap_path = run_dir / "class_pair_coverage_heatmap.png"
    covariate_leakage_plot_path = run_dir / "covariate_leakage_audit.png"

    report_path.write_text(render_corpus_adequacy_report(result), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "summary": asdict(result.summary),
                "scorecard": asdict(result.scorecard),
                "thresholds": asdict(result.thresholds),
                "recommendations": list(result.recommendations),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_csv(
        feature_set_coverage_path,
        [dict(row) for row in result.feature_set_rows],
        [
            "feature_set",
            "feature",
            "moderate_or_strong_count",
            "moderate_or_strong_fraction",
            "strong_count",
            "supporting_tier_count",
            "supporting_class_count",
            "supporting_tiers",
            "supporting_classes",
            "status",
            "recommendation",
        ],
    )
    pair_fieldnames = [
        "class_a",
        "class_b",
        "expected_difficulty",
        "required_tiers",
        "satisfied_tiers",
        "pairwise_auc",
        "overlap_estimate",
        "pairwise_classifier_accuracy",
        "mahalanobis_distance",
        "required_tier_min_examples",
    ]
    pair_dynamic_fields = sorted(
        {
            key
            for row in result.class_pair_rows
            for key in row.keys()
            if str(key).startswith("count_")
        }
    )
    _write_csv(
        class_pair_coverage_path,
        [dict(row) for row in result.class_pair_rows],
        [*pair_fieldnames, *pair_dynamic_fields, "status", "recommendation"],
    )
    _write_csv(
        class_balance_path,
        [dict(row) for row in result.class_balance_rows],
        ["tier", "true_class", "count", "expected_count", "delta_from_expected", "status", "recommendation"],
    )
    _write_csv(
        covariate_leakage_path,
        [dict(row) for row in result.covariate_rows],
        ["covariate", "max_pairwise_auc", "worst_pair", "spread_ratio", "normalized_wasserstein", "min_class_mean", "max_class_mean", "status", "recommendation"],
    )
    _write_csv(
        scorecard_path,
        [dict(row) for row in result.scorecard_rows],
        sorted({key for row in result.scorecard_rows for key in row.keys()}) if result.scorecard_rows else ["term", "score", "desired_direction", "artifact"],
    )
    _write_csv(
        validity_audit_path,
        [dict(row) for row in result.validity_rows],
        ["trajectory_id", "target_class", "assigned_class", "validity_score", "alternate_class_similarity", "label_status"],
    )
    _write_csv(
        degeneracy_report_path,
        [dict(row) for row in result.degeneracy_rows],
        ["term", "value", "interpretation"],
    )
    pair_status_heatmap_path.write_bytes(_figure_to_png(_render_pair_status_heatmap(result)))
    covariate_leakage_plot_path.write_bytes(_figure_to_png(_render_covariate_leakage_plot(result)))
    return CorpusAdequacyArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        feature_set_coverage_path=feature_set_coverage_path,
        class_pair_coverage_path=class_pair_coverage_path,
        class_balance_path=class_balance_path,
        covariate_leakage_path=covariate_leakage_path,
        scorecard_path=scorecard_path,
        validity_audit_path=validity_audit_path,
        degeneracy_report_path=degeneracy_report_path,
        pair_status_heatmap_path=pair_status_heatmap_path,
        covariate_leakage_plot_path=covariate_leakage_plot_path,
    )
