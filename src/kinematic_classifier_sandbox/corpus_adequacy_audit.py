from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import io
import json
import os
from pathlib import Path

from .feature_analysis import (
    FeatureAnalysisResult,
    FEATURE_SET_MANIFEST_PATH,
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


@dataclass(frozen=True, slots=True)
class CorpusAdequacyResult:
    feature_analysis: FeatureAnalysisResult
    feature_set_rows: tuple[dict[str, object], ...]
    class_pair_rows: tuple[dict[str, object], ...]
    class_balance_rows: tuple[dict[str, object], ...]
    covariate_rows: tuple[dict[str, object], ...]
    recommendations: tuple[str, ...]
    summary: CorpusAdequacySummary
    thresholds: CorpusAdequacyThresholds


@dataclass(frozen=True, slots=True)
class CorpusAdequacyArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    feature_set_coverage_path: Path
    class_pair_coverage_path: Path
    class_balance_path: Path
    covariate_leakage_path: Path
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
        min_mean = min(means_by_class.values()) if means_by_class else 0.0
        max_mean = max(means_by_class.values()) if means_by_class else 0.0
        overall_mean = _mean(list(means_by_class.values())) if means_by_class else 0.0
        spread_ratio = (max_mean - min_mean) / max(overall_mean, 1e-6)
        for index, class_a in enumerate(class_names):
            for class_b in class_names[index + 1 :]:
                metrics = _pairwise_metrics(numeric_rows, (covariate_name,), class_a, class_b)
                pairwise_values.append((float(metrics["pairwise_auc"]), class_a, class_b))
        worst_auc, worst_a, worst_b = max(pairwise_values, key=lambda item: item[0])
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
                "min_class_mean": min_mean,
                "max_class_mean": max_mean,
                "status": status,
                "recommendation": recommendation,
            }
        )
    return rows, recommendations


def analyze_corpus_adequacy(
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    thresholds: CorpusAdequacyThresholds | None = None,
) -> CorpusAdequacyResult:
    selected_thresholds = thresholds or CorpusAdequacyThresholds()
    feature_analysis = analyze_feature_datasets(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
    )
    feature_set_rows, feature_recommendations = _feature_set_coverage_rows(feature_analysis, selected_thresholds)
    class_pair_rows, pair_recommendations = _class_pair_rows(feature_analysis, selected_thresholds)
    class_balance_rows, balance_recommendations = _class_balance_rows(feature_analysis)
    covariate_rows, covariate_recommendations = _covariate_rows(feature_analysis, selected_thresholds)

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
    overall_status = "fail" if overall_detail_status == "red" else ("warn" if yellow_count else "pass")
    recommendations = tuple(
        dict.fromkeys(
            [
                *feature_recommendations,
                *pair_recommendations,
                *balance_recommendations,
                *covariate_recommendations,
            ]
        )
    )
    summary = CorpusAdequacySummary(
        overall_status=overall_status,
        overall_pass=(red_count == 0),
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
    )
    return CorpusAdequacyResult(
        feature_analysis=feature_analysis,
        feature_set_rows=tuple(feature_set_rows),
        class_pair_rows=tuple(class_pair_rows),
        class_balance_rows=tuple(class_balance_rows),
        covariate_rows=tuple(covariate_rows),
        recommendations=recommendations,
        summary=summary,
        thresholds=selected_thresholds,
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
        "This audit turns the current corpus diagnostics into enforceable gates over feature coverage, declared hard pairs, class balance, and covariate leakage.",
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
        "",
        "## Feature Coverage by Feature Set",
        "",
        "| feature_set | feature | moderate_or_strong_fraction | strong_count | tiers | classes | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
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
            "| covariate | max_pairwise_auc | spread_ratio | worst_pair | status |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in result.covariate_rows:
        lines.append(
            f"| {row['covariate']} | {row['max_pairwise_auc']:.3f} | {row['spread_ratio']:.3f} | {row['worst_pair']} | {row['status']} |"
        )
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
    pair_status_heatmap_path = run_dir / "class_pair_coverage_heatmap.png"
    covariate_leakage_plot_path = run_dir / "covariate_leakage_audit.png"

    report_path.write_text(render_corpus_adequacy_report(result), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "summary": asdict(result.summary),
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
        ["covariate", "max_pairwise_auc", "worst_pair", "spread_ratio", "min_class_mean", "max_class_mean", "status", "recommendation"],
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
        pair_status_heatmap_path=pair_status_heatmap_path,
        covariate_leakage_plot_path=covariate_leakage_plot_path,
    )
