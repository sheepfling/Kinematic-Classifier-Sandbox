from __future__ import annotations

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..analysis.feature_analysis import (
    FEATURE_SET_MANIFEST_PATH,
    FeatureAnalysisResult,
    analyze_feature_datasets,
    load_feature_set_manifest,
)
from ..utils.math import _mean
from ..utils.plotting import plt
from .adequacy_audit_utils import (
    _class_balance_rows,
    _class_pair_rows,
    _clip01,
    _covariate_rows,
    _degeneracy_rows,
    _distribution_balance_score,
    _feature_excitation_score,
    _feature_set_coverage_rows,
    _pair_boundary_score,
    _status_label,
    _triviality_penalty,
    _validity_rows,
    _worst_status,
    load_class_pair_manifest,
)
from .adequacy_contracts import (
    CorpusAdequacyResult,
    CorpusAdequacyScorecard,
    CorpusAdequacySummary,
    CorpusAdequacyThresholds,
)


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
    report = MarkdownDocument("Corpus Adequacy Audit")
    report.paragraph(
        "This audit turns the current corpus diagnostics into a formal scorecard over balance, feature excitation, class-pair boundary coverage, class validity, leakage, triviality, and degeneracy."
    )
    report.heading("Overall Gate", level=2)
    report.bullet_list(
        [
            f"Overall status: {result.summary.overall_status}",
            f"Overall pass: {result.summary.overall_pass}",
            f"Feature coverage: {result.summary.feature_status}",
            f"Class-pair coverage: {result.summary.class_pair_status}",
            f"Class balance: {result.summary.class_balance_status}",
            f"Covariate leakage: {result.summary.covariate_status}",
            f"Trajectories analyzed: {result.summary.total_trajectories}",
            f"Red findings: {result.summary.red_count}",
            f"Yellow findings: {result.summary.yellow_count}",
            f"Q_corpus: {result.summary.q_corpus:.3f}",
            f"Leakage penalty: {result.summary.leakage_penalty:.3f}",
            f"Triviality penalty: {result.summary.triviality_penalty:.3f}",
            f"Class-validity score: {result.summary.class_validity_score:.3f}",
            f"Degeneracy penalty: {result.summary.degeneracy_penalty:.3f}",
        ]
    )
    report.heading("Corpus Scorecard", level=2)
    report.table(
        ["term", "score", "desired_direction", "artifact"],
        [
            (
                row['term'],
                f"{float(row['score']):.3f}",
                row['desired_direction'],
                row['artifact'],
            )
            for row in result.scorecard_rows[:10]
        ],
    )
    report.heading("Feature Coverage by Feature Set", level=2)
    report.table(
        [
            "feature_set",
            "feature",
            "moderate_or_strong_fraction",
            "strong_count",
            "tiers",
            "classes",
            "status",
        ],
        [
            (
                row["feature_set"],
                row["feature"],
                f"{row['moderate_or_strong_fraction']:.3f}",
                row["strong_count"],
                row["supporting_tier_count"],
                row["supporting_class_count"],
                row["status"],
            )
            for row in result.feature_set_rows
        ],
    )
    report.heading("Declared Class-Pair Boundary Coverage", level=2)
    report.table(
        ["class_a", "class_b", "difficulty", "pairwise_auc", "overlap", "required_tiers", "status"],
        [
            (
                row['class_a'],
                row['class_b'],
                row['expected_difficulty'],
                f"{row['pairwise_auc']:.3f}",
                f"{row['overlap_estimate']:.3f}",
                row['required_tiers'],
                row['status'],
            )
            for row in result.class_pair_rows
        ],
    )
    report.heading("Class Balance", level=2)
    report.table(
        ["tier", "true_class", "count", "expected_count", "status"],
        [
            (
                row["tier"],
                row["true_class"],
                row["count"],
                row["expected_count"],
                row["status"],
            )
            for row in result.class_balance_rows
        ],
    )
    report.heading("Covariate Leakage", level=2)
    report.table(
        ["covariate", "max_pairwise_auc", "spread_ratio", "normalized_wasserstein", "worst_pair", "status"],
        [
            (
                row["covariate"],
                f"{row['max_pairwise_auc']:.3f}",
                f"{row['spread_ratio']:.3f}",
                f"{row['normalized_wasserstein']:.3f}",
                row["worst_pair"],
                row["status"],
            )
            for row in result.covariate_rows
        ],
    )
    report.heading("Class Validity", level=2)
    report.table(
        ["label_status", "count"],
        [(status, sum(1 for row in result.validity_rows if row["label_status"] == status)) for status in ("valid_target_class", "ambiguous", "invalid", "relabel_candidate")],
    )
    report.heading("Degeneracy", level=2)
    report.table(
        ["term", "value", "interpretation"],
        [
            (
                row["term"],
                f"{float(row['value']):.3f}",
                row["interpretation"],
            )
            for row in result.degeneracy_rows
        ],
    )
    report.heading("Recommendations", level=2)
    if result.recommendations:
        report.bullet_list(result.recommendations)
    else:
        report.bullet_list(["No missing-coverage recommendations. The current corpus clears every enforced gate."])
    return report.text()
