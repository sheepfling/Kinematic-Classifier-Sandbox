from __future__ import annotations

from statistics import mean
from pathlib import Path

from ..markdown_builder import MarkdownDocument
from ..utils.plotting import plt
from ..utils.plotting import _figure_to_png
from .autodevelopment_types import CorpusAutodevelopmentResult


def _selected_corpus_evaluation(result: CorpusAutodevelopmentResult):
    return next(
        evaluation for evaluation in result.candidate_evaluations if evaluation.spec.candidate_id == result.selected_candidate_id
    )


def _reference_rejected_evaluation(result: CorpusAutodevelopmentResult):
    selected_id = result.selected_candidate_id
    for evaluation in sorted(result.candidate_evaluations, key=lambda item: float(item.score_row["overall_score"]), reverse=True):
        if evaluation.spec.candidate_id != selected_id:
            return evaluation
    return None


def _render_report(result: CorpusAutodevelopmentResult) -> str:
    selected = next(row for row in result.candidate_score_rows if row["candidate_id"] == result.selected_candidate_id)
    rejected_ids = [row["candidate_id"] for row in result.rejected_candidate_rows[:3]]
    pareto_ids = [row["candidate_id"] for row in result.pareto_front_rows]

    doc = MarkdownDocument("Corpus Autodevelopment")
    doc.paragraph(
        "This artifact proposes, scores, compares, and selects candidate corpora against explicit adequacy objectives."
    )

    doc.heading("Selection Summary", level=2)
    doc.bullet_list(
        [
            f"Selected candidate: `{result.selected_candidate_id}`",
            f"Overall score: `{float(selected['overall_score']):.3f}`",
            f"Adequacy status: `{selected['adequacy_status']}`",
            f"Pareto-front candidates: `{', '.join(pareto_ids)}`",
            f"Example rejected candidates: `{', '.join(rejected_ids)}`",
        ]
    )

    doc.heading("Scoring Logic", level=2)
    doc.bullet_list(
        [
            f"Policy: `{selected.get('policy_id', 'unknown')}`",
            "`overall_score` is computed through `CorpusPolicySpec` corpus-autodevelopment weights.",
            "Higher is better.",
            "The Pareto front preserves non-dominated candidates even if only one is selected for default use.",
        ]
    )

    doc.heading("What This Proves", level=2)
    doc.bullet_list(
        [
            "The repo can compare multiple corpus candidates instead of only auditing the default corpus.",
            "Selection is tied to declared adequacy objectives rather than manual tweaking alone.",
            "Rejected candidates remain inspectable, which makes the corpus-development story defensible.",
        ]
    )
    return doc.text()


def render_corpus_autodevelopment_numeric_walkthrough_markdown(result: CorpusAutodevelopmentResult) -> str:
    selected = _selected_corpus_evaluation(result)
    rejected = _reference_rejected_evaluation(result)
    objectives = result.objectives
    difficulty_targets = dict(objectives.get("difficulty_distribution", {}))
    leakage_targets = dict(objectives.get("covariate_leakage", {}))
    selected_difficulty = {key: value for key, value in selected.manifest_row.items() if key.endswith("_fraction")}
    objective_rows = [
        ("balance_score", float(selected.score_row["balance_score"]), "Higher is better; mean class-balance status score."),
        ("boundary_coverage_score", float(selected.score_row["boundary_coverage_score"]), "Higher is better; mean hard-pair coverage status score."),
        ("feature_excitation_score", float(selected.score_row["feature_excitation_score"]), "Higher is better; combines excitation fraction and adequacy status."),
        ("difficulty_diversity_score", float(selected.score_row["difficulty_diversity_score"]), "Higher is better; closeness to the configured tier-fraction target."),
        ("leakage_penalty", float(selected.score_row["leakage_penalty"]), "Lower is better; covariate-only separability and spread."),
        ("triviality_penalty", float(selected.score_row["triviality_penalty"]), "Lower is better; penalizes hard pairs that become too easy."),
        ("degeneracy_penalty", float(selected.score_row["degeneracy_penalty"]), "Lower is better; penalizes red or weakly excited feature sets."),
    ]
    selected_is_pareto = any(str(row["candidate_id"]) == selected.spec.candidate_id for row in result.pareto_front_rows)

    doc = MarkdownDocument("Corpus Autodevelopment Numeric Walkthrough")
    doc.paragraph(
        "This worked example decomposes the selected corpus candidate's real score using the exact terms implemented in `corpus/autodevelopment.py`."
    )

    doc.heading("Selected Candidate", level=2)
    doc.bullet_list(
        [
            f"Candidate: `{selected.spec.candidate_id}`",
            f"Sampling method: `{selected.spec.sampling_method}`",
            f"Adequacy status: `{selected.score_row['adequacy_status']}`",
            f"Overall score: `{float(selected.score_row['overall_score']):.3f}`",
            f"Pareto-front member: `{'yes' if selected_is_pareto else 'no'}`",
        ]
    )

    doc.heading("Score Equation", level=2)
    doc.fence("S_k = n_+\\sum_r w^+_r x_{k,r} - n_-\\sum_u w^-_u p_{k,u}", language="tex")
    doc.paragraph("where the implemented selected-candidate values are:")

    doc.table(
        ["term", "value", "interpretation"],
        [(f"`{name}`", f"`{value:.3f}`", interpretation) for name, value, interpretation in objective_rows],
    )

    doc.paragraph("Substituting those values gives:")
    doc.fence(
        f"S_{{{selected.spec.candidate_id}}} = "
        f"{float(selected.score_row['balance_score']):.3f}"
        f" + {float(selected.score_row['boundary_coverage_score']):.3f}"
        f" + {float(selected.score_row['feature_excitation_score']):.3f}"
        f" + {float(selected.score_row['difficulty_diversity_score']):.3f}"
        f" - {float(selected.score_row['leakage_penalty']):.3f}"
        f" - {float(selected.score_row['triviality_penalty']):.3f}"
        f" - {float(selected.score_row['degeneracy_penalty']):.3f}"
        f" = {float(selected.score_row['overall_score']):.3f}",
        language="tex",
    )

    doc.heading("Difficulty-Diversity Subscore", level=2)
    doc.paragraph(
        "The difficulty term compares the selected corpus distribution with the configured target fractions from `experiments/corpus_objectives/common_1d_corpus_objectives.yaml`."
    )

    tier_rows = []
    total_abs_error = 0.0
    for base_tier in ("easy", "boundary", "adversarial", "stress", "realistic"):
        target = float(difficulty_targets.get(f"{base_tier}_fraction", 0.0))
        selected_fraction = float(selected_difficulty.get(f"{base_tier}_v1_fraction", 0.0))
        error = abs(selected_fraction - target)
        total_abs_error += error
        tier_rows.append((f"`{base_tier}`", f"`{target:.3f}`", f"`{selected_fraction:.3f}`", f"`{error:.3f}`"))

    doc.table(["tier", "target fraction", "selected fraction", "absolute error"], tier_rows)
    doc.fence(
        f"D_k = 1 - \\frac{{{total_abs_error:.3f}}}{{2}} = {float(selected.score_row['difficulty_diversity_score']):.3f}",
        language="tex",
    )

    doc.heading("Leakage and Triviality Checks", level=2)
    doc.bullet_list(
        [
            f"Leakage objective limits: duration `{float(leakage_targets.get('max_duration_class_correlation', 0.20)):.2f}`, sample-count `{float(leakage_targets.get('max_sample_count_class_correlation', 0.20)):.2f}`, noise `{float(leakage_targets.get('max_noise_class_correlation', 0.20)):.2f}`",
            f"Selected leakage penalty: `{float(selected.score_row['leakage_penalty']):.3f}`",
            f"Selected triviality penalty: `{float(selected.score_row['triviality_penalty']):.3f}`",
            f"Selected degeneracy penalty: `{float(selected.score_row['degeneracy_penalty']):.3f}`",
            f"Reference rejected candidate: `{rejected.spec.candidate_id if rejected else 'none'}`",
        ]
    )

    return doc.text()


def _render_corpus_score_pareto(result: CorpusAutodevelopmentResult):
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    pareto_ids = {row["candidate_id"] for row in result.pareto_front_rows}
    for row in result.candidate_score_rows:
        candidate_id = str(row["candidate_id"])
        color = "#dc2626" if candidate_id == result.selected_candidate_id else ("#2563eb" if candidate_id in pareto_ids else "#6b7280")
        ax.scatter(float(row["leakage_penalty"]), float(row["feature_excitation_score"]), s=90, color=color, alpha=0.9)
        ax.text(float(row["leakage_penalty"]) + 0.01, float(row["feature_excitation_score"]) + 0.01, candidate_id, fontsize=8)
    ax.set_xlabel("Leakage penalty")
    ax.set_ylabel("Feature excitation score")
    ax.set_title("Corpus Candidate Pareto View", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _render_feature_excitation_heatmap(result: CorpusAutodevelopmentResult):
    candidate_ids = [row["candidate_id"] for row in result.candidate_score_rows]
    feature_sets = sorted({str(row["feature_set"]) for row in result.feature_excitation_comparison_rows})
    matrix = []
    for candidate_id in candidate_ids:
        candidate_lookup = {
            str(row["feature_set"]): float(row["mean_moderate_or_strong_fraction"])
            for row in result.feature_excitation_comparison_rows
            if row["candidate_id"] == candidate_id
        }
        matrix.append([candidate_lookup.get(feature_set, 0.0) for feature_set in feature_sets])
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(feature_sets)))
    ax.set_xticklabels(feature_sets, rotation=30, ha="right")
    ax.set_yticks(range(len(candidate_ids)))
    ax.set_yticklabels(candidate_ids)
    ax.set_title("Feature Excitation By Candidate", loc="left", fontweight="bold")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _render_leakage_by_candidate(result: CorpusAutodevelopmentResult):
    candidate_ids = [row["candidate_id"] for row in result.candidate_score_rows]
    worst_auc = []
    for candidate_id in candidate_ids:
        values = [float(row["max_pairwise_auc"]) for row in result.leakage_comparison_rows if row["candidate_id"] == candidate_id]
        worst_auc.append(max(values) if values else 0.0)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    colors = ["#dc2626" if candidate_id == result.selected_candidate_id else "#2563eb" for candidate_id in candidate_ids]
    ax.bar(candidate_ids, worst_auc, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Worst covariate-only AUC")
    ax.set_title("Leakage By Candidate", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_difficulty_distribution(result: CorpusAutodevelopmentResult):
    candidate_ids = [row["candidate_id"] for row in result.candidate_score_rows]
    tiers = ["easy_v1", "boundary_v1", "adversarial_v1", "stress_v1", "realistic_v1"]
    colors = ["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8", "#1e3a8a"]
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    bottoms = [0.0] * len(candidate_ids)
    manifest_lookup = {str(row["candidate_id"]): row for row in result.candidate_manifest_rows}
    for tier, color in zip(tiers, colors):
        values = [float(manifest_lookup[candidate_id].get(f"{tier}_fraction", 0.0)) for candidate_id in candidate_ids]
        ax.bar(candidate_ids, values, bottom=bottoms, label=tier, color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of trajectories")
    ax.set_title("Difficulty Distribution By Candidate", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    return fig
