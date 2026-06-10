from __future__ import annotations

from dataclasses import replace
from statistics import mean

from ...analysis.feature_analysis import analyze_feature_datasets
from ...trajectory_generator import generate_trajectory_datasets
from ..adequacy_audit import analyze_corpus_adequacy
from ..autodevelopment_types import CorpusCandidateEvaluation
from ..autodevelopment_utils import (
    DEFAULT_OBJECTIVES_PATH,
    CorpusCandidateSpec,
    _balance_score,
    _boundary_coverage_score,
    _candidate_manifest_row,
    _candidate_tier_definitions,
    _default_candidate_specs,
    _degeneracy_penalty,
    _difficulty_distribution_rows,
    _difficulty_diversity_score,
    _feature_excitation_score,
    _leakage_penalty,
    _pareto_objectives,
    _triviality_penalty,
    load_corpus_objectives,
)
from ..policy import (
    CorpusPolicySpec,
    load_corpus_policy_spec,
    score_corpus_autodevelopment_candidate,
)
from .feature_gap_trajectory_explorer_types import (
    FeatureGapIterationSummary,
    FeatureGapRecommendation,
    FeatureGapRow,
    FeatureGapTrajectoryExplorerResult,
)


def _status_weight(status: str) -> float:
    return {"red": 1.0, "yellow": 0.6, "green": 0.0}.get(status, 0.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _fieldnames(rows: tuple[dict[str, object], ...]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def _evaluate_candidate_spec(
    spec: CorpusCandidateSpec,
    *,
    objectives: dict[str, object],
    policy: CorpusPolicySpec,
) -> CorpusCandidateEvaluation:
    datasets = generate_trajectory_datasets(tier_definitions=_candidate_tier_definitions(spec), seed=spec.seed)
    distribution_rows = _difficulty_distribution_rows(datasets)
    difficulty_diversity_score = _difficulty_diversity_score(distribution_rows, objectives)
    manifest_row = _candidate_manifest_row(spec, distribution_rows)
    feature_analysis = analyze_feature_datasets(datasets=datasets)
    adequacy = analyze_corpus_adequacy(datasets=datasets, feature_analysis_result=feature_analysis)
    balance_score = _balance_score(adequacy)
    boundary_coverage_score = _boundary_coverage_score(adequacy)
    feature_excitation_score = _feature_excitation_score(adequacy)
    leakage_penalty = _leakage_penalty(adequacy, objectives)
    triviality_penalty = _triviality_penalty(adequacy)
    degeneracy_penalty = _degeneracy_penalty(adequacy)
    recommendation_count = len(adequacy.recommendations)
    provenance_completeness_score = 1.0 if recommendation_count else 0.9
    score_row = {
        "candidate_id": spec.candidate_id,
        "adequacy_status": adequacy.summary.overall_status,
        "policy_id": policy.policy_id,
        "balance_score": balance_score,
        "boundary_coverage_score": boundary_coverage_score,
        "feature_excitation_score": feature_excitation_score,
        "difficulty_diversity_score": difficulty_diversity_score,
        "provenance_completeness_score": provenance_completeness_score,
        "leakage_penalty": leakage_penalty,
        "triviality_penalty": triviality_penalty,
        "degeneracy_penalty": degeneracy_penalty,
        "q_corpus": adequacy.summary.q_corpus,
        "recommendation_count": recommendation_count,
        "overall_score": score_corpus_autodevelopment_candidate(
            policy,
            balance_score=balance_score,
            boundary_coverage_score=boundary_coverage_score,
            feature_excitation_score=feature_excitation_score,
            difficulty_diversity_score=difficulty_diversity_score,
            provenance_completeness_score=provenance_completeness_score,
            leakage_penalty=leakage_penalty,
            triviality_penalty=triviality_penalty,
            degeneracy_penalty=degeneracy_penalty,
        ),
    }
    return CorpusCandidateEvaluation(
        spec=spec,
        feature_analysis=feature_analysis,
        adequacy=adequacy,
        manifest_row=manifest_row,
        score_row=score_row,
        adequacy_row={"candidate_id": spec.candidate_id, "adequacy_status": adequacy.summary.overall_status},
        feature_excitation_rows=(),
        leakage_rows=tuple(adequacy.covariate_rows),
        pareto_objectives=_pareto_objectives(score_row),
    )


def _feature_gap_rows(adequacy, iteration: int) -> list[FeatureGapRow]:
    rows: list[FeatureGapRow] = []
    feature_rows = sorted(
        adequacy.feature_set_rows,
        key=lambda row: (
            str(row["status"]) == "green",
            float(row["moderate_or_strong_fraction"]),
            str(row["feature"]),
        ),
    )
    for feature_index, row in enumerate(feature_rows):
        status = str(row["status"])
        if status == "green" and feature_index >= 3:
            continue
        moderate_fraction = float(row["moderate_or_strong_fraction"])
        severity = _status_weight(status) + (1.0 - moderate_fraction)
        feature_name = str(row["feature"])
        rows.append(
            FeatureGapRow(
                iteration=iteration,
                gap_id=f"feature_set:{feature_name}",
                gap_kind="feature_set",
                target_id=feature_name,
                status=status,
                severity=severity,
                observed_value=moderate_fraction,
                target_value=0.45,
                recommendation_hint="increase feature excitation through stress and adversarial trajectories",
            )
        )
    for row in adequacy.class_pair_rows:
        status = str(row["status"])
        if status == "green":
            continue
        target_id = f"{row['class_a']}|{row['class_b']}"
        pairwise_auc = float(row["pairwise_auc"])
        overlap = float(row["overlap_estimate"])
        tier_count_fields = [key for key in row if str(key).startswith("count_")]
        tier_min = min((int(row[key]) for key in tier_count_fields), default=0)
        severity = _status_weight(status) + max(0.0, 0.75 - pairwise_auc) + max(0.0, 0.18 - overlap) + max(0, 2 - tier_min) * 0.15
        rows.append(
            FeatureGapRow(
                iteration=iteration,
                gap_id=f"class_pair:{target_id}",
                gap_kind="class_pair",
                target_id=target_id,
                status=status,
                severity=severity,
                observed_value=pairwise_auc,
                target_value=0.80,
                recommendation_hint="add more boundary and adversarial pair evidence",
            )
        )
    for row in adequacy.covariate_rows:
        status = str(row["status"])
        if status == "green":
            continue
        auc = float(row["max_pairwise_auc"])
        spread = float(row["spread_ratio"])
        severity = _status_weight(status) + max(0.0, auc - 0.70) + max(0.0, spread - 0.85)
        rows.append(
            FeatureGapRow(
                iteration=iteration,
                gap_id=f"covariate:{row['covariate']}",
                gap_kind="covariate",
                target_id=str(row["covariate"]),
                status=status,
                severity=severity,
                observed_value=auc,
                target_value=0.70,
                recommendation_hint="reduce measurement noise, outliers, and irregularity to lower leakage",
            )
        )
    rows.sort(key=lambda row: (-row.severity, row.gap_kind, row.target_id))
    return rows


def _class_balance_gap_row(adequacy, iteration: int) -> FeatureGapRow | None:
    problematic = [row for row in adequacy.class_balance_rows if str(row["status"]) != "green"]
    if not problematic:
        return None
    worst = max(problematic, key=lambda row: int(row["delta_from_expected"]))
    delta = int(worst["delta_from_expected"])
    return FeatureGapRow(
        iteration=iteration,
        gap_id=f"class_balance:{worst['tier']}|{worst['true_class']}",
        gap_kind="class_balance",
        target_id=f"{worst['tier']}|{worst['true_class']}",
        status=str(worst["status"]),
        severity=_status_weight(str(worst["status"])) + 0.20 * delta,
        observed_value=float(worst["count"]),
        target_value=float(worst["expected_count"]),
        recommendation_hint="reallocate tier counts toward underrepresented classes",
    )


def _shift_tier_counts(
    tier_counts: dict[str, int],
    *,
    updates: dict[str, int],
) -> dict[str, int]:
    values = dict(tier_counts)
    for key, delta in updates.items():
        values[key] = max(2, min(10, values.get(key, 2) + delta))
    return values


def _recommendations_for_gaps(
    current_spec: CorpusCandidateSpec,
    gap_rows: tuple[FeatureGapRow, ...],
    *,
    iteration: int,
    seed: int,
    selected_families: set[str],
) -> tuple[FeatureGapRecommendation, ...]:
    recommendations: list[FeatureGapRecommendation] = []
    seen_kinds: set[str] = set()
    for gap_index, gap in enumerate(gap_rows):
        if gap.gap_kind in seen_kinds:
            continue
        seen_kinds.add(gap.gap_kind)
        if gap.gap_kind == "feature_set":
            family = "stress_feature_excitation"
            sampler_name = "stress_mutation"
            updates = {"adversarial_v1": 2, "stress_v1": 2, "easy_v1": -1}
            measurement_scale = current_spec.measurement_scale * 1.12
            irregularity_scale = current_spec.irregularity_scale * 1.12
            outlier_scale = current_spec.outlier_scale * 1.20
            step_scale = current_spec.step_scale * 0.94
            expected_effect = "raise moderate or strong feature excitation across weak feature families"
        elif gap.gap_kind == "class_pair":
            family = "boundary_pair_focus"
            sampler_name = "boundary_mutation"
            updates = {"boundary_v1": 2, "adversarial_v1": 1, "realistic_v1": 1, "easy_v1": -1}
            measurement_scale = current_spec.measurement_scale * 1.05
            irregularity_scale = current_spec.irregularity_scale * 1.08
            outlier_scale = current_spec.outlier_scale * 1.02
            step_scale = current_spec.step_scale * 0.96
            expected_effect = "increase hard pair evidence near decision boundaries"
        elif gap.gap_kind == "covariate":
            family = "leakage_reduction"
            sampler_name = "archive_mutation"
            updates = {"realistic_v1": 1, "boundary_v1": 1, "stress_v1": -1}
            measurement_scale = current_spec.measurement_scale * 0.92
            irregularity_scale = current_spec.irregularity_scale * 0.82
            outlier_scale = current_spec.outlier_scale * 0.78
            step_scale = current_spec.step_scale * 1.04
            expected_effect = "reduce covariate leakage while preserving corpus diversity"
        else:
            family = "balance_repair"
            sampler_name = "grid"
            updates = {"boundary_v1": 1, "realistic_v1": 1}
            measurement_scale = current_spec.measurement_scale
            irregularity_scale = current_spec.irregularity_scale * 0.96
            outlier_scale = current_spec.outlier_scale
            step_scale = current_spec.step_scale
            expected_effect = "restore tier balance without collapsing overall coverage"
        novelty_bonus = 0.15 if family not in selected_families else 0.0
        tier_counts = _shift_tier_counts(current_spec.tier_counts, updates=updates)
        recommendations.append(
            FeatureGapRecommendation(
                iteration=iteration,
                recommendation_id=f"iter{iteration}_{family}_{gap_index}",
                source_gap_id=gap.gap_id,
                source_gap_kind=gap.gap_kind,
                trajectory_family=family,
                sampler_name=sampler_name,
                priority=gap.severity + novelty_bonus,
                description=f"Address `{gap.target_id}` via {family}.",
                expected_effect=expected_effect,
                tier_counts=tier_counts,
                measurement_scale=measurement_scale,
                irregularity_scale=irregularity_scale,
                outlier_scale=outlier_scale,
                step_scale=step_scale,
            )
        )
    recommendations.sort(key=lambda row: (-row.priority, row.trajectory_family, row.recommendation_id))
    return tuple(recommendations)


def _spec_from_recommendation(
    current_spec: CorpusCandidateSpec,
    recommendation: FeatureGapRecommendation,
    *,
    iteration: int,
    candidate_index: int,
    seed: int,
) -> CorpusCandidateSpec:
    return replace(
        current_spec,
        candidate_id=f"feature_gap_iter{iteration}_{recommendation.sampler_name}_{candidate_index}",
        description=recommendation.description,
        sampling_method=recommendation.sampler_name,
        seed=seed + iteration * 100 + candidate_index,
        tier_counts=dict(recommendation.tier_counts),
        measurement_scale=_clamp(recommendation.measurement_scale, 0.70, 1.40),
        irregularity_scale=_clamp(recommendation.irregularity_scale, 0.60, 1.40),
        outlier_scale=_clamp(recommendation.outlier_scale, 0.60, 1.50),
        step_scale=_clamp(recommendation.step_scale, 0.80, 1.20),
    )


def _selection_pressure(
    evaluation: CorpusCandidateEvaluation,
    recommendation: FeatureGapRecommendation,
    *,
    policy: CorpusPolicySpec,
    selected_families: set[str],
) -> float:
    weights = policy.generic_explorer_weights
    novelty = 1.0 if recommendation.trajectory_family not in selected_families else 0.45
    validity = float(evaluation.adequacy.summary.class_validity_score)
    coverage_novelty = min(recommendation.priority / 2.5, 1.0) * novelty
    boundary = float(evaluation.score_row["boundary_coverage_score"])
    stress = float(evaluation.score_row["feature_excitation_score"])
    environment = float(evaluation.score_row["difficulty_diversity_score"])
    provenance = float(evaluation.score_row["provenance_completeness_score"])
    total = sum(float(value) for value in weights.values())
    return (
        weights["validity"] * validity
        + weights["coverage_novelty"] * coverage_novelty
        + weights["boundary_score"] * boundary
        + weights["classifier_stress"] * stress
        + weights["environment_score"] * environment
        + weights["provenance_completeness"] * provenance
    ) / max(total, 1e-9)


def _candidate_score_row(
    evaluation: CorpusCandidateEvaluation,
    recommendation: FeatureGapRecommendation,
    *,
    iteration: int,
    selection_score: float,
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "candidate_id": evaluation.spec.candidate_id,
        "recommendation_id": recommendation.recommendation_id,
        "trajectory_family": recommendation.trajectory_family,
        "sampler_name": recommendation.sampler_name,
        "source_gap_kind": recommendation.source_gap_kind,
        "source_gap_id": recommendation.source_gap_id,
        "adequacy_status": evaluation.adequacy.summary.overall_status,
        "q_corpus": evaluation.score_row["q_corpus"],
        "feature_excitation_score": evaluation.score_row["feature_excitation_score"],
        "boundary_coverage_score": evaluation.score_row["boundary_coverage_score"],
        "difficulty_diversity_score": evaluation.score_row["difficulty_diversity_score"],
        "leakage_penalty": evaluation.score_row["leakage_penalty"],
        "triviality_penalty": evaluation.score_row["triviality_penalty"],
        "degeneracy_penalty": evaluation.score_row["degeneracy_penalty"],
        "overall_score": evaluation.score_row["overall_score"],
        "selection_score": selection_score,
        "measurement_scale": evaluation.spec.measurement_scale,
        "irregularity_scale": evaluation.spec.irregularity_scale,
        "outlier_scale": evaluation.spec.outlier_scale,
        "step_scale": evaluation.spec.step_scale,
    }


def _accepted_candidate(
    current_evaluation: CorpusCandidateEvaluation,
    candidate_evaluations: tuple[tuple[CorpusCandidateEvaluation, FeatureGapRecommendation, float], ...],
) -> tuple[CorpusCandidateEvaluation, FeatureGapRecommendation, float, bool]:
    selected = max(
        candidate_evaluations,
        key=lambda item: (
            item[2],
            float(item[0].score_row["overall_score"]),
            float(item[0].score_row["q_corpus"]),
        ),
    )
    accepted = (
        float(selected[0].score_row["overall_score"]) > float(current_evaluation.score_row["overall_score"])
        or float(selected[0].score_row["q_corpus"]) > float(current_evaluation.score_row["q_corpus"])
        or float(selected[0].score_row["feature_excitation_score"]) > float(current_evaluation.score_row["feature_excitation_score"])
    )
    return selected[0], selected[1], selected[2], accepted


def analyze_feature_gap_trajectory_explorer(
    *,
    seed: int = 7,
    policy: CorpusPolicySpec | None = None,
    max_iterations: int = 3,
    plateau_patience: int = 1,
) -> FeatureGapTrajectoryExplorerResult:
    objectives = load_corpus_objectives(DEFAULT_OBJECTIVES_PATH)
    resolved_policy = policy or load_corpus_policy_spec()
    initial_spec = _default_candidate_specs(seed)[0]
    current_evaluation = _evaluate_candidate_spec(initial_spec, objectives=objectives, policy=resolved_policy)
    selected_evaluations: list[CorpusCandidateEvaluation] = [current_evaluation]
    gap_rows: list[FeatureGapRow] = []
    recommendation_rows: list[FeatureGapRecommendation] = []
    iteration_rows: list[FeatureGapIterationSummary] = []
    candidate_score_rows: list[dict[str, object]] = []
    selected_families: set[str] = set()
    plateau_count = 0
    stop_reason = "budget_exhausted"

    for iteration in range(1, max_iterations + 1):
        starting_evaluation = current_evaluation
        current_gaps = _feature_gap_rows(current_evaluation.adequacy, iteration)
        balance_gap = _class_balance_gap_row(current_evaluation.adequacy, iteration)
        if balance_gap is not None:
            current_gaps.append(balance_gap)
        current_gaps.sort(key=lambda row: (-row.severity, row.gap_kind, row.target_id))
        gap_rows.extend(current_gaps)
        if not current_gaps:
            stop_reason = "no_gaps_remaining"
            break
        recommendations = _recommendations_for_gaps(
            current_evaluation.spec,
            tuple(current_gaps),
            iteration=iteration,
            seed=seed,
            selected_families=selected_families,
        )
        recommendation_rows.extend(recommendations)
        if not recommendations:
            stop_reason = "no_recommendations_generated"
            break

        candidate_evaluations: list[tuple[CorpusCandidateEvaluation, FeatureGapRecommendation, float]] = []
        for candidate_index, recommendation in enumerate(recommendations):
            spec = _spec_from_recommendation(current_evaluation.spec, recommendation, iteration=iteration, candidate_index=candidate_index, seed=seed)
            evaluation = _evaluate_candidate_spec(spec, objectives=objectives, policy=resolved_policy)
            selection_score = _selection_pressure(
                evaluation,
                recommendation,
                policy=resolved_policy,
                selected_families=selected_families,
            )
            candidate_evaluations.append((evaluation, recommendation, selection_score))
            candidate_score_rows.append(
                _candidate_score_row(
                    evaluation,
                    recommendation,
                    iteration=iteration,
                    selection_score=selection_score,
                )
            )

        selected_evaluation, selected_recommendation, _, accepted = _accepted_candidate(
            current_evaluation,
            tuple(candidate_evaluations),
        )
        q_delta = float(selected_evaluation.score_row["q_corpus"]) - float(starting_evaluation.score_row["q_corpus"])
        overall_delta = float(selected_evaluation.score_row["overall_score"]) - float(starting_evaluation.score_row["overall_score"])
        feature_delta = float(selected_evaluation.score_row["feature_excitation_score"]) - float(starting_evaluation.score_row["feature_excitation_score"])
        boundary_delta = float(selected_evaluation.score_row["boundary_coverage_score"]) - float(starting_evaluation.score_row["boundary_coverage_score"])
        iteration_stop_reason = "accepted"
        if not accepted:
            plateau_count += 1
            iteration_stop_reason = "plateau_candidate"
        else:
            plateau_count = 0
            current_evaluation = selected_evaluation
            selected_evaluations.append(selected_evaluation)
            selected_families.add(selected_recommendation.trajectory_family)
        if plateau_count > plateau_patience:
            iteration_stop_reason = "plateau_stop"
            stop_reason = "plateau_reached"
        iteration_rows.append(
            FeatureGapIterationSummary(
                iteration=iteration,
                starting_candidate_id=starting_evaluation.spec.candidate_id,
                selected_candidate_id=selected_evaluation.spec.candidate_id,
                selected_recommendation_id=selected_recommendation.recommendation_id,
                accepted=accepted,
                stop_reason=iteration_stop_reason,
                starting_q_corpus=float(starting_evaluation.score_row["q_corpus"]),
                selected_q_corpus=float(selected_evaluation.score_row["q_corpus"]),
                q_corpus_delta=q_delta,
                starting_feature_excitation=float(starting_evaluation.score_row["feature_excitation_score"]),
                selected_feature_excitation=float(selected_evaluation.score_row["feature_excitation_score"]),
                feature_excitation_delta=feature_delta,
                starting_boundary_coverage=float(starting_evaluation.score_row["boundary_coverage_score"]),
                selected_boundary_coverage=float(selected_evaluation.score_row["boundary_coverage_score"]),
                boundary_coverage_delta=boundary_delta,
                starting_overall_score=float(starting_evaluation.score_row["overall_score"]),
                selected_overall_score=float(selected_evaluation.score_row["overall_score"]),
                overall_score_delta=overall_delta,
            )
        )
        if stop_reason == "plateau_reached":
            break

    final_evaluation = current_evaluation
    if iteration_rows and stop_reason == "budget_exhausted":
        if any(row.accepted for row in iteration_rows):
            stop_reason = "iteration_budget_exhausted"
        else:
            stop_reason = "no_improving_candidate"
    report_markdown = _render_report(
        initial_candidate_id=initial_spec.candidate_id,
        final_evaluation=final_evaluation,
        gap_rows=tuple(gap_rows),
        recommendation_rows=tuple(recommendation_rows),
        iteration_rows=tuple(iteration_rows),
        stop_reason=stop_reason,
    )
    return FeatureGapTrajectoryExplorerResult(
        initial_candidate_id=initial_spec.candidate_id,
        final_candidate_id=final_evaluation.spec.candidate_id,
        stop_reason=stop_reason,
        selected_candidate_ids=tuple(evaluation.spec.candidate_id for evaluation in selected_evaluations),
        gap_rows=tuple(gap_rows),
        recommendation_rows=tuple(recommendation_rows),
        iteration_rows=tuple(iteration_rows),
        candidate_score_rows=tuple(candidate_score_rows),
        selected_evaluations=tuple(selected_evaluations),
        report_markdown=report_markdown,
    )


def _render_report(
    *,
    initial_candidate_id: str,
    final_evaluation: CorpusCandidateEvaluation,
    gap_rows: tuple[FeatureGapRow, ...],
    recommendation_rows: tuple[FeatureGapRecommendation, ...],
    iteration_rows: tuple[FeatureGapIterationSummary, ...],
    stop_reason: str,
) -> str:
    top_gap_summary = ", ".join(f"`{row.target_id}` ({row.gap_kind})" for row in gap_rows[:4]) or "none"
    accepted_rows = [row for row in iteration_rows if row.accepted]
    mean_q_delta = mean(row.q_corpus_delta for row in accepted_rows) if accepted_rows else 0.0
    return "\n".join(
        [
            "# Feature Gap Trajectory Explorer",
            "",
            "This closed-loop explorer converts adequacy gaps into concrete trajectory-family mutations and keeps the best improving corpus candidate at each iteration.",
            "",
            "## Summary",
            f"- initial candidate id: `{initial_candidate_id}`",
            f"- final candidate id: `{final_evaluation.spec.candidate_id}`",
            f"- stop reason: `{stop_reason}`",
            f"- iterations attempted: `{len(iteration_rows)}`",
            f"- accepted iterations: `{sum(1 for row in iteration_rows if row.accepted)}`",
            f"- final q_corpus: `{float(final_evaluation.score_row['q_corpus']):.3f}`",
            f"- final overall score: `{float(final_evaluation.score_row['overall_score']):.3f}`",
            f"- mean accepted q_corpus delta: `{mean_q_delta:.3f}`",
            "",
            "## Gap Signals",
            f"- total gap rows recorded: `{len(gap_rows)}`",
            f"- top gap targets: {top_gap_summary}",
            "",
            "## Notes",
            f"- recommendations generated: `{len(recommendation_rows)}`",
            "- Trajectory families are deterministic and map to existing sampler concepts: boundary, stress, archive, and grid-style balancing.",
            "- Selection uses corpus autodevelopment scoring plus a generic-explorer-style novelty pressure over recommendation families.",
        ]
    )
