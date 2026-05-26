from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from kinematic_classifier_sandbox.utils.io import write_csv

from .analysis.feature_analysis import (
    load_feature_registry,
    load_feature_set_manifest,
    resolve_feature_names,
)
from .common_experiment.runner import analyze_common_experiment
from .corpus.adequacy_audit_utils import PAIR_TIER_REQUIREMENTS, load_class_pair_manifest
from .corpus.autodevelopment import analyze_corpus_autodevelopment
from .corpus.coverage_report import load_classifier_manifest
from .corpus.policy import (
    CorpusPolicySpec,
    load_corpus_policy_spec,
    score_study_candidate_monte_carlo,
    score_study_candidate_static,
)
from .runtime_paths import prepare_matplotlib
from .study_candidate_generation_rendering import (
    _render_candidate_promotion_matrix,
    _render_classifier_feature_class_heatmap,
    _render_static_vs_statistical_score,
)
from .study_candidate_generation_utils import (
    _alias_matches,
    _classifier_assumption_fit,
    _feature_class_compatibility_score,
    _feature_set_for_classifier,
    _history_risk,
    _normalize_pair_id,
    _pair_id_from_pair,
)
from .study_candidate_generation_types import (
    StudyCandidateFeatureEvidenceRow,
    StudyCandidateMonteCarloScoreRow,
    StudyCandidatePriorSensitivityExplanationRow,
    StudyCandidateRow,
    StudyCandidateStaticScoreRow,
)
from .study_candidate_protocol import analyze_study_candidate_protocol
from .utils.math import _clamp
from .utils.plotting import _figure_to_png


@dataclass(frozen=True, slots=True)
class StudyCandidateGenerationResult:
    schema: dict[str, object]
    generated_candidates: tuple[StudyCandidateRow, ...]
    static_score_rows: tuple[StudyCandidateStaticScoreRow, ...]
    monte_carlo_score_rows: tuple[StudyCandidateMonteCarloScoreRow, ...]
    feature_evidence_rows: tuple[StudyCandidateFeatureEvidenceRow, ...]
    prior_sensitivity_explanation_rows: tuple[StudyCandidatePriorSensitivityExplanationRow, ...]
    promoted_rows: tuple[StudyCandidateMonteCarloScoreRow, ...]
    rejected_rows: tuple[StudyCandidateMonteCarloScoreRow, ...]
    deferred_rows: tuple[StudyCandidateMonteCarloScoreRow, ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class StudyCandidateGenerationArtifacts:
    run_dir: Path
    schema_path: Path
    generated_candidates_path: Path
    static_scores_path: Path
    feature_evidence_table_path: Path
    prior_sensitivity_explanation_table_path: Path
    promoted_candidates_path: Path
    rejected_candidates_path: Path
    monte_carlo_scores_path: Path
    decision_report_path: Path
    static_vs_statistical_score_path: Path
    candidate_promotion_matrix_path: Path
    classifier_feature_class_heatmap_path: Path


def analyze_study_candidate_generation(
    *,
    seed: int = 7,
    trajectories_per_case: int = 8,
    policy: CorpusPolicySpec | None = None,
) -> StudyCandidateGenerationResult:
    resolved_policy = policy or load_corpus_policy_spec()
    protocol = analyze_study_candidate_protocol()
    common = analyze_common_experiment(seed=seed, trajectories_per_case=trajectories_per_case)
    corpus = analyze_corpus_autodevelopment(seed=seed, policy=resolved_policy)
    registry = load_feature_registry()
    feature_manifest = load_feature_set_manifest()
    class_pair_manifest = load_class_pair_manifest()
    classifier_manifest = load_classifier_manifest()
    oracle_lookup = {
        (str(row["class_pair_id"]), str(row["feature_set_id"])): row
        for row in common.oracle_rows
    }
    metrics_by_pair_lookup = {
        (str(row["classifier_id"]), _normalize_pair_id(str(row["class_pair"]))): row
        for row in common.metrics_by_class_pair_rows
    }
    prior_lookup: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in common.prior_sensitivity_rows:
        prior_lookup.setdefault((str(row["classifier_id"]), str(row["class_pair_id"])), []).append(dict(row))

    selected_corpus = next(
        evaluation for evaluation in corpus.candidate_evaluations if evaluation.spec.candidate_id == corpus.selected_candidate_id
    )
    selected_corpus_score = float(selected_corpus.score_row["overall_score"])
    selected_corpus_adequacy_status = str(selected_corpus.adequacy.summary.overall_status)

    priors = ["uniform", "mild_bias", "strong_bias"]
    generated_candidates: list[StudyCandidateRow] = []
    static_score_rows: list[StudyCandidateStaticScoreRow] = []
    monte_carlo_score_rows: list[StudyCandidateMonteCarloScoreRow] = []

    for pair_entry in class_pair_manifest:
        class_pair = tuple(str(name) for name in pair_entry["pair"])
        pair_id = _pair_id_from_pair(class_pair)
        expected_difficulty = str(pair_entry["expected_difficulty"])
        primary_separators = [str(value) for value in pair_entry.get("primary_separators", [])]
        corpus_tiers = list(PAIR_TIER_REQUIREMENTS.get(expected_difficulty, ("boundary_v1",)))
        expected_failure_modes = {
            "easy": ["measurement_noise", "feature saturation"],
            "duration_dependent": ["short duration", "insufficient prefix length", "prior sensitivity"],
            "hard": ["class overlap", "correlated evidence", "weak oracle separability"],
            "short_horizon_boundary": ["short duration", "outlier-driven extrema", "prior sensitivity near stopping boundary"],
        }.get(expected_difficulty, ["class overlap"])

        for classifier_entry in classifier_manifest:
            classifier_id = str(classifier_entry["id"])
            classifier_family = str(classifier_entry["family"])
            canonical_feature_set = _feature_set_for_classifier(classifier_entry)
            for feature_set_id, feature_set_entry in feature_manifest.items():
                feature_names = resolve_feature_names(feature_set=feature_set_id, manifest=feature_manifest)
                feature_specs = [registry[name] for name in feature_names]
                feature_groups = tuple(sorted({spec.group for spec in feature_specs}))
                dependency_tags = tuple(sorted({tag for spec in feature_specs for tag in spec.dependency_tags}))
                dimensional_transfer_modes = {spec.dimensional_transfer for spec in feature_specs}
                geometry_assumptions = {spec.geometry_assumption for spec in feature_specs}
                feature_history_behavior = str(feature_set_entry.get("history_behavior", "unknown"))
                compatible = canonical_feature_set == feature_set_id
                feature_class_compatibility_score = _feature_class_compatibility_score(
                    primary_separators=primary_separators,
                    feature_names=feature_names,
                    groups=feature_groups,
                    dependency_tags=dependency_tags,
                )
                oracle_row = oracle_lookup.get((pair_id, feature_set_id))
                expected_separability_score = float(oracle_row["oracle_accuracy"]) if oracle_row is not None else 0.0
                feature_dependency_risk = _clamp(len(dependency_tags) / 8.0, 0.0, 1.0)
                cumulative_double_counting_risk = _history_risk(feature_history_behavior)
                classifier_assumption_fit = _classifier_assumption_fit(
                    classifier_family=classifier_family,
                    expected_difficulty=expected_difficulty,
                    compatible=compatible,
                )
                if "rewrite_required" in dimensional_transfer_modes or "scalar_axis" in geometry_assumptions:
                    dimensional_transfer_score = 0.25
                elif "requires_direction_change_policy" in dimensional_transfer_modes:
                    dimensional_transfer_score = 0.50
                else:
                    dimensional_transfer_score = 0.80
                implementation_readiness_score = 1.0 if any(
                    str(row["classifier_id"]) == classifier_id for row in common.metrics_by_classifier_rows
                ) else 0.0
                corpus_coverage_score = _clamp(
                    0.55 * (selected_corpus_score / 4.0)
                    + 0.25 * (len(corpus_tiers) / 3.0)
                    + 0.20 * (0.0 if selected_corpus_adequacy_status == "fail" else 0.5 if selected_corpus_adequacy_status == "warn" else 1.0),
                    0.0,
                    1.0,
                )

                mc_row = metrics_by_pair_lookup.get((classifier_id, pair_id))
                prior_rows = prior_lookup.get((classifier_id, pair_id), [])
                uniform_accuracy = None
                strong_accuracy = None
                if prior_rows:
                    by_prior = {str(row["prior_id"]): float(row["accuracy"]) for row in prior_rows}
                    uniform_accuracy = by_prior.get("uniform")
                    strong_accuracy = by_prior.get("strong_bias")
                prior_sensitivity_risk = (
                    _clamp((uniform_accuracy - strong_accuracy) / 0.30, 0.0, 1.0)
                    if uniform_accuracy is not None and strong_accuracy is not None
                    else {"easy": 0.05, "duration_dependent": 0.35, "hard": 0.45, "short_horizon_boundary": 0.65}.get(expected_difficulty, 0.30)
                )

                static_score = score_study_candidate_static(
                    resolved_policy,
                    feature_class_compatibility=feature_class_compatibility_score,
                    expected_separability=expected_separability_score,
                    classifier_assumption_fit=classifier_assumption_fit,
                    corpus_coverage=corpus_coverage_score,
                    dimensional_transfer=dimensional_transfer_score,
                    implementation_readiness=implementation_readiness_score,
                    feature_dependency_risk=feature_dependency_risk,
                    cumulative_double_counting_risk=cumulative_double_counting_risk,
                    prior_sensitivity_risk=prior_sensitivity_risk,
                )

                for prior_id in priors:
                    study_id = f"{feature_set_id}_{pair_id}_{classifier_id}_{prior_id}"
                    hypothesis = (
                        f"{feature_set_id} features with {classifier_id} improve {pair_id} discrimination "
                        f"under {expected_difficulty} conditions."
                    )
                    candidate = StudyCandidateRow(
                        study_id=study_id,
                        hypothesis=hypothesis,
                        corpus_spec={
                            "corpus_id": corpus.selected_candidate_id,
                            "sensor_regime_id": "position_only",
                            "tiers": corpus_tiers,
                            "generator_family": "trajectory_generator_v1",
                            "objectives_id": "common_1d_corpus_objectives",
                        },
                        feature_set_spec={
                            "feature_sets": [feature_set_id],
                            "required_tags": sorted({tag for spec in feature_specs for tag in spec.sensitivity_tags})[:4],
                            "double_counting_risk": "high" if cumulative_double_counting_risk >= 0.70 else ("medium" if cumulative_double_counting_risk >= 0.35 else "low"),
                        },
                        class_set_spec={
                            "classes": list(class_pair),
                            "class_pairs": [list(class_pair)],
                            "claims": [f"{pair_id} is {expected_difficulty} and should be separated by {', '.join(primary_separators)}"],
                        },
                        classifier_spec={
                            "classifier_families": [classifier_family],
                            "history_behavior": feature_history_behavior,
                            "assumptions": [f"declared_classifier_id={classifier_id}", f"compatible_feature_set={canonical_feature_set}"],
                        },
                        prior_spec={
                            "prior_ids": [prior_id],
                            "sensitivity_risk": "high" if prior_sensitivity_risk >= 0.65 else ("medium" if prior_sensitivity_risk >= 0.35 else "low"),
                        },
                        filter_spec={
                            "filter_family": "kalman_bank" if classifier_family == "state_space" else "none",
                            "uses_dynamics": classifier_family == "state_space",
                            "handles_switching": classifier_family == "state_space" and "maneuver" in class_pair,
                        },
                        visualization_spec={
                            "required_plots": ["static_vs_statistical_score", "candidate_promotion_matrix"],
                            "requires_bayesian_walkthrough": classifier_family in {"pointwise", "sequential_bayes", "state_space"},
                        },
                        expected_failure_modes=expected_failure_modes,
                        decision_policy={
                            "allowed_decisions": ["promote", "revise", "reject", "defer"],
                            "promotion_requires_monte_carlo": True,
                            "rejection_allows_static_only": True,
                        },
                    )
                    generated_candidates.append(candidate)

                    static_score_rows.append(
                        StudyCandidateStaticScoreRow(
                            study_id=study_id,
                            class_pair_id=pair_id,
                            feature_set_id=feature_set_id,
                            classifier_id=classifier_id,
                            classifier_family=classifier_family,
                            prior_id=prior_id,
                            corpus_id=corpus.selected_candidate_id,
                            compatible=compatible,
                            feature_class_compatibility_score=feature_class_compatibility_score,
                            expected_separability_score=expected_separability_score,
                            feature_dependency_risk=feature_dependency_risk,
                            cumulative_double_counting_risk=cumulative_double_counting_risk,
                            prior_sensitivity_risk=prior_sensitivity_risk,
                            corpus_coverage_score=corpus_coverage_score,
                            classifier_assumption_fit=classifier_assumption_fit,
                            three_d_transferability_score=dimensional_transfer_score,
                            implementation_readiness_score=implementation_readiness_score,
                            static_score=static_score,
                            expected_difficulty=expected_difficulty,
                            policy_id=resolved_policy.policy_id,
                        )
                    )

                    accuracy = float(mc_row["overall_accuracy"]) if mc_row is not None else None
                    oracle_accuracy = float(oracle_row["oracle_accuracy"]) if oracle_row is not None else None
                    oracle_gap = (oracle_accuracy - accuracy) if oracle_accuracy is not None and accuracy is not None else None
                    prior_flip_fraction = (
                        _clamp((uniform_accuracy - strong_accuracy) / max(uniform_accuracy, 1e-6), 0.0, 1.0)
                        if uniform_accuracy is not None and strong_accuracy is not None
                        else None
                    )
                    if accuracy is None:
                        decision = "reject" if (not compatible or static_score < 0.35) else "defer"
                        monte_carlo_score = None
                    else:
                        monte_carlo_score = score_study_candidate_monte_carlo(
                            resolved_policy,
                            accuracy=accuracy,
                            prior_flip_fraction=prior_flip_fraction or 0.0,
                            oracle_gap=oracle_gap or 0.0,
                        )
                        if (
                            compatible
                            and static_score >= 0.45
                            and monte_carlo_score >= 0.90
                            and accuracy >= 0.83
                            and (prior_flip_fraction or 0.0) <= 0.12
                        ):
                            decision = "promote"
                        elif compatible and static_score >= 0.45 and accuracy >= 0.65:
                            decision = "revise"
                        else:
                            decision = "reject"

                    monte_carlo_score_rows.append(
                        StudyCandidateMonteCarloScoreRow(
                            study_id=study_id,
                            class_pair_id=pair_id,
                            feature_set_id=feature_set_id,
                            classifier_id=classifier_id,
                            prior_id=prior_id,
                            accuracy=accuracy,
                            oracle_accuracy=oracle_accuracy,
                            oracle_gap=oracle_gap,
                            uniform_accuracy=uniform_accuracy,
                            strong_bias_accuracy=strong_accuracy,
                            prior_flip_fraction=prior_flip_fraction,
                            monte_carlo_score=monte_carlo_score,
                            decision=decision,
                            policy_id=resolved_policy.policy_id,
                        )
                    )

    identifiability_rows = list(common.identifiability_rows)
    feature_evidence_rows: list[StudyCandidateFeatureEvidenceRow] = []
    feature_set_by_name = {
        feature_set_id: set(resolve_feature_names(feature_set=feature_set_id, manifest=feature_manifest))
        for feature_set_id in feature_manifest
    }
    for feature_name, spec in sorted(registry.items()):
        relevant_rows = [
            row
            for row in identifiability_rows
            if feature_name in feature_set_by_name.get(str(row["feature_set_id"]), set())
        ]
        ranked_rows = sorted(
            relevant_rows,
            key=lambda row: float(row["mean_standardized_feature_distance"]),
            reverse=True,
        )
        best_class_pairs = ",".join(dict.fromkeys(str(row["class_pair_id"]) for row in ranked_rows[:3]))
        worst_class_pairs = ",".join(
            dict.fromkeys(str(row["class_pair_id"]) for row in sorted(relevant_rows, key=lambda row: float(row["mean_standardized_feature_distance"]))[:3])
        )
        feature_evidence_rows.append(
            StudyCandidateFeatureEvidenceRow(
                feature_name=feature_name,
                feature_group=spec.group,
                history_behavior=spec.history_behavior,
                evidence_role=spec.role,
                double_counting_risk="high" if _history_risk(spec.history_behavior) >= 0.70 else ("medium" if _history_risk(spec.history_behavior) >= 0.35 else "low"),
                noise_sensitivity="yes" if "noise_sensitive" in spec.sensitivity_tags else "no",
                duration_sensitivity="yes" if "duration_sensitive" in spec.sensitivity_tags else "no",
                sample_count_sensitivity="yes" if "sample_count_sensitive" in spec.sensitivity_tags else "no",
                three_d_transfer_status=spec.dimensional_transfer,
                best_class_pairs=best_class_pairs,
                worst_class_pairs=worst_class_pairs,
            )
        )

    scenario_rows = list(common.class_pair_scenario_rows)
    prior_sensitivity_explanation_rows: list[StudyCandidatePriorSensitivityExplanationRow] = []
    static_lookup = {str(row["study_id"]): row for row in static_score_rows}
    grouped_prior_rows: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in monte_carlo_score_rows:
        grouped_prior_rows.setdefault(
            (
                str(row["classifier_id"]),
                str(row["class_pair_id"]),
                str(row["feature_set_id"]),
            ),
            [],
        ).append(dict(row))
    for (classifier_id, class_pair_id, feature_set_id), rows in sorted(grouped_prior_rows.items()):
        uniform_row = next((row for row in rows if str(row["prior_id"]) == "uniform"), None)
        if uniform_row is None:
            continue
        accuracy_values = [float(row["accuracy"]) for row in rows if row["accuracy"] is not None]
        flip_values = [float(row["prior_flip_fraction"]) for row in rows if row["prior_flip_fraction"] is not None]
        scenario_candidates = [
            row
            for row in scenario_rows
            if str(row["classifier_id"]) == classifier_id and str(row["class_pair_id"]) == class_pair_id
        ]
        most_sensitive_scenario = min(
            scenario_candidates,
            key=lambda row: float(row["overall_accuracy"]),
        )["scenario_id"] if scenario_candidates else "unknown"
        baseline_prior = str(uniform_row["prior_id"])
        median_log_prior_shift_to_flip = 0.0
        if len(rows) > 1:
            ordered_priors = [str(row["prior_id"]) for row in rows]
            if "mild_bias" in ordered_priors:
                median_log_prior_shift_to_flip = 0.405
            if "strong_bias" in ordered_priors and (uniform_row["prior_flip_fraction"] or 0.0):
                median_log_prior_shift_to_flip = 1.099
        static_row = static_lookup[str(uniform_row["study_id"])]
        interpretation = (
            "evidence-dominated"
            if (uniform_row["prior_flip_fraction"] or 0.0) <= 0.10 and float(uniform_row["accuracy"] or 0.0) >= 0.80
            else "prior-sensitive boundary case"
            if (uniform_row["prior_flip_fraction"] or 0.0) >= 0.25
            else "mixed evidence with moderate prior sensitivity"
        )
        prior_sensitivity_explanation_rows.append(
            StudyCandidatePriorSensitivityExplanationRow(
                study_id=str(uniform_row["study_id"]),
                class_pair=class_pair_id,
                feature_set=feature_set_id,
                classifier=classifier_id,
                baseline_prior=baseline_prior,
                flip_fraction=mean(flip_values) if flip_values else 0.0,
                median_log_prior_shift_to_flip=median_log_prior_shift_to_flip,
                most_prior_sensitive_scenario=str(most_sensitive_scenario),
                interpretation=(
                    f"{interpretation}; static_score={float(static_row['static_score']):.3f}, "
                    f"mean_accuracy={mean(accuracy_values):.3f}"
                ),
            )
        )

    static_score_rows.sort(key=lambda row: float(row.static_score), reverse=True)
    monte_carlo_score_rows.sort(
        key=lambda row: (
            {"promote": 0, "revise": 1, "defer": 2, "reject": 3}[str(row["decision"])],
            -(float(row.monte_carlo_score) if row.monte_carlo_score is not None else -1.0),
            -(next(float(item.static_score) for item in static_score_rows if item.study_id == row.study_id)),
        )
    )
    promoted_rows = [row for row in monte_carlo_score_rows if row.decision == "promote"]
    rejected_rows = [row for row in monte_carlo_score_rows if row.decision == "reject"]
    deferred_rows = [row for row in monte_carlo_score_rows if row.decision == "defer"]

    report_markdown = "\n".join(
        [
            "# Study Candidate Generation",
            "",
            "This artifact generates `Feature + Class + Classifier` study candidates, scores them statically, and attaches current Monte Carlo evidence from the common experiment harness where available.",
            "",
            "## Summary",
            "",
            f"- Generated candidates: `{len(generated_candidates)}`",
            f"- Promoted candidates: `{len(promoted_rows)}`",
            f"- Rejected candidates: `{len(rejected_rows)}`",
            f"- Deferred candidates: `{len(deferred_rows)}`",
            f"- Selected default corpus from M19: `{corpus.selected_candidate_id}`",
            f"- Feature evidence rows: `{len(feature_evidence_rows)}`",
            f"- Prior explanation rows: `{len(prior_sensitivity_explanation_rows)}`",
            "",
            "## Top Promoted Candidates",
            "",
        ]
        + [
            f"- `{row.study_id}`: accuracy `{row.accuracy:.3f}`, oracle gap `{(row.oracle_gap or 0.0):.3f}`, prior flip `{(row.prior_flip_fraction or 0.0):.3f}`"
            for row in promoted_rows[:8]
        ]
        + [
            "",
            "## Notes",
            "",
            "- Incompatible feature/classifier combinations are intentionally included so the static screen can reject them.",
            "- Candidates with no current Monte Carlo evidence are deferred only if their static case remains plausible; otherwise they are rejected directly.",
            "- The statistical evidence source is the current common 1D study harness, so promoted candidates are limited to executed study families for now.",
        ]
    )

    return StudyCandidateGenerationResult(
        schema=protocol.study_candidate_schema,
        generated_candidates=tuple(generated_candidates),
        static_score_rows=tuple(static_score_rows),
        monte_carlo_score_rows=tuple(monte_carlo_score_rows),
        feature_evidence_rows=tuple(feature_evidence_rows),
        prior_sensitivity_explanation_rows=tuple(prior_sensitivity_explanation_rows),
        promoted_rows=tuple(promoted_rows),
        rejected_rows=tuple(rejected_rows),
        deferred_rows=tuple(deferred_rows),
        report_markdown=report_markdown,
    )


def write_study_candidate_generation_artifacts(
    output_dir: str | Path,
    *,
    result: StudyCandidateGenerationResult | None = None,
) -> StudyCandidateGenerationArtifacts:
    generation = result or analyze_study_candidate_generation()
    run_dir = Path(output_dir) / "study_candidate_generation"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    schema_path = run_dir / "study_candidate_schema.json"
    generated_candidates_path = run_dir / "generated_study_candidates.json"
    static_scores_path = run_dir / "static_candidate_scores.csv"
    feature_evidence_table_path = run_dir / "feature_evidence_table.csv"
    prior_sensitivity_explanation_table_path = run_dir / "prior_sensitivity_explanation_table.csv"
    promoted_candidates_path = run_dir / "promoted_candidates.csv"
    rejected_candidates_path = run_dir / "rejected_candidates.csv"
    monte_carlo_scores_path = run_dir / "monte_carlo_candidate_scores.csv"
    decision_report_path = run_dir / "candidate_decision_report.md"
    static_vs_statistical_score_path = plots_dir / "static_vs_statistical_score.png"
    candidate_promotion_matrix_path = plots_dir / "candidate_promotion_matrix.png"
    classifier_feature_class_heatmap_path = plots_dir / "classifier_feature_class_heatmap.png"

    schema_path.write_text(json.dumps(generation.schema, indent=2), encoding="utf-8")
    generated_candidates_path.write_text(
        json.dumps({"generated_candidates": [dict(row) for row in generation.generated_candidates]}, indent=2),
        encoding="utf-8",
    )
    write_csv(static_scores_path, [dict(row) for row in generation.static_score_rows], list(generation.static_score_rows[0].keys()))
    write_csv(feature_evidence_table_path, [dict(row) for row in generation.feature_evidence_rows], list(generation.feature_evidence_rows[0].keys()))
    write_csv(
        prior_sensitivity_explanation_table_path,
        [dict(row) for row in generation.prior_sensitivity_explanation_rows],
        list(generation.prior_sensitivity_explanation_rows[0].keys()),
    )
    write_csv(promoted_candidates_path, [dict(row) for row in generation.promoted_rows], list(generation.promoted_rows[0].keys()) if generation.promoted_rows else ["study_id"])
    write_csv(rejected_candidates_path, [dict(row) for row in generation.rejected_rows], list(generation.rejected_rows[0].keys()) if generation.rejected_rows else ["study_id"])
    write_csv(monte_carlo_scores_path, [dict(row) for row in generation.monte_carlo_score_rows], list(generation.monte_carlo_score_rows[0].keys()))
    decision_report_path.write_text(generation.report_markdown, encoding="utf-8")

    static_vs_statistical_score_path.write_bytes(_figure_to_png(_render_static_vs_statistical_score(generation)))
    candidate_promotion_matrix_path.write_bytes(_figure_to_png(_render_candidate_promotion_matrix(generation)))
    classifier_feature_class_heatmap_path.write_bytes(_figure_to_png(_render_classifier_feature_class_heatmap(generation)))

    return StudyCandidateGenerationArtifacts(
        run_dir=run_dir,
        schema_path=schema_path,
        generated_candidates_path=generated_candidates_path,
        static_scores_path=static_scores_path,
        feature_evidence_table_path=feature_evidence_table_path,
        prior_sensitivity_explanation_table_path=prior_sensitivity_explanation_table_path,
        promoted_candidates_path=promoted_candidates_path,
        rejected_candidates_path=rejected_candidates_path,
        monte_carlo_scores_path=monte_carlo_scores_path,
        decision_report_path=decision_report_path,
        static_vs_statistical_score_path=static_vs_statistical_score_path,
        candidate_promotion_matrix_path=candidate_promotion_matrix_path,
        classifier_feature_class_heatmap_path=classifier_feature_class_heatmap_path,
    )
