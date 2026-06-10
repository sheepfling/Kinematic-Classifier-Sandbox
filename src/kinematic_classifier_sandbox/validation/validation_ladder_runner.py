from __future__ import annotations

from ..common_experiment.runner import analyze_common_experiment
from ..common_experiment.contracts import CommonExperimentResult
from ..corpus.autodevelopment import analyze_corpus_autodevelopment
from ..corpus.autodevelopment_types import CorpusAutodevelopmentResult
from ..study_candidate_generation import analyze_study_candidate_generation
from ..study_candidate_generation import StudyCandidateGenerationResult
from ..study_candidate_protocol import analyze_study_candidate_protocol
from ..study_candidate_protocol import StudyCandidateProtocolResult
from .validation_ladder_contracts import ValidationLadderResult
from .validation_ladder_rendering import render_validation_ladder_report


def _status_from_score(score: float, *, pass_threshold: float, partial_threshold: float) -> str:
    if score >= pass_threshold:
        return "pass"
    if score >= partial_threshold:
        return "partial"
    return "fail"


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _canonical_pair_id(class_a: str, class_b: str) -> str:
    ordered = sorted((class_a, class_b))
    return f"{ordered[0]}_vs_{ordered[1]}"


def analyze_validation_ladder(
    *,
    seed: int = 7,
    trajectories_per_case: int = 6,
    protocol_result: StudyCandidateProtocolResult | None = None,
    common_result: CommonExperimentResult | None = None,
    corpus_result: CorpusAutodevelopmentResult | None = None,
    study_generation_result: StudyCandidateGenerationResult | None = None,
) -> ValidationLadderResult:
    protocol = protocol_result or analyze_study_candidate_protocol()
    common = common_result or analyze_common_experiment(seed=seed, trajectories_per_case=trajectories_per_case)
    corpus = corpus_result or analyze_corpus_autodevelopment(seed=seed)
    study_generation = study_generation_result or analyze_study_candidate_generation(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        protocol_result=protocol,
        common_result=common,
        corpus_result=corpus,
    )
    selected_corpus = next(
        evaluation for evaluation in corpus.candidate_evaluations if evaluation.spec.candidate_id == corpus.selected_candidate_id
    )

    static_lookup = {
        str(row["study_id"]): dict(row)
        for row in study_generation.static_score_rows
        if str(row["prior_id"]) == "uniform"
    }
    monte_carlo_lookup = {
        str(row["study_id"]): dict(row)
        for row in study_generation.monte_carlo_score_rows
        if str(row["prior_id"]) == "uniform"
    }
    pair_adequacy_lookup = {
        _canonical_pair_id(str(row["class_a"]), str(row["class_b"])): dict(row)
        for row in selected_corpus.adequacy.class_pair_rows
    }
    scenario_lookup: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in common.class_pair_scenario_rows:
        scenario_lookup.setdefault((str(row["classifier_id"]), str(row["class_pair_id"])), []).append(dict(row))
    duration_lookup: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in common.class_pair_duration_rows:
        duration_lookup.setdefault((str(row["classifier_id"]), str(row["class_pair_id"])), []).append(dict(row))
    prediction_lookup: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in common.pair_prediction_rows:
        prediction_lookup.setdefault((str(row["classifier_id"]), str(row["class_pair_id"])), []).append(dict(row))

    score_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    canonical_studies = sorted(static_lookup.keys())
    for study_id in canonical_studies:
        static_row = static_lookup[study_id]
        monte_carlo_row = monte_carlo_lookup.get(study_id)
        classifier_id = str(static_row["classifier_id"])
        class_pair_id = str(static_row["class_pair_id"])
        feature_set_id = str(static_row["feature_set_id"])
        prediction_rows = prediction_lookup.get((classifier_id, class_pair_id), [])
        scenario_rows = scenario_lookup.get((classifier_id, class_pair_id), [])
        duration_rows = duration_lookup.get((classifier_id, class_pair_id), [])
        class_a, class_b = class_pair_id.split("_vs_", 1)
        pair_adequacy = pair_adequacy_lookup.get(_canonical_pair_id(class_a, class_b))

        levels: list[dict[str, object]] = []

        static_score = float(static_row["static_score"])
        static_status = (
            "fail"
            if not bool(static_row["compatible"])
            else _status_from_score(static_score, pass_threshold=0.55, partial_threshold=0.40)
        )
        levels.append(
            {
                "level_id": 1,
                "level_name": "static_compatibility",
                "status": static_status,
                "score": static_score,
                "evidence_summary": f"compatible={static_row['compatible']}, static_score={static_score:.3f}",
                "linked_artifacts": ["artifacts/study_candidate_generation/static_candidate_scores.csv"],
            }
        )

        if pair_adequacy is None:
            corpus_score = 0.0
            corpus_status = "defer"
            corpus_summary = "pair-specific adequacy row missing"
        else:
            pair_status = str(pair_adequacy["status"])
            corpus_score = {"green": 0.85, "yellow": 0.60, "red": 0.25}[pair_status]
            if str(selected_corpus.adequacy.summary.covariate_status) == "fail":
                corpus_score = max(0.0, corpus_score - 0.15)
            corpus_status = _status_from_score(corpus_score, pass_threshold=0.75, partial_threshold=0.45)
            corpus_summary = (
                f"pair_status={pair_status}, pairwise_auc={float(pair_adequacy['pairwise_auc']):.3f}, "
                f"covariate_status={selected_corpus.adequacy.summary.covariate_status}"
            )
        levels.append(
            {
                "level_id": 2,
                "level_name": "corpus_adequacy",
                "status": corpus_status,
                "score": corpus_score,
                "evidence_summary": corpus_summary,
                "linked_artifacts": [
                    "artifacts/corpus_autodevelopment_v1/corpus_adequacy_comparison.csv",
                    "artifacts/corpus_autodevelopment_v1/selected_corpus_manifest.json",
                ],
            }
        )

        separability_score = float(static_row["expected_separability_score"])
        separability_status = _status_from_score(separability_score, pass_threshold=0.65, partial_threshold=0.50)
        levels.append(
            {
                "level_id": 3,
                "level_name": "feature_separability",
                "status": separability_status,
                "score": separability_score,
                "evidence_summary": f"expected_separability_score={separability_score:.3f}",
                "linked_artifacts": ["artifacts/study_candidate_generation/static_candidate_scores.csv"],
            }
        )

        oracle_accuracy = float(monte_carlo_row["oracle_accuracy"]) if monte_carlo_row and monte_carlo_row["oracle_accuracy"] is not None else 0.0
        oracle_status = _status_from_score(oracle_accuracy, pass_threshold=0.75, partial_threshold=0.55)
        levels.append(
            {
                "level_id": 4,
                "level_name": "oracle_separability",
                "status": oracle_status,
                "score": oracle_accuracy,
                "evidence_summary": f"oracle_accuracy={oracle_accuracy:.3f}",
                "linked_artifacts": ["artifacts/common_1d_classifier_study/oracle_classifier_results.csv"],
            }
        )

        if monte_carlo_row is None or monte_carlo_row["accuracy"] is None:
            performance_score = 0.0
            performance_status = "defer"
            performance_summary = "classifier performance unavailable"
        else:
            performance_score = float(monte_carlo_row["accuracy"])
            performance_status = _status_from_score(performance_score, pass_threshold=0.88, partial_threshold=0.72)
            performance_summary = f"accuracy={performance_score:.3f}, oracle_gap={float(monte_carlo_row['oracle_gap']):.3f}"
        levels.append(
            {
                "level_id": 5,
                "level_name": "classifier_performance",
                "status": performance_status,
                "score": performance_score,
                "evidence_summary": performance_summary,
                "linked_artifacts": [
                    "artifacts/study_candidate_generation/monte_carlo_candidate_scores.csv",
                    "artifacts/common_1d_classifier_study/metrics_by_class_pair.csv",
                ],
            }
        )

        if prediction_rows:
            overall_accuracy = sum(1.0 if row["predicted_class"] == row["true_class"] else 0.0 for row in prediction_rows) / len(
                prediction_rows
            )
            mean_confidence = sum(float(row["confidence"]) for row in prediction_rows) / len(prediction_rows)
            calibration_gap = abs(mean_confidence - overall_accuracy)
            confident_error_rate = (
                sum(
                    1
                    for row in prediction_rows
                    if float(row["confidence"]) >= 0.80 and row["predicted_class"] != row["true_class"]
                )
                / len(prediction_rows)
            )
            final_duration_row = max(duration_rows, key=lambda row: float(row["time"])) if duration_rows else None
            posterior_margin = float(final_duration_row["posterior_margin"]) if final_duration_row is not None else 0.0
            posterior_score = max(
                0.0,
                min(
                    1.0,
                    0.45 * (1.0 - calibration_gap)
                    + 0.35 * (1.0 - confident_error_rate)
                    + 0.20 * min(1.0, posterior_margin / 0.35),
                ),
            )
            posterior_status = _status_from_score(posterior_score, pass_threshold=0.80, partial_threshold=0.60)
            posterior_summary = (
                f"mean_confidence={mean_confidence:.3f}, calibration_gap={calibration_gap:.3f}, "
                f"confident_error_rate={confident_error_rate:.3f}, posterior_margin={posterior_margin:.3f}"
            )
        else:
            posterior_score = 0.0
            posterior_status = "defer"
            posterior_summary = "prediction rows unavailable"
        levels.append(
            {
                "level_id": 6,
                "level_name": "posterior_and_calibration_quality",
                "status": posterior_status,
                "score": posterior_score,
                "evidence_summary": posterior_summary,
                "linked_artifacts": ["artifacts/common_1d_classifier_study/unified_predictions.csv"],
            }
        )

        if monte_carlo_row is None or monte_carlo_row["prior_flip_fraction"] is None:
            prior_score = 0.0
            prior_status = "defer"
            prior_summary = "prior sensitivity unavailable"
        else:
            prior_flip_fraction = float(monte_carlo_row["prior_flip_fraction"])
            prior_score = max(0.0, 1.0 - prior_flip_fraction)
            prior_status = _status_from_score(prior_score, pass_threshold=0.92, partial_threshold=0.75)
            prior_summary = (
                f"uniform_accuracy={_format_score(float(monte_carlo_row['uniform_accuracy']) if monte_carlo_row['uniform_accuracy'] is not None else None)}, "
                f"strong_bias_accuracy={_format_score(float(monte_carlo_row['strong_bias_accuracy']) if monte_carlo_row['strong_bias_accuracy'] is not None else None)}, "
                f"prior_flip_fraction={prior_flip_fraction:.3f}"
            )
        levels.append(
            {
                "level_id": 7,
                "level_name": "prior_sensitivity",
                "status": prior_status,
                "score": prior_score,
                "evidence_summary": prior_summary,
                "linked_artifacts": ["artifacts/common_1d_classifier_study/prior_sensitivity_by_class_pair.csv"],
            }
        )

        if scenario_rows:
            stress_rows = [
                row for row in scenario_rows
                if str(row["scenario_family"]) in {"boundary", "short_horizon", "noise_stress", "outlier_stress", "irregular_sampling"}
            ]
            selected_rows = stress_rows or scenario_rows
            worst_accuracy = min(float(row["overall_accuracy"]) for row in selected_rows)
            mean_accuracy = sum(float(row["overall_accuracy"]) for row in selected_rows) / len(selected_rows)
            robustness_score = max(0.0, min(1.0, 0.60 * worst_accuracy + 0.40 * mean_accuracy))
            robustness_status = _status_from_score(robustness_score, pass_threshold=0.82, partial_threshold=0.62)
            worst_scenario = min(selected_rows, key=lambda row: float(row["overall_accuracy"]))
            robustness_summary = (
                f"worst_scenario={worst_scenario['scenario_id']} ({worst_scenario['scenario_family']}), "
                f"worst_accuracy={worst_accuracy:.3f}, mean_accuracy={mean_accuracy:.3f}"
            )
        else:
            robustness_score = 0.0
            robustness_status = "defer"
            robustness_summary = "scenario robustness rows unavailable"
        levels.append(
            {
                "level_id": 8,
                "level_name": "stress_and_adversarial_robustness",
                "status": robustness_status,
                "score": robustness_score,
                "evidence_summary": robustness_summary,
                "linked_artifacts": ["artifacts/common_1d_classifier_study/class_pair_scenario_study.csv"],
            }
        )

        dimensional_score = float(static_row["three_d_transferability_score"])
        dimensional_status = _status_from_score(dimensional_score, pass_threshold=0.70, partial_threshold=0.20)
        dimensional_summary = f"three_d_transferability_score={dimensional_score:.3f}"
        levels.append(
            {
                "level_id": 9,
                "level_name": "dimensional_transfer_assessment",
                "status": dimensional_status,
                "score": dimensional_score,
                "evidence_summary": dimensional_summary,
                "linked_artifacts": ["artifacts/dimensional_lift_audit/dimensional_lift_audit.md"],
            }
        )

        level_status_by_name = {str(level["level_name"]): str(level["status"]) for level in levels}
        known_gaps: list[str] = []
        if level_status_by_name["corpus_adequacy"] != "pass":
            known_gaps.append("selected corpus still has unresolved adequacy issues for this study family")
        if level_status_by_name["dimensional_transfer_assessment"] != "pass":
            known_gaps.append("3D transfer is partial and still needs vector-policy work")
        if level_status_by_name["stress_and_adversarial_robustness"] == "partial":
            known_gaps.append("stress robustness is not yet consistently strong across hard scenarios")

        if any(level_status_by_name[name] == "defer" for name in level_status_by_name if name != "promotion_decision"):
            final_decision = "defer"
            rationale = "Some validation levels still lack evidence, so the study cannot be finalized."
        elif any(
            level_status_by_name[name] == "fail"
            for name in ("static_compatibility", "feature_separability", "oracle_separability", "classifier_performance")
        ):
            final_decision = "reject"
            rationale = "The study fails one or more core static or statistical gates."
        elif (
            static_score >= 0.45
            and performance_score >= 0.88
            and posterior_score >= 0.75
            and prior_score >= 0.90
            and robustness_score >= 0.65
            and corpus_score >= 0.45
            and level_status_by_name["feature_separability"] != "fail"
            and level_status_by_name["oracle_separability"] != "fail"
        ):
            final_decision = "promote"
            rationale = "The study is strong on the core 1D ladder and is promotable despite remaining dimensional-transfer caveats."
        elif any(
            level_status_by_name[name] == "fail"
            for name in ("corpus_adequacy", "posterior_and_calibration_quality", "prior_sensitivity", "stress_and_adversarial_robustness")
        ):
            final_decision = "revise"
            rationale = "The study is viable but needs corpus, calibration, or robustness work before promotion."
        elif all(
            level_status_by_name[name] == "pass"
            for name in (
                "static_compatibility",
                "feature_separability",
                "oracle_separability",
                "classifier_performance",
                "posterior_and_calibration_quality",
                "prior_sensitivity",
            )
        ):
            final_decision = "promote"
            rationale = "The study passes the core ladder and is strong enough to promote within the current 1D methodology stack."
        else:
            final_decision = "revise"
            rationale = "The study has promising evidence but still contains partial gates that should be improved."

        decision_level_status = {"promote": "pass", "revise": "partial", "reject": "fail", "defer": "defer"}[final_decision]
        levels.append(
            {
                "level_id": 10,
                "level_name": "promotion_decision",
                "status": decision_level_status,
                "score": {"promote": 1.0, "revise": 0.65, "reject": 0.15, "defer": 0.0}[final_decision],
                "evidence_summary": rationale,
                "linked_artifacts": ["artifacts/study_candidate_generation/candidate_decision_report.md"],
            }
        )

        for level in levels:
            score_rows.append(
                {
                    "study_id": study_id,
                    "class_pair_id": class_pair_id,
                    "feature_set_id": feature_set_id,
                    "classifier_id": classifier_id,
                    "level_id": level["level_id"],
                    "level_name": level["level_name"],
                    "status": level["status"],
                    "score": level["score"],
                    "evidence_summary": level["evidence_summary"],
                }
            )

        decision_rows.append(
            {
                "study_id": study_id,
                "class_pair_id": class_pair_id,
                "feature_set_id": feature_set_id,
                "classifier_id": classifier_id,
                "static_score": static_score,
                "corpus_score": corpus_score,
                "oracle_accuracy": oracle_accuracy,
                "classifier_accuracy": performance_score,
                "posterior_quality_score": posterior_score,
                "prior_sensitivity_score": prior_score,
                "robustness_score": robustness_score,
                "dimensional_transfer_score": dimensional_score,
                "final_decision": final_decision,
                "decision_rationale": rationale,
                "known_gaps": " | ".join(known_gaps),
            }
        )

    decision_rows.sort(
        key=lambda row: (
            {"promote": 0, "revise": 1, "defer": 2, "reject": 3}[str(row["final_decision"])],
            -float(row["classifier_accuracy"]),
            -float(row["static_score"]),
        )
    )
    score_rows.sort(key=lambda row: (str(row["study_id"]), int(row["level_id"])))

    report_markdown = render_validation_ladder_report(
        ValidationLadderResult(
            contract_schema=protocol.validation_ladder_schema,
            score_rows=tuple(score_rows),
            decision_rows=tuple(decision_rows),
            report_markdown="",
        )
    )
    return ValidationLadderResult(
        contract_schema=protocol.validation_ladder_schema,
        score_rows=tuple(score_rows),
        decision_rows=tuple(decision_rows),
        report_markdown=report_markdown,
    )


__all__ = ["analyze_validation_ladder"]
