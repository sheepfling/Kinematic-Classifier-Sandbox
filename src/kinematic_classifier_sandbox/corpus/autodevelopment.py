from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ..analysis.feature_analysis import analyze_feature_datasets
from ..runtime_paths import prepare_matplotlib
from ..trajectory_generator import generate_trajectory_datasets
from ..utils.plotting import _figure_to_png
from .adequacy_audit import analyze_corpus_adequacy
from .autodevelopment_rendering import (
    _render_corpus_score_pareto,
    _render_difficulty_distribution,
    _render_feature_excitation_heatmap,
    _render_leakage_by_candidate,
    _render_report,
    render_corpus_autodevelopment_numeric_walkthrough_markdown,
)
from .autodevelopment_types import (
    CorpusAutodevelopmentArtifacts,
    CorpusAutodevelopmentResult,
    CorpusCandidateEvaluation,
)
from .autodevelopment_utils import (
    CorpusCandidateSpec,
    DEFAULT_OBJECTIVES_PATH,
    _balance_score,
    _boundary_coverage_score,
    _candidate_manifest_row,
    _candidate_tier_definitions,
    _degeneracy_penalty,
    _difficulty_diversity_score,
    _difficulty_distribution_rows,
    _feature_excitation_comparison_rows,
    _feature_excitation_score,
    _is_dominated,
    _leakage_penalty,
    _pareto_front_rows,
    _pareto_objectives,
    _scale_range,
    _status_score,
    _triviality_penalty,
    _default_candidate_specs,
    load_corpus_objectives,
)
from .policy import CorpusPolicySpec, load_corpus_policy_spec, score_corpus_autodevelopment_candidate


def analyze_corpus_autodevelopment(
    *,
    seed: int = 7,
    policy: CorpusPolicySpec | None = None,
) -> CorpusAutodevelopmentResult:
    objectives_path = Path(DEFAULT_OBJECTIVES_PATH)
    objectives = load_corpus_objectives(objectives_path)
    resolved_policy = policy or load_corpus_policy_spec()
    candidate_specs = _default_candidate_specs(seed)
    candidate_evaluations: list[CorpusCandidateEvaluation] = []
    candidate_manifest_rows: list[dict[str, object]] = []
    candidate_score_rows: list[dict[str, object]] = []
    rejected_candidate_rows: list[dict[str, object]] = []
    adequacy_rows: list[dict[str, object]] = []
    feature_excitation_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []

    for spec in candidate_specs:
        datasets = generate_trajectory_datasets(tier_definitions=_candidate_tier_definitions(spec), seed=spec.seed)
        distribution_rows = _difficulty_distribution_rows(datasets)
        difficulty_diversity_score = _difficulty_diversity_score(distribution_rows, objectives)
        manifest_row = _candidate_manifest_row(spec, distribution_rows)
        feature_analysis = analyze_feature_datasets(datasets=datasets)
        adequacy = analyze_corpus_adequacy(
            datasets=datasets,
            feature_analysis_result=feature_analysis,
        )
        balance_score = _balance_score(adequacy)
        boundary_coverage_score = _boundary_coverage_score(adequacy)
        feature_excitation_score = _feature_excitation_score(adequacy)
        leakage_penalty = _leakage_penalty(adequacy, objectives)
        triviality_penalty = _triviality_penalty(adequacy)
        degeneracy_penalty = _degeneracy_penalty(adequacy)
        score_row = {
            "candidate_id": spec.candidate_id,
            "adequacy_status": adequacy.summary.overall_status,
            "policy_id": resolved_policy.policy_id,
            "balance_score": balance_score,
            "boundary_coverage_score": boundary_coverage_score,
            "feature_excitation_score": feature_excitation_score,
            "difficulty_diversity_score": difficulty_diversity_score,
            "leakage_penalty": leakage_penalty,
            "triviality_penalty": triviality_penalty,
            "degeneracy_penalty": degeneracy_penalty,
            "overall_score": score_corpus_autodevelopment_candidate(
                resolved_policy,
                balance_score=balance_score,
                boundary_coverage_score=boundary_coverage_score,
                feature_excitation_score=feature_excitation_score,
                difficulty_diversity_score=difficulty_diversity_score,
                leakage_penalty=leakage_penalty,
                triviality_penalty=triviality_penalty,
                degeneracy_penalty=degeneracy_penalty,
            ),
        }
        candidate_score_rows.append(score_row)
        if adequacy.summary.overall_status in {"red", "fail"}:
            rejected_candidate_rows.append({"candidate_id": spec.candidate_id, "overall_score": score_row["overall_score"], "adequacy_status": adequacy.summary.overall_status})
        adequacy_rows.append({"candidate_id": spec.candidate_id, "adequacy_status": adequacy.summary.overall_status, "overall_score": score_row["overall_score"]})
        feature_excitation_rows.extend(_feature_excitation_comparison_rows(spec.candidate_id, adequacy))
        leakage_rows.extend([{**row, "candidate_id": spec.candidate_id} for row in adequacy.covariate_rows])
        candidate_manifest_rows.append(manifest_row)
        candidate_evaluations.append(
            CorpusCandidateEvaluation(
                spec=spec,
                feature_analysis=feature_analysis,
                adequacy=adequacy,
                manifest_row=manifest_row,
                score_row=score_row,
                adequacy_row={"candidate_id": spec.candidate_id, "adequacy_status": adequacy.summary.overall_status},
                feature_excitation_rows=tuple(_feature_excitation_comparison_rows(spec.candidate_id, adequacy)),
                leakage_rows=tuple(adequacy.covariate_rows),
                pareto_objectives=_pareto_objectives(score_row),
            )
        )

    pareto_front_rows = _pareto_front_rows(candidate_evaluations)
    selected = max(candidate_evaluations, key=lambda evaluation: float(evaluation.score_row["overall_score"]))
    result = CorpusAutodevelopmentResult(
        objectives_path=objectives_path,
        objectives=objectives,
        candidate_evaluations=tuple(candidate_evaluations),
        selected_candidate_id=selected.spec.candidate_id,
        candidate_manifest_rows=tuple(candidate_manifest_rows),
        candidate_score_rows=tuple(candidate_score_rows),
        rejected_candidate_rows=tuple(rejected_candidate_rows),
        pareto_front_rows=tuple(pareto_front_rows),
        adequacy_comparison_rows=tuple(adequacy_rows),
        feature_excitation_comparison_rows=tuple(feature_excitation_rows),
        leakage_comparison_rows=tuple(leakage_rows),
        report_markdown="",
    )
    report_markdown = _render_report(result)
    return CorpusAutodevelopmentResult(
        objectives_path=objectives_path,
        objectives=objectives,
        candidate_evaluations=tuple(candidate_evaluations),
        selected_candidate_id=selected.spec.candidate_id,
        candidate_manifest_rows=tuple(candidate_manifest_rows),
        candidate_score_rows=tuple(candidate_score_rows),
        rejected_candidate_rows=tuple(rejected_candidate_rows),
        pareto_front_rows=tuple(pareto_front_rows),
        adequacy_comparison_rows=tuple(adequacy_rows),
        feature_excitation_comparison_rows=tuple(feature_excitation_rows),
        leakage_comparison_rows=tuple(leakage_rows),
        report_markdown=report_markdown,
    )


def write_corpus_autodevelopment_artifacts(
    output_dir: str | Path,
    *,
    result: CorpusAutodevelopmentResult | None = None,
) -> CorpusAutodevelopmentArtifacts:
    autodevelopment = result or analyze_corpus_autodevelopment()
    run_dir = Path(output_dir) / "corpus_autodevelopment_v1"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    objectives_path = run_dir / "corpus_objectives.yaml"
    candidate_manifest_path = run_dir / "candidate_corpus_manifest.csv"
    candidate_scores_path = run_dir / "corpus_candidate_scores.csv"
    selected_manifest_path = run_dir / "selected_corpus_manifest.json"
    rejected_manifest_path = run_dir / "rejected_corpus_manifest.csv"
    pareto_front_path = run_dir / "corpus_pareto_front.csv"
    adequacy_comparison_path = run_dir / "corpus_adequacy_comparison.csv"
    feature_excitation_comparison_path = run_dir / "feature_excitation_comparison.csv"
    leakage_comparison_path = run_dir / "leakage_comparison.csv"
    report_path = run_dir / "corpus_autodevelopment_report.md"
    numeric_walkthrough_path = run_dir / "corpus_autodevelopment_numeric_walkthrough.md"
    corpus_score_pareto_path = plots_dir / "corpus_score_pareto.png"
    feature_excitation_heatmap_path = plots_dir / "feature_excitation_heatmap.png"
    leakage_by_candidate_path = plots_dir / "leakage_by_candidate.png"
    difficulty_distribution_by_candidate_path = plots_dir / "difficulty_distribution_by_candidate.png"

    objectives_path.write_text(Path(autodevelopment.objectives_path).read_text(encoding="utf-8"), encoding="utf-8")
    write_csv(candidate_manifest_path, list(autodevelopment.candidate_manifest_rows), list(autodevelopment.candidate_manifest_rows[0].keys()))
    write_csv(candidate_scores_path, list(autodevelopment.candidate_score_rows), list(autodevelopment.candidate_score_rows[0].keys()))
    write_csv(rejected_manifest_path, list(autodevelopment.rejected_candidate_rows), list(autodevelopment.rejected_candidate_rows[0].keys()))
    write_csv(pareto_front_path, list(autodevelopment.pareto_front_rows), list(autodevelopment.pareto_front_rows[0].keys()))
    write_csv(adequacy_comparison_path, list(autodevelopment.adequacy_comparison_rows), list(autodevelopment.adequacy_comparison_rows[0].keys()))
    write_csv(feature_excitation_comparison_path, list(autodevelopment.feature_excitation_comparison_rows), list(autodevelopment.feature_excitation_comparison_rows[0].keys()))
    write_csv(leakage_comparison_path, list(autodevelopment.leakage_comparison_rows), list(autodevelopment.leakage_comparison_rows[0].keys()))

    selected_evaluation = next(
        evaluation for evaluation in autodevelopment.candidate_evaluations if evaluation.spec.candidate_id == autodevelopment.selected_candidate_id
    )
    selected_manifest_payload = {
        "selected_candidate_id": autodevelopment.selected_candidate_id,
        "objectives_path": str(autodevelopment.objectives_path),
        "selected_spec": asdict(selected_evaluation.spec),
        "selected_score": selected_evaluation.score_row,
        "selected_adequacy_summary": asdict(selected_evaluation.adequacy.summary),
    }
    selected_manifest_path.write_text(json.dumps(selected_manifest_payload, indent=2), encoding="utf-8")
    report_path.write_text(autodevelopment.report_markdown, encoding="utf-8")
    numeric_walkthrough_path.write_text(render_corpus_autodevelopment_numeric_walkthrough_markdown(autodevelopment), encoding="utf-8")

    corpus_score_pareto_path.write_bytes(_figure_to_png(_render_corpus_score_pareto(autodevelopment)))
    feature_excitation_heatmap_path.write_bytes(_figure_to_png(_render_feature_excitation_heatmap(autodevelopment)))
    leakage_by_candidate_path.write_bytes(_figure_to_png(_render_leakage_by_candidate(autodevelopment)))
    difficulty_distribution_by_candidate_path.write_bytes(_figure_to_png(_render_difficulty_distribution(autodevelopment)))

    return CorpusAutodevelopmentArtifacts(
        run_dir=run_dir,
        objectives_path=objectives_path,
        candidate_manifest_path=candidate_manifest_path,
        candidate_scores_path=candidate_scores_path,
        selected_manifest_path=selected_manifest_path,
        rejected_manifest_path=rejected_manifest_path,
        pareto_front_path=pareto_front_path,
        adequacy_comparison_path=adequacy_comparison_path,
        feature_excitation_comparison_path=feature_excitation_comparison_path,
        leakage_comparison_path=leakage_comparison_path,
        report_path=report_path,
        numeric_walkthrough_path=numeric_walkthrough_path,
        corpus_score_pareto_path=corpus_score_pareto_path,
        feature_excitation_heatmap_path=feature_excitation_heatmap_path,
        leakage_by_candidate_path=leakage_by_candidate_path,
        difficulty_distribution_by_candidate_path=difficulty_distribution_by_candidate_path,
    )
