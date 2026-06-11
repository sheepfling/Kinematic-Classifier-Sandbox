from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from kinematic_classifier_sandbox.markdown_builder import (
    MarkdownDocument,
    MermaidEdge,
    MermaidFlow,
    MermaidNode,
)
from kinematic_classifier_sandbox.utils.io import _write_json, _write_text, write_csv
from kinematic_classifier_sandbox.utils.runtime import repo_root

ROOT = repo_root()


@dataclass(frozen=True, slots=True)
class ClaimEvidence:
    claim_id: str
    claim: str
    pillar: str
    evidence_doc: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    current_status: str
    limitations: str
    next_work: str
    showcase_plot: str
    showcase_table: str
    supporting_equation: str


@dataclass(frozen=True, slots=True)
class ArtifactManifestEntry:
    path: str
    generated_by: str
    depends_on: tuple[str, ...]
    question_answered: str
    claim_supported: str
    status: str
    known_limitation: str


@dataclass(frozen=True, slots=True)
class WitnessProblem:
    witness: str
    purpose: str
    class_set: str
    feature_set: str
    classifier_or_filter: str
    prior_regime: str
    corpus_objective: str
    proves: str
    does_not_prove: str
    key_plot: str
    key_table: str
    status: str
    next_3d_extension: str


@dataclass(frozen=True, slots=True)
class LadderRung:
    rung: str
    algorithm: str
    evidence_source: str
    adds: str
    failure_addressed: str
    witness: str
    status: str


@dataclass(frozen=True, slots=True)
class RepoStoryArtifacts:
    run_dir: Path
    claim_matrix_path: Path
    artifact_manifest_path: Path
    artifact_graph_path: Path
    witness_matrix_path: Path
    ladder_matrix_path: Path
    status_report_path: Path
    repo_layer_diagram_path: Path
    artifact_dependency_graph_path: Path
    docs_written: tuple[Path, ...]


CLAIMS: tuple[ClaimEvidence, ...] = (
    ClaimEvidence(
        claim_id="C01",
        claim="Corpus quality is evaluated before classifier claims.",
        pillar="Corpus Explorer",
        evidence_doc=("docs/surveys/methodology_evaluation_framework.md", "docs/story/corpus_explorer.md"),
        artifact_paths=(
            "artifacts/corpus_adequacy_audit_v1/corpus_adequacy_scorecard.csv",
            "artifacts/corpus_adequacy_audit_v1/covariate_leakage_audit.csv",
            "artifacts/corpus_adequacy_audit_v1/class_pair_coverage_heatmap.png",
        ),
        test_paths=("tests/corpus/test_corpus_adequacy_audit.py", "tests/validation/test_class_validity.py"),
        current_status="strong",
        limitations="Closed-loop adequacy reruns now exist, but failing selected corpora still require additional generation or search passes.",
        next_work="Use rerun recommendations to regenerate or reselect corpora until the selected packet clears the intended gates.",
        showcase_plot="plots/corpus_adequacy_scorecard.png",
        showcase_table="tables/covariate_leakage_audit.csv",
        supporting_equation="S_k balances validity, excitation, difficulty, leakage, triviality, and degeneracy terms.",
    ),
    ClaimEvidence(
        claim_id="C02",
        claim="Feature/class separability can be inspected statically.",
        pillar="Study Candidate Evaluator",
        evidence_doc=("docs/surveys/methodology_evaluation_framework.md", "docs/story/study_candidate_evaluator.md"),
        artifact_paths=(
            "artifacts/feature_analysis_v1/feature_separation_scores.csv",
            "artifacts/feature_analysis_v1/pairwise_auc_matrix.csv",
            "artifacts/feature_analysis_v1/pairwise_overlap_heatmap.png",
        ),
        test_paths=("tests/analysis/test_feature_analysis.py", "tests/analysis/test_pca_analysis.py"),
        current_status="strong",
        limitations="Feature reports now warn on history-bearing and correlated bundles, but governance warnings do not by themselves remove dependence risk.",
        next_work="Carry the same caveat policy through every downstream evidence-study report.",
        showcase_plot="plots/pairwise_overlap_heatmap.png",
        showcase_table="tables/feature_separation_scores.csv",
        supporting_equation="Pairwise overlap and AUC compare class distributions before classifier blame.",
    ),
    ClaimEvidence(
        claim_id="C03",
        claim="Priors are explicitly tested for fragility.",
        pillar="Evaluation And Promotion",
        evidence_doc=("docs/story/how_to_interpret_results.md", "docs/surveys/posterior_update_math.md"),
        artifact_paths=(
            "artifacts/prior_sensitivity_pointwise_v1/prior_sensitivity.csv",
            "artifacts/prior_sensitivity_windowed_robust_v1/prior_flip_thresholds.csv",
            "artifacts/prior_sensitivity_pointwise_v1/decision_flip_thresholds.png",
        ),
        test_paths=("tests/inference/test_prior_sensitivity_analysis.py",),
        current_status="strong",
        limitations="Prior sweeps are only as good as the plausible prior range.",
        next_work="Define study-specific prior-range policies.",
        showcase_plot="plots/prior_sensitivity.png",
        showcase_table="tables/posterior_flip_thresholds.csv",
        supporting_equation="Posterior log odds equal evidence log odds plus prior log odds.",
    ),
    ClaimEvidence(
        claim_id="C04",
        claim="Classifiers share a posterior/evidence contract.",
        pillar="Classifier Ladder",
        evidence_doc=("docs/surveys/classifier_ladder_and_contracts.md", "docs/story/algorithm_ladder.md"),
        artifact_paths=(
            "artifacts/generic_inference_contract/evidence_provider_schema.json",
            "artifacts/generic_inference_contract/posterior_history_schema.json",
            "artifacts/classification_evidence_proof/evidence_provider_manifest.json",
        ),
        test_paths=("tests/methodology/test_generic_inference_contract.py", "tests/methodology/test_generic_classification_evidence_proof.py"),
        current_status="strong",
        limitations="Core metrics are now normalized, but some rung-specific diagnostics still live in backend-specific side tables.",
        next_work="Keep advanced-diagnostic side tables aligned while preserving the shared required metric surface.",
        showcase_plot="plots/pointwise_vs_accumulator_posterior_timelines.png",
        showcase_table="tables/algorithm_ladder_proof.csv",
        supporting_equation="Each rung emits class evidence that normalizes into posterior history.",
    ),
    ClaimEvidence(
        claim_id="C05",
        claim="1D witness problems prove ladder layers.",
        pillar="1D Witness Suite",
        evidence_doc=("docs/witnesses/index.md", "docs/story/00_repo_story.md"),
        artifact_paths=(
            "artifacts/pointwise_baseline/confusion_final.csv",
            "artifacts/pointwise_baseline/pointwise_baseline_diagnostics.png",
            "artifacts/kalman_filter_bank/kalman_bank_diagnostics.png",
        ),
        test_paths=("tests/inference/test_pointwise_baseline.py", "tests/inference/test_windowed_baseline.py", "tests/inference/test_kalman_filter_bank.py"),
        current_status="strong",
        limitations="Witnesses do not prove 3D dynamic completeness.",
        next_work="Add matching 3D witness problems after vector adapters land.",
        showcase_plot="plots/pointwise_vs_accumulator_posterior_timelines.png",
        showcase_table="tables/toy_problem_summary.csv",
        supporting_equation="Each witness exercises one evidence construction and the shared posterior update.",
    ),
    ClaimEvidence(
        claim_id="C06",
        claim="Corpus Explorer can generate and score candidate data.",
        pillar="Corpus Explorer",
        evidence_doc=("docs/surveys/corpus_generation_and_search.md", "docs/story/corpus_explorer.md"),
        artifact_paths=(
            "artifacts/generic_corpus_exploration/candidate_scores.csv",
            "artifacts/generic_corpus_exploration/archive_coverage_heatmap.png",
            "artifacts/selected_generated_corpus/corpus_manifest.json",
        ),
        test_paths=("tests/analysis/test_feature_analysis.py", "tests/analysis/test_pca_analysis.py"),
        current_status="v1 complete",
        limitations="QD selection now triggers adequacy reruns, but archive objectives and search dimensions are still 1D-oriented.",
        next_work="Use adequacy reruns as an optimization loop and broaden objective/search dimensions before claiming broader generality.",
        showcase_plot="plots/candidate_corpus_comparison.png",
        showcase_table="tables/class_pair_coverage.csv",
        supporting_equation="theta ~ q(theta | o,b) and tau ~ G_b(theta,xi).",
    ),
    ClaimEvidence(
        claim_id="C07",
        claim="Advanced filters are promoted by demonstrated failure evidence.",
        pillar="Classifier Ladder",
        evidence_doc=("docs/surveys/dimensional_lift_and_advanced_filter_gates.md", "docs/story/algorithm_ladder.md"),
        artifact_paths=(
            "artifacts/advanced_filter_decision_v1/advanced_filter_decision_report.md",
            "artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv",
            "artifacts/filter_trace_validation_v1/filter_trace_validation_report.md",
        ),
        test_paths=("tests/methodology/test_generic_inference_contract.py", "tests/methodology/test_generic_classification_evidence_proof.py"),
        current_status="trace_validated + witness_supported by case",
        limitations="Trace validation, witness support, and broader study justification are separate layers; current promotions do not imply universal dominance over simpler methods.",
        next_work="Lift the witness and trace packets to vector PVA and 3D backend adapters, then expand study-justification sweeps.",
        showcase_plot="plots/advanced_filter_decision_matrix.png",
        showcase_table="tables/advanced_filter_method_comparison.csv",
        supporting_equation="Advanced filters emit the shared posterior evidence contract.",
    ),
    ClaimEvidence(
        claim_id="C08",
        claim="3D transition is a controlled lift, not a full rewrite.",
        pillar="Evaluation And Promotion",
        evidence_doc=("docs/surveys/dimensional_lift_and_advanced_filter_gates.md", "docs/showcase/09_3d_transition_plan.md"),
        artifact_paths=(
            "artifacts/dimensional_lift_audit/module_dimension_status.csv",
            "artifacts/dimensional_lift_audit/required_3d_adapters.md",
            "artifacts/dimensional_lift_audit/vector_proof_posterior_history.csv",
        ),
        test_paths=("tests/analysis/test_dimensional_lift_audit.py", "tests/corpus/exploration/test_backend_adapter_proof.py"),
        current_status="architectural",
        limitations="The repo now states a precise 3D-ready boundary: vector corpus, vector features, and vector contract proofs remain incomplete.",
        next_work="Implement the listed vector corpus, feature, and contract proofs before advancing to full 3D backend work.",
        showcase_plot="plots/dimension_lift_audit_chart.png",
        showcase_table="tables/module_dimension_status.csv",
        supporting_equation="The study candidate remains s = (D, f, C, m, pi, b).",
    ),
)


WITNESSES: tuple[WitnessProblem, ...] = (
    WitnessProblem("pointwise_overlap", "Baseline likelihood and prior machinery", "current 1D pointwise classes", "instantaneous observations", "pointwise likelihood", "prior_sensitivity_pointwise", "overlapping local evidence", "posterior flips and prior effects", "history or dynamics", "artifacts/pointwise_baseline/pointwise_baseline_diagnostics.png", "artifacts/pointwise_baseline/confusion_final.csv", "promote", "vector-valued pointwise observations"),
    WitnessProblem("windowed_outlier_extrema", "Raw versus robust feature behavior", "current 1D windowed classes", "raw and robust extrema", "windowed likelihood", "windowed raw and robust sweeps", "outlier extrema stress", "feature design changes posterior stability", "independence of overlapping windows", "artifacts/windowed_baseline/windowed_baseline_diagnostics.png", "artifacts/windowed_baseline/confusion_robust.csv", "revise/promote by case", "robust 3D extrema features"),
    WitnessProblem("sequential_history", "History beyond pointwise evidence", "current 1D accumulator classes", "pointwise evidence history", "sequential Bayes", "accumulator priors", "time-series ambiguity", "history improves confidence", "dynamics residuals", "artifacts/bayes_accumulator/bayes_accumulator_diagnostics.png", "artifacts/bayes_accumulator/confidence_crossings.csv", "promote", "3D evidence histories"),
    WitnessProblem("kalman_endpoint_match", "Dynamics evidence under endpoint ambiguity", "Kalman model classes", "innovation residuals", "Kalman bank", "Kalman config priors", "matched endpoint ambiguity", "innovation likelihood evidence", "general nonlinear 3D sufficiency", "artifacts/kalman_filter_bank/kalman_bank_diagnostics.png", "artifacts/kalman_filter_bank/confusion_final.csv", "promote", "3D state-space model bank"),
    WitnessProblem("transition_switching", "Mode transition logic before IMM", "switching scenarios", "mode transition evidence", "transition matrix accumulator", "transition matrix config", "switching trajectories", "transition evidence for static-class failure", "IMM/PF/RBPF implementation", "artifacts/transition_matrix_accumulator_v1/transition_matrix_diagnostics.png", "artifacts/transition_matrix_accumulator_v1/transition_matrix_scenario_summary.csv", "pass", "3D maneuver transition states"),
    WitnessProblem("generated_corpus_stress", "Generated hard or fragile examples", "selected generated corpus classes", "generated feature matrix", "corpus classifier scoring", "downstream study priors", "stress adequacy leakage validity and excitation", "corpus generation scoring and selection", "final corpus completeness", "artifacts/generic_corpus_exploration/selected_trajectory_gallery.png", "artifacts/generic_corpus_exploration/candidate_scores.csv", "v1 complete", "3D backend adapters and QD dimensions"),
)


LADDER: tuple[LadderRung, ...] = (
    LadderRung("0", "Pointwise", "log p(y_t | c)", "local likelihood", "no baseline", "pointwise_overlap", "promote"),
    LadderRung("1", "Windowed", "log p(phi_t | c)", "local history", "outliers and noise", "windowed_outlier_extrema", "revise/promote by case"),
    LadderRung("2", "Sequential Bayes", "recursive evidence", "memory", "pointwise ignores history", "sequential_history", "promote"),
    LadderRung("3", "Kalman bank", "innovation likelihood", "dynamics", "endpoint ambiguity", "kalman_endpoint_match", "promote"),
    LadderRung("4", "Transition matrix", "T_ij mode transition", "switching", "static class assumption", "transition_switching", "pass"),
    LadderRung("5", "IMM/PF/RBPF", "advanced state inference", "nonlinear switching and mixed latent state inference", "transition static nonlinear and latent-event failures", "advanced_filter_witnesses", "witness_supported / study_justified by case"),
)


ARTIFACT_MANIFEST: tuple[ArtifactManifestEntry, ...] = (
    ArtifactManifestEntry("artifacts/corpus_objectives/objective_validation_report.md", "src/kinematic_classifier_sandbox/corpus/objectives.py", ("experiments/corpus_objectives/common_1d_corpus_objectives.yaml",), "What corpus objectives are valid enough to drive generation?", "C06", "implemented", "Objectives still need richer 3D geometry and dynamics terms."),
    ArtifactManifestEntry("artifacts/candidate_generation/generated_candidates.csv", "src/kinematic_classifier_sandbox/corpus/exploration/candidate_generation.py", ("artifacts/corpus_objectives/objective_validation_report.md",), "Which candidate parameterizations were generated?", "C06", "implemented", "Candidate diversity is limited by current 1D sampler families."),
    ArtifactManifestEntry("artifacts/generic_corpus_exploration/candidate_scores.csv", "src/kinematic_classifier_sandbox/corpus/exploration/generic_corpus_exploration.py", ("artifacts/corpus_objectives/objective_validation_report.md",), "Can candidates be generated and scored?", "C06", "v1 complete", "Closed-loop QD hardening remains future work."),
    ArtifactManifestEntry("artifacts/generic_corpus_exploration/archive_coverage_heatmap.png", "src/kinematic_classifier_sandbox/corpus/exploration/generic_corpus_exploration.py", ("artifacts/generic_corpus_exploration/archive_cells.csv",), "Does the corpus explorer cover archive cells?", "C06", "v1 complete", "Archive dimensions are still 1D-oriented."),
    ArtifactManifestEntry("artifacts/selected_generated_corpus/corpus_manifest.json", "src/kinematic_classifier_sandbox/selected_generated_corpus.py", ("artifacts/generic_corpus_exploration/selected_corpus_manifest.json",), "What generated corpus was selected for evaluation?", "C06", "v1 complete", "Selection should be rerun after every adequacy-hardening pass."),
    ArtifactManifestEntry("artifacts/corpus_adequacy_audit_v1/corpus_adequacy_scorecard.csv", "src/kinematic_classifier_sandbox/corpus/adequacy_audit.py", ("artifacts/common_1d_classifier_study/dataset_manifest.json",), "Does the corpus pass adequacy gates before classifier claims?", "C01", "strong", "Some generated corpora still require hardening."),
    ArtifactManifestEntry("artifacts/corpus_adequacy_audit_v1/covariate_leakage_audit.csv", "src/kinematic_classifier_sandbox/corpus/adequacy_audit.py", ("artifacts/common_1d_classifier_study/dataset_manifest.json",), "Are there corpus shortcuts or leakage risks?", "C01", "strong", "Leakage checks are only as complete as the recorded covariates."),
    ArtifactManifestEntry("artifacts/class_validity/class_validity_scores.csv", "src/kinematic_classifier_sandbox/validation/class_validity.py", ("artifacts/class_validity/class_definition_schema.json",), "Are class definitions internally valid?", "C01", "implemented", "Class validity scores do not replace domain review."),
    ArtifactManifestEntry("artifacts/feature_analysis_v1/feature_separation_scores.csv", "src/kinematic_classifier_sandbox/analysis/feature_analysis.py", ("artifacts/feature_analysis_v1/feature_matrix.csv",), "Are classes separable under the current feature set?", "C02", "strong", "Static separability does not prove independent evidence."),
    ArtifactManifestEntry("artifacts/feature_analysis_v1/pairwise_auc_matrix.csv", "src/kinematic_classifier_sandbox/analysis/feature_analysis.py", ("artifacts/feature_analysis_v1/feature_matrix.csv",), "Which class pairs are separable by feature?", "C02", "strong", "Pairwise separability can hide multiclass interactions."),
    ArtifactManifestEntry("artifacts/feature_analysis_v1/pairwise_overlap_heatmap.png", "src/kinematic_classifier_sandbox/analysis/feature_analysis.py", ("artifacts/feature_analysis_v1/pairwise_overlap_matrix.csv",), "Where do feature distributions overlap?", "C02", "strong", "Overlap depends on the sampled corpus."),
    ArtifactManifestEntry("artifacts/prior_sensitivity_pointwise_v1/prior_sensitivity.csv", "src/kinematic_classifier_sandbox/inference/prior_sensitivity_analysis.py", ("artifacts/pointwise_baseline/posterior_history.csv",), "Do pointwise decisions flip under plausible priors?", "C03", "strong", "Prior ranges need study-specific policy."),
    ArtifactManifestEntry("artifacts/prior_sensitivity_windowed_robust_v1/prior_flip_thresholds.csv", "src/kinematic_classifier_sandbox/inference/prior_sensitivity_analysis.py", ("artifacts/windowed_baseline/posterior_history.csv",), "Where do robust windowed decisions flip?", "C03", "strong", "Robust features still inherit window-dependence assumptions."),
    ArtifactManifestEntry("artifacts/generic_inference_contract/evidence_provider_schema.json", "src/kinematic_classifier_sandbox/generic_inference_contract.py", ("artifacts/generic_inference_contract/classifier_output_schema.json",), "Do evidence providers share a contract?", "C04", "strong", "Optional metrics vary by rung."),
    ArtifactManifestEntry("artifacts/classification_evidence_proof/evidence_provider_manifest.json", "src/kinematic_classifier_sandbox/generic_classification_evidence_proof.py", ("artifacts/generic_inference_contract/evidence_provider_schema.json",), "Which evidence providers satisfy the shared contract?", "C04", "strong", "Contract coverage does not prove performance quality."),
    ArtifactManifestEntry("artifacts/common_1d_classifier_study/unified_posterior_history.csv", "src/kinematic_classifier_sandbox/common_experiment_harness.py", ("artifacts/common_1d_classifier_study/unified_likelihood_history.csv",), "Can different classifier families produce comparable posterior histories?", "C04", "implemented", "Comparability still depends on aligned corpus slices and feature sets."),
    ArtifactManifestEntry("artifacts/pointwise_baseline/pointwise_baseline_diagnostics.png", "src/kinematic_classifier_sandbox/inference/pointwise_baseline.py", ("artifacts/pointwise_baseline/posterior_history.csv",), "Does the pointwise witness expose likelihood and posterior behavior?", "C05", "promote", "No history or dynamics evidence."),
    ArtifactManifestEntry("artifacts/windowed_baseline/windowed_baseline_diagnostics.png", "src/kinematic_classifier_sandbox/inference/windowed_baseline.py", ("artifacts/windowed_baseline/feature_matrix.csv",), "Does the windowed witness expose raw versus robust feature behavior?", "C05", "revise/promote by case", "Overlapping windows can double-count evidence if misinterpreted."),
    ArtifactManifestEntry("artifacts/bayes_accumulator/bayes_accumulator_diagnostics.png", "src/kinematic_classifier_sandbox/inference/sequential_bayes_accumulator.py", ("artifacts/bayes_accumulator/posterior_history.csv",), "Does sequential history improve evidence over time?", "C05", "promote", "Does not model dynamics residuals."),
    ArtifactManifestEntry("artifacts/kalman_filter_bank/kalman_bank_diagnostics.png", "src/kinematic_classifier_sandbox/inference/kalman_filter_bank.py", ("artifacts/kalman_filter_bank/innovation_history.csv",), "Can innovation likelihoods serve as class evidence?", "C05", "promote", "Does not prove general nonlinear 3D performance."),
    ArtifactManifestEntry("artifacts/transition_matrix_accumulator_v1/transition_matrix_diagnostics.png", "src/kinematic_classifier_sandbox/inference/transition_matrix_accumulator.py", ("artifacts/transition_matrix_accumulator_v1/transition_matrix_posterior_history.csv",), "Does transition logic address static-class switching failures?", "C05", "pass", "Transition logic is now a baseline for IMM rather than a replacement."),
    ArtifactManifestEntry("artifacts/advanced_filter_decision_v1/advanced_filter_decision_summary.json", "src/kinematic_classifier_sandbox/advanced_filters/evaluation.py", ("artifacts/advanced_filter_decision_v1/advanced_filter_decision_evidence.json",), "What does the conservative go/no-go gate say before broader witness promotion claims?", "C07", "implemented", "This remains the conservative gate, not the full witness and trace story."),
    ArtifactManifestEntry("artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv", "src/kinematic_classifier_sandbox/advanced_filters/evaluation.py", ("artifacts/advanced_filter_comparison_v1/method_comparison.csv", "artifacts/advanced_filter_comparison_v1/advanced_filter_comparison_report.md"), "Which advanced methods are only witness-supported versus justified for broader study use?", "C07", "implemented", "Witness support and study justification remain witness-family specific."),
    ArtifactManifestEntry("artifacts/filter_trace_validation_v1/filter_trace_validation_report.md", "src/kinematic_classifier_sandbox/tracing/filter_trace_validation_packet.py", ("artifacts/filter_trace_validation_v1/method_trace_matrix.csv", "artifacts/filter_trace_validation_v1/filter_step_trace_schema.json"), "Which methods expose auditable prior, prediction, likelihood, posterior, and diagnostic trace packets?", "C07", "implemented", "Trace validation proves mechanical auditability, not promotion by itself."),
    ArtifactManifestEntry("artifacts/dimensional_lift_audit/module_dimension_status.csv", "src/kinematic_classifier_sandbox/analysis/dimensional_lift_audit.py", ("src/kinematic_classifier_sandbox",), "What must change for 3D transition?", "C08", "architectural", "3D adapters and dynamics are incomplete."),
    ArtifactManifestEntry("artifacts/validation_ladder/validation_ladder_decisions.csv", "src/kinematic_classifier_sandbox/validation/validation_ladder.py", ("artifacts/validation_ladder/validation_ladder_scores.csv",), "Which studies promote, revise, reject, or defer?", "C01;C02;C03;C04", "implemented", "Promotion quality depends on upstream corpus, feature, prior, and evidence checks."),
)


TRACKED_METHOD_SURFACES: tuple[tuple[str, str, str], ...] = (
    (
        "Algorithm coverage matrix",
        "artifacts/algorithm_coverage_matrix_v1/algorithm_coverage_matrix_report.md",
        "Broader algorithm map covering physics, benchmark TSC, neural sequence, representation, uncertainty, optimizer, and roadmap lanes.",
    ),
    (
        "Method validation operating system",
        "artifacts/method_validation_os_v1/method_validation_os_report.md",
        "Lane, witness, and promotion-status registry tying methods to controlled evidence rather than complexity preference.",
    ),
    (
        "Trajectory exploration backend registry",
        "artifacts/trajectory_exploration_backend_registry_v1/report.md",
        "Current and planned generator/search backends, including heuristic, CEM, MAP-Elites, PPO, CMA-ES, SAC, TD3, and MPC-style lanes.",
    ),
)





def _tuple_join(values: tuple[str, ...]) -> str:
    return ";".join(values)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _center_text(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    lines = text.split("\n")
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    heights = [box[3] - box[1] for box in line_boxes]
    total_height = sum(heights) + 6 * (len(lines) - 1)
    y = rect[1] + (rect[3] - rect[1] - total_height) / 2
    for line, box, height in zip(lines, line_boxes, heights):
        width = box[2] - box[0]
        x = rect[0] + (rect[2] - rect[0] - width) / 2
        draw.text((x, y), line, font=font, fill=(24, 32, 38))
        y += height + 6


def _draw_box(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], label: str, font: ImageFont.ImageFont) -> None:
    draw.rounded_rectangle(rect, radius=12, fill=(244, 247, 245), outline=(50, 65, 72), width=3)
    _center_text(draw, rect, label, font)


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line((*start, *end), fill=(50, 65, 72), width=3)
    if abs(end[1] - start[1]) >= abs(end[0] - start[0]):
        points = [(end[0], end[1]), (end[0] - 8, end[1] - 16), (end[0] + 8, end[1] - 16)]
    else:
        points = [(end[0], end[1]), (end[0] - 16, end[1] - 8), (end[0] - 16, end[1] + 8)]
    draw.polygon(points, fill=(50, 65, 72))


def render_repo_layer_diagram(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1400, 1750), (252, 252, 248))
    draw = ImageDraw.Draw(image)
    font = _font(24)
    labels = (
        "Study Candidate s\n(D, f, C, m, pi, b)",
        "Corpus Objective",
        "Backend / Generator / CorpusGym / Search / QD",
        "Validated Corpus",
        "Feature Set + Class Validity",
        "Evidence Provider",
        "Posterior Updater + Prior",
        "Metrics / Validation Ladder / Decision",
    )
    rects: list[tuple[int, int, int, int]] = []
    y = 70
    for label in labels:
        rect = (285, y, 1115, y + 130)
        _draw_box(draw, rect, label, font)
        rects.append(rect)
        y += 210
    for first, second in zip(rects, rects[1:]):
        _draw_arrow(draw, (700, first[3] + 10), (700, second[1] - 10))
    image.save(path)


def render_artifact_dependency_graph(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1700, 1000), (252, 252, 248))
    draw = ImageDraw.Draw(image)
    font = _font(20)
    nodes = [
        ("Objectives", (70, 120, 310, 230)),
        ("Candidates", (390, 120, 630, 230)),
        ("Selected\nCorpus", (710, 120, 950, 230)),
        ("Adequacy\nAudit", (1030, 120, 1270, 230)),
        ("Feature\nAnalysis", (1350, 120, 1590, 230)),
        ("Evidence\nContract", (230, 560, 470, 670)),
        ("Posterior\nHistory", (550, 560, 790, 670)),
        ("Evaluation", (870, 560, 1110, 670)),
        ("Validation\nLadder", (1190, 560, 1430, 670)),
    ]
    for label, rect in nodes:
        _draw_box(draw, rect, label, font)
    top = [rect for _, rect in nodes[:5]]
    for first, second in zip(top, top[1:]):
        _draw_arrow(draw, (first[2] + 8, (first[1] + first[3]) // 2), (second[0] - 8, (second[1] + second[3]) // 2))
    _draw_arrow(draw, (1470, 240), (350, 550))
    bottom = [rect for _, rect in nodes[5:]]
    for first, second in zip(bottom, bottom[1:]):
        _draw_arrow(draw, (first[2] + 8, (first[1] + first[3]) // 2), (second[0] - 8, (second[1] + second[3]) // 2))
    image.save(path)


def render_claim_evidence_markdown() -> str:
    report = MarkdownDocument("Claim Evidence Matrix")
    report.paragraph(
        "The machine-readable source is `artifacts/repo_story/claim_evidence_matrix.csv`. This page is generated from `kinematic_classifier_sandbox.repo_story.CLAIMS`."
    )
    report.table(
        [
            "Claim ID",
            "Claim",
            "Pillar",
            "Evidence docs",
            "Artifact paths",
            "Tests",
            "Status",
            "Limitation",
            "Next work",
        ],
        [
            (
                claim.claim_id,
                claim.claim,
                claim.pillar,
                "; ".join(report.inline_code(path) for path in claim.evidence_doc),
                "; ".join(report.inline_code(path) for path in claim.artifact_paths),
                "; ".join(report.inline_code(path) for path in claim.test_paths),
                claim.current_status,
                claim.limitations,
                claim.next_work,
            )
            for claim in CLAIMS
        ],
    )
    report.heading("Use Rule", level=2)
    report.paragraph(
        "Every new major repo claim should add a row with docs, artifacts, tests, status, limitation, and next work. "
        "A claim without an artifact and a limitation is not ready for the front door."
    )
    return report.text()


def render_artifact_graph_markdown() -> str:
    report_rows = (
        ("artifacts/corpus_adequacy_audit_v1/corpus_adequacy_report.md", "corpus_adequacy_scorecard.csv, class_pair_coverage.csv, covariate_leakage_audit.csv, class_balance.csv"),
        ("artifacts/feature_analysis_v1/feature_analysis_report.md", "feature_separation_scores.csv, pairwise_auc_matrix.csv, pairwise_overlap_matrix.csv, identifiability_matrix.csv"),
        ("artifacts/prior_sensitivity_pointwise_v1/prior_sensitivity_report.md", "prior_sensitivity.csv, prior_flip_thresholds.csv, prior_dominance_metrics.json"),
        ("artifacts/generic_corpus_exploration/corpus_exploration_report.md", "candidate_scores.csv, archive_cells.csv, backend_comparison.csv, selected_corpus_manifest.json"),
        ("artifacts/kalman_filter_bank/kalman_bank_report.md", "innovation_history.csv, posterior_history.csv, confusion_final.csv, kalman_model_definitions.json"),
        ("artifacts/transition_matrix_accumulator_v1/transition_matrix_accumulator_report.md", "transition_matrix_scenario_summary.csv, transition_matrix_posterior_history.csv, transition_matrix_config.yaml"),
        ("artifacts/advanced_filter_decision_v1/advanced_filter_decision_report.md", "advanced_filter_decision_summary.json, advanced_filter_decision_evidence.json"),
        ("artifacts/advanced_filter_comparison_v1/advanced_filter_comparison_report.md", "advanced_method_gate_matrix.csv, method_comparison.csv, advanced_method_promotion_cards.md"),
        ("artifacts/filter_trace_validation_v1/filter_trace_validation_report.md", "method_trace_matrix.csv, trace_requirement_matrix.csv, filter_step_trace_schema.json"),
        ("artifacts/dimensional_lift_audit/dimensional_lift_audit.md", "module_dimension_status.csv, scalar_assumption_inventory.csv, validation_results.json"),
    )
    flow = MermaidFlow(
        nodes=(
            MermaidNode("obj", "corpus\nobjectives"),
            MermaidNode("cand", "candidate\ngeneration"),
            MermaidNode("corpus", "selected\ngenerated\ncorpus"),
            MermaidNode("adequacy", "corpus\nadequacy\naudit"),
            MermaidNode("feat", "feature\nanalysis"),
            MermaidNode("evidence", "evidence\ncontract"),
            MermaidNode("posterior", "posterior\nhistories"),
            MermaidNode("evaluation", "evaluation\nartifacts"),
            MermaidNode("ladder", "validation\nladder"),
            MermaidNode("decision", "promotion\ndecision"),
        ),
        edges=(
            MermaidEdge("obj", "cand"),
            MermaidEdge("cand", "corpus"),
            MermaidEdge("corpus", "adequacy"),
            MermaidEdge("adequacy", "feat"),
            MermaidEdge("feat", "evidence"),
            MermaidEdge("evidence", "posterior"),
            MermaidEdge("posterior", "evaluation"),
            MermaidEdge("evaluation", "ladder"),
            MermaidEdge("ladder", "decision"),
        ),
    )

    report = MarkdownDocument("Artifact Graph")
    report.paragraph(
        "The graph source is `artifacts/repo_story/artifact_graph.json`; rendered diagrams are:"
    )
    report.bullet_list(
        [
            "`artifacts/repo_story/repo_layer_diagram.png`",
            "`artifacts/repo_story/artifact_dependency_graph.png`",
        ]
    )
    report.heading("Dependency Flow", level=2)
    report.mermaid(flow)
    report.heading("Manifest Dependencies", level=2)
    report.table(
        ["Artifact", "Depends on"],
        [
            (
                report.inline_code(entry.path),
                ", ".join(report.inline_code(item) for item in entry.depends_on),
            )
            for entry in ARTIFACT_MANIFEST
        ],
    )
    report.heading("Reports And Their Tables", level=2)
    report.table(
        ["Report", "Primary tables / structured inputs"],
        [(report.inline_code(report_path), tables) for report_path, tables in report_rows],
    )
    report.heading("Plots Supporting Claims", level=2)
    report.table(
        ["Claim", "Plot"],
        [(f"{claim.claim_id} {claim.claim}", report.inline_code(claim.showcase_plot)) for claim in CLAIMS],
    )
    report.paragraph(
        "The canonical machine-readable dependency set is generated from `kinematic_classifier_sandbox.repo_story.ARTIFACT_MANIFEST`."
    )
    return report.text()


def render_artifact_index_markdown() -> str:
    sections = (
        ("Corpus Objective And Explorer", {"C06"}),
        ("Corpus Adequacy And Class Validity", {"C01"}),
        ("Feature And Class Separability", {"C02"}),
        ("Prior And Posterior Behavior", {"C03", "C04"}),
        ("Witness Problems", {"C05"}),
        ("Promotion, Advanced Filters, And 3D Lift", {"C07", "C08", "C01;C02;C03;C04"}),
    )
    report = MarkdownDocument("Artifact Index")
    report.paragraph("This is the human-readable view of `artifact_manifest.json`. It is generated from the repo-story manifest.")
    for title, claim_ids in sections:
        rows = [
            (entry.path, entry.question_answered, entry.claim_supported)
            for entry in ARTIFACT_MANIFEST
            if entry.claim_supported in claim_ids
        ]
        report.heading(title, level=2)
        report.table(["Artifact", "Question answered", "Claim"], rows)
    report.heading("Tracked Method Surfaces", level=2)
    report.table(
        ["Surface", "Bundle", "Why it matters"],
        [(label, report.inline_code(path), summary) for label, path, summary in TRACKED_METHOD_SURFACES],
    )
    report.paragraph(
        "Every manifest entry includes `path`, `generated_by`, `depends_on`, `question_answered`, `claim_supported`, `status`, and `known_limitation`."
    )
    return report.text()


def render_proof_gallery() -> str:
    report = MarkdownDocument("Proof Gallery")
    report.paragraph("This gallery is generated from the canonical PLN-024 claim matrix.")
    for index, claim in enumerate(CLAIMS, start=1):
        report.heading(f"Claim {index}: {claim.claim}", level=2)
        report.bullet_list(
            [
                f"Pillar: {claim.pillar}",
                f"Supporting equation: {claim.supporting_equation}",
                f"Supporting plot: {report.markdown_link(claim.showcase_plot)}",
                f"Supporting table: {report.markdown_link(claim.showcase_table)}",
                f"Supporting artifact: {report.inline_code(claim.artifact_paths[0])}",
                f"Current limitation: {claim.limitations}",
                f"Next work: {claim.next_work}",
            ]
        )
    return report.text()


def render_story_index() -> str:
    report = MarkdownDocument("Claim-Oriented Showcase Index")
    report.paragraph("Use this as the team-facing showcase front door.")
    for index, claim in enumerate(CLAIMS, start=1):
        report.paragraph(f"{index}. {claim.claim}")
        report.bullet_list(
            [
                report.inline_code(claim.showcase_plot),
                report.inline_code(claim.showcase_table),
                report.inline_code(claim.artifact_paths[0]),
            ]
        )
    report.heading("Tracked Method Surfaces", level=2)
    report.bullet_list(
        [
            f"{label}: {report.inline_code(path)}"
            for label, path, _summary in TRACKED_METHOD_SURFACES
        ]
    )
    return report.text()


def render_team_packet_index() -> str:
    report = MarkdownDocument("Team Packet")
    report.paragraph(
        "This packet should be readable without opening source code. The repo is a methodology workbench "
        "for kinematic classification studies, currently proven through 1D witness problems."
    )
    report.heading("Front Door", level=2)
    report.ordered_list(
        [
            "`docs/story/00_repo_story.md`",
            "`docs/story/01_methodology_map.md`",
            "`docs/story/02_reading_order.md`",
            "`artifacts/showcase/story_index.md`",
            "`artifacts/repo_story/claim_evidence_matrix.csv`",
        ]
    )
    report.heading("Headline Products", level=2)
    report.ordered_list(
        [
            "Study Candidate Evaluator: Feature + Class + Classifier/Filter + Prior evaluation.",
            "Corpus Explorer: objective-driven corpus generation, search, adequacy, leakage, and class-validity evaluation.",
            "1D Witness Suite: controlled problems that prove methodology layers before 3D.",
        ]
    )
    report.heading("Claim-Oriented Review", level=2)
    report.bullet_list([f"{claim.claim}" for claim in CLAIMS])
    report.heading("Most Useful Artifacts", level=2)
    report.bullet_list(
        [
            "`artifacts/repo_story/top_20_artifacts.md`",
            "`artifacts/showcase/proof_gallery.md`",
            "`artifacts/showcase/story_index.md`",
            "`artifacts/repo_story/artifact_index.md`",
            "`artifacts/repo_story/witness_problem_matrix.csv`",
            "`artifacts/algorithm_coverage_matrix_v1/algorithm_coverage_matrix_report.md`",
            "`artifacts/method_validation_os_v1/method_validation_os_report.md`",
            "`artifacts/trajectory_exploration_backend_registry_v1/report.md`",
        ]
    )
    return report.text()


def render_status_report() -> str:
    report = MarkdownDocument("PLN-024 Status Report")
    report.paragraph("Last Updated: 2026-05-25")
    report.heading("Scope Covered", level=2)
    report.bullet_list(
        [
            "Front-door repo story under `docs/story/`.",
            "Canonical reading order and document roles.",
            "Canonical vocabulary and alias rules.",
            "Study Candidate Evaluator explainer.",
            "Corpus Explorer explainer.",
            "Algorithm ladder page and matrix.",
            "Algorithm-map and method-validation registry surfaces.",
            "Trajectory-exploration backend registry surface.",
            "Result interpretation checklist.",
            "Claim-to-evidence matrix with docs, artifacts, tests, limitations, and next work.",
            "Witness-problem cards for all six current 1D witnesses.",
            "Repo layer diagram and artifact dependency graph.",
            "Consolidated artifact manifest and artifact index.",
            "Claim-oriented showcase and team packet front doors.",
            "Repo-story artifacts generated from `src/kinematic_classifier_sandbox/repo_story.py`.",
        ]
    )
    report.heading("Artifacts Generated", level=2)
    report.bullet_list(
        [
            "`artifacts/repo_story/index.md`",
            "`artifacts/repo_story/glossary.json`",
            "`artifacts/repo_story/claim_evidence_matrix.csv`",
            "`artifacts/repo_story/witness_problem_matrix.csv`",
            "`artifacts/repo_story/algorithm_ladder_matrix.csv`",
            "`artifacts/repo_story/study_candidate_evaluator_summary.csv`",
            "`artifacts/repo_story/corpus_explorer_summary.csv`",
            "`artifacts/repo_story/result_interpretation_checklist.md`",
            "`artifacts/repo_story/artifact_graph.json`",
            "`artifacts/repo_story/artifact_manifest.json`",
            "`artifacts/repo_story/artifact_index.md`",
            "`artifacts/repo_story/top_20_artifacts.md`",
            "`artifacts/repo_story/repo_layer_diagram.png`",
            "`artifacts/repo_story/artifact_dependency_graph.png`",
            "`artifacts/repo_story/pln024_status_report.md`",
            "`artifacts/algorithm_coverage_matrix_v1/algorithm_coverage_matrix_report.md`",
            "`artifacts/method_validation_os_v1/method_validation_os_report.md`",
            "`artifacts/trajectory_exploration_backend_registry_v1/report.md`",
        ]
    )
    report.heading("Validation Evidence", level=2)
    report.bullet_list(
        [
            "Path audit: all referenced docs, artifacts, tests, witness plots, witness tables, and manifest dependencies exist.",
            (
                f"Manifest audit: `artifact_manifest.json` has {len(ARTIFACT_MANIFEST)} entries and every entry includes path, generator, "
                "dependencies, question, claim, status, and limitation."
            ),
            (
                "Witness audit: every witness card links to at least one plot and one table through "
                "`witness_problem_matrix.csv`."
            ),
            (
                "Regression evidence: run `PYTHONPYCACHEPREFIX=/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache "
                "MPLCONFIGDIR=/private/tmp/kinematic-classifier-sandbox-mpl python3 scripts/all.py`."
            ),
        ]
    )
    report.heading("Top 10 Review Artifacts", level=2)
    report.ordered_list(
        [
            "`docs/story/00_repo_story.md`",
            "`docs/story/01_methodology_map.md`",
            "`docs/story/02_reading_order.md`",
            "`docs/story/study_candidate_evaluator.md`",
            "`docs/story/corpus_explorer.md`",
            "`docs/story/algorithm_ladder.md`",
            "`docs/story/algorithm_map.md`",
            "`docs/story/how_to_interpret_results.md`",
            "`docs/witnesses/index.md`",
            "`artifacts/repo_story/claim_evidence_matrix.csv`",
            "`artifacts/showcase/proof_gallery.md`",
        ]
    )
    report.heading("Stale Or Duplicate Docs Found", level=2)
    report.bullet_list(
        [
            "`docs/surveys/toy_1d_bayesian_baseline.md` is explicitly marked superseded by the witness-problem framing.",
            "`docs/showcase/*.md` remains useful supporting material, but `artifacts/showcase/story_index.md` is the claim-oriented front door.",
        ]
    )
    report.heading("Remaining Engineering Follow-Up", level=2)
    report.paragraph(
        "The story layer is generated and validated. Future work should extend the methodology itself: 3D vector backends, 3D feature families, richer dynamics, and closed-loop QD search."
    )
    return report.text()


def render_repo_story_index() -> str:
    report = MarkdownDocument("Repo Story Artifact Index")
    report.paragraph(
        "This directory is the proof-navigation layer for the repo story. It is generated by `src/kinematic_classifier_sandbox/repo_story.py`."
    )
    report.heading("Front Door", level=2)
    report.bullet_list(
        [
            "`claim_evidence_matrix.csv`: headline claims mapped to docs, artifacts, tests, limitations, and next work.",
            "`artifact_manifest.json`: curated artifact entries with provenance and claim support.",
            "`artifact_index.md`: human-readable artifact index.",
            "`top_20_artifacts.md`: highest-signal artifacts for review.",
            "`witness_problem_matrix.csv`: witness-problem summary and evidence links.",
            "`algorithm_ladder_matrix.csv`: ladder rung summary.",
            "`study_candidate_evaluator_summary.csv`: canonical study-candidate layers and primary artifacts.",
            "`corpus_explorer_summary.csv`: canonical corpus-explorer layers and primary artifacts.",
            "`artifact_graph.json`: dependency graph source.",
            "`repo_layer_diagram.png`: repo layer diagram.",
            "`artifact_dependency_graph.png`: artifact dependency graph.",
            "`pln024_status_report.md`: status, validation evidence, and next methodology follow-up.",
        ]
    )
    report.heading("Canonical Docs", level=2)
    report.bullet_list(
        [
            "`docs/story/00_repo_story.md`",
            "`docs/story/01_methodology_map.md`",
            "`docs/story/02_reading_order.md`",
            "`docs/story/algorithm_map.md`",
            "`docs/story/corpus_explorer.md`",
            "`docs/story/claim_evidence_matrix.md`",
            "`docs/witnesses/index.md`",
        ]
    )
    report.heading("Adjacent Generated Bundles", level=2)
    report.bullet_list(
        [
            f"{label}: {report.inline_code(path)}"
            for label, path, _summary in TRACKED_METHOD_SURFACES
        ]
    )
    return report.text()


def render_top_20_artifacts() -> str:
    entries = [
        "docs/story/00_repo_story.md",
        "docs/story/01_methodology_map.md",
        "docs/story/02_reading_order.md",
        "docs/story/algorithm_map.md",
        "docs/story/study_candidate_evaluator.md",
        "docs/story/corpus_explorer.md",
        "docs/story/algorithm_ladder.md",
        "docs/story/how_to_interpret_results.md",
        "docs/witnesses/index.md",
        "artifacts/repo_story/claim_evidence_matrix.csv",
        "artifacts/repo_story/artifact_manifest.json",
        "artifacts/repo_story/witness_problem_matrix.csv",
        "artifacts/algorithm_coverage_matrix_v1/algorithm_coverage_matrix_report.md",
        "artifacts/method_validation_os_v1/method_validation_os_report.md",
        "artifacts/trajectory_exploration_backend_registry_v1/report.md",
        "artifacts/corpus_adequacy_audit_v1/corpus_adequacy_scorecard.csv",
        "artifacts/feature_analysis_v1/feature_separation_scores.csv",
        "artifacts/prior_sensitivity_pointwise_v1/prior_sensitivity.csv",
        "artifacts/generic_inference_contract/evidence_provider_schema.json",
        "artifacts/generic_corpus_exploration/candidate_scores.csv",
        "artifacts/selected_generated_corpus/corpus_manifest.json",
        "artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv",
    ]
    report = MarkdownDocument("Top 20 Story Artifacts To Review")
    report.ordered_list([f"`{entry}`" for entry in entries])
    report.paragraph("Status and validation summary: `artifacts/repo_story/pln024_status_report.md`.")
    return report.text()


def render_result_interpretation_checklist() -> str:
    report = MarkdownDocument("Result Interpretation Checklist")
    report.ordered_list(
        [
            "Did the corpus pass adequacy?",
            "Are the class labels valid?",
            "Are features excited?",
            "Are classes separable by oracle or static analysis?",
            "Does the classifier underperform the oracle?",
            "Are decisions prior-sensitive?",
            "Is confusion localized by class pair, time phase, or sensor regime?",
            "Does a more advanced method address a demonstrated failure?",
        ]
    )
    report.paragraph(
        "Do not promote a leaderboard claim until corpus adequacy, class validity, feature excitation, prior sensitivity, and oracle gap are understood."
    )
    return report.text()


def write_repo_story_artifacts(
    output_dir: str | Path = ROOT / "artifacts",
    *,
    docs_root: str | Path | None = ROOT / "docs",
    source_root: str | Path = ROOT,
    write_showcase: bool = True,
) -> RepoStoryArtifacts:
    output_root = Path(output_dir)
    source_root = Path(source_root)
    run_dir = output_root / "repo_story"
    run_dir.mkdir(parents=True, exist_ok=True)

    claim_rows = [
        {
            "claim_id": claim.claim_id,
            "claim": claim.claim,
            "pillar": claim.pillar,
            "evidence_doc": _tuple_join(claim.evidence_doc),
            "artifact_paths": _tuple_join(claim.artifact_paths),
            "test_paths": _tuple_join(claim.test_paths),
            "current_status": claim.current_status,
            "limitations": claim.limitations,
            "next_work": claim.next_work,
        }
        for claim in CLAIMS
    ]
    claim_matrix_path = run_dir / "claim_evidence_matrix.csv"
    write_csv(claim_matrix_path, claim_rows, list(claim_rows[0]))

    witness_rows = [asdict(witness) for witness in WITNESSES]
    witness_matrix_path = run_dir / "witness_problem_matrix.csv"
    write_csv(witness_matrix_path, witness_rows, list(witness_rows[0]))

    ladder_rows = [asdict(rung) for rung in LADDER]
    ladder_matrix_path = run_dir / "algorithm_ladder_matrix.csv"
    write_csv(ladder_matrix_path, ladder_rows, list(ladder_rows[0]))

    artifact_manifest_path = run_dir / "artifact_manifest.json"
    _write_json(artifact_manifest_path, [asdict(entry) for entry in ARTIFACT_MANIFEST])

    artifact_graph = {
        "nodes": [
            {"id": re.sub(r"[^a-z0-9]+", "_", entry.path.lower()).strip("_"), "path": entry.path, "claim": entry.claim_supported}
            for entry in ARTIFACT_MANIFEST
        ],
        "edges": [
            [dependency, entry.path]
            for entry in ARTIFACT_MANIFEST
            for dependency in entry.depends_on
        ],
    }
    artifact_graph_path = run_dir / "artifact_graph.json"
    _write_json(artifact_graph_path, artifact_graph)

    _write_json(run_dir / "glossary.json", {
        "StudyCandidate": "A proposed study unit s = (D, f, C, m, pi, b) combining corpus, feature set, class set or class pair, classifier/filter family, prior regime, and optional backend.",
        "CorpusObjective": "A declarative target for corpus coverage, class validity, feature excitation, stress, backend capability, and leakage constraints.",
        "CorpusCandidate": "A generated or selected trajectory bundle before final adequacy and selection gates.",
        "SelectedCorpus": "A corpus that has passed enough objective, validity, leakage, and adequacy checks to feed a study candidate.",
        "FeatureSet": "A named set of computed features used as evidence inputs or separability diagnostics.",
        "ClassSet": "The full set of labels a study is allowed to distinguish.",
        "ClassPair": "A two-class slice used for pairwise separability, overlap, AUC, and prior-sensitivity analysis.",
        "EvidenceProvider": "A pointwise, windowed, sequential, filter, or transition model that converts observations, features, or residuals into class evidence.",
        "PosteriorUpdater": "Shared machinery that combines priors and evidence into normalized posterior histories.",
        "ClassifierFamily": "A family of evidence providers with a common construction.",
        "FilterBackend": "A dynamics-aware model or backend that produces residuals, innovations, state estimates, or likelihoods.",
        "PriorRegime": "The class prior configuration and prior stress used to test decision fragility.",
        "ValidationLadder": "The ordered checks that turn a study candidate into a promotion decision.",
        "WitnessProblem": "A controlled small problem used to prove a methodology layer before 3D lift.",
        "PromotionDecision": "The final disposition assigned to a study candidate: promote, revise, reject, or defer.",
    })
    write_csv(run_dir / "study_candidate_evaluator_summary.csv", [
        {"section": "study_candidate", "summary": "s = (D, f, C, m, pi, b)", "primary_doc": "docs/story/study_candidate_evaluator.md", "primary_artifact": "artifacts/study_candidate_generation/m18_validation_summary.json", "status": "canonical"},
        {"section": "static_checks", "summary": "class validity feature excitation leakage separability", "primary_doc": "docs/story/study_candidate_evaluator.md", "primary_artifact": "artifacts/feature_analysis_v1/feature_separation_scores.csv", "status": "implemented"},
        {"section": "monte_carlo_checks", "summary": "calibration accuracy over time confusion confidence", "primary_doc": "docs/story/study_candidate_evaluator.md", "primary_artifact": "artifacts/monte_carlo_accumulator/metrics_by_time.csv", "status": "implemented"},
        {"section": "prior_sensitivity", "summary": "flip thresholds and prior dominance", "primary_doc": "docs/story/how_to_interpret_results.md", "primary_artifact": "artifacts/prior_sensitivity_pointwise_v1/prior_flip_thresholds.csv", "status": "implemented"},
        {"section": "promotion_decision", "summary": "promote revise reject defer", "primary_doc": "docs/story/study_candidate_evaluator.md", "primary_artifact": "artifacts/validation_ladder/validation_ladder_decisions.csv", "status": "implemented"},
    ], ["section", "summary", "primary_doc", "primary_artifact", "status"])
    write_csv(run_dir / "corpus_explorer_summary.csv", [
        {"section": "corpus_objective", "summary": "declarative target for corpus coverage stress validity and leakage", "primary_doc": "docs/story/corpus_explorer.md", "primary_artifact": "artifacts/corpus_objectives/objective_validation_report.md", "status": "implemented"},
        {"section": "backend_adapter", "summary": "generator or simulator interface that produces trajectory candidates", "primary_doc": "docs/story/corpus_explorer.md", "primary_artifact": "artifacts/backend_adapter_proof/backend_manifest.json", "status": "implemented"},
        {"section": "backend_registry", "summary": "tracked exploration and generator backends with capability and phase labels", "primary_doc": "docs/story/corpus_explorer.md", "primary_artifact": "artifacts/trajectory_exploration_backend_registry_v1/backend_registry.csv", "status": "implemented"},
        {"section": "candidate_sampler", "summary": "samples theta under objective and backend constraints", "primary_doc": "docs/story/corpus_explorer.md", "primary_artifact": "artifacts/candidate_generation/generated_candidates.csv", "status": "implemented"},
        {"section": "class_validity", "summary": "scores whether labels are meaningful", "primary_doc": "docs/story/corpus_explorer.md", "primary_artifact": "artifacts/class_validity/class_validity_scores.csv", "status": "implemented"},
        {"section": "feature_excitation", "summary": "checks whether features are exercised", "primary_doc": "docs/story/corpus_explorer.md", "primary_artifact": "artifacts/generated_corpus_features/feature_excitation_scores.csv", "status": "implemented"},
        {"section": "leakage", "summary": "audits covariate shortcuts and degeneracy", "primary_doc": "docs/story/corpus_explorer.md", "primary_artifact": "artifacts/corpus_adequacy_audit_v1/covariate_leakage_audit.csv", "status": "implemented"},
        {"section": "qd_archive", "summary": "tracks coverage and elites", "primary_doc": "docs/story/corpus_explorer.md", "primary_artifact": "artifacts/quality_diversity_corpus/archive_coverage.csv", "status": "v1 complete"},
        {"section": "selected_corpus", "summary": "validated generated corpus for study evaluation", "primary_doc": "docs/story/corpus_explorer.md", "primary_artifact": "artifacts/selected_generated_corpus/corpus_manifest.json", "status": "v1 complete"},
    ], ["section", "summary", "primary_doc", "primary_artifact", "status"])
    _write_text(run_dir / "result_interpretation_checklist.md", render_result_interpretation_checklist())

    _write_text(run_dir / "index.md", render_repo_story_index())
    _write_text(run_dir / "artifact_index.md", render_artifact_index_markdown())
    _write_text(run_dir / "top_20_artifacts.md", render_top_20_artifacts())
    status_report_path = run_dir / "pln024_status_report.md"
    _write_text(status_report_path, render_status_report())
    repo_layer_diagram_path = run_dir / "repo_layer_diagram.png"
    artifact_dependency_graph_path = run_dir / "artifact_dependency_graph.png"
    render_repo_layer_diagram(repo_layer_diagram_path)
    render_artifact_dependency_graph(artifact_dependency_graph_path)

    docs_written: list[Path] = []
    if docs_root is not None:
        docs_root = Path(docs_root)
        claim_doc = docs_root / "story" / "claim_evidence_matrix.md"
        artifact_doc = docs_root / "story" / "artifact_graph.md"
        _write_text(claim_doc, render_claim_evidence_markdown())
        _write_text(artifact_doc, render_artifact_graph_markdown())
        docs_written.extend([claim_doc, artifact_doc])

    if write_showcase:
        showcase_dir = output_root / "showcase"
        team_packet_dir = output_root / "team_packet"
        _write_text(showcase_dir / "proof_gallery.md", render_proof_gallery())
        _write_text(showcase_dir / "story_index.md", render_story_index())
        _write_text(team_packet_dir / "index.md", render_team_packet_index())

    validation = validate_repo_story_references(source_root=source_root, output_dir=output_root)
    if validation["missing"]:
        missing = "\n".join(f"{where}: {target}" for where, target in validation["missing"])
        raise FileNotFoundError(f"repo-story references are missing:\n{missing}")

    return RepoStoryArtifacts(
        run_dir=run_dir,
        claim_matrix_path=claim_matrix_path,
        artifact_manifest_path=artifact_manifest_path,
        artifact_graph_path=artifact_graph_path,
        witness_matrix_path=witness_matrix_path,
        ladder_matrix_path=ladder_matrix_path,
        status_report_path=status_report_path,
        repo_layer_diagram_path=repo_layer_diagram_path,
        artifact_dependency_graph_path=artifact_dependency_graph_path,
        docs_written=tuple(docs_written),
    )


def validate_repo_story_references(
    *,
    source_root: str | Path = ROOT,
    output_dir: str | Path = ROOT / "artifacts",
) -> dict[str, object]:
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    missing: list[tuple[str, str]] = []
    for claim in CLAIMS:
        for path in (*claim.evidence_doc, *claim.artifact_paths, *claim.test_paths):
            if not (source_root / path).exists() and not (output_dir / path.removeprefix("artifacts/")).exists():
                missing.append((claim.claim_id, path))
    for witness in WITNESSES:
        for path in (witness.key_plot, witness.key_table):
            if not (source_root / path).exists() and not (output_dir / path.removeprefix("artifacts/")).exists():
                missing.append((witness.witness, path))
    for entry in ARTIFACT_MANIFEST:
        for path in (entry.path, *entry.depends_on):
            if not (source_root / path).exists() and not (output_dir / path.removeprefix("artifacts/")).exists():
                missing.append((entry.path, path))
    return {
        "status": "pass" if not missing else "fail",
        "claim_count": len(CLAIMS),
        "witness_count": len(WITNESSES),
        "artifact_manifest_count": len(ARTIFACT_MANIFEST),
        "missing": tuple(missing),
    }
