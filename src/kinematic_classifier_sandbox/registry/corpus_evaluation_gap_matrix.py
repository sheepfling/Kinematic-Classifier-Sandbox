from __future__ import annotations

import importlib
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Literal

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..utils.io import _write_json, _write_text, write_csv
from ..utils.plotting import _figure_to_png, plt
from ..utils.runtime import repo_root

ROOT = repo_root()

CapabilityStatus = Literal["implemented", "partial", "doc_only", "missing"]
CapabilityScope = Literal["default_common_study_only", "generic_api", "selected_corpus_only"]


@dataclass(frozen=True, slots=True)
class CapabilityDocRef:
    path: str
    required_snippets: tuple[str, ...] = ()
    forbidden_snippets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CorpusEvaluationCapabilitySpec:
    capability_id: str
    title: str
    implementation_module: str | None
    implementation_function: str | None
    artifact_module: str | None
    artifact_writer: str | None
    canonical_artifacts: tuple[str, ...]
    test_paths: tuple[str, ...]
    markdown_docs: tuple[CapabilityDocRef, ...]
    latex_docs: tuple[CapabilityDocRef, ...]
    scope: CapabilityScope
    limitation_note: str
    force_status: CapabilityStatus | None = None
    materialize_invoker: Callable[[Path], object] | None = None


@dataclass(frozen=True, slots=True)
class CorpusEvaluationCapabilityRow:
    capability_id: str
    title: str
    implementation_target: str
    artifact_writer_target: str
    scope: CapabilityScope
    current_status: CapabilityStatus
    implementation_callable: bool
    artifact_writer_callable: bool
    tests_exist: bool
    markdown_docs_exist: bool
    latex_docs_exist: bool
    markdown_docs_coherent: bool
    latex_docs_coherent: bool
    canonical_artifact_count: int
    materialized: bool
    observed_artifact_classes: tuple[str, ...]
    coherence_issues: tuple[str, ...]
    limitation_note: str


@dataclass(frozen=True, slots=True)
class CorpusEvaluationGapMatrixResult:
    capability_rows: tuple[CorpusEvaluationCapabilityRow, ...]
    summary: dict[str, object]
    matrix_rows: tuple[dict[str, object], ...]
    issue_rows: tuple[dict[str, object], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CorpusEvaluationGapMatrixArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    matrix_path: Path
    coherence_issues_path: Path
    inventory_path: Path
    status_plot_path: Path


def _load_callable(module_name: str | None, function_name: str | None) -> Callable[..., object] | None:
    if module_name is None or function_name is None:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    value = getattr(module, function_name, None)
    return value if callable(value) else None


def _path_exists(relative_path: str) -> bool:
    return (ROOT / relative_path).exists()


def _doc_ref_status(refs: tuple[CapabilityDocRef, ...]) -> tuple[bool, bool, list[str]]:
    if not refs:
        return False, False, ["docs_missing"]
    doc_exists = True
    coherent = True
    issues: list[str] = []
    for ref in refs:
        path = ROOT / ref.path
        if not path.exists():
            doc_exists = False
            coherent = False
            issues.append(f"missing_doc:{ref.path}")
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in ref.required_snippets:
            if snippet not in text:
                coherent = False
                issues.append(f"missing_snippet:{ref.path}:{snippet}")
        for snippet in ref.forbidden_snippets:
            if snippet in text:
                coherent = False
                issues.append(f"forbidden_snippet:{ref.path}:{snippet}")
    return doc_exists, coherent, issues


def _extract_paths(value: object) -> list[Path]:
    if isinstance(value, Path):
        return [value]
    if is_dataclass(value):
        paths: list[Path] = []
        for field in fields(value):
            paths.extend(_extract_paths(getattr(value, field.name)))
        return paths
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        paths: list[Path] = []
        for item in value:
            paths.extend(_extract_paths(item))
        return paths
    if isinstance(value, tuple | list):
        paths: list[Path] = []
        for item in value:
            paths.extend(_extract_paths(item))
        return paths
    return []


def _artifact_class(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in {".md", ".pdf", ".tex", ".html"}:
        return "report"
    if suffix in {".csv", ".tsv"}:
        return "tabular"
    if suffix in {".json", ".yaml", ".yml"}:
        return "summary"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}:
        return "visual"
    return None


def _materialize(spec: CorpusEvaluationCapabilitySpec) -> tuple[tuple[str, ...], bool]:
    writer = spec.materialize_invoker or _load_callable(spec.artifact_module, spec.artifact_writer)
    if writer is None:
        return (), False
    with tempfile.TemporaryDirectory(prefix=f"kcs-corpus-gap-{spec.capability_id}-") as temp_dir:
        artifacts = writer(Path(temp_dir))
        observed = sorted(
            {
                artifact_class
                for path in _extract_paths(artifacts)
                if path.exists() and (artifact_class := _artifact_class(path)) is not None
            }
        )
    return tuple(observed), True


def _status_for_spec(
    spec: CorpusEvaluationCapabilitySpec,
    *,
    implementation_callable: bool,
    artifact_writer_callable: bool,
    tests_exist: bool,
    markdown_docs_exist: bool,
    latex_docs_exist: bool,
    markdown_docs_coherent: bool,
    latex_docs_coherent: bool,
) -> CapabilityStatus:
    if spec.force_status is not None:
        return spec.force_status
    if implementation_callable or artifact_writer_callable:
        if all((artifact_writer_callable, tests_exist, markdown_docs_exist, latex_docs_exist, markdown_docs_coherent, latex_docs_coherent)):
            return "implemented"
        return "partial"
    if markdown_docs_exist or latex_docs_exist:
        return "doc_only"
    return "missing"


def _selected_generated_corpus_invoker(output_dir: Path) -> object:
    from ..corpus.selected_generated_corpus import write_selected_generated_corpus_artifacts

    return write_selected_generated_corpus_artifacts(output_dir)


CAPABILITY_SPECS: tuple[CorpusEvaluationCapabilitySpec, ...] = (
    CorpusEvaluationCapabilitySpec(
        capability_id="corpus_adequacy_scoring",
        title="Corpus adequacy scoring",
        implementation_module="kinematic_classifier_sandbox.corpus.adequacy_audit",
        implementation_function="analyze_corpus_adequacy",
        artifact_module="kinematic_classifier_sandbox.corpus.adequacy_artifact_io",
        artifact_writer="write_corpus_adequacy_artifacts",
        canonical_artifacts=(
            "artifacts/corpus_adequacy_audit_v1/corpus_adequacy_scorecard.csv",
            "artifacts/corpus_adequacy_audit_v1/corpus_adequacy_summary.json",
            "artifacts/corpus_adequacy_audit_v1/corpus_adequacy_report.md",
        ),
        test_paths=("tests/corpus/test_corpus_adequacy_audit.py",),
        markdown_docs=(
            CapabilityDocRef(
                "docs/story/corpus_explorer.md",
                required_snippets=("default generated/common-study corpus evaluation", "selected generated corpus packet rerun"),
            ),
            CapabilityDocRef(
                "docs/surveys/corpus_generation_and_search.md",
                required_snippets=("current strongest supported corpus-evaluation path", "default generated/common-study corpora"),
            ),
        ),
        latex_docs=(
            CapabilityDocRef(
                "docs/latex/kinematic_classifier_workflow.tex",
                required_snippets=("default generated/common-study path", "selected-corpus rerun path"),
            ),
        ),
        scope="default_common_study_only",
        limitation_note="Automatic adequacy scoring is strong for the shipped generated/common-study path, but that should not be overstated as full arbitrary-corpus evaluation.",
    ),
    CorpusEvaluationCapabilitySpec(
        capability_id="feature_excitation_coverage",
        title="Feature excitation coverage",
        implementation_module="kinematic_classifier_sandbox.analysis.feature_analysis",
        implementation_function="analyze_feature_datasets",
        artifact_module="kinematic_classifier_sandbox.analysis.feature_analysis_artifact_io",
        artifact_writer="write_feature_analysis_artifacts",
        canonical_artifacts=(
            "artifacts/feature_analysis_v1/feature_excitation_matrix.csv",
            "artifacts/feature_analysis_v1/feature_excitation_summary.json",
            "artifacts/feature_analysis_v1/feature_excitation_heatmap.png",
        ),
        test_paths=("tests/analysis/test_feature_analysis.py",),
        markdown_docs=(
            CapabilityDocRef(
                "docs/surveys/feature_workflow.md",
                required_snippets=("feature_excitation_matrix.csv", "class_confusability_heatmap.png"),
            ),
        ),
        latex_docs=(
            CapabilityDocRef(
                "docs/latex/kinematic_classifier_methodology.tex",
                required_snippets=("feature excitation",),
            ),
        ),
        scope="default_common_study_only",
        limitation_note="The feature-analysis framework is reusable, but the shipped engineered feature registry is still explicitly 1D-scoped.",
    ),
    CorpusEvaluationCapabilitySpec(
        capability_id="class_pair_boundary_coverage",
        title="Class-pair boundary coverage",
        implementation_module="kinematic_classifier_sandbox.corpus.adequacy_audit",
        implementation_function="analyze_corpus_adequacy",
        artifact_module="kinematic_classifier_sandbox.corpus.adequacy_artifact_io",
        artifact_writer="write_corpus_adequacy_artifacts",
        canonical_artifacts=(
            "artifacts/corpus_adequacy_audit_v1/class_pair_coverage.csv",
            "artifacts/corpus_adequacy_audit_v1/class_pair_coverage_heatmap.png",
        ),
        test_paths=("tests/corpus/test_corpus_adequacy_audit.py",),
        markdown_docs=(
            CapabilityDocRef(
                "docs/surveys/corpus_generation_and_search.md",
                required_snippets=("class-pair boundary coverage",),
            ),
        ),
        latex_docs=(
            CapabilityDocRef(
                "docs/surveys/methodology_evaluation_framework.tex",
                required_snippets=("class_pair_coverage_heatmap.png",),
            ),
        ),
        scope="default_common_study_only",
        limitation_note="Boundary coverage is declared and audited for the current manifest, but the manifest remains coupled to the shipped common-study class geometry.",
    ),
    CorpusEvaluationCapabilitySpec(
        capability_id="class_balance",
        title="Class balance",
        implementation_module="kinematic_classifier_sandbox.corpus.adequacy_audit",
        implementation_function="analyze_corpus_adequacy",
        artifact_module="kinematic_classifier_sandbox.corpus.adequacy_artifact_io",
        artifact_writer="write_corpus_adequacy_artifacts",
        canonical_artifacts=("artifacts/corpus_adequacy_audit_v1/class_balance.csv",),
        test_paths=("tests/corpus/test_corpus_adequacy_audit.py",),
        markdown_docs=(
            CapabilityDocRef("docs/surveys/corpus_generation_and_search.md", required_snippets=("class balance and class-pair balance",)),
        ),
        latex_docs=(
            CapabilityDocRef("docs/surveys/corpus_generation_and_search.tex", required_snippets=("class balance and class-pair balance",)),
        ),
        scope="default_common_study_only",
        limitation_note="Balance checks are automatic for generated/common-study corpora, but they do not by themselves validate broader external-corpus semantics.",
    ),
    CorpusEvaluationCapabilitySpec(
        capability_id="leakage_audit",
        title="Leakage audit",
        implementation_module="kinematic_classifier_sandbox.corpus.adequacy_audit",
        implementation_function="analyze_corpus_adequacy",
        artifact_module="kinematic_classifier_sandbox.corpus.adequacy_artifact_io",
        artifact_writer="write_corpus_adequacy_artifacts",
        canonical_artifacts=(
            "artifacts/corpus_adequacy_audit_v1/covariate_leakage_audit.csv",
            "artifacts/corpus_adequacy_audit_v1/covariate_leakage_audit.png",
        ),
        test_paths=("tests/corpus/test_corpus_adequacy_audit.py", "tests/corpus/test_selected_generated_corpus.py"),
        markdown_docs=(
            CapabilityDocRef("docs/story/corpus_explorer.md", required_snippets=("audits leakage",)),
            CapabilityDocRef("docs/surveys/corpus_generation_and_search.md", required_snippets=("leakage control",)),
        ),
        latex_docs=(
            CapabilityDocRef("docs/latex/kinematic_classifier_workflow.tex", required_snippets=("balance and leakage reports",)),
        ),
        scope="default_common_study_only",
        limitation_note="Leakage is treated as a first-class penalty, but leakage control is strongest on generated/common-study corpora and selected reruns.",
    ),
    CorpusEvaluationCapabilitySpec(
        capability_id="triviality_and_degeneracy",
        title="Triviality and degeneracy checks",
        implementation_module="kinematic_classifier_sandbox.corpus.adequacy_audit",
        implementation_function="analyze_corpus_adequacy",
        artifact_module="kinematic_classifier_sandbox.corpus.adequacy_artifact_io",
        artifact_writer="write_corpus_adequacy_artifacts",
        canonical_artifacts=("artifacts/corpus_adequacy_audit_v1/corpus_degeneracy_report.csv",),
        test_paths=("tests/corpus/test_corpus_adequacy_audit.py",),
        markdown_docs=(
            CapabilityDocRef("docs/surveys/corpus_generation_and_search.md", required_snippets=("degeneracy control",)),
        ),
        latex_docs=(
            CapabilityDocRef("docs/latex/kinematic_classifier_methodology.tex", required_snippets=("triviality", "degeneracy"),),
        ),
        scope="default_common_study_only",
        limitation_note="These penalties are automated inside the adequacy score, but the thresholds are still tuned around the current witness corpus families.",
    ),
    CorpusEvaluationCapabilitySpec(
        capability_id="feature_class_confusability",
        title="Feature/class confusability",
        implementation_module="kinematic_classifier_sandbox.analysis.feature_analysis",
        implementation_function="analyze_feature_datasets",
        artifact_module="kinematic_classifier_sandbox.analysis.feature_analysis_artifact_io",
        artifact_writer="write_feature_analysis_artifacts",
        canonical_artifacts=(
            "artifacts/feature_analysis_v1/pairwise_auc_matrix.csv",
            "artifacts/feature_analysis_v1/pairwise_overlap_matrix.csv",
            "artifacts/feature_analysis_v1/class_confusability_heatmap.png",
        ),
        test_paths=("tests/analysis/test_feature_analysis.py",),
        markdown_docs=(
            CapabilityDocRef("docs/surveys/feature_workflow.md", required_snippets=("class-pair confusion pressure view",)),
        ),
        latex_docs=(
            CapabilityDocRef("docs/latex/kinematic_classifier_methodology.tex", required_snippets=("Class confusability map",)),
        ),
        scope="default_common_study_only",
        limitation_note="Confusability is well surfaced for the current feature geometry, but that geometry is still the shipped 1D/common-study feature space.",
    ),
    CorpusEvaluationCapabilitySpec(
        capability_id="pca_and_dimensionality_diagnostics",
        title="PCA and dimensionality diagnostics",
        implementation_module="kinematic_classifier_sandbox.analysis.pca_dimensionality_audit",
        implementation_function="analyze_pca_dimensionality",
        artifact_module="kinematic_classifier_sandbox.analysis.pca_dimensionality_audit",
        artifact_writer="write_pca_dimensionality_audit_artifacts",
        canonical_artifacts=(
            "artifacts/pca_analysis_v1/pca_report.md",
            "artifacts/pca_dimensionality_audit_v1/pca_component_sweep.csv",
            "artifacts/pca_dimensionality_audit_v1/variance_vs_error.png",
        ),
        test_paths=("tests/analysis/test_pca_analysis.py", "tests/analysis/test_pca_dimensionality_audit.py"),
        markdown_docs=(
            CapabilityDocRef("docs/surveys/feature_workflow.md", required_snippets=("Use PCA as a diagnostic",)),
        ),
        latex_docs=(
            CapabilityDocRef("docs/latex/kinematic_classifier_methodology.tex", required_snippets=("AUC, confusion, PCA, adequacy, and leakage methodology",)),
        ),
        scope="default_common_study_only",
        limitation_note="PCA/clusterability diagnostics are implemented, but they still summarize the shipped feature sets rather than a fully general external feature pipeline.",
    ),
    CorpusEvaluationCapabilitySpec(
        capability_id="selected_corpus_closed_loop_rerun",
        title="Selected-corpus closed-loop rerun",
        implementation_module="kinematic_classifier_sandbox.corpus.selected_generated_corpus",
        implementation_function="analyze_selected_generated_corpus",
        artifact_module="kinematic_classifier_sandbox.corpus.selected_generated_corpus_artifact_io",
        artifact_writer="write_selected_generated_corpus_artifacts",
        canonical_artifacts=(
            "artifacts/selected_generated_corpus/corpus_manifest.json",
            "artifacts/selected_generated_corpus/adequacy_summary.json",
            "artifacts/selected_generated_corpus/adequacy_regressions.csv",
        ),
        test_paths=("tests/corpus/test_selected_generated_corpus.py",),
        markdown_docs=(
            CapabilityDocRef("docs/story/corpus_explorer.md", required_snippets=("selected generated corpus packet rerun",)),
            CapabilityDocRef("docs/surveys/corpus_generation_and_search.md", required_snippets=("selected generated corpus packet rerun",)),
        ),
        latex_docs=(
            CapabilityDocRef("docs/latex/kinematic_classifier_workflow.tex", required_snippets=("selected-corpus rerun path",)),
        ),
        scope="selected_corpus_only",
        limitation_note="Closed-loop reruns are strong for the repo’s selected generated corpus packet, but that is narrower than arbitrary provided-corpus intake.",
        materialize_invoker=_selected_generated_corpus_invoker,
    ),
    CorpusEvaluationCapabilitySpec(
        capability_id="classifier_support_coverage_report",
        title="Classifier-support coverage report",
        implementation_module="kinematic_classifier_sandbox.corpus.coverage_report",
        implementation_function="analyze_coverage_report",
        artifact_module="kinematic_classifier_sandbox.corpus.coverage_artifact_io",
        artifact_writer="write_coverage_report_artifacts",
        canonical_artifacts=(
            "artifacts/coverage_report_v1/feature_set_summary.csv",
            "artifacts/coverage_report_v1/classifier_support.csv",
            "artifacts/coverage_report_v1/coverage_report.md",
        ),
        test_paths=("tests/corpus/test_coverage_report.py",),
        markdown_docs=(
            CapabilityDocRef("docs/surveys/feature_workflow.md", required_snippets=("coverage report artifacts",)),
        ),
        latex_docs=(
            CapabilityDocRef("docs/latex/kinematic_classifier_workflow.tex", required_snippets=("coverage\\_report\\_v1",)),
        ),
        scope="default_common_study_only",
        limitation_note="Classifier-support coverage is automatic for the declared classifier manifest, but that manifest is still the shipped common-study ladder surface.",
    ),
    CorpusEvaluationCapabilitySpec(
        capability_id="arbitrary_given_corpus_evaluation",
        title="Arbitrary given corpus evaluation",
        implementation_module="kinematic_classifier_sandbox.corpus.trajectory_backend_contract",
        implementation_function="analyze_trajectory_backend_contract",
        artifact_module="kinematic_classifier_sandbox.corpus.trajectory_backend_contract_rendering",
        artifact_writer="write_trajectory_backend_contract_artifacts",
        canonical_artifacts=(
            "artifacts/trajectory_backend_contract/backend_contract.json",
            "artifacts/trajectory_backend_contract/capability_matrix.csv",
            "artifacts/external_backend_examples/external_backend_examples_report.md",
        ),
        test_paths=("tests/corpus/test_trajectory_backend_contract.py", "tests/external_backend_examples/test_external_backend_examples.py"),
        markdown_docs=(
            CapabilityDocRef(
                "docs/story/corpus_explorer.md",
                required_snippets=("adapter contract for provided or external corpus sources", "not yet a full arbitrary-corpus adequacy pipeline"),
                forbidden_snippets=("full generic corpus evaluation for any provided corpus",),
            ),
            CapabilityDocRef(
                "docs/surveys/corpus_generation_and_search.md",
                required_snippets=("adapter/contract layer", "not yet a full arbitrary-corpus adequacy pipeline"),
                forbidden_snippets=("full generic corpus evaluation for any provided corpus",),
            ),
        ),
        latex_docs=(
            CapabilityDocRef(
                "docs/latex/kinematic_classifier_workflow.tex",
                required_snippets=("adapter/contract layer exists for provided corpus sources", "not yet a full arbitrary-corpus adequacy pipeline"),
                forbidden_snippets=("full generic corpus evaluation for any provided corpus",),
            ),
        ),
        scope="generic_api",
        limitation_note="The repo has backend-contract and external-adapter proofs, but not a hardened full adequacy/coverage/confusability pipeline for any arbitrary provided corpus.",
        force_status="partial",
    ),
)


def analyze_corpus_evaluation_gap_matrix(
    *,
    capability_ids: Iterable[str] | None = None,
    materialize: bool = False,
) -> CorpusEvaluationGapMatrixResult:
    selected = set(capability_ids) if capability_ids is not None else None
    rows: list[CorpusEvaluationCapabilityRow] = []
    issue_rows: list[dict[str, object]] = []
    for spec in CAPABILITY_SPECS:
        if selected is not None and spec.capability_id not in selected:
            continue
        implementation_callable = _load_callable(spec.implementation_module, spec.implementation_function) is not None
        artifact_writer_callable = _load_callable(spec.artifact_module, spec.artifact_writer) is not None
        tests_exist = all(_path_exists(path) for path in spec.test_paths)
        markdown_docs_exist, markdown_docs_coherent, markdown_issues = _doc_ref_status(spec.markdown_docs)
        latex_docs_exist, latex_docs_coherent, latex_issues = _doc_ref_status(spec.latex_docs)
        observed_artifact_classes: tuple[str, ...] = ()
        was_materialized = False
        if materialize:
            observed_artifact_classes, was_materialized = _materialize(spec)
        coherence_issues = tuple(
            dict.fromkeys(
                [
                    *(["code_missing"] if not implementation_callable and not artifact_writer_callable else []),
                    *(["tests_missing"] if not tests_exist else []),
                    *(["markdown_docs_missing"] if not markdown_docs_exist else []),
                    *(["latex_docs_missing"] if not latex_docs_exist else []),
                    *(["markdown_docs_incoherent"] if markdown_docs_exist and not markdown_docs_coherent else []),
                    *(["latex_docs_incoherent"] if latex_docs_exist and not latex_docs_coherent else []),
                    *markdown_issues,
                    *latex_issues,
                    *(["implemented_without_artifact_writer"] if spec.force_status != "partial" and implementation_callable and not artifact_writer_callable else []),
                ]
            )
        )
        current_status = _status_for_spec(
            spec,
            implementation_callable=implementation_callable,
            artifact_writer_callable=artifact_writer_callable,
            tests_exist=tests_exist,
            markdown_docs_exist=markdown_docs_exist,
            latex_docs_exist=latex_docs_exist,
            markdown_docs_coherent=markdown_docs_coherent,
            latex_docs_coherent=latex_docs_coherent,
        )
        row = CorpusEvaluationCapabilityRow(
            capability_id=spec.capability_id,
            title=spec.title,
            implementation_target=f"{spec.implementation_module}.{spec.implementation_function}" if spec.implementation_module and spec.implementation_function else "",
            artifact_writer_target=f"{spec.artifact_module}.{spec.artifact_writer}" if spec.artifact_module and spec.artifact_writer else "",
            scope=spec.scope,
            current_status=current_status,
            implementation_callable=implementation_callable,
            artifact_writer_callable=artifact_writer_callable,
            tests_exist=tests_exist,
            markdown_docs_exist=markdown_docs_exist,
            latex_docs_exist=latex_docs_exist,
            markdown_docs_coherent=markdown_docs_coherent,
            latex_docs_coherent=latex_docs_coherent,
            canonical_artifact_count=len(spec.canonical_artifacts),
            materialized=was_materialized,
            observed_artifact_classes=observed_artifact_classes,
            coherence_issues=coherence_issues,
            limitation_note=spec.limitation_note,
        )
        rows.append(row)
        for issue in coherence_issues:
            issue_rows.append(
                {
                    "capability_id": spec.capability_id,
                    "title": spec.title,
                    "issue": issue,
                    "scope": spec.scope,
                    "current_status": current_status,
                }
            )

    summary = {
        "capability_count": len(rows),
        "materialized": materialize,
        "status_counts": dict(sorted(Counter(row.current_status for row in rows).items())),
        "scope_counts": dict(sorted(Counter(row.scope for row in rows).items())),
        "issue_counts": dict(sorted(Counter(issue["issue"] for issue in issue_rows).items())),
        "automatic_evaluation_verdict": "yes for default/common-study and selected-corpus paths",
        "artifact_coverage_verdict": "strong" if any(row.current_status == "implemented" for row in rows) else "partial",
        "documentation_coherence_verdict": "mostly strong" if len(issue_rows) <= 4 else "mixed",
        "arbitrary_given_corpus_verdict": "partial",
    }
    matrix_rows = tuple(
        {
            "capability_id": row.capability_id,
            "title": row.title,
            "implementation_target": row.implementation_target,
            "artifact_writer_target": row.artifact_writer_target,
            "scope": row.scope,
            "current_status": row.current_status,
            "implementation_callable": row.implementation_callable,
            "artifact_writer_callable": row.artifact_writer_callable,
            "tests_exist": row.tests_exist,
            "markdown_docs_exist": row.markdown_docs_exist,
            "latex_docs_exist": row.latex_docs_exist,
            "markdown_docs_coherent": row.markdown_docs_coherent,
            "latex_docs_coherent": row.latex_docs_coherent,
            "canonical_artifact_count": row.canonical_artifact_count,
            "materialized": row.materialized,
            "observed_artifact_classes": "|".join(row.observed_artifact_classes),
            "coherence_issues": "|".join(row.coherence_issues),
            "limitation_note": row.limitation_note,
        }
        for row in rows
    )
    payload = CorpusEvaluationGapMatrixResult(
        capability_rows=tuple(rows),
        summary=summary,
        matrix_rows=matrix_rows,
        issue_rows=tuple(issue_rows),
        report_markdown="",
    )
    report_markdown = render_corpus_evaluation_gap_matrix_report(payload)
    return CorpusEvaluationGapMatrixResult(
        capability_rows=payload.capability_rows,
        summary=payload.summary,
        matrix_rows=payload.matrix_rows,
        issue_rows=payload.issue_rows,
        report_markdown=report_markdown,
    )


def _render_status_plot(result: CorpusEvaluationGapMatrixResult):
    labels = [row.capability_id for row in result.capability_rows]
    values = [{"implemented": 3, "partial": 2, "doc_only": 1, "missing": 0}[row.current_status] for row in result.capability_rows]
    colors = {
        "implemented": "#16a34a",
        "partial": "#d97706",
        "doc_only": "#2563eb",
        "missing": "#dc2626",
    }
    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.42 * len(labels) + 1.5)))
    ax.barh(range(len(labels)), values, color=[colors[row.current_status] for row in result.capability_rows])
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlim(0, 3.2)
    ax.set_xticks([0, 1, 2, 3], labels=["missing", "doc_only", "partial", "implemented"])
    ax.set_title("Corpus-Evaluation Capability Status", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def render_corpus_evaluation_gap_matrix_report(result: CorpusEvaluationGapMatrixResult) -> str:
    report = MarkdownDocument("Corpus Evaluation Gap Matrix")
    report.paragraph(
        "This audit answers from repo state whether the repository can automatically evaluate corpus adequacy, coverage, leakage, and confusability; "
        "whether the supporting artifact families exist; and whether code, Markdown docs, and LaTeX tell the same story."
    )
    report.heading("Executive Summary", level=2)
    report.bullet_list(
        [
            f"automatic evaluation: `{result.summary['automatic_evaluation_verdict']}`",
            f"artifact coverage: `{result.summary['artifact_coverage_verdict']}`",
            f"documentation coherence: `{result.summary['documentation_coherence_verdict']}`",
            f"generic arbitrary-corpus intake: `{result.summary['arbitrary_given_corpus_verdict']}`",
        ]
    )
    report.heading("Capability Matrix", level=2)
    report.table(
        ["capability", "status", "scope", "code", "artifacts", "tests", "docs", "latex"],
        [
            (
                f"`{row.capability_id}`",
                row.current_status,
                row.scope,
                "yes" if row.implementation_callable else "no",
                "yes" if row.artifact_writer_callable else "no",
                "yes" if row.tests_exist else "no",
                "yes" if row.markdown_docs_exist and row.markdown_docs_coherent else "no",
                "yes" if row.latex_docs_exist and row.latex_docs_coherent else "no",
            )
            for row in result.capability_rows
        ],
    )
    if result.issue_rows:
        report.heading("Coherence Issues", level=2)
        report.bullet_list(
            [
                f"`{issue['capability_id']}`: `{issue['issue']}`"
                for issue in result.issue_rows
            ]
        )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "The strongest supported path is the shipped generated/common-study corpus workflow.",
            "Selected generated corpus packets are also re-audited in a closed loop and are stronger than arbitrary provided-corpus intake.",
            "Adapter/contract proofs for provided or external corpus sources exist, but they should not be described as a full arbitrary-corpus adequacy pipeline.",
        ]
    )
    return report.text()


def write_corpus_evaluation_gap_matrix_artifacts(
    output_dir: str | Path,
    *,
    result: CorpusEvaluationGapMatrixResult | None = None,
    capability_ids: Iterable[str] | None = None,
    materialize: bool = False,
) -> CorpusEvaluationGapMatrixArtifacts:
    payload = result or analyze_corpus_evaluation_gap_matrix(capability_ids=capability_ids, materialize=materialize)
    output_root = Path(output_dir)
    run_dir = output_root / "corpus_evaluation_gap_matrix_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "corpus_evaluation_gap_matrix_report.md"
    summary_path = run_dir / "corpus_evaluation_gap_matrix_summary.json"
    matrix_path = run_dir / "corpus_evaluation_gap_matrix.csv"
    coherence_issues_path = run_dir / "corpus_evaluation_coherence_issues.csv"
    inventory_path = run_dir / "corpus_evaluation_gap_matrix_inventory.json"
    status_plot_path = run_dir / "corpus_evaluation_capability_status.png"

    _write_text(report_path, payload.report_markdown)
    _write_json(summary_path, payload.summary)
    write_csv(matrix_path, list(payload.matrix_rows), list(payload.matrix_rows[0]) if payload.matrix_rows else [])
    write_csv(
        coherence_issues_path,
        list(payload.issue_rows),
        ["capability_id", "title", "issue", "scope", "current_status"] if payload.issue_rows else [],
    )
    _write_json(
        inventory_path,
        [
            {
                "capability_id": spec.capability_id,
                "title": spec.title,
                "implementation_module": spec.implementation_module,
                "implementation_function": spec.implementation_function,
                "artifact_module": spec.artifact_module,
                "artifact_writer": spec.artifact_writer,
                "canonical_artifacts": list(spec.canonical_artifacts),
                "test_paths": list(spec.test_paths),
                "markdown_docs": [ref.path for ref in spec.markdown_docs],
                "latex_docs": [ref.path for ref in spec.latex_docs],
                "scope": spec.scope,
                "limitation_note": spec.limitation_note,
            }
            for spec in CAPABILITY_SPECS
        ],
    )
    status_plot_path.write_bytes(_figure_to_png(_render_status_plot(payload)))
    return CorpusEvaluationGapMatrixArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        matrix_path=matrix_path,
        coherence_issues_path=coherence_issues_path,
        inventory_path=inventory_path,
        status_plot_path=status_plot_path,
    )
