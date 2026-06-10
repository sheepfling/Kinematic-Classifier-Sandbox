from __future__ import annotations

import ast
import importlib
import json
import shlex
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

from ..utils.runtime import repo_root
from typing import Any, Literal

from ..markdown_builder import MarkdownDocument
from ..utils.io import _write_json, _write_text, write_csv
from ..utils.plotting import _figure_to_png, plt

ROOT = repo_root()
EXPORT_SCRIPT_PATH = ROOT / "scripts" / "export_artifacts.py"

ArtifactClass = Literal["report", "tabular", "summary", "visual"]
Audience = Literal["user_facing", "mixed", "maintainer_facing"]
Category = Literal["analysis", "study", "corpus", "inference", "validation", "methodology", "registry", "showcase"]
VisualizationPolicy = Literal["required", "exempt"]
ArtifactInvoker = Callable[[Path], object]

REPORT_SUFFIXES = {".md", ".pdf", ".tex", ".html"}
TABULAR_SUFFIXES = {".csv", ".tsv"}
SUMMARY_SUFFIXES = {".json", ".yaml", ".yml"}
VISUAL_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}


@dataclass(frozen=True, slots=True)
class ExportedSurfaceCallsite:
    surface_id: str
    writer_name: str
    module: str
    line_number: int
    target_name: str | None


@dataclass(frozen=True, slots=True)
class ExportedSurfaceMetadata:
    category: Category | None = None
    audience: Audience | None = None
    analysis_function: str | None = None
    required_artifact_classes: tuple[ArtifactClass, ...] | None = None
    visualization_policy: VisualizationPolicy | None = None
    visualization_exemption_reason: str | None = None
    rerun_command: str | None = None
    notes: str = ""
    invoker: ArtifactInvoker | None = None


@dataclass(frozen=True, slots=True)
class ExportedSurfaceSpec:
    surface_id: str
    module: str
    writer_name: str
    line_number: int
    category: Category
    audience: Audience
    analysis_function: str | None
    required_artifact_classes: tuple[ArtifactClass, ...]
    visualization_policy: VisualizationPolicy
    visualization_exemption_reason: str | None
    rerun_command: str
    notes: str
    invoker: ArtifactInvoker | None


@dataclass(frozen=True, slots=True)
class ExportedSurfaceAuditRow:
    surface_id: str
    module: str
    writer_name: str
    category: Category
    audience: Audience
    analysis_function: str | None
    required_artifact_classes: tuple[ArtifactClass, ...]
    declared_machine_artifact: bool
    visualization_policy: VisualizationPolicy
    visualization_exemption_reason: str | None
    rerun_command: str
    rerun_command_target_exists: bool
    writer_callable: bool
    exported_by_script: bool
    materialized: bool
    run_scope_name: str | None
    observed_artifact_classes: tuple[ArtifactClass, ...]
    missing_requirements: tuple[str, ...]
    notes: str


@dataclass(frozen=True, slots=True)
class ExportedSurfaceCoverageResult:
    surface_rows: tuple[ExportedSurfaceAuditRow, ...]
    summary: dict[str, object]
    coverage_matrix_rows: tuple[dict[str, object], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class ExportedSurfaceCoverageArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    coverage_matrix_path: Path
    missing_coverage_path: Path
    visualization_exemptions_path: Path
    rerun_commands_path: Path
    category_plot_path: Path
    inventory_path: Path


def _default_rerun_command() -> str:
    return "PYTHONPYCACHEPREFIX=../active/CACHE/kinematic-classifier-sandbox/.pycache python3 scripts/export_artifacts.py"


def _surface_id_from_name(name: str) -> str:
    surface_id = name
    for prefix in ("write_", "build_"):
        if surface_id.startswith(prefix):
            surface_id = surface_id[len(prefix) :]
    for suffix in ("_artifacts", "_artifact", "_path", "_png"):
        if surface_id.endswith(suffix):
            surface_id = surface_id[: -len(suffix)]
            break
    return surface_id


def _parse_export_script_calls(script_path: Path = EXPORT_SCRIPT_PATH) -> tuple[ExportedSurfaceCallsite, ...]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    import_map: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                import_map[alias.asname or alias.name] = (node.module, alias.name)

    main_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    callsites: list[ExportedSurfaceCallsite] = []
    for stmt in main_fn.body:
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(stmt, ast.Assign):
            value = stmt.value
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                target_name = stmt.targets[0].id
        elif isinstance(stmt, ast.Expr):
            value = stmt.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
            continue
        local_name = value.func.id
        if local_name not in import_map:
            continue
        module, imported_name = import_map[local_name]
        is_writer = imported_name.startswith("write_") and (
            imported_name.endswith("_artifact") or imported_name.endswith("_artifacts")
        )
        is_builder = imported_name.startswith("build_") and imported_name.endswith("_artifacts")
        if not (is_writer or is_builder):
            continue
        surface_id = _surface_id_from_name(target_name or imported_name)
        callsites.append(
            ExportedSurfaceCallsite(
                surface_id=surface_id,
                writer_name=imported_name,
                module=module,
                line_number=getattr(stmt, "lineno", 0),
                target_name=target_name,
            )
        )
    return tuple(callsites)


def _derive_category(module: str, surface_id: str) -> Category:
    if ".analysis." in module:
        return "analysis"
    if ".corpus." in module:
        return "corpus"
    if ".inference." in module:
        return "inference"
    if ".validation." in module:
        return "validation"
    if ".methodology." in module:
        return "methodology"
    if ".showcase." in module or ".story." in module:
        return "showcase"
    if ".registry." in module or "formal_math" in module or "strict_equation_audit" in module:
        return "registry"
    if "common_experiment" in module or "witnesses." in module or module.endswith(".artifacts"):
        return "study"
    return "study"


def _derive_audience(category: Category) -> Audience:
    if category in {"registry"}:
        return "maintainer_facing"
    if category in {"methodology", "showcase"}:
        return "mixed"
    return "user_facing"


REPORT_AND_MACHINE = ("report", "summary")
FULL_ANALYSIS_BUNDLE = ("report", "tabular", "summary", "visual")
REPORT_AND_VISUAL = ("report", "visual")
VISUAL_ONLY = ("visual",)


def _prior_variant_invoker(feature_mode: str | None = None) -> ArtifactInvoker:
    def _invoke(output_dir: Path) -> object:
        from ..inference.prior_sensitivity.artifact_io import write_prior_sensitivity_artifacts
        from ..inference.prior_sensitivity_analysis import (
            analyze_pointwise_prior_sensitivity,
            analyze_windowed_prior_sensitivity,
        )

        if feature_mode is None:
            result = analyze_pointwise_prior_sensitivity(seed=7)
        else:
            result = analyze_windowed_prior_sensitivity(seed=7, feature_mode=feature_mode)
        return write_prior_sensitivity_artifacts(output_dir, result=result)

    return _invoke


def _boundary_common_experiment_invoker(output_dir: Path) -> object:
    from ..common_experiment.artifact_io import write_common_experiment_artifacts

    return write_common_experiment_artifacts(
        output_dir,
        config_path=ROOT / "experiments" / "common_1d_boundary_study" / "common_experiment_config.yaml",
    )


def _showcase_invoker(output_dir: Path) -> object:
    from ..showcase.builder import build_showcase_artifacts

    return build_showcase_artifacts(output_dir, refresh=False, create_zip=False)


def _repo_story_invoker(output_dir: Path) -> object:
    from ..story.repo_story import write_repo_story_artifacts

    docs_root = output_dir / "_repo_story_docs"
    return write_repo_story_artifacts(output_dir, docs_root=docs_root, write_showcase=False)


METADATA_OVERRIDES: dict[str, ExportedSurfaceMetadata] = {
    "method_survey": ExportedSurfaceMetadata(
        category="study",
        audience="mixed",
        required_artifact_classes=("report",),
        visualization_policy="exempt",
        visualization_exemption_reason="Legacy survey export currently emits only a Markdown summary bundle.",
        notes="Legacy single-report export kept in the canonical inventory so the audit can surface its lighter contract.",
    ),
    "survey": ExportedSurfaceMetadata(
        category="study",
        audience="mixed",
        required_artifact_classes=("report",),
        visualization_policy="exempt",
        visualization_exemption_reason="Legacy survey export currently emits only a Markdown summary bundle.",
        notes="Legacy single-report export kept in the canonical inventory so the audit can surface its lighter contract.",
    ),
    "advanced_filter_contract": ExportedSurfaceMetadata(
        category="inference",
        analysis_function="analyze_advanced_filter_contract",
    ),
    "advanced_state_inference": ExportedSurfaceMetadata(
        category="inference",
        analysis_function="analyze_advanced_state_inference",
    ),
    "pointwise_prior": ExportedSurfaceMetadata(
        category="inference",
        analysis_function="analyze_pointwise_prior_sensitivity",
        invoker=_prior_variant_invoker(None),
    ),
    "windowed_raw_prior": ExportedSurfaceMetadata(
        category="inference",
        analysis_function="analyze_windowed_prior_sensitivity",
        invoker=_prior_variant_invoker("raw"),
    ),
    "windowed_robust_prior": ExportedSurfaceMetadata(
        category="inference",
        analysis_function="analyze_windowed_prior_sensitivity",
        invoker=_prior_variant_invoker("robust"),
    ),
    "boundary_common_experiment": ExportedSurfaceMetadata(
        category="study",
        invoker=_boundary_common_experiment_invoker,
        notes="Boundary-study rerun uses the dedicated boundary config instead of the default common study config.",
    ),
    "study_candidate_protocol": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=("report", "summary"),
        visualization_policy="exempt",
        visualization_exemption_reason="Protocol export is schema/report oriented and intentionally omits plots.",
    ),
    "generic_classification_evidence_proof": ExportedSurfaceMetadata(
        category="methodology",
        required_artifact_classes=("report", "summary"),
        visualization_policy="exempt",
        visualization_exemption_reason="Evidence proof export is a methodology/report surface without standalone plots.",
    ),
    "generic_filtering_contract": ExportedSurfaceMetadata(
        category="methodology",
        required_artifact_classes=("report", "summary"),
        visualization_policy="exempt",
        visualization_exemption_reason="Filtering contract export is a schema/report surface without standalone plots.",
    ),
    "generic_inference_contract": ExportedSurfaceMetadata(
        category="methodology",
        required_artifact_classes=("report", "summary", "tabular"),
        visualization_policy="exempt",
        visualization_exemption_reason="Inference contract export is a schema/report surface without standalone plots.",
    ),
    "generic_feature_taxonomy": ExportedSurfaceMetadata(
        category="methodology",
        required_artifact_classes=("report", "summary", "tabular"),
        visualization_policy="exempt",
        visualization_exemption_reason="Feature taxonomy export is currently report and registry oriented without standalone plots.",
    ),
    "methodology_latex": ExportedSurfaceMetadata(
        category="methodology",
        audience="mixed",
    ),
    "methodology_section_symbol_audit": ExportedSurfaceMetadata(
        category="methodology",
        audience="mixed",
    ),
    "functional_surface_catalog": ExportedSurfaceMetadata(
        category="registry",
        required_artifact_classes=("report", "summary", "tabular", "visual"),
        visualization_policy="required",
    ),
    "formal_math_registry": ExportedSurfaceMetadata(
        category="registry",
        required_artifact_classes=("report", "summary", "tabular", "visual"),
        visualization_policy="required",
    ),
    "formal_math_visual_registry": ExportedSurfaceMetadata(
        category="registry",
        required_artifact_classes=("report", "summary", "tabular", "visual"),
        visualization_policy="required",
    ),
    "strict_equation_audit": ExportedSurfaceMetadata(
        category="registry",
        required_artifact_classes=("report", "summary", "tabular"),
        visualization_policy="exempt",
        visualization_exemption_reason="Strict equation audit is a report/table audit without a dedicated plot artifact.",
    ),
    "showcase": ExportedSurfaceMetadata(
        category="showcase",
        audience="mixed",
        required_artifact_classes=("report", "summary", "tabular", "visual"),
        visualization_policy="required",
        invoker=_showcase_invoker,
        rerun_command="PYTHONPYCACHEPREFIX=../active/CACHE/kinematic-classifier-sandbox/.pycache python3 scripts/build_showcase.py --output-dir artifacts",
    ),
    "repo_story": ExportedSurfaceMetadata(
        category="showcase",
        audience="mixed",
        required_artifact_classes=("report", "summary", "tabular", "visual"),
        visualization_policy="required",
        invoker=_repo_story_invoker,
        rerun_command="PYTHONPYCACHEPREFIX=../active/CACHE/kinematic-classifier-sandbox/.pycache python3 scripts/render/render_repo_story.py --output-dir artifacts",
    ),
    "posterior_math": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=REPORT_AND_VISUAL,
        visualization_policy="required",
        notes="Math explainer currently omits machine-readable sidecars; the audit keeps that limitation visible.",
    ),
    "probability_primitives": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=REPORT_AND_VISUAL,
        visualization_policy="required",
        notes="Math explainer currently omits machine-readable sidecars; the audit keeps that limitation visible.",
    ),
    "posterior_numeric": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=REPORT_AND_VISUAL,
        visualization_policy="required",
        notes="Math explainer currently omits machine-readable sidecars; the audit keeps that limitation visible.",
    ),
    "identity_feature_confusion": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=VISUAL_ONLY,
        visualization_policy="required",
    ),
    "identity_posterior_explainer": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=REPORT_AND_VISUAL,
        visualization_policy="required",
    ),
    "identity_posterior_failure": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=REPORT_AND_VISUAL,
        visualization_policy="required",
    ),
    "identity_posterior_comparison": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=REPORT_AND_VISUAL,
        visualization_policy="required",
    ),
    "identity_posterior_margin_trace": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=REPORT_AND_VISUAL,
        visualization_policy="required",
    ),
    "toy_benchmark_plot": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=VISUAL_ONLY,
        visualization_policy="required",
    ),
    "toy_feature_confusion": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=VISUAL_ONLY,
        visualization_policy="required",
    ),
    "posterior": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=REPORT_AND_VISUAL,
        visualization_policy="required",
    ),
    "posterior_failure": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=REPORT_AND_VISUAL,
        visualization_policy="required",
    ),
    "posterior_comparison": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=REPORT_AND_VISUAL,
        visualization_policy="required",
    ),
    "posterior_margin": ExportedSurfaceMetadata(
        category="study",
        required_artifact_classes=REPORT_AND_VISUAL,
        visualization_policy="required",
    ),
    "external_backend_examples": ExportedSurfaceMetadata(
        category="corpus",
        required_artifact_classes=("report", "summary"),
        visualization_policy="exempt",
        visualization_exemption_reason="External backend examples export is an index/report surface without local plots.",
    ),
}


SURFACE_ID_ALIASES = {
    "survey": "method_survey",
    "write_advanced_filter_contract": "advanced_filter_contract",
    "write_advanced_state_inference": "advanced_state_inference",
    "write_external_backend_examples": "external_backend_examples",
    "write_rung_sufficiency": "rung_sufficiency",
    "identity_feature_confusion_png": "identity_feature_confusion",
    "benchmark_plot_png": "toy_benchmark_plot",
    "toy_feature_confusion_png": "toy_feature_confusion",
    "write_identity_posterior_explainer": "identity_posterior_explainer",
    "write_identity_posterior_failure": "identity_posterior_failure",
    "write_identity_posterior_comparison": "identity_posterior_comparison",
    "write_identity_posterior_margin_trace": "identity_posterior_margin_trace",
}


def _canonical_surface_id(surface_id: str) -> str:
    return SURFACE_ID_ALIASES.get(surface_id, surface_id)


def _default_required_classes(category: Category, visualization_policy: VisualizationPolicy) -> tuple[ArtifactClass, ...]:
    classes: list[ArtifactClass] = ["report", "tabular", "summary"]
    if category in {"methodology", "registry"}:
        classes = ["report", "summary"]
    if visualization_policy == "required":
        classes.append("visual")
    return tuple(classes)


def build_exported_surface_inventory(script_path: Path = EXPORT_SCRIPT_PATH) -> tuple[ExportedSurfaceSpec, ...]:
    specs: list[ExportedSurfaceSpec] = []
    for callsite in _parse_export_script_calls(script_path):
        surface_id = _canonical_surface_id(callsite.surface_id)
        override = METADATA_OVERRIDES.get(surface_id, ExportedSurfaceMetadata())
        category = override.category or _derive_category(callsite.module, surface_id)
        audience = override.audience or _derive_audience(category)
        visualization_policy = override.visualization_policy or "required"
        required_artifact_classes = override.required_artifact_classes or _default_required_classes(
            category, visualization_policy
        )
        if visualization_policy == "exempt" and override.visualization_exemption_reason is None:
            raise ValueError(f"{surface_id} is visualization-exempt but has no exemption reason")
        specs.append(
            ExportedSurfaceSpec(
                surface_id=surface_id,
                module=callsite.module,
                writer_name=callsite.writer_name,
                line_number=callsite.line_number,
                category=category,
                audience=audience,
                analysis_function=override.analysis_function,
                required_artifact_classes=required_artifact_classes,
                visualization_policy=visualization_policy,
                visualization_exemption_reason=override.visualization_exemption_reason,
                rerun_command=override.rerun_command or _default_rerun_command(),
                notes=override.notes,
                invoker=override.invoker,
            )
        )
    return tuple(specs)


def _load_callable(module_name: str, function_name: str) -> Callable[..., object] | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    value = getattr(module, function_name, None)
    if callable(value):
        return value
    return None


def _rerun_command_target_exists(command: str) -> bool:
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token == "-m" and index + 1 < len(tokens):
            return True
        if token.startswith("scripts/") or token.endswith(".py"):
            candidate = ROOT / token if not token.startswith("/") else Path(token)
            return candidate.exists()
    return False


def _classify_path(path: Path) -> ArtifactClass | None:
    suffix = path.suffix.lower()
    if suffix in REPORT_SUFFIXES:
        return "report"
    if suffix in TABULAR_SUFFIXES:
        return "tabular"
    if suffix in SUMMARY_SUFFIXES:
        return "summary"
    if suffix in VISUAL_SUFFIXES:
        return "visual"
    return None


def _extract_paths(value: object) -> list[Path]:
    if isinstance(value, Path):
        return [value]
    if is_dataclass(value):
        paths: list[Path] = []
        for field in fields(value):
            paths.extend(_extract_paths(getattr(value, field.name)))
        return paths
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        paths = []
        for item in value:
            paths.extend(_extract_paths(item))
        return paths
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        paths = []
        for item in value:
            paths.extend(_extract_paths(item))
        return paths
    if hasattr(value, "__dict__"):
        paths = []
        for attribute, attribute_value in vars(value).items():
            if attribute.startswith("_"):
                continue
            paths.extend(_extract_paths(attribute_value))
        return paths
    return []


def _materialize_surface(spec: ExportedSurfaceSpec) -> tuple[tuple[ArtifactClass, ...], str | None]:
    writer = _load_callable(spec.module, spec.writer_name)
    if writer is None:
        return (), None
    invoker = spec.invoker or writer
    with tempfile.TemporaryDirectory(prefix=f"kcs-surface-{spec.surface_id}-") as temp_dir:
        output_dir = Path(temp_dir) / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = invoker(output_dir)
        run_scope_name: str | None = None
        if hasattr(artifacts, "run_dir"):
            run_dir = getattr(artifacts, "run_dir")
            if isinstance(run_dir, Path):
                run_scope_name = run_dir.name
        paths = [path for path in _extract_paths(artifacts) if path.exists()]
        classes = sorted({artifact_class for path in paths if (artifact_class := _classify_path(path)) is not None})
        return tuple(classes), run_scope_name


def _missing_requirements(
    required_artifact_classes: tuple[ArtifactClass, ...],
    visualization_policy: VisualizationPolicy,
    visualization_exemption_reason: str | None,
) -> tuple[str, ...]:
    issues: list[str] = []
    required = set(required_artifact_classes)
    if "report" not in required:
        issues.append("missing_report_class")
    if "tabular" not in required and "summary" not in required:
        issues.append("missing_machine_class")
    if visualization_policy == "required" and "visual" not in required:
        issues.append("missing_visual_class")
    if visualization_policy == "exempt" and not visualization_exemption_reason:
        issues.append("missing_visual_exemption_reason")
    return tuple(issues)


def analyze_exported_surface_coverage(
    *,
    surface_ids: Iterable[str] | None = None,
    materialize: bool = False,
    script_path: Path = EXPORT_SCRIPT_PATH,
) -> ExportedSurfaceCoverageResult:
    inventory = build_exported_surface_inventory(script_path)
    selected_ids = {surface_id for surface_id in surface_ids} if surface_ids is not None else None

    rows: list[ExportedSurfaceAuditRow] = []
    for spec in inventory:
        if selected_ids is not None and spec.surface_id not in selected_ids:
            continue
        observed_artifact_classes: tuple[ArtifactClass, ...] = ()
        run_scope_name: str | None = None
        if materialize:
            observed_artifact_classes, run_scope_name = _materialize_surface(spec)
        rows.append(
            ExportedSurfaceAuditRow(
                surface_id=spec.surface_id,
                module=spec.module,
                writer_name=spec.writer_name,
                category=spec.category,
                audience=spec.audience,
                analysis_function=spec.analysis_function,
                required_artifact_classes=spec.required_artifact_classes,
                declared_machine_artifact=(
                    "tabular" in spec.required_artifact_classes or "summary" in spec.required_artifact_classes
                ),
                visualization_policy=spec.visualization_policy,
                visualization_exemption_reason=spec.visualization_exemption_reason,
                rerun_command=spec.rerun_command,
                rerun_command_target_exists=_rerun_command_target_exists(spec.rerun_command),
                writer_callable=_load_callable(spec.module, spec.writer_name) is not None,
                exported_by_script=True,
                materialized=materialize,
                run_scope_name=run_scope_name,
                observed_artifact_classes=observed_artifact_classes,
                missing_requirements=_missing_requirements(
                    spec.required_artifact_classes,
                    spec.visualization_policy,
                    spec.visualization_exemption_reason,
                ),
                notes=spec.notes,
            )
        )

    category_counts = Counter(row.category for row in rows)
    missing_requirement_counts = Counter(issue for row in rows for issue in row.missing_requirements)
    summary = {
        "surface_count": len(rows),
        "materialized": materialize,
        "categories": dict(sorted(category_counts.items())),
        "audiences": dict(sorted(Counter(row.audience for row in rows).items())),
        "visualization_exemption_count": sum(1 for row in rows if row.visualization_policy == "exempt"),
        "writer_callable_count": sum(1 for row in rows if row.writer_callable),
        "rerun_target_ok_count": sum(1 for row in rows if row.rerun_command_target_exists),
        "missing_requirement_counts": dict(sorted(missing_requirement_counts.items())),
        "surface_ids": [row.surface_id for row in rows],
    }

    coverage_matrix_rows = tuple(
        {
            "surface_id": row.surface_id,
            "category": row.category,
            "audience": row.audience,
            "writer_name": row.writer_name,
            "analysis_function": row.analysis_function or "",
            "required_artifact_classes": "|".join(row.required_artifact_classes),
            "declared_machine_artifact": row.declared_machine_artifact,
            "visualization_policy": row.visualization_policy,
            "visualization_exemption_reason": row.visualization_exemption_reason or "",
            "writer_callable": row.writer_callable,
            "exported_by_script": row.exported_by_script,
            "rerun_command": row.rerun_command,
            "rerun_command_target_exists": row.rerun_command_target_exists,
            "materialized": row.materialized,
            "run_scope_name": row.run_scope_name or "",
            "observed_artifact_classes": "|".join(row.observed_artifact_classes),
            "missing_requirements": "|".join(row.missing_requirements),
            "notes": row.notes,
        }
        for row in rows
    )
    report_markdown = render_exported_surface_coverage_report(
        ExportedSurfaceCoverageResult(
            surface_rows=tuple(rows),
            summary=summary,
            coverage_matrix_rows=coverage_matrix_rows,
            report_markdown="",
        )
    )
    return ExportedSurfaceCoverageResult(
        surface_rows=tuple(rows),
        summary=summary,
        coverage_matrix_rows=coverage_matrix_rows,
        report_markdown=report_markdown,
    )


def _render_category_plot(result: ExportedSurfaceCoverageResult):
    category_counts = Counter(row.category for row in result.surface_rows)
    labels = list(category_counts)
    values = [category_counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(labels, values, color="#2563eb")
    ax.set_title("Exported Surface Count By Category", loc="left", fontweight="bold")
    ax.set_ylabel("surface count")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def render_exported_surface_coverage_report(result: ExportedSurfaceCoverageResult) -> str:
    report = MarkdownDocument("Exported Surface Coverage Audit")
    report.paragraph(
        "This audit treats `scripts/export_artifacts.py` as the canonical exported-surface list and checks whether each exported "
        "surface has a declared artifact-class contract, a stable rerun command, and an explicit visualization policy."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Surface count: `{result.summary['surface_count']}`",
            f"Materialized subset: `{result.summary['materialized']}`",
            f"Writer callables resolved: `{result.summary['writer_callable_count']}`",
            f"Rerun command targets found: `{result.summary['rerun_target_ok_count']}`",
            f"Visualization exemptions: `{result.summary['visualization_exemption_count']}`",
        ]
    )
    if result.summary["missing_requirement_counts"]:
        report.heading("Missing Coverage Signals", level=2)
        report.bullet_list(
            [
                f"`{issue}`: `{count}` surface(s)"
                for issue, count in result.summary["missing_requirement_counts"].items()
            ]
        )
    report.heading("Coverage Matrix", level=2)
    report.table(
        [
            "surface_id",
            "category",
            "required",
            "visual_policy",
            "writer",
            "rerun",
            "missing",
        ],
        [
            (
                f"`{row.surface_id}`",
                row.category,
                ", ".join(row.required_artifact_classes),
                row.visualization_policy,
                "yes" if row.writer_callable else "no",
                "yes" if row.rerun_command_target_exists else "no",
                ", ".join(row.missing_requirements) if row.missing_requirements else "none",
            )
            for row in result.surface_rows
        ],
    )
    report.heading("Policy", level=2)
    report.bullet_list(
        [
            "Every exported surface should declare a durable report artifact.",
            "Every exported surface should declare at least one machine-consumable artifact class: tabular or summary.",
            "Visualization is required unless the inventory explicitly records an exemption reason.",
            "Materialized subset checks are optional and exist to prove the static contract against real output bundles.",
        ]
    )
    return report.text()


def write_exported_surface_coverage_artifacts(
    output_dir: str | Path,
    *,
    result: ExportedSurfaceCoverageResult | None = None,
    surface_ids: Iterable[str] | None = None,
    materialize: bool = False,
) -> ExportedSurfaceCoverageArtifacts:
    payload = result or analyze_exported_surface_coverage(surface_ids=surface_ids, materialize=materialize)
    output_root = Path(output_dir)
    run_dir = output_root / "exported_surface_coverage_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "exported_surface_coverage_report.md"
    summary_path = run_dir / "exported_surface_coverage_summary.json"
    coverage_matrix_path = run_dir / "exported_surface_coverage_matrix.csv"
    missing_coverage_path = run_dir / "exported_surface_missing_coverage.csv"
    visualization_exemptions_path = run_dir / "exported_surface_visualization_exemptions.csv"
    rerun_commands_path = run_dir / "exported_surface_rerun_commands.csv"
    category_plot_path = run_dir / "exported_surface_categories.png"
    inventory_path = run_dir / "exported_surface_inventory.json"

    _write_text(report_path, payload.report_markdown)
    _write_json(summary_path, payload.summary)
    _write_json(
        inventory_path,
        [
            {
                "surface_id": row.surface_id,
                "module": row.module,
                "writer_name": row.writer_name,
                "category": row.category,
                "audience": row.audience,
                "analysis_function": row.analysis_function,
                "required_artifact_classes": list(row.required_artifact_classes),
                "visualization_policy": row.visualization_policy,
                "visualization_exemption_reason": row.visualization_exemption_reason,
                "rerun_command": row.rerun_command,
                "notes": row.notes,
            }
            for row in payload.surface_rows
        ],
    )
    write_csv(coverage_matrix_path, list(payload.coverage_matrix_rows), list(payload.coverage_matrix_rows[0]) if payload.coverage_matrix_rows else [])
    write_csv(
        missing_coverage_path,
        [
            {
                "surface_id": row.surface_id,
                "category": row.category,
                "missing_requirements": "|".join(row.missing_requirements),
                "required_artifact_classes": "|".join(row.required_artifact_classes),
                "visualization_policy": row.visualization_policy,
            }
            for row in payload.surface_rows
            if row.missing_requirements
        ],
        [
            "surface_id",
            "category",
            "missing_requirements",
            "required_artifact_classes",
            "visualization_policy",
        ],
    )
    write_csv(
        visualization_exemptions_path,
        [
            {
                "surface_id": row.surface_id,
                "category": row.category,
                "visualization_exemption_reason": row.visualization_exemption_reason or "",
            }
            for row in payload.surface_rows
            if row.visualization_policy == "exempt"
        ],
        ["surface_id", "category", "visualization_exemption_reason"],
    )
    write_csv(
        rerun_commands_path,
        [
            {
                "surface_id": row.surface_id,
                "rerun_command": row.rerun_command,
                "rerun_command_target_exists": row.rerun_command_target_exists,
            }
            for row in payload.surface_rows
        ],
        ["surface_id", "rerun_command", "rerun_command_target_exists"],
    )
    category_plot_path.write_bytes(_figure_to_png(_render_category_plot(payload)))
    return ExportedSurfaceCoverageArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        coverage_matrix_path=coverage_matrix_path,
        missing_coverage_path=missing_coverage_path,
        visualization_exemptions_path=visualization_exemptions_path,
        rerun_commands_path=rerun_commands_path,
        category_plot_path=category_plot_path,
        inventory_path=inventory_path,
    )
