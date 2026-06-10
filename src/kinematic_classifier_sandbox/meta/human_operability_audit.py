from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from ..utils.runtime import repo_root

from ..registry.formal_math_registry import load_equation_registry
from ..showcase.validation import extract_markdown_relative_targets
from ..story.repo_story import ARTIFACT_MANIFEST, CLAIMS, validate_repo_story_references
from ..utils.io import _write_json, _write_text, union_fieldnames, write_csv
from .repo_shape_audit import analyze_repo_shape

ROOT = repo_root()
ARTIFACT_DIR = ROOT / "artifacts" / "human_operability_audit_v1"

CORE_FRONT_DOORS: tuple[tuple[str, str, str], ...] = (
    ("repo_story", "Repo front door", "docs/story/00_repo_story.md"),
    ("reading_order", "Story reading order", "docs/story/02_reading_order.md"),
    ("package_map", "Package map", "src/kinematic_classifier_sandbox/README.md"),
    ("test_map", "Test suite map", "tests/README.md"),
    ("scripts_map", "Scripts map", "scripts/README.md"),
    ("claim_matrix", "Claim matrix", "docs/story/claim_evidence_matrix.md"),
)

SHOWCASE_FRONT_DOORS: tuple[tuple[str, str, str], ...] = (
    ("showcase_story_index", "Showcase story index", "artifacts/showcase/story_index.md"),
    ("showcase_proof_gallery", "Showcase proof gallery", "artifacts/showcase/proof_gallery.md"),
)

RERUN_SURFACES: tuple[tuple[str, str, tuple[str, ...], bool], ...] = (
    ("repo_checks", "Repo checks", ("scripts/check.py",), False),
    ("repo_shape_audit", "Repo-shape audit", ("scripts/audit/audit_repo_shape.py",), False),
    (
        "artifact_showcase_validation",
        "Artifact/showcase validation",
        ("scripts/audit/validate_artifacts.py", "scripts/build_showcase.py"),
        True,
    ),
    ("corpus_audit", "Corpus audit", ("scripts/audit/audit_corpus.py",), False),
    ("dimensional_audit", "Dimensional audit", ("scripts/audit/audit_dimensions.py",), False),
    (
        "methodology_docs",
        "Methodology docs / LaTeX packet",
        (
            "methodology-latex",
            "scripts/render/render_methodology_latex.py",
            "scripts/export_artifacts.py",
            "scripts/build/build_methodology_docs.sh",
        ),
        False,
    ),
)


@dataclass(frozen=True, slots=True)
class HumanOperabilityAuditResult:
    front_door_rows: tuple[dict[str, object], ...]
    claim_traceability_rows: tuple[dict[str, object], ...]
    equation_graph_rows: tuple[dict[str, object], ...]
    rerun_command_rows: tuple[dict[str, object], ...]
    issue_rows: tuple[dict[str, object], ...]
    summary: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class HumanOperabilityAuditArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    issues_path: Path
    front_door_coverage_path: Path
    claim_traceability_path: Path
    equation_graph_link_audit_path: Path
    rerun_command_coverage_path: Path


def analyze_human_operability_audit(
    *,
    source_root: str | Path = ROOT,
    output_dir: str | Path = ROOT / "artifacts",
) -> HumanOperabilityAuditResult:
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    issue_rows: list[dict[str, object]] = []

    front_door_rows = list(_front_door_rows(source_root, output_dir, issue_rows))
    claim_traceability_rows = list(_claim_traceability_rows(source_root, output_dir, issue_rows))
    equation_graph_rows = list(_equation_graph_rows(source_root, output_dir, issue_rows))
    rerun_command_rows = list(_rerun_command_rows(source_root, issue_rows))

    repo_story_validation = validate_repo_story_references(source_root=source_root, output_dir=output_dir)
    if repo_story_validation["status"] != "pass":
        for claim_id, path in repo_story_validation["missing"]:
            issue_rows.append(
                _issue(
                    severity="fail",
                    code="repo_story_reference_missing",
                    subject=str(claim_id),
                    detail=f"Claim/story metadata references a missing path: `{path}`.",
                    path=str(path),
                )
            )

    repo_shape_result = analyze_repo_shape()
    for row in repo_shape_result.root_module_rows:
        if row["status"] == "legacy_root_debt":
            issue_rows.append(
                _issue(
                    severity="warn",
                    code="legacy_root_navigation_debt",
                    subject=str(row["module"]),
                    detail="Legacy root-level implementation debt remains visible at the package front door.",
                    path=str(row["path"]),
                )
            )

    summary = {
        "overall_status": "fail" if any(row["severity"] == "fail" for row in issue_rows) else "pass",
        "hard_fail_count": sum(1 for row in issue_rows if row["severity"] == "fail"),
        "warning_count": sum(1 for row in issue_rows if row["severity"] == "warn"),
        "front_door_count": len(front_door_rows),
        "claim_count": len(claim_traceability_rows),
        "equation_graph_link_count": len(equation_graph_rows),
        "rerun_surface_count": len(rerun_command_rows),
    }
    report = _render_report(
        front_door_rows=front_door_rows,
        claim_traceability_rows=claim_traceability_rows,
        equation_graph_rows=equation_graph_rows,
        rerun_command_rows=rerun_command_rows,
        issue_rows=issue_rows,
        summary=summary,
    )
    return HumanOperabilityAuditResult(
        front_door_rows=tuple(front_door_rows),
        claim_traceability_rows=tuple(claim_traceability_rows),
        equation_graph_rows=tuple(equation_graph_rows),
        rerun_command_rows=tuple(rerun_command_rows),
        issue_rows=tuple(issue_rows),
        summary=summary,
        report_markdown=report,
    )


def write_human_operability_audit_artifacts(
    output_dir: str | Path = ROOT / "artifacts",
    *,
    result: HumanOperabilityAuditResult | None = None,
) -> HumanOperabilityAuditArtifacts:
    payload = result or analyze_human_operability_audit(output_dir=output_dir)
    run_dir = Path(output_dir) / "human_operability_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "human_operability_audit_report.md"
    summary_path = run_dir / "human_operability_audit_summary.json"
    issues_path = run_dir / "human_operability_audit_issues.csv"
    front_door_coverage_path = run_dir / "front_door_coverage.csv"
    claim_traceability_path = run_dir / "claim_traceability.csv"
    equation_graph_link_audit_path = run_dir / "equation_graph_link_audit.csv"
    rerun_command_coverage_path = run_dir / "rerun_command_coverage.csv"

    _write_text(report_path, payload.report_markdown)
    _write_json(summary_path, payload.summary)
    _write_rows(front_door_coverage_path, payload.front_door_rows)
    _write_rows(claim_traceability_path, payload.claim_traceability_rows)
    _write_rows(equation_graph_link_audit_path, payload.equation_graph_rows)
    _write_rows(rerun_command_coverage_path, payload.rerun_command_rows)
    _write_rows(issues_path, payload.issue_rows)
    return HumanOperabilityAuditArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        issues_path=issues_path,
        front_door_coverage_path=front_door_coverage_path,
        claim_traceability_path=claim_traceability_path,
        equation_graph_link_audit_path=equation_graph_link_audit_path,
        rerun_command_coverage_path=rerun_command_coverage_path,
    )


def _front_door_rows(
    source_root: Path,
    output_dir: Path,
    issue_rows: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    showcase_declared_canonical = (source_root / "docs" / "story" / "document_roles.md").exists()
    for row_id, label, rel_path in (*CORE_FRONT_DOORS, *(SHOWCASE_FRONT_DOORS if showcase_declared_canonical else ())):
        path = _resolve_path(source_root, output_dir, rel_path)
        exists = path is not None and path.exists()
        text = path.read_text(encoding="utf-8") if exists else ""
        nonempty = bool(text.strip())
        link_count = len(_extract_links(path, text)) if exists else 0
        status = "pass" if exists and nonempty else "fail"
        if status == "fail":
            issue_rows.append(
                _issue(
                    severity="fail",
                    code="missing_front_door",
                    subject=label,
                    detail=f"Missing or empty human front-door document: `{rel_path}`.",
                    path=rel_path,
                )
            )
        elif link_count == 0:
            issue_rows.append(
                _issue(
                    severity="warn",
                    code="front_door_missing_breadcrumbs",
                    subject=label,
                    detail="Front-door document exists but does not link onward to the next layer.",
                    path=rel_path,
                )
            )
        rows.append(
            {
                "front_door_id": row_id,
                "label": label,
                "path": rel_path,
                "exists": exists,
                "nonempty": nonempty,
                "link_count": link_count,
                "status": status,
            }
        )
    return tuple(rows)


def _claim_traceability_rows(
    source_root: Path,
    output_dir: Path,
    issue_rows: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for claim in CLAIMS:
        missing_docs = [path for path in claim.evidence_doc if _resolve_path(source_root, output_dir, path) is None]
        missing_artifacts = [path for path in claim.artifact_paths if _resolve_path(source_root, output_dir, path) is None]
        missing_tests = [path for path in claim.test_paths if _resolve_path(source_root, output_dir, path) is None]
        generating_paths = [
            entry.generated_by
            for entry in ARTIFACT_MANIFEST
            if entry.claim_supported == claim.claim_id and _resolve_path(source_root, output_dir, entry.generated_by) is not None
        ]
        limitations_present = bool(claim.limitations.strip())
        next_work_present = bool(claim.next_work.strip())
        status = "pass"
        if (
            missing_docs
            or missing_artifacts
            or missing_tests
            or not generating_paths
            or not limitations_present
            or not next_work_present
        ):
            status = "fail"
            issue_rows.append(
                _issue(
                    severity="fail",
                    code="claim_traceability_failure",
                    subject=claim.claim_id,
                    detail=(
                        "Claim is missing at least one required proof leg across docs, artifacts, tests, generating code, "
                        "limitations, or next-work text."
                    ),
                    path="docs/story/claim_evidence_matrix.md",
                )
            )
        rows.append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.claim,
                "doc_count": len(claim.evidence_doc),
                "artifact_count": len(claim.artifact_paths),
                "test_count": len(claim.test_paths),
                "generating_code_count": len(generating_paths),
                "limitations_present": limitations_present,
                "next_work_present": next_work_present,
                "missing_docs": "; ".join(missing_docs),
                "missing_artifacts": "; ".join(missing_artifacts),
                "missing_tests": "; ".join(missing_tests),
                "status": status,
            }
        )
    return tuple(rows)


def _equation_graph_rows(
    source_root: Path,
    output_dir: Path,
    issue_rows: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    crosswalk_path = source_root / "docs" / "math" / "code_equation_crosswalk.md"
    if not crosswalk_path.exists() or not crosswalk_path.read_text(encoding="utf-8").strip():
        issue_rows.append(
            _issue(
                severity="fail",
                code="missing_equation_crosswalk",
                subject="code_equation_crosswalk",
                detail="The code/equation crosswalk is missing or empty.",
                path="docs/math/code_equation_crosswalk.md",
            )
        )
    else:
        crosswalk_text = crosswalk_path.read_text(encoding="utf-8")
        for phrase in ("Bayes recursive update", "CorpusGym reward", "Advanced-filter gate"):
            if phrase not in crosswalk_text:
                issue_rows.append(
                    _issue(
                        severity="warn",
                        code="crosswalk_missing_core_phrase",
                        subject=phrase,
                        detail="The code/equation crosswalk does not mention a core implemented equation family by name.",
                        path="docs/math/code_equation_crosswalk.md",
                    )
                )

    for row in load_equation_registry():
        implementation = row["implementation"]
        module_path = source_root / implementation["module"]
        function_exists = module_path.exists() and _module_has_function(module_path, implementation["function"])
        artifact_missing = [path for path in row.get("artifacts", []) if _resolve_path(source_root, output_dir, path) is None]
        test_missing = [path for path in row.get("tests", []) if _resolve_path(source_root, output_dir, path) is None]
        status = "pass"
        if row["status"] == "implemented" and (not function_exists or test_missing):
            status = "fail"
            issue_rows.append(
                _issue(
                    severity="fail",
                    code="equation_registry_broken_reference",
                    subject=str(row["id"]),
                    detail="Implemented equation metadata points to missing code, artifact, or test paths.",
                    path="docs/math/equation_registry.yaml",
                )
            )
        elif row["status"] == "implemented" and artifact_missing:
            issue_rows.append(
                _issue(
                    severity="warn",
                    code="equation_registry_artifact_missing",
                    subject=str(row["id"]),
                    detail="Implemented equation metadata points to artifacts that are not currently materialized in the workspace.",
                    path="docs/math/equation_registry.yaml",
                )
            )
        rows.append(
            {
                "link_type": "equation_registry",
                "source": "docs/math/equation_registry.yaml",
                "reference_id": row["id"],
                "target": f"{implementation['module']}::{implementation['function']}",
                "artifact_count": len(row.get("artifacts", [])),
                "missing_artifact_count": len(artifact_missing),
                "missing_test_count": len(test_missing),
                "status": status,
            }
        )

    for rel_path in (
        "docs/story/00_repo_story.md",
        "docs/story/02_reading_order.md",
        "artifacts/showcase/proof_gallery.md",
        "artifacts/showcase/story_index.md",
    ):
        path = _resolve_path(source_root, output_dir, rel_path)
        if path is None or not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for target in _extract_links(path, text):
            exists = target.exists()
            if not exists:
                issue_rows.append(
                    _issue(
                        severity="fail" if "showcase" in rel_path else "warn",
                        code="broken_markdown_reference",
                        subject=rel_path,
                        detail=f"Markdown reference points to a missing file: `{target}`.",
                        path=rel_path,
                    )
                )
            rows.append(
                {
                    "link_type": "markdown_reference",
                    "source": rel_path,
                    "reference_id": target.name,
                    "target": _display_target(target, source_root),
                    "artifact_count": "",
                    "missing_artifact_count": 0 if exists else 1,
                    "missing_test_count": "",
                    "status": "pass" if exists else "fail",
                }
            )

    methodology_path = source_root / "docs" / "latex" / "kinematic_classifier_methodology.tex"
    if not methodology_path.exists():
        issue_rows.append(
            _issue(
                severity="fail",
                code="missing_methodology_tex",
                subject="kinematic_classifier_methodology",
                detail="Canonical methodology LaTeX source is missing.",
                path="docs/latex/kinematic_classifier_methodology.tex",
            )
        )
    else:
        for ref_kind, target in _tex_references(methodology_path, source_root):
            exists = target.exists()
            if not exists:
                issue_rows.append(
                    _issue(
                        severity="fail",
                        code="broken_latex_reference",
                        subject=ref_kind,
                        detail=f"LaTeX reference points to a missing file: `{target}`.",
                        path="docs/latex/kinematic_classifier_methodology.tex",
                    )
                )
            rows.append(
                {
                    "link_type": f"latex_{ref_kind}",
                    "source": "docs/latex/kinematic_classifier_methodology.tex",
                    "reference_id": target.name,
                    "target": _display_target(target, source_root),
                    "artifact_count": "",
                    "missing_artifact_count": 0 if exists else 1,
                    "missing_test_count": "",
                    "status": "pass" if exists else "fail",
                }
            )
    return tuple(rows)


def _rerun_command_rows(source_root: Path, issue_rows: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    texts = {
        "scripts/README.md": (source_root / "scripts" / "README.md").read_text(encoding="utf-8")
        if (source_root / "scripts" / "README.md").exists()
        else "",
        "tests/README.md": (source_root / "tests" / "README.md").read_text(encoding="utf-8")
        if (source_root / "tests" / "README.md").exists()
        else "",
        "src/kinematic_classifier_sandbox/README.md": (
            source_root / "src" / "kinematic_classifier_sandbox" / "README.md"
        ).read_text(encoding="utf-8")
        if (source_root / "src" / "kinematic_classifier_sandbox" / "README.md").exists()
        else "",
    }
    rows: list[dict[str, object]] = []
    for surface_id, label, markers, required_hard_fail in RERUN_SURFACES:
        matches = [doc for doc, text in texts.items() if any(marker in text for marker in markers)]
        status = "pass" if matches else "warn"
        if not matches:
            issue_rows.append(
                _issue(
                    severity="fail" if required_hard_fail else "warn",
                    code="missing_rerun_command",
                    subject=surface_id,
                    detail=f"No concise rerun command was found for {label.lower()}.",
                    path="scripts/README.md",
                )
            )
        rows.append(
            {
                "surface_id": surface_id,
                "label": label,
                "markers": "; ".join(markers),
                "source_docs": "; ".join(matches),
                "status": status,
            }
        )
    return tuple(rows)


def _render_report(
    *,
    front_door_rows: list[dict[str, object]],
    claim_traceability_rows: list[dict[str, object]],
    equation_graph_rows: list[dict[str, object]],
    rerun_command_rows: list[dict[str, object]],
    issue_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> str:
    lines = [
        "# Human Operability Audit",
        "",
        "This audit asks whether a strong human engineer can find the repo front door, follow the claim story, trace each major claim into code/tests/artifacts, and rerun the important workflows without agent-only discovery.",
        "",
        "## Summary",
        "",
        f"- Overall status: `{summary['overall_status']}`",
        f"- Hard failures: `{summary['hard_fail_count']}`",
        f"- Warnings: `{summary['warning_count']}`",
        f"- Front-door docs checked: `{summary['front_door_count']}`",
        f"- Claims checked: `{summary['claim_count']}`",
        f"- Equation/graph links checked: `{summary['equation_graph_link_count']}`",
        f"- Rerun surfaces checked: `{summary['rerun_surface_count']}`",
        "",
    ]
    hard_failures = [row for row in issue_rows if row["severity"] == "fail"]
    warnings = [row for row in issue_rows if row["severity"] == "warn"]
    lines.extend(["## Hard Failures", ""])
    if hard_failures:
        lines.extend([f"- `{row['code']}`: {row['detail']}" for row in hard_failures])
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend([f"- `{row['code']}`: {row['detail']}" for row in warnings[:20]])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Front-Door Coverage",
            "",
            f"- Passing front doors: `{sum(1 for row in front_door_rows if row['status'] == 'pass')}` / `{len(front_door_rows)}`",
            f"- Breadcrumb warnings: `{sum(1 for row in front_door_rows if row['link_count'] == 0)}`",
            "",
            "## Claim Traceability",
            "",
            f"- Fully traced claims: `{sum(1 for row in claim_traceability_rows if row['status'] == 'pass')}` / `{len(claim_traceability_rows)}`",
            "",
            "## Equation And Graph Linkage",
            "",
            f"- Passing equation/graph rows: `{sum(1 for row in equation_graph_rows if row['status'] == 'pass')}` / `{len(equation_graph_rows)}`",
            "",
            "## Rerunability",
            "",
            f"- Documented rerun surfaces: `{sum(1 for row in rerun_command_rows if row['status'] == 'pass')}` / `{len(rerun_command_rows)}`",
            "",
            "## Recommended Next Cleanup",
            "",
            "- Add onward links in any front-door docs that still read like isolated notes.",
            "- Keep rerun commands in `scripts/README.md` synchronized with canonical audit/build surfaces.",
            "- Continue shrinking legacy root-level module debt so package navigation matches the documented boundaries.",
        ]
    )
    return "\n".join(lines) + "\n"


def _resolve_path(source_root: Path, output_dir: Path, rel_path: str) -> Path | None:
    candidate = source_root / rel_path
    if candidate.exists():
        return candidate
    if rel_path.startswith("artifacts/"):
        artifact_candidate = output_dir / rel_path.removeprefix("artifacts/")
        if artifact_candidate.exists():
            return artifact_candidate
    return None


def _extract_links(path: Path, text: str) -> tuple[Path, ...]:
    return tuple((path.parent / target).resolve() for target in extract_markdown_relative_targets(text))


def _module_has_function(path: Path, function_name: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return True
    return False


def _tex_references(path: Path, source_root: Path) -> tuple[tuple[str, Path], ...]:
    text = path.read_text(encoding="utf-8")
    rows: list[tuple[str, Path]] = []
    graphic_dirs = [path.parent]
    for group in re.findall(r"\\graphicspath\{(.+?)\}", text):
        for raw_dir in re.findall(r"\{([^}]+)\}", group):
            graphic_dirs.append((path.parent / raw_dir).resolve())
    for raw_target in re.findall(r"\\input\{([^}]+)\}", text):
        target = path.parent / raw_target
        if target.suffix != ".tex":
            target = target.with_suffix(".tex")
        rows.append(("input", target.resolve()))
    for raw_target in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        base = Path(raw_target)
        if base.suffix:
            candidates = [graphic_dir / base for graphic_dir in graphic_dirs]
        else:
            candidates = [
                graphic_dir / base.with_suffix(suffix)
                for graphic_dir in graphic_dirs
                for suffix in (".png", ".pdf", ".jpg", ".jpeg")
            ]
        artifact_candidates = list((source_root / "artifacts").rglob(base.name)) if (source_root / "artifacts").exists() else []
        resolved = next(
            (candidate.resolve() for candidate in [*candidates, *artifact_candidates] if candidate.exists()),
            candidates[0].resolve(),
        )
        rows.append(("figure", resolved))
    return tuple(rows)


def _write_rows(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    rows_list = list(rows)
    fieldnames = union_fieldnames(rows_list) if rows_list else ["status"]
    write_csv(path, rows_list, fieldnames)


def _display_target(target: Path, source_root: Path) -> str:
    try:
        return str(target.relative_to(source_root))
    except ValueError:
        return str(target)


def _issue(*, severity: str, code: str, subject: str, detail: str, path: str) -> dict[str, object]:
    return {
        "severity": severity,
        "code": code,
        "subject": subject,
        "detail": detail,
        "path": path,
    }
