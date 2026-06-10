from __future__ import annotations

from dataclasses import dataclass
import ast
import csv
import json
from pathlib import Path

from kinematic_classifier_sandbox.utils.runtime import repo_root


REPO_ROOT = repo_root()
PACKAGE_ROOT = REPO_ROOT / "src" / "kinematic_classifier_sandbox"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "import_simplicity_audit_v1"

ALLOWED_PATH_SNIFFING = {
    "scripts/_bootstrap.py",
    "scripts/audit/_bootstrap.py",
    "scripts/render/_bootstrap.py",
    "scripts/run/_bootstrap.py",
    "scripts/workflows/_bootstrap.py",
    "src/kinematic_classifier_sandbox/utils/runtime.py",
}

ALLOWED_SCRIPT_BOOTSTRAP = {
    "scripts/_bootstrap.py",
    "scripts/audit/_bootstrap.py",
    "scripts/render/_bootstrap.py",
    "scripts/run/_bootstrap.py",
    "scripts/workflows/_bootstrap.py",
}

LEGACY_ROOT_WRAPPER_MODULES = {
    "advanced_filter_decision",
    "bayesian_walkthroughs",
    "catalog",
    "common_dataset_comparison",
    "corpus_adequacy_audit",
    "corpus_autodevelopment",
    "coverage_report",
    "dimensional_lift_audit",
    "feature_analysis",
    "feature_rows",
    "formal_math_registry",
    "formal_math_visual_registry",
    "functional_surface_catalog",
    "generic_classification_evidence_proof",
    "generic_feature_taxonomy",
    "generic_filtering_contract",
    "generic_inference_contract",
    "identity_1d",
    "identity_posterior_explainer",
    "inspection_bundle",
    "irregular_window_comparison",
    "kalman_filter_bank",
    "kalman_observable_comparison",
    "kalman_variant_comparison",
    "methodology_compendium",
    "methodology_latex",
    "monte_carlo_benchmark",
    "pca_analysis",
    "pca_dimensionality_audit",
    "pointwise_baseline",
    "posterior_explainer",
    "prior_sensitivity_analysis",
    "sequential_bayes_accumulator",
    "shared_evaluation",
    "short_horizon_identifiability",
    "strict_equation_audit",
    "technique_comparison",
    "toy_1d",
    "transition_matrix_accumulator",
    "validation_ladder",
    "velocity_aided_kalman_comparison",
    "windowed_baseline",
}


@dataclass(frozen=True, slots=True)
class ImportSimplicityAuditResult:
    issue_rows: tuple[dict[str, object], ...]
    summary: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class ImportSimplicityAuditArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    issues_path: Path


def analyze_import_simplicity() -> ImportSimplicityAuditResult:
    rows = tuple(_issue_rows())
    summary = {
        "passes": not any(row["status"] == "violation" for row in rows),
        "issue_count": len(rows),
        "violation_count": sum(1 for row in rows if row["status"] == "violation"),
        "debt_count": sum(1 for row in rows if row["status"] == "debt"),
        "wildcard_import_count": sum(1 for row in rows if row["kind"] == "wildcard_import"),
        "path_sniffing_count": sum(1 for row in rows if row["kind"] == "path_sniffing"),
        "script_bootstrap_count": sum(1 for row in rows if row["kind"] == "script_bootstrap"),
        "package_side_effect_count": sum(1 for row in rows if row["kind"] == "package_import_side_effect"),
        "legacy_root_import_count": sum(1 for row in rows if row["kind"] == "legacy_root_import"),
    }
    result = ImportSimplicityAuditResult(
        issue_rows=rows,
        summary=summary,
        report_markdown="",
    )
    return ImportSimplicityAuditResult(
        issue_rows=rows,
        summary=summary,
        report_markdown=render_import_simplicity_audit_report(result),
    )


def write_import_simplicity_audit_artifacts(
    output_dir: str | Path = ARTIFACT_DIR.parent,
    *,
    result: ImportSimplicityAuditResult | None = None,
) -> ImportSimplicityAuditArtifacts:
    payload = result or analyze_import_simplicity()
    run_dir = Path(output_dir) / "import_simplicity_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "import_simplicity_audit_report.md"
    summary_path = run_dir / "import_simplicity_audit_summary.json"
    issues_path = run_dir / "import_simplicity_issues.csv"

    report_path.write_text(payload.report_markdown, encoding="utf-8")
    summary_path.write_text(json.dumps(payload.summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(issues_path, payload.issue_rows)
    return ImportSimplicityAuditArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        issues_path=issues_path,
    )


def render_import_simplicity_audit_report(result: ImportSimplicityAuditResult) -> str:
    return "\n".join(
        [
            "# Import Simplicity Audit",
            "",
            "This audit inventories clever import, path, and package-surface patterns that PLN-034 removes.",
            "",
            "## Summary",
            "",
            f"- Passes: `{result.summary['passes']}`",
            f"- Issues: `{result.summary['issue_count']}`",
            f"- Violations: `{result.summary['violation_count']}`",
            f"- Debt rows: `{result.summary['debt_count']}`",
            f"- Wildcard imports: `{result.summary['wildcard_import_count']}`",
            f"- Path sniffing rows: `{result.summary['path_sniffing_count']}`",
            f"- Script bootstrap rows: `{result.summary['script_bootstrap_count']}`",
            f"- Package import side effects: `{result.summary['package_side_effect_count']}`",
            f"- Legacy root imports: `{result.summary['legacy_root_import_count']}`",
            "",
            "## Interpretation",
            "",
            "- `violation` rows block strict mode.",
            "- `debt` rows are known compatibility surfaces to remove or explicitly justify.",
            "- Explicit static `__all__` lists are allowed; wildcard facades and dynamic public surfaces are not.",
        ]
    )


def _issue_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in _python_files():
        relative = _relative(path)
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            rows.append(_row(relative, exc.lineno or 0, "parse_error", "violation", str(exc)))
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.names and any(alias.name == "*" for alias in node.names):
                rows.append(_row(relative, node.lineno, "wildcard_import", "violation", _node_text(text, node)))
            elif isinstance(node, ast.Call) and _is_sys_path_mutation(node):
                status = "allowed" if relative in ALLOWED_SCRIPT_BOOTSTRAP else "violation"
                rows.append(_row(relative, node.lineno, "script_bootstrap", status, _node_text(text, node)))
            elif isinstance(node, ast.Assign) and _assigns_pythonpath(node):
                status = "allowed" if relative in ALLOWED_SCRIPT_BOOTSTRAP else "violation"
                rows.append(_row(relative, node.lineno, "script_bootstrap", status, _node_text(text, node)))
            elif isinstance(node, ast.Call) and _looks_like_path_parents_sniff(node):
                status = "allowed" if relative in ALLOWED_PATH_SNIFFING else "violation"
                rows.append(_row(relative, node.lineno, "path_sniffing", status, _node_text(text, node)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__":
                if path.name == "__init__.py":
                    rows.append(_row(relative, node.lineno, "module_getattr", "violation", _node_text(text, node)))
            elif isinstance(node, ast.Assign) and _assigns_dynamic_all(node):
                rows.append(_row(relative, node.lineno, "dynamic_all", "violation", _node_text(text, node)))

        rows.extend(_package_side_effect_rows(relative, text, tree))
        rows.extend(_legacy_root_import_rows(relative, tree, text))
    return rows


def _python_files() -> tuple[Path, ...]:
    roots = [REPO_ROOT / "src", REPO_ROOT / "scripts"]
    return tuple(sorted(path for root in roots if root.exists() for path in root.rglob("*.py")))


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _row(path: str, line: int, kind: str, status: str, detail: str) -> dict[str, object]:
    return {
        "path": path,
        "line": line,
        "kind": kind,
        "status": status,
        "detail": detail.strip(),
    }


def _node_text(text: str, node: ast.AST) -> str:
    if not hasattr(node, "lineno"):
        return ""
    lines = text.splitlines()
    index = max(int(node.lineno) - 1, 0)
    return lines[index].strip() if index < len(lines) else ""


def _is_sys_path_mutation(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in {"insert", "append", "extend"}
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "path"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "sys"
    )


def _assigns_pythonpath(node: ast.Assign) -> bool:
    for target in node.targets:
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "environ"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "os"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "PYTHONPATH"
        ):
            return True
    return False


def _looks_like_path_parents_sniff(node: ast.Call) -> bool:
    text = ast.dump(node)
    return "Path" in text and "__file__" in text and "parents" in text


def _assigns_dynamic_all(node: ast.Assign) -> bool:
    if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
        return False
    return not isinstance(node.value, (ast.List, ast.Tuple))


def _package_side_effect_rows(path: str, text: str, tree: ast.AST) -> list[dict[str, object]]:
    if path != "src/kinematic_classifier_sandbox/__init__.py":
        return []
    rows: list[dict[str, object]] = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            rows.append(_row(path, node.lineno, "package_import_side_effect", "violation", _node_text(text, node)))
    return rows


def _legacy_root_import_rows(path: str, tree: ast.AST, text: str) -> list[dict[str, object]]:
    if not path.startswith("src/kinematic_classifier_sandbox/"):
        return []
    if path.count("/") == 2:
        return []
    rows: list[dict[str, object]] = []
    for node in ast.walk(tree):
        module = None
        if isinstance(node, ast.ImportFrom):
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("kinematic_classifier_sandbox."):
                    module = alias.name
        if not module:
            continue
        parts = module.split(".")
        if len(parts) >= 2 and parts[0] == "kinematic_classifier_sandbox" and parts[1] in LEGACY_ROOT_WRAPPER_MODULES:
            rows.append(_row(path, node.lineno, "legacy_root_import", "violation", _node_text(text, node)))
    return rows


def _write_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    fieldnames = ["path", "line", "kind", "status", "detail"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
