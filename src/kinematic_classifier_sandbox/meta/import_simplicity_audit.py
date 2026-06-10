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

COMMON_ACCIDENTAL_EXPORT_NAMES = {
    "Callable",
    "Path",
    "Protocol",
    "TYPE_CHECKING",
    "annotations",
    "asdict",
    "csv",
    "dataclass",
    "erf",
    "io",
    "json",
    "log",
    "median",
    "plt",
    "repo_root",
    "sqrt",
}

BROAD_PACKAGE_SURFACE_EXPORT_LIMIT = 25


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
        "root_wrapper_count": sum(1 for row in rows if row["kind"] == "root_wrapper_surface"),
        "accidental_export_count": sum(1 for row in rows if row["kind"] == "accidental_export"),
        "broad_package_surface_count": sum(1 for row in rows if row["kind"] == "broad_package_surface"),
        "import_cycle_count": sum(1 for row in rows if row["kind"] == "import_cycle"),
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
            f"- Root wrapper surfaces: `{result.summary['root_wrapper_count']}`",
            f"- Accidental exports: `{result.summary['accidental_export_count']}`",
            f"- Broad package surfaces: `{result.summary['broad_package_surface_count']}`",
            f"- Import cycles: `{result.summary['import_cycle_count']}`",
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
                status = _script_bootstrap_status(relative)
                rows.append(_row(relative, node.lineno, "script_bootstrap", status, _node_text(text, node)))
            elif isinstance(node, ast.Assign) and _assigns_pythonpath(node):
                status = _script_bootstrap_status(relative)
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
        rows.extend(_root_wrapper_surface_rows(path, relative, text, tree))
        rows.extend(_accidental_export_rows(relative, tree))
        rows.extend(_broad_package_surface_rows(relative, tree))
    rows.extend(_import_cycle_rows())
    return rows


def _python_files() -> tuple[Path, ...]:
    roots = [REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "tests"]
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


def _script_bootstrap_status(path: str) -> str:
    if path in ALLOWED_SCRIPT_BOOTSTRAP:
        return "allowed"
    if path.startswith("tests/"):
        return "debt"
    return "violation"


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


def _root_wrapper_surface_rows(path: Path, relative: str, text: str, tree: ast.AST) -> list[dict[str, object]]:
    if path.parent != PACKAGE_ROOT or path.stem not in LEGACY_ROOT_WRAPPER_MODULES:
        return []
    exported = _static_all_names(tree)
    detail = f"{path.stem} exports {len(exported)} names"
    if exported:
        detail = f"{detail}: {', '.join(exported[:12])}"
        if len(exported) > 12:
            detail = f"{detail}, ..."
    line = _all_assignment_line(tree) or 1
    return [_row(relative, line, "root_wrapper_surface", "debt", detail)]


def _accidental_export_rows(path: str, tree: ast.AST) -> list[dict[str, object]]:
    exported = set(_static_all_names(tree))
    if not exported:
        return []
    imported_names = _top_level_imported_names(tree)
    rows: list[dict[str, object]] = []
    for name in sorted(exported & COMMON_ACCIDENTAL_EXPORT_NAMES & imported_names.keys()):
        rows.append(
            _row(
                path,
                imported_names[name],
                "accidental_export",
                "debt",
                f"`{name}` is imported and re-exported; keep public surfaces domain-specific",
            )
        )
    return rows


def _broad_package_surface_rows(path: str, tree: ast.AST) -> list[dict[str, object]]:
    if not path.endswith("/__init__.py"):
        return []
    exported = _static_all_names(tree)
    if len(exported) <= BROAD_PACKAGE_SURFACE_EXPORT_LIMIT:
        return []
    return [
        _row(
            path,
            _all_assignment_line(tree) or 1,
            "broad_package_surface",
            "debt",
            f"package initializer exports {len(exported)} names; prefer a named API module",
        )
    ]


def _static_all_names(tree: ast.AST) -> tuple[str, ...]:
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            return ()
        names = []
        for element in node.value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                names.append(element.value)
        return tuple(names)
    return ()


def _all_assignment_line(tree: ast.AST) -> int | None:
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return node.lineno
    return None


def _top_level_imported_names(tree: ast.AST) -> dict[str, int]:
    imported: dict[str, int] = {}
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported[alias.asname or alias.name.split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    imported[alias.asname or alias.name] = node.lineno
    return imported


def _import_cycle_rows() -> list[dict[str, object]]:
    graph = _package_import_graph()
    rows: list[dict[str, object]] = []
    for cycle in _strongly_connected_components(graph):
        if len(cycle) <= 1:
            continue
        paths = sorted(_module_to_relative_path(module) for module in cycle)
        rows.append(
            _row(
                paths[0],
                1,
                "import_cycle",
                "debt",
                " -> ".join(sorted(cycle)),
            )
        )
    return rows


def _package_import_graph() -> dict[str, set[str]]:
    files = tuple(sorted(PACKAGE_ROOT.rglob("*.py")))
    modules = {_module_name(path): path for path in files}
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for module, path in modules.items():
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in _top_level_import_nodes(tree):
            for imported in _internal_import_targets(module, node):
                target = _nearest_known_module(imported, modules)
                if target and target != module:
                    graph[module].add(target)
    return graph


def _top_level_import_nodes(tree: ast.AST) -> tuple[ast.Import | ast.ImportFrom, ...]:
    if not isinstance(tree, ast.Module):
        return ()
    imports: list[ast.Import | ast.ImportFrom] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
    return tuple(imports)


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT)
    parts = ("kinematic_classifier_sandbox", *relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_to_relative_path(module: str) -> str:
    suffix = module.removeprefix("kinematic_classifier_sandbox").lstrip(".")
    path = PACKAGE_ROOT / Path(*suffix.split(".")) if suffix else PACKAGE_ROOT / "__init__.py"
    py_path = path.with_suffix(".py")
    if py_path.exists():
        return _relative(py_path)
    return _relative(path / "__init__.py")


def _internal_import_targets(current_module: str, node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names if alias.name.startswith("kinematic_classifier_sandbox"))
    if not isinstance(node, ast.ImportFrom):
        return ()
    if node.level == 0:
        return (node.module,) if node.module and node.module.startswith("kinematic_classifier_sandbox") else ()

    current_package = current_module if _is_package_module(current_module) else current_module.rsplit(".", 1)[0]
    package_parts = current_package.split(".")
    keep_count = len(package_parts) - node.level + 1
    if keep_count < 1:
        return ()
    base = ".".join(package_parts[:keep_count])
    if node.module:
        return (f"{base}.{node.module}",)
    return tuple(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")


def _is_package_module(module: str) -> bool:
    suffix = module.removeprefix("kinematic_classifier_sandbox").lstrip(".")
    path = PACKAGE_ROOT if not suffix else PACKAGE_ROOT / Path(*suffix.split("."))
    return (path / "__init__.py").exists()


def _nearest_known_module(module: str | None, modules: dict[str, Path]) -> str | None:
    if not module:
        return None
    candidate = module
    while candidate.startswith("kinematic_classifier_sandbox"):
        if candidate in modules:
            return candidate
        if "." not in candidate:
            return None
        candidate = candidate.rsplit(".", 1)[0]
    return None


def _strongly_connected_components(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph[node]:
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] != indices[node]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        components.append(tuple(component))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return components


def _write_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    fieldnames = ["path", "line", "kind", "status", "detail"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
