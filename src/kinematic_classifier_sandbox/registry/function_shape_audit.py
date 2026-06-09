from __future__ import annotations

import ast
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..markdown_builder import MarkdownDocument
from ..utils.io import write_csv

PACKAGE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PACKAGE_DIR.parent
REPO_ROOT = SRC_DIR.parent
PACKAGE_SRC_DIR = SRC_DIR / "kinematic_classifier_sandbox"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "function_shape_audit_v1"
OVERRIDES_PATH = REPO_ROOT / "docs" / "protocols" / "function_shape_overrides.yaml"

SCENARIO_TOKENS = {
    "stationary",
    "constant_velocity",
    "constant_acceleration",
    "braking",
    "maneuver",
    "toy_1d",
    "identity_1d",
    "boundary_v1",
    "stress_v1",
    "adversarial_v1",
    "realistic_v1",
    "stationary_regular",
    "constant_velocity_regular",
    "constant_velocity_irregular",
    "constant_acceleration_regular",
    "ambiguous",
    "late_flip",
    "switching",
}


@dataclass(frozen=True, slots=True)
class FunctionShapeRow:
    module_path: str
    qualified_name: str
    symbol_kind: str
    line_number: int
    visibility: str
    heuristic_role: str
    heuristic_specificity: str
    role: str
    specificity: str
    scenario_tokens: tuple[str, ...]
    override_source: str | None
    rationale: str


@dataclass(frozen=True, slots=True)
class FileShapeSummaryRow:
    module_path: str
    callable_count: int
    function_count: int
    method_count: int
    generic_count: int
    study_specific_count: int
    scenario_specific_count: int
    dominant_specificity: str
    dominant_role: str


@dataclass(frozen=True, slots=True)
class FunctionShapeAuditResult:
    function_rows: tuple[FunctionShapeRow, ...]
    file_rows: tuple[FileShapeSummaryRow, ...]
    summary: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class FunctionShapeAuditArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    function_rows_path: Path
    file_rows_path: Path


def load_function_shape_overrides() -> dict[str, dict[str, dict[str, object]]]:
    if not OVERRIDES_PATH.exists():
        return {"modules": {}, "functions": {}}
    payload = yaml.safe_load(OVERRIDES_PATH.read_text(encoding="utf-8")) or {}
    modules = payload.get("modules", {})
    functions = payload.get("functions", {})
    if not isinstance(modules, dict) or not isinstance(functions, dict):
        raise ValueError("function shape overrides must define 'modules' and 'functions' mappings")
    return {"modules": modules, "functions": functions}


def _module_family(module_path: str) -> str:
    relative = module_path.removeprefix("src/kinematic_classifier_sandbox/")
    return relative.split("/", 1)[0]


def _iter_callables(
    body: list[ast.stmt],
    *,
    parent_name: str | None = None,
) -> list[tuple[ast.AST, str, str]]:
    rows: list[tuple[ast.AST, str, str]] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualified_name = node.name if parent_name is None else f"{parent_name}.{node.name}"
            rows.append((node, qualified_name, "method" if parent_name is not None else "function"))
        elif isinstance(node, ast.ClassDef):
            class_name = node.name if parent_name is None else f"{parent_name}.{node.name}"
            rows.extend(_iter_callables(node.body, parent_name=class_name))
    return rows


def _string_constants(node: ast.AST) -> tuple[str, ...]:
    values: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.add(child.value)
    return tuple(sorted(values))


def _role_for_callable(module_path: str, qualified_name: str) -> str:
    leaf_name = qualified_name.split(".")[-1]
    if leaf_name == "main":
        return "entrypoint"
    if leaf_name.startswith("analyze_"):
        return "analysis"
    if leaf_name.startswith("render_"):
        return "reporting"
    if leaf_name.startswith("write_"):
        return "artifact_io"
    if leaf_name.startswith("load_"):
        return "loader"
    if leaf_name.startswith("generate_"):
        return "generator"
    if leaf_name.startswith("default_"):
        return "config"
    if leaf_name.startswith("run_"):
        return "runner"
    if any(token in leaf_name for token in ("plot", "figure", "png", "svg", "chart")):
        return "plotting"
    if "/utils/" in module_path or module_path.endswith("_utils.py") or module_path.endswith("/utils.py"):
        return "utility"
    if leaf_name.startswith("_"):
        return "helper"
    return "public_api"


def _specificity_for_callable(
    module_path: str,
    qualified_name: str,
    string_constants: tuple[str, ...],
) -> tuple[str, tuple[str, ...], str]:
    matched_tokens = tuple(
        sorted(
            {
                token
                for token in SCENARIO_TOKENS
                if token in module_path or token in qualified_name or token in string_constants
            }
        )
    )
    if "/witnesses/" in module_path:
        return "scenario_specific", matched_tokens, "module lives under witnesses"
    if matched_tokens:
        return "scenario_specific", matched_tokens, f"callable references scenario tokens: {', '.join(matched_tokens)}"
    generic_markers = (
        "/utils/",
        "/schema/",
        "/common_experiment/",
        "/registry/",
        "/methodology/",
        "/showcase/",
    )
    if any(marker in module_path for marker in generic_markers):
        return "generic", matched_tokens, "module belongs to shared infrastructure layer"
    if module_path.endswith(("contracts.py", "protocols.py", "config.py", "config_models.py", "types.py")):
        return "generic", matched_tokens, "module is a shared contract/config surface"
    return "study_specific", matched_tokens, "module is repo-specific but not tied to a single named scenario"


def _scan_function_rows() -> list[FunctionShapeRow]:
    overrides = load_function_shape_overrides()
    module_overrides = overrides["modules"]
    function_overrides = overrides["functions"]
    rows: list[FunctionShapeRow] = []
    for path in sorted(PACKAGE_SRC_DIR.rglob("*.py")):
        module_path = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node, qualified_name, symbol_kind in _iter_callables(tree.body):
            assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            visibility = "private" if qualified_name.split(".")[-1].startswith("_") else "public"
            string_constants = _string_constants(node)
            heuristic_role = _role_for_callable(module_path, qualified_name)
            heuristic_specificity, scenario_tokens, rationale = _specificity_for_callable(
                module_path,
                qualified_name,
                string_constants,
            )
            role = heuristic_role
            specificity = heuristic_specificity
            module_override = module_overrides.get(module_path, {})
            function_override = function_overrides.get(f"{module_path}::{qualified_name}", {})
            override_source: str | None = None
            if module_override:
                override_source = f"module:{module_path}"
            if function_override:
                override_source = f"function:{module_path}::{qualified_name}"
            role = str(function_override.get("role", module_override.get("role", role)))
            specificity = str(
                function_override.get("specificity", module_override.get("specificity", specificity))
            )
            override_rationale = function_override.get("rationale", module_override.get("rationale"))
            if override_rationale is not None:
                rationale = str(override_rationale)
            rows.append(
                FunctionShapeRow(
                    module_path=module_path,
                    qualified_name=qualified_name,
                    symbol_kind=symbol_kind,
                    line_number=node.lineno,
                    visibility=visibility,
                    heuristic_role=heuristic_role,
                    heuristic_specificity=heuristic_specificity,
                    role=role,
                    specificity=specificity,
                    scenario_tokens=scenario_tokens,
                    override_source=override_source,
                    rationale=rationale,
                )
            )
    return rows


def _scan_file_rows(function_rows: list[FunctionShapeRow]) -> list[FileShapeSummaryRow]:
    rows_by_module: dict[str, list[FunctionShapeRow]] = {}
    for row in function_rows:
        rows_by_module.setdefault(row.module_path, []).append(row)

    file_rows: list[FileShapeSummaryRow] = []
    for module_path, rows in sorted(rows_by_module.items()):
        specificity_counts = Counter(row.specificity for row in rows)
        role_counts = Counter(row.role for row in rows)
        file_rows.append(
            FileShapeSummaryRow(
                module_path=module_path,
                callable_count=len(rows),
                function_count=sum(1 for row in rows if row.symbol_kind == "function"),
                method_count=sum(1 for row in rows if row.symbol_kind == "method"),
                generic_count=specificity_counts.get("generic", 0),
                study_specific_count=specificity_counts.get("study_specific", 0),
                scenario_specific_count=specificity_counts.get("scenario_specific", 0),
                dominant_specificity=max(
                    ("generic", "study_specific", "scenario_specific"),
                    key=lambda key: (specificity_counts.get(key, 0), key),
                ),
                dominant_role=max(sorted(role_counts), key=lambda key: (role_counts[key], key)),
            )
        )
    return file_rows


def render_function_shape_audit_report(result: FunctionShapeAuditResult) -> str:
    report = MarkdownDocument("Function Shape Audit")
    report.paragraph(
        "This audit classifies every top-level function and class method under "
        "`src/kinematic_classifier_sandbox/` by role and by specificity."
    )
    report.paragraph(
        "The specificity buckets are intentionally coarse: "
        "`generic`, `study_specific`, and `scenario_specific`."
    )
    report.paragraph(
        f"Manual overrides are loaded from `{OVERRIDES_PATH.relative_to(REPO_ROOT)}` when present."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Files scanned: `{result.summary['file_count']}`",
            f"Callables scanned: `{result.summary['callable_count']}`",
            f"Generic callables: `{result.summary['generic_count']}`",
            f"Study-specific callables: `{result.summary['study_specific_count']}`",
            f"Scenario-specific callables: `{result.summary['scenario_specific_count']}`",
            f"Overridden callables: `{result.summary['override_count']}`",
        ]
    )
    report.heading("By Family", level=2)
    report.table(
        ["family", "callables", "generic", "study_specific", "scenario_specific"],
        [
            (
                family,
                counts["callables"],
                counts["generic"],
                counts["study_specific"],
                counts["scenario_specific"],
            )
            for family, counts in sorted(result.summary["family_counts"].items())
        ],
    )
    report.heading("Files", level=2)
    report.table(
        [
            "module",
            "callables",
            "functions",
            "methods",
            "generic",
            "study_specific",
            "scenario_specific",
            "dominant_specificity",
            "dominant_role",
        ],
        [
            (
                f"`{row.module_path}`",
                row.callable_count,
                row.function_count,
                row.method_count,
                row.generic_count,
                row.study_specific_count,
                row.scenario_specific_count,
                row.dominant_specificity,
                row.dominant_role,
            )
            for row in result.file_rows
        ],
    )
    report.heading("Overrides", level=2)
    override_rows = [row for row in result.function_rows if row.override_source is not None]
    if override_rows:
        report.table(
            [
                "callable",
                "heuristic_role",
                "final_role",
                "heuristic_specificity",
                "final_specificity",
                "override_source",
                "rationale",
            ],
            [
                (
                    f"`{row.module_path}::{row.qualified_name}`",
                    row.heuristic_role,
                    row.role,
                    row.heuristic_specificity,
                    row.specificity,
                    row.override_source or "",
                    row.rationale,
                )
                for row in override_rows
            ],
        )
    else:
        report.paragraph("No explicit override rows were applied.")
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "`generic` means the callable looks reusable across studies or package layers.",
            "`study_specific` means the callable belongs to this repo's methodology lab but is not tied to a single named scenario.",
            "`scenario_specific` means the callable is tied to named classes, tiers, witnesses, or benchmark scenarios.",
            "Explicit overrides are for audit curation only; they should expose repo-shape intent, not hide structural problems.",
            "Use the CSV outputs when deciding which modules should be promoted into cleaner shared layers versus kept as witness or benchmark code.",
        ]
    )
    return report.text()


def analyze_function_shape_audit() -> FunctionShapeAuditResult:
    function_rows = tuple(_scan_function_rows())
    file_rows = tuple(_scan_file_rows(list(function_rows)))
    family_counts: dict[str, dict[str, int]] = {}
    for row in function_rows:
        family = _module_family(row.module_path)
        counts = family_counts.setdefault(
            family,
            {"callables": 0, "generic": 0, "study_specific": 0, "scenario_specific": 0},
        )
        counts["callables"] += 1
        counts[row.specificity] += 1
    summary = {
        "file_count": len(file_rows),
        "callable_count": len(function_rows),
        "generic_count": sum(1 for row in function_rows if row.specificity == "generic"),
        "study_specific_count": sum(1 for row in function_rows if row.specificity == "study_specific"),
        "scenario_specific_count": sum(1 for row in function_rows if row.specificity == "scenario_specific"),
        "override_count": sum(1 for row in function_rows if row.override_source is not None),
        "family_counts": family_counts,
        "override_path": str(OVERRIDES_PATH.relative_to(REPO_ROOT)),
    }
    report_markdown = render_function_shape_audit_report(
        FunctionShapeAuditResult(
            function_rows=function_rows,
            file_rows=file_rows,
            summary=summary,
            report_markdown="",
        )
    )
    return FunctionShapeAuditResult(
        function_rows=function_rows,
        file_rows=file_rows,
        summary=summary,
        report_markdown=report_markdown,
    )


def write_function_shape_audit_artifacts(
    output_dir: str | Path,
    *,
    result: FunctionShapeAuditResult | None = None,
) -> FunctionShapeAuditArtifacts:
    audit = result or analyze_function_shape_audit()
    run_dir = Path(output_dir) / "function_shape_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "function_shape_audit_report.md"
    summary_path = run_dir / "function_shape_audit_summary.json"
    function_rows_path = run_dir / "function_shape_rows.csv"
    file_rows_path = run_dir / "file_shape_summary.csv"

    report_path.write_text(audit.report_markdown, encoding="utf-8")
    summary_path.write_text(json.dumps(audit.summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(
        function_rows_path,
        [
            {
                "module_path": row.module_path,
                "qualified_name": row.qualified_name,
                "symbol_kind": row.symbol_kind,
                "line_number": row.line_number,
                "visibility": row.visibility,
                "heuristic_role": row.heuristic_role,
                "heuristic_specificity": row.heuristic_specificity,
                "role": row.role,
                "specificity": row.specificity,
                "scenario_tokens": ";".join(row.scenario_tokens),
                "override_source": row.override_source or "",
                "rationale": row.rationale,
            }
            for row in audit.function_rows
        ],
        [
            "module_path",
            "qualified_name",
            "symbol_kind",
            "line_number",
            "visibility",
            "heuristic_role",
            "heuristic_specificity",
            "role",
            "specificity",
            "scenario_tokens",
            "override_source",
            "rationale",
        ],
    )
    write_csv(
        file_rows_path,
        [
            {
                "module_path": row.module_path,
                "callable_count": row.callable_count,
                "function_count": row.function_count,
                "method_count": row.method_count,
                "generic_count": row.generic_count,
                "study_specific_count": row.study_specific_count,
                "scenario_specific_count": row.scenario_specific_count,
                "dominant_specificity": row.dominant_specificity,
                "dominant_role": row.dominant_role,
            }
            for row in audit.file_rows
        ],
        [
            "module_path",
            "callable_count",
            "function_count",
            "method_count",
            "generic_count",
            "study_specific_count",
            "scenario_specific_count",
            "dominant_specificity",
            "dominant_role",
        ],
    )
    return FunctionShapeAuditArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        function_rows_path=function_rows_path,
        file_rows_path=file_rows_path,
    )
