from __future__ import annotations

from dataclasses import dataclass
import ast
import csv
import io
import json
import os
from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
REPO_ROOT = SRC_DIR.parent
PACKAGE_SRC_DIR = SRC_DIR / "kinematic_classifier_sandbox"
DOCS_MATH_DIR = REPO_ROOT / "docs" / "math"
EQUATION_REGISTRY_PATH = DOCS_MATH_DIR / "equation_registry.yaml"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "formal_math_registry_v1"


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _figure_to_png(fig) -> bytes:
    plt = _prepare_matplotlib()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass(frozen=True, slots=True)
class FormalMathFunctionRow:
    module_path: str
    function_name: str
    line_number: int
    visibility: str
    role: str
    doc_summary: str
    equation_ids: tuple[str, ...]
    equation_link_count: int


@dataclass(frozen=True, slots=True)
class FormalMathEquationRow:
    equation_id: str
    status: str
    implementation_module: str
    implementation_function: str
    linked_function_exists: bool
    artifact_count: int
    test_count: int


@dataclass(frozen=True, slots=True)
class FormalMathRegistryResult:
    function_rows: tuple[FormalMathFunctionRow, ...]
    equation_rows: tuple[FormalMathEquationRow, ...]
    summary: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class FormalMathRegistryArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    function_registry_path: Path
    equation_registry_path: Path
    crosswalk_path: Path
    plot_path: Path


def load_equation_registry() -> list[dict[str, object]]:
    payload = yaml.safe_load(EQUATION_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    return payload


def _role_for_function(function_name: str) -> str:
    if function_name == "main":
        return "entrypoint"
    if function_name.startswith("analyze_"):
        return "analysis"
    if function_name.startswith("render_"):
        return "render"
    if function_name.startswith("write_"):
        return "artifact_writer"
    if function_name.startswith("load_"):
        return "loader"
    if function_name.startswith("generate_"):
        return "generator"
    if function_name.startswith("run_"):
        return "runner"
    if function_name.startswith("_"):
        return "helper"
    return "public_api"


def _scan_function_rows() -> list[FormalMathFunctionRow]:
    equation_lookup: dict[tuple[str, str], list[str]] = {}
    for row in load_equation_registry():
        implementation = row["implementation"]
        key = (implementation["module"], implementation["function"])
        equation_lookup.setdefault(key, []).append(row["id"])

    rows: list[FormalMathFunctionRow] = []
    for path in sorted(PACKAGE_SRC_DIR.rglob("*.py")):
        module_path = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            function_name = node.name
            equation_ids = tuple(sorted(equation_lookup.get((module_path, function_name), ())))
            doc = ast.get_docstring(node) or ""
            doc_summary = doc.splitlines()[0].strip() if doc else ""
            rows.append(
                FormalMathFunctionRow(
                    module_path=module_path,
                    function_name=function_name,
                    line_number=node.lineno,
                    visibility="private" if function_name.startswith("_") else "public",
                    role=_role_for_function(function_name),
                    doc_summary=doc_summary,
                    equation_ids=equation_ids,
                    equation_link_count=len(equation_ids),
                )
            )
    return rows


def _scan_equation_rows(function_rows: list[FormalMathFunctionRow]) -> list[FormalMathEquationRow]:
    function_lookup = {(row.module_path, row.function_name) for row in function_rows}
    rows: list[FormalMathEquationRow] = []
    for row in load_equation_registry():
        implementation = row["implementation"]
        module_path = implementation["module"]
        function_name = implementation["function"]
        rows.append(
            FormalMathEquationRow(
                equation_id=row["id"],
                status=row["status"],
                implementation_module=module_path,
                implementation_function=function_name,
                linked_function_exists=(module_path, function_name) in function_lookup,
                artifact_count=len(row.get("artifacts", [])),
                test_count=len(row.get("tests", [])),
            )
        )
    return rows


def analyze_formal_math_registry() -> FormalMathRegistryResult:
    function_rows = tuple(_scan_function_rows())
    equation_rows = tuple(_scan_equation_rows(list(function_rows)))

    summary = {
        "function_count": len(function_rows),
        "public_function_count": sum(1 for row in function_rows if row.visibility == "public"),
        "private_function_count": sum(1 for row in function_rows if row.visibility == "private"),
        "analysis_function_count": sum(1 for row in function_rows if row.role == "analysis"),
        "render_function_count": sum(1 for row in function_rows if row.role == "render"),
        "artifact_writer_count": sum(1 for row in function_rows if row.role == "artifact_writer"),
        "helper_count": sum(1 for row in function_rows if row.role == "helper"),
        "generator_count": sum(1 for row in function_rows if row.role == "generator"),
        "runner_count": sum(1 for row in function_rows if row.role == "runner"),
        "entrypoint_count": sum(1 for row in function_rows if row.role == "entrypoint"),
        "equation_count": len(equation_rows),
        "implemented_equation_count": sum(1 for row in equation_rows if row.status == "implemented"),
        "conceptual_equation_count": sum(1 for row in equation_rows if row.status == "conceptual"),
        "linked_equation_count": sum(1 for row in equation_rows if row.linked_function_exists),
        "modules_with_functions": len({row.module_path for row in function_rows}),
        "modules_with_equations": len({row.implementation_module for row in equation_rows if row.linked_function_exists}),
    }
    report_markdown = render_formal_math_registry_report(
        FormalMathRegistryResult(
            function_rows=function_rows,
            equation_rows=equation_rows,
            summary=summary,
            report_markdown="",
        )
    )
    return FormalMathRegistryResult(
        function_rows=function_rows,
        equation_rows=equation_rows,
        summary=summary,
        report_markdown=report_markdown,
    )


def _render_role_plot(result: FormalMathRegistryResult):
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    roles = ["analysis", "render", "artifact_writer", "loader", "generator", "runner", "helper", "public_api", "entrypoint"]
    counts = [sum(1 for row in result.function_rows if row.role == role) for role in roles]
    colors = ["#2563eb", "#0f766e", "#7c3aed", "#d97706", "#dc2626", "#0891b2", "#6b7280", "#111827", "#059669"]
    ax.bar(roles, counts, color=colors)
    ax.set_ylabel("function count")
    ax.set_title("Formal Math Registry Role Coverage", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def render_formal_math_registry_report(result: FormalMathRegistryResult) -> str:
    lines = [
        "# Formal Math Registry",
        "",
        "This registry is the formal inventory of math- and evaluation-facing helper functions plus the equation definitions that point back into code. It is generated from the source tree and the equation registry, so the result is auditable instead of hand-edited.",
        "",
        "## Summary",
        "",
        f"- Functions scanned: `{result.summary['function_count']}`",
        f"- Public functions: `{result.summary['public_function_count']}`",
        f"- Private helpers: `{result.summary['private_function_count']}`",
        f"- Analysis functions: `{result.summary['analysis_function_count']}`",
        f"- Render functions: `{result.summary['render_function_count']}`",
        f"- Artifact writers: `{result.summary['artifact_writer_count']}`",
        f"- Equation entries: `{result.summary['equation_count']}`",
        f"- Implemented equations: `{result.summary['implemented_equation_count']}`",
        f"- Conceptual equations: `{result.summary['conceptual_equation_count']}`",
        f"- Linked equations: `{result.summary['linked_equation_count']}`",
        "",
        "## Equation Registry",
        "",
        "| equation_id | status | implementation | linked | artifacts | tests |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in result.equation_rows:
        implementation = f"`{row.implementation_module}::{row.implementation_function}`"
        linked = "yes" if row.linked_function_exists else "no"
        lines.append(
            f"| `{row.equation_id}` | `{row.status}` | {implementation} | {linked} | {row.artifact_count} | {row.test_count} |"
        )
    lines.extend(
        [
            "",
            "## Function Registry",
            "",
            "The table below lists every top-level function found under `src/kinematic_classifier_sandbox/`.",
            "",
            "| module | function | line | visibility | role | equations | doc summary |",
            "| --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in result.function_rows:
        equation_ids = "<br>".join(f"`{equation_id}`" for equation_id in row.equation_ids) or "none"
        lines.append(
            f"| `{row.module_path}` | `{row.function_name}` | {row.line_number} | {row.visibility} | {row.role} | {equation_ids} | {row.doc_summary or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The equation registry shows which mathematics are implemented versus still conceptual.",
            "- The function registry shows where the real callable surfaces live, including helpers that are not part of the public API.",
            "- The linked-function column is the bridge between derivation-level documentation and executable code.",
        ]
    )
    return "\n".join(lines)


def write_formal_math_registry_artifacts(
    output_dir: str | Path,
    *,
    result: FormalMathRegistryResult | None = None,
) -> FormalMathRegistryArtifacts:
    payload = result or analyze_formal_math_registry()
    output_root = Path(output_dir)
    run_dir = output_root / "formal_math_registry_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "formal_math_registry_report.md"
    summary_path = run_dir / "formal_math_registry_summary.json"
    function_registry_path = run_dir / "function_registry.csv"
    equation_registry_path = run_dir / "equation_registry.csv"
    crosswalk_path = run_dir / "function_equation_crosswalk.csv"
    plot_path = run_dir / "formal_math_registry_role_counts.png"

    report_path.write_text(payload.report_markdown, encoding="utf-8")
    summary_path.write_text(json.dumps(payload.summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(
        function_registry_path,
        [
            {
                "module_path": row.module_path,
                "function_name": row.function_name,
                "line_number": row.line_number,
                "visibility": row.visibility,
                "role": row.role,
                "equation_ids": ";".join(row.equation_ids),
                "equation_link_count": row.equation_link_count,
                "doc_summary": row.doc_summary,
            }
            for row in payload.function_rows
        ],
        [
            "module_path",
            "function_name",
            "line_number",
            "visibility",
            "role",
            "equation_ids",
            "equation_link_count",
            "doc_summary",
        ],
    )
    _write_csv(
        equation_registry_path,
        [
            {
                "equation_id": row.equation_id,
                "status": row.status,
                "implementation_module": row.implementation_module,
                "implementation_function": row.implementation_function,
                "linked_function_exists": row.linked_function_exists,
                "artifact_count": row.artifact_count,
                "test_count": row.test_count,
            }
            for row in payload.equation_rows
        ],
        [
            "equation_id",
            "status",
            "implementation_module",
            "implementation_function",
            "linked_function_exists",
            "artifact_count",
            "test_count",
        ],
    )
    _write_csv(
        crosswalk_path,
        [
            {
                "equation_id": row.equation_id,
                "implementation": f"{row.implementation_module}::{row.implementation_function}",
                "status": row.status,
                "linked_function_exists": row.linked_function_exists,
                "artifact_count": row.artifact_count,
                "test_count": row.test_count,
            }
            for row in payload.equation_rows
        ],
        [
            "equation_id",
            "implementation",
            "status",
            "linked_function_exists",
            "artifact_count",
            "test_count",
        ],
    )
    plot_path.write_bytes(_figure_to_png(_render_role_plot(payload)))

    return FormalMathRegistryArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        function_registry_path=function_registry_path,
        equation_registry_path=equation_registry_path,
        crosswalk_path=crosswalk_path,
        plot_path=plot_path,
    )
