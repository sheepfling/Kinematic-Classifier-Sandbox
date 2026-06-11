from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from ..utils.runtime import repo_root

REPO_ROOT = repo_root()
PACKAGE_ROOT = REPO_ROOT / "src" / "kinematic_classifier_sandbox"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "repo_shape_audit_v1"

FRONT_DOOR_ROOT_MODULES = {
    "__init__.py",
    "__main__.py",
    "api_core.py",
}

ORCHESTRATION_ROOT_MODULES = {
    "artifacts.py",
    "common_1d_study_adapter.py",
    "common_experiment_classifier_registry.py",
    "contracts_rendering.py",
    "markdown_builder.py",
    "runtime_paths.py",
    "scenarios.py",
    "showcase_builder.py",
    "study_candidate_generation.py",
    "study_candidate_generation_rendering.py",
    "study_candidate_generation_types.py",
    "study_candidate_generation_utils.py",
    "study_candidate_protocol.py",
    "study_candidate_protocol_utils.py",
    "trajectory_generator.py",
    "trajectory_generator_rendering.py",
    "trajectory_series.py",
}

LEGACY_COMPAT_WRAPPER_MODULES = {
    "advanced_filter_decision.py",
    "bayesian_walkthroughs.py",
    "catalog.py",
    "common_dataset_comparison.py",
    "common_experiment_harness.py",
    "contracts.py",
    "corpus_adequacy_audit.py",
    "corpus_autodevelopment.py",
    "coverage_report.py",
    "dimensional_lift_audit.py",
    "feature_analysis.py",
    "feature_rows.py",
    "formal_math_registry.py",
    "formal_math_visual_registry.py",
    "functional_surface_catalog.py",
    "generic_classification_evidence_proof.py",
    "generic_feature_taxonomy.py",
    "generic_filtering_contract.py",
    "generic_inference_contract.py",
    "identity_1d.py",
    "identity_posterior_explainer.py",
    "inspection_bundle.py",
    "irregular_window_comparison.py",
    "kalman_filter_bank.py",
    "kalman_observable_comparison.py",
    "kalman_variant_comparison.py",
    "methodology_compendium.py",
    "methodology_latex.py",
    "milestones.py",
    "monte_carlo_benchmark.py",
    "pca_analysis.py",
    "pca_dimensionality_audit.py",
    "pointwise_baseline.py",
    "posterior_explainer.py",
    "prior_sensitivity_analysis.py",
    "repo_story.py",
    "sequential_bayes_accumulator.py",
    "shared_evaluation.py",
    "short_horizon_identifiability.py",
    "strict_equation_audit.py",
    "technique_comparison.py",
    "toy_1d.py",
    "transition_matrix_accumulator.py",
    "validation_ladder.py",
    "velocity_aided_kalman_comparison.py",
    "windowed_baseline.py",
}

LEGACY_ROOT_DEBT_MODULES = {
    "adaptive_stress_corpus.py",
    "advanced_state_inference.py",
    "backend_adapter_proof.py",
    "candidate_generation.py",
    "capability_aware_search.py",
    "class_validity.py",
    "corpus_classifier_scoring.py",
    "corpus_gym.py",
    "corpus_objectives.py",
    "corpus_policy.py",
    "corpus_policy_sweep.py",
    "corpus_search_baseline.py",
    "corpus_synthesis_comparison.py",
    "environment_aware_corpus.py",
    "external_backend_examples.py",
    "external_backend_examples_rendering.py",
    "generated_corpus_features.py",
    "generic_corpus_exploration.py",
    "objective_corpus_gym_runner.py",
    "objective_driven_qd_archive.py",
    "quality_diversity_corpus.py",
    "rl_backend_decision.py",
    "selected_generated_corpus.py",
}

SCAN_ROOTS = ("src", "docs", "tests", "scripts", "experiments", "templates")
CRUFT_SUFFIXES = (".pyc", ".aux", ".fdb_latexmk", ".fls", ".log", ".out")
CRUFT_NAMES = {".DS_Store", "__pycache__"}


@dataclass(frozen=True, slots=True)
class RepoShapeAuditResult:
    root_module_rows: tuple[dict[str, object], ...]
    duplicate_module_rows: tuple[dict[str, object], ...]
    duplicate_script_rows: tuple[dict[str, object], ...]
    generated_cruft_rows: tuple[dict[str, object], ...]
    oversized_module_rows: tuple[dict[str, object], ...]
    issue_rows: tuple[dict[str, object], ...]
    summary: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class RepoShapeAuditArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    root_module_inventory_path: Path
    duplicate_module_inventory_path: Path
    duplicate_script_inventory_path: Path
    generated_cruft_inventory_path: Path
    oversized_module_inventory_path: Path
    issues_path: Path


def analyze_repo_shape() -> RepoShapeAuditResult:
    root_module_rows = tuple(_root_module_rows())
    duplicate_module_rows = tuple(_duplicate_module_rows())
    duplicate_script_rows = tuple(_duplicate_script_rows())
    generated_cruft_rows = tuple(_generated_cruft_rows())
    oversized_module_rows = tuple(_oversized_module_rows())
    issue_rows = tuple(
        [row for row in root_module_rows if row["status"] == "violation"]
        + [row for row in duplicate_module_rows if row["status"] == "violation"]
        + [row for row in duplicate_script_rows if row["status"] == "violation"]
        + [row for row in generated_cruft_rows if row["status"] == "violation"]
    )
    summary = {
        "passes": not issue_rows,
        "issue_count": len(issue_rows),
        "root_module_count": len(root_module_rows),
        "legacy_wrapper_count": sum(1 for row in root_module_rows if row["status"] == "legacy_wrapper"),
        "legacy_root_debt_count": sum(1 for row in root_module_rows if row["status"] == "legacy_root_debt"),
        "duplicate_module_count": len(duplicate_module_rows),
        "duplicate_script_count": len(duplicate_script_rows),
        "generated_cruft_count": len(generated_cruft_rows),
        "oversized_module_count": len(oversized_module_rows),
    }
    result = RepoShapeAuditResult(
        root_module_rows=root_module_rows,
        duplicate_module_rows=duplicate_module_rows,
        duplicate_script_rows=duplicate_script_rows,
        generated_cruft_rows=generated_cruft_rows,
        oversized_module_rows=oversized_module_rows,
        issue_rows=issue_rows,
        summary=summary,
        report_markdown="",
    )
    return RepoShapeAuditResult(
        root_module_rows=root_module_rows,
        duplicate_module_rows=duplicate_module_rows,
        duplicate_script_rows=duplicate_script_rows,
        generated_cruft_rows=generated_cruft_rows,
        oversized_module_rows=oversized_module_rows,
        issue_rows=issue_rows,
        summary=summary,
        report_markdown=render_repo_shape_audit_report(result),
    )


def write_repo_shape_audit_artifacts(
    output_dir: str | Path = ARTIFACT_DIR.parent,
    *,
    result: RepoShapeAuditResult | None = None,
) -> RepoShapeAuditArtifacts:
    payload = result or analyze_repo_shape()
    run_dir = Path(output_dir) / "repo_shape_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "repo_shape_audit_report.md"
    summary_path = run_dir / "repo_shape_audit_summary.json"
    root_module_inventory_path = run_dir / "root_module_inventory.csv"
    duplicate_module_inventory_path = run_dir / "duplicate_module_inventory.csv"
    duplicate_script_inventory_path = run_dir / "duplicate_script_inventory.csv"
    generated_cruft_inventory_path = run_dir / "generated_cruft_inventory.csv"
    oversized_module_inventory_path = run_dir / "oversized_module_inventory.csv"
    issues_path = run_dir / "repo_shape_issues.csv"

    report_path.write_text(payload.report_markdown, encoding="utf-8")
    summary_path.write_text(json.dumps(payload.summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(root_module_inventory_path, payload.root_module_rows)
    _write_csv(duplicate_module_inventory_path, payload.duplicate_module_rows)
    _write_csv(duplicate_script_inventory_path, payload.duplicate_script_rows)
    _write_csv(generated_cruft_inventory_path, payload.generated_cruft_rows)
    _write_csv(oversized_module_inventory_path, payload.oversized_module_rows)
    _write_csv(issues_path, payload.issue_rows)
    return RepoShapeAuditArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        root_module_inventory_path=root_module_inventory_path,
        duplicate_module_inventory_path=duplicate_module_inventory_path,
        duplicate_script_inventory_path=duplicate_script_inventory_path,
        generated_cruft_inventory_path=generated_cruft_inventory_path,
        oversized_module_inventory_path=oversized_module_inventory_path,
        issues_path=issues_path,
    )


def render_repo_shape_audit_report(result: RepoShapeAuditResult) -> str:
    return "\n".join(
        [
            "# Repo Shape Audit",
            "",
            "This audit checks whether source layout follows the declared methodology package map.",
            "",
            "## Summary",
            "",
            f"- Passes: `{result.summary['passes']}`",
            f"- Issues: `{result.summary['issue_count']}`",
            f"- Root modules: `{result.summary['root_module_count']}`",
            f"- Legacy wrappers: `{result.summary['legacy_wrapper_count']}`",
            f"- Legacy root debt modules: `{result.summary['legacy_root_debt_count']}`",
            f"- Duplicate module names: `{result.summary['duplicate_module_count']}`",
            f"- Duplicate root scripts: `{result.summary['duplicate_script_count']}`",
            f"- Generated cruft entries: `{result.summary['generated_cruft_count']}`",
            f"- Oversized modules: `{result.summary['oversized_module_count']}`",
            "",
            "## Interpretation",
            "",
            "- `violation` rows fail the audit.",
            "- `legacy_wrapper` rows are allowed compatibility debt and should shrink over time.",
            "- `legacy_root_debt` rows are currently tolerated implementation debt and should be migrated by domain.",
        ]
    )


def _root_module_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        name = path.name
        if name in FRONT_DOOR_ROOT_MODULES:
            status = "front_door"
        elif name in ORCHESTRATION_ROOT_MODULES:
            status = "orchestration"
        elif name in LEGACY_COMPAT_WRAPPER_MODULES and _is_wrapper(path):
            status = "legacy_wrapper"
        elif name in LEGACY_ROOT_DEBT_MODULES:
            status = "legacy_root_debt"
        else:
            status = "violation"
        rows.append(
            {
                "path": _rel(path),
                "module": path.stem,
                "line_count": _line_count(path),
                "status": status,
                "is_wrapper": _is_wrapper(path),
            }
        )
    return rows


def _duplicate_module_rows() -> list[dict[str, object]]:
    root_modules = {path.stem: path for path in PACKAGE_ROOT.glob("*.py")}
    rows: list[dict[str, object]] = []
    for name, root_path in sorted(root_modules.items()):
        duplicates = [
            path
            for path in PACKAGE_ROOT.glob(f"*/*{name}.py")
            if path.parent.name != "__pycache__"
        ]
        duplicates.extend(
            path
            for path in PACKAGE_ROOT.glob(f"*/*/{name}.py")
            if "__pycache__" not in path.parts
        )
        if not duplicates:
            continue
        if root_path.name == "__init__.py":
            continue
        allowed = (
            root_path.name in LEGACY_COMPAT_WRAPPER_MODULES
            or root_path.name in ORCHESTRATION_ROOT_MODULES
            or root_path.name in LEGACY_ROOT_DEBT_MODULES
        )
        rows.append(
            {
                "root_path": _rel(root_path),
                "duplicate_paths": " | ".join(_rel(path) for path in sorted(set(duplicates))),
                "status": "allowed_legacy_debt" if allowed else "violation",
            }
        )
    return rows


def _duplicate_script_rows() -> list[dict[str, object]]:
    scripts_root = REPO_ROOT / "scripts"
    root_scripts = {path.name: path for path in scripts_root.glob("*.py")}
    rows: list[dict[str, object]] = []
    for subdir in ("audit", "build", "render", "run", "workflows"):
        path = scripts_root / subdir
        if not path.exists():
            continue
        for script in sorted(path.glob("*.py")):
            if script.name == "_bootstrap.py":
                continue
            root_script = root_scripts.get(script.name)
            if root_script is None:
                continue
            rows.append(
                {
                    "root_script": _rel(root_script),
                    "canonical_script": _rel(script),
                    "status": "violation",
                }
            )
    return rows


def _generated_cruft_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root_name in SCAN_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.name in CRUFT_NAMES or (path.is_file() and path.suffix in CRUFT_SUFFIXES):
                rows.append({"path": _rel(path), "kind": "directory" if path.is_dir() else "file", "status": "violation"})
    return rows


def _oversized_module_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        line_count = _line_count(path)
        if line_count >= 700:
            rows.append(
                {
                    "path": _rel(path),
                    "line_count": line_count,
                    "status": "split_candidate",
                }
            )
    return rows


def _write_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _is_wrapper(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    body = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if any(" import *" in line for line in body[:8]):
        return True
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    meaningful_nodes = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        )
    ]
    if not meaningful_nodes:
        return False
    return all(
        isinstance(node, ast.ImportFrom)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        )
        for node in meaningful_nodes
    )


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
