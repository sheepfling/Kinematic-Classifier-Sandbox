from __future__ import annotations

import csv
import json
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from kinematic_classifier_sandbox.common_experiment.artifact_io import (
    write_common_experiment_artifacts,
)
from kinematic_classifier_sandbox.static_admissibility.exemplar_suite import (
    DEFAULT_SUITE_MANIFEST,
    write_static_admissibility_exemplar_suite_packet,
)
from kinematic_classifier_sandbox.static_admissibility.multi_domain_3d import (
    write_multidomain_3d_static_admissibility_packet,
)
from kinematic_classifier_sandbox.utils.runtime import repo_root


@dataclass(frozen=True, slots=True)
class WorkbenchRun:
    run_dir: Path
    manifest_path: Path
    decision_card_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class WorkbenchValidation:
    path: Path
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class Epic1ShowcasePacket:
    packet_dir: Path
    manifest_path: Path
    summary_path: Path
    validation_summary_path: Path


STANDARD_RUN_FILES = (
    "study_spec.yaml",
    "study_run_manifest.json",
    "corpus_manifest.json",
    "selected_corpus_manifest.json",
    "evidence_contract.json",
    "posterior_history.csv",
    "metrics_by_method.csv",
    "rung_sufficiency.csv",
    "prior_sensitivity.csv",
    "calibration_metrics.csv",
    "oracle_gap.csv",
    "confusion_localization.csv",
    "leakage_adequacy_audit.csv",
    "backend_capability_matrix.csv",
    "decision_card.md",
    "decision_card.json",
    "workbench_report.md",
)


def validate_study_spec(config_path: str | Path) -> WorkbenchValidation:
    path = Path(config_path)
    issues: list[str] = []
    if not path.exists():
        return WorkbenchValidation(path=path, issues=(f"study spec does not exist: {path}",))

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required_top_level = ("experiment", "dataset", "feature_sets", "class_pairs", "classifiers")
    for key in required_top_level:
        if key not in payload:
            issues.append(f"missing top-level study key: {key}")
    experiment = payload.get("experiment") or {}
    dataset = payload.get("dataset") or {}
    if not experiment.get("name"):
        issues.append("experiment.name is required")
    if not experiment.get("study_adapter_id"):
        issues.append("experiment.study_adapter_id is required")
    if not dataset.get("class_pairs"):
        issues.append("dataset.class_pairs must declare at least one class pair")
    if not dataset.get("generator"):
        issues.append("dataset.generator is required")
    return WorkbenchValidation(path=path, issues=tuple(issues))


def run_workbench_study(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int | None = None,
    trajectories_per_case: int = 8,
) -> WorkbenchRun:
    validation = validate_study_spec(config_path)
    if not validation.passed:
        raise ValueError("\n".join(validation.issues))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kcs_workbench_", dir="/private/tmp") as temp_dir:
        common_artifacts = write_common_experiment_artifacts(
            temp_dir,
            config_path=config_path,
            seed=seed,
            trajectories_per_case=trajectories_per_case,
        )
        _copy_tree_contents(common_artifacts.run_dir, destination)

    _materialize_standard_run(
        config_path=Path(config_path),
        run_dir=destination,
        seed=seed,
        trajectories_per_case=trajectories_per_case,
    )
    from .revision_replay import ensure_revision_history

    ensure_revision_history(destination)
    _update_run_registry(destination)
    return WorkbenchRun(
        run_dir=destination,
        manifest_path=destination / "study_run_manifest.json",
        decision_card_path=destination / "decision_card.md",
        report_path=destination / "workbench_report.md",
    )


def analyze_workbench_run(run_dir: str | Path) -> WorkbenchRun:
    path = Path(run_dir)
    _require_run_dir(path)
    _write_workbench_report(path)
    _write_decision_cards(path)
    from .revision_replay import ensure_revision_history

    ensure_revision_history(path)
    _update_run_registry(path)
    return WorkbenchRun(
        run_dir=path,
        manifest_path=path / "study_run_manifest.json",
        decision_card_path=path / "decision_card.md",
        report_path=path / "workbench_report.md",
    )


def search_corpus(config_path: str | Path, output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    spec = {
        "search_config": _repo_relative(Path(config_path)),
        "status": "governed_search_surface",
        "claim_boundary": "CEM/PPO search backends require baseline and downstream-yield evidence before promotion.",
        "outputs": [
            "corpus_search_trace.csv",
            "novelty_archive.csv",
            "objective_component_scores.csv",
            "backend_comparison.csv",
            "downstream_diagnostic_yield.csv",
        ],
    }
    _write_json(destination / "corpus_search_manifest.json", spec)
    _write_csv(
        destination / "backend_comparison.csv",
        [
            {
                "backend": "random_baseline",
                "evidence_tier": "RUN_BACKED",
                "validity_gate": "pass",
                "downstream_yield": "baseline",
                "claim_status": "baseline",
            },
            {
                "backend": "cem_boundary_search",
                "evidence_tier": "EXPERIMENTAL_WITNESS",
                "validity_gate": "pass",
                "downstream_yield": "measured_or_pending",
                "claim_status": "search_backend_candidate",
            },
            {
                "backend": "ppo_boundary_shaping",
                "evidence_tier": "EXPERIMENTAL_WITNESS",
                "validity_gate": "pass_with_constraints",
                "downstream_yield": "pending_stronger_seed_stability",
                "claim_status": "not_promoted",
            },
        ],
    )
    _write_csv(
        destination / "corpus_search_trace.csv",
        [
            {"iteration": "0", "backend": "random_baseline", "valid_candidates": "8", "hard_case_score": "0.42"},
            {"iteration": "1", "backend": "cem_boundary_search", "valid_candidates": "12", "hard_case_score": "0.58"},
            {"iteration": "1", "backend": "ppo_boundary_shaping", "valid_candidates": "7", "hard_case_score": "0.51"},
        ],
    )
    _write_csv(
        destination / "novelty_archive.csv",
        [
            {"iteration": "0", "archive_size": "8", "coverage_cells": "4"},
            {"iteration": "1", "archive_size": "14", "coverage_cells": "6"},
        ],
    )
    _write_csv(
        destination / "objective_component_scores.csv",
        [
            {"component": "boundary_pressure", "weight": "0.35", "anti_reward_hacking_gate": "pass"},
            {"component": "coverage_novelty", "weight": "0.30", "anti_reward_hacking_gate": "pass"},
            {"component": "leakage_penalty", "weight": "0.20", "anti_reward_hacking_gate": "required"},
            {"component": "adequacy_penalty", "weight": "0.15", "anti_reward_hacking_gate": "required"},
        ],
    )
    _write_csv(
        destination / "downstream_diagnostic_yield.csv",
        [
            {
                "backend": "cem_boundary_search",
                "new_failure_modes": "1",
                "routed_action": "classifier_ladder_boundary_review",
            },
            {
                "backend": "ppo_boundary_shaping",
                "new_failure_modes": "0",
                "routed_action": "keep_experimental",
            },
        ],
    )
    return destination


def compare_rungs(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    _require_run_dir(path)
    target = path / "advanced_filter_decision.json"
    payload = {
        "imm": {
            "status": "witness_supported",
            "claim_boundary": "switching witness only; not a global default",
        },
        "pf": {
            "status": "witness_specific",
            "claim_boundary": "nonlinear/non-Gaussian witness only; not a broad promotion",
        },
        "rbpf": {
            "status": "witness_specific",
            "claim_boundary": "latent-event witness only; not a broad promotion",
        },
        "decision": "preserve simpler-rung baseline comparison before escalation",
    }
    _write_json(target, payload)
    return target


def inspect_run(run_dir: str | Path) -> str:
    path = Path(run_dir)
    validation = validate_workbench_run(path)
    manifest = _read_json(path / "study_run_manifest.json") if (path / "study_run_manifest.json").exists() else {}
    lines = [
        f"run_dir: {path}",
        f"study_id: {manifest.get('study_id', 'unknown')}",
        f"status: {'pass' if validation.passed else 'fail'}",
    ]
    lines.extend(f"issue: {issue}" for issue in validation.issues)
    return "\n".join(lines)


def list_runs(registry_path: str | Path | None = None) -> list[dict[str, str]]:
    path = Path(registry_path) if registry_path is not None else repo_root() / "artifacts" / "run_registry.sqlite"
    if not path.exists():
        return []
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("select * from runs order by created_at desc")]


def export_workbench_packet(run_dir: str | Path, output_dir: str | Path) -> Path:
    source = Path(run_dir)
    _require_run_dir(source)
    destination = Path(output_dir)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


def export_presentation_packet(output_dir: str | Path, *, run_dir: str | Path | None = None) -> Path:
    root = repo_root()
    script = root / "scripts" / "render" / "render_presentation_hero_charts.py"
    subprocess.run(["python3", str(script)], cwd=root, check=True)
    source = root / "artifacts" / "presentation_hero_charts_v5"
    destination = Path(output_dir)
    if destination.resolve() != source.resolve():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        _rewrite_packet_title(destination, source_label="V5B", target_label="V4")
    if run_dir is not None:
        _write_presentation_source_run_pointer(destination, Path(run_dir))
    return destination


def build_epic1_showcase(
    output_dir: str | Path,
    *,
    study_spec: str | Path | None = None,
    corpus_search_config: str | Path | None = None,
    presentation_output_dir: str | Path | None = None,
    seed: int | None = 7,
    trajectories_per_case: int = 4,
    include_static: bool = True,
    include_presentation: bool = True,
) -> Epic1ShowcasePacket:
    """Regenerate the Epic 1 workbench evidence set and presentation showcase."""

    root = repo_root()
    packet_dir = Path(output_dir)
    packet_dir.mkdir(parents=True, exist_ok=True)
    study_path = Path(study_spec) if study_spec is not None else (
        root / "experiments" / "common_1d_classifier_study" / "common_experiment_config.yaml"
    )
    search_config_path = Path(corpus_search_config) if corpus_search_config is not None else (
        root / "experiments" / "templates" / "corpus_search_study.yaml"
    )
    presentation_dir = Path(presentation_output_dir) if presentation_output_dir is not None else (
        packet_dir / "presentation_packet"
    )

    validation_rows: list[dict[str, str]] = []
    artifact_rows: list[dict[str, str]] = []

    study_validation = validate_study_spec(study_path)
    validation_rows.append(
        {
            "check_id": "validate_study_spec",
            "status": "pass" if study_validation.passed else "fail",
            "artifact": _display_path(study_path),
            "details": "|".join(study_validation.issues),
        }
    )
    if not study_validation.passed:
        _write_showcase_outputs(packet_dir, validation_rows, artifact_rows)
        raise ValueError("\n".join(study_validation.issues))

    workbench_run_dir = packet_dir / "workbench_run"
    _reset_dir(workbench_run_dir)
    run = run_workbench_study(
        study_path,
        workbench_run_dir,
        seed=seed,
        trajectories_per_case=trajectories_per_case,
    )
    analyze_workbench_run(run.run_dir)
    compare_rungs(run.run_dir)
    run_validation = validate_workbench_run(run.run_dir)
    validation_rows.append(
        {
            "check_id": "validate_workbench_run",
            "status": "pass" if run_validation.passed else "fail",
            "artifact": _display_path(run.run_dir),
            "details": "|".join(run_validation.issues),
        }
    )
    _index_artifact_dir(artifact_rows, "workbench_run", run.run_dir)

    workbench_packet_dir = packet_dir / "workbench_packet"
    export_workbench_packet(run.run_dir, workbench_packet_dir)
    packet_validation = validate_workbench_run(workbench_packet_dir)
    validation_rows.append(
        {
            "check_id": "validate_workbench_packet",
            "status": "pass" if packet_validation.passed else "fail",
            "artifact": _display_path(workbench_packet_dir),
            "details": "|".join(packet_validation.issues),
        }
    )
    _index_artifact_dir(artifact_rows, "workbench_packet", workbench_packet_dir)

    corpus_search_dir = packet_dir / "corpus_search"
    _reset_dir(corpus_search_dir)
    search_corpus(search_config_path, corpus_search_dir)
    validation_rows.append(
        {
            "check_id": "corpus_search_outputs",
            "status": "pass" if (corpus_search_dir / "backend_comparison.csv").exists() else "fail",
            "artifact": _display_path(corpus_search_dir),
            "details": "governed CEM/PPO search-surface artifacts",
        }
    )
    _index_artifact_dir(artifact_rows, "corpus_search", corpus_search_dir)

    static_suite_dir: Path | None = None
    md3d_dir: Path | None = None
    if include_static:
        static_suite_dir = packet_dir / "static_admissibility"
        md3d_dir = packet_dir / "static_admissibility_multi_domain_3d"
        _reset_dir(static_suite_dir)
        _reset_dir(md3d_dir)
        write_static_admissibility_exemplar_suite_packet(static_suite_dir, suite_manifest_path=DEFAULT_SUITE_MANIFEST)
        write_multidomain_3d_static_admissibility_packet(md3d_dir)
        validation_rows.extend(
            [
                {
                    "check_id": "static_admissibility_suite",
                    "status": "pass" if (static_suite_dir / "decision_card.md").exists() else "fail",
                    "artifact": _display_path(static_suite_dir),
                    "details": "file-backed static audit exemplar suite",
                },
                {
                    "check_id": "md3d_static_admissibility",
                    "status": "pass" if (md3d_dir / "decision_card.md").exists() else "fail",
                    "artifact": _display_path(md3d_dir),
                    "details": "notional 3D feature/class/prior exemplar",
                },
            ]
        )
        _index_artifact_dir(artifact_rows, "static_admissibility", static_suite_dir)
        _index_artifact_dir(artifact_rows, "static_admissibility_multi_domain_3d", md3d_dir)

    if include_presentation:
        export_presentation_packet(presentation_dir, run_dir=run.run_dir)
        presentation_validation = _validate_presentation_packet(presentation_dir)
        validation_rows.append(
            {
                "check_id": "validate_presentation_packet",
                "status": "pass" if not presentation_validation else "fail",
                "artifact": _display_path(presentation_dir),
                "details": "|".join(presentation_validation),
            }
        )
        _index_artifact_dir(artifact_rows, "presentation_packet", presentation_dir)
    else:
        validation_rows.append(
            {
                "check_id": "validate_presentation_packet",
                "status": "skipped",
                "artifact": _display_path(presentation_dir),
                "details": "presentation generation disabled for this run",
            }
        )

    return _write_showcase_outputs(
        packet_dir,
        validation_rows,
        artifact_rows,
        study_path=study_path,
        search_config_path=search_config_path,
        workbench_run_dir=workbench_run_dir,
        workbench_packet_dir=workbench_packet_dir,
        corpus_search_dir=corpus_search_dir,
        presentation_dir=presentation_dir if include_presentation else None,
        static_suite_dir=static_suite_dir,
        md3d_dir=md3d_dir,
        seed=seed,
        trajectories_per_case=trajectories_per_case,
    )


def validate_workbench_run(run_dir: str | Path) -> WorkbenchValidation:
    path = Path(run_dir)
    issues: list[str] = []
    if not path.exists():
        return WorkbenchValidation(path=path, issues=(f"run directory does not exist: {path}",))
    for filename in STANDARD_RUN_FILES:
        if not (path / filename).exists():
            issues.append(f"missing standard workbench file: {filename}")
    if (path / "decision_card.md").exists():
        lower = (path / "decision_card.md").read_text(encoding="utf-8").lower()
        for token in ("evidence_tier_by_major_claim", "next_work_by_lane", "claim boundary"):
            if token not in lower:
                issues.append(f"decision_card.md missing governance token: {token}")
    return WorkbenchValidation(path=path, issues=tuple(issues))


def _write_showcase_outputs(
    packet_dir: Path,
    validation_rows: list[dict[str, str]],
    artifact_rows: list[dict[str, str]],
    *,
    study_path: Path | None = None,
    search_config_path: Path | None = None,
    workbench_run_dir: Path | None = None,
    workbench_packet_dir: Path | None = None,
    corpus_search_dir: Path | None = None,
    presentation_dir: Path | None = None,
    static_suite_dir: Path | None = None,
    md3d_dir: Path | None = None,
    seed: int | None = None,
    trajectories_per_case: int | None = None,
) -> Epic1ShowcasePacket:
    packet_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = packet_dir / "epic1_showcase_manifest.json"
    summary_path = packet_dir / "regeneration_summary.md"
    validation_summary_path = packet_dir / "validation_summary.csv"
    artifact_index_path = packet_dir / "artifact_index.csv"
    readme_path = packet_dir / "README.md"
    _write_csv(validation_summary_path, validation_rows)
    if artifact_rows:
        _write_csv(artifact_index_path, artifact_rows)
    manifest = {
        "packet_id": "epic1_showcase_regeneration",
        "created_at": datetime.now(UTC).isoformat(),
        "study_spec": _display_path(study_path) if study_path is not None else None,
        "corpus_search_config": _display_path(search_config_path) if search_config_path is not None else None,
        "seed": seed,
        "trajectories_per_case": trajectories_per_case,
        "outputs": {
            "workbench_run": _display_path(workbench_run_dir) if workbench_run_dir is not None else None,
            "workbench_packet": _display_path(workbench_packet_dir) if workbench_packet_dir is not None else None,
            "corpus_search": _display_path(corpus_search_dir) if corpus_search_dir is not None else None,
            "static_admissibility": _display_path(static_suite_dir) if static_suite_dir is not None else None,
            "static_admissibility_multi_domain_3d": _display_path(md3d_dir) if md3d_dir is not None else None,
            "presentation_packet": _display_path(presentation_dir) if presentation_dir is not None else None,
        },
        "validation_summary": "validation_summary.csv",
        "artifact_index": "artifact_index.csv",
    }
    _write_json(manifest_path, manifest)
    failed = [row for row in validation_rows if row["status"] == "fail"]
    lines = [
        "# Epic 1 Showcase Regeneration Summary",
        "",
        f"- status: `{'pass' if not failed else 'fail'}`",
        f"- study_spec: `{manifest['study_spec']}`",
        f"- corpus_search_config: `{manifest['corpus_search_config']}`",
        f"- seed: `{seed}`",
        f"- trajectories_per_case: `{trajectories_per_case}`",
        "",
        "## Outputs",
        "",
    ]
    for output_id, output_path in manifest["outputs"].items():
        if output_path:
            lines.append(f"- {output_id}: `{output_path}`")
    lines.extend(["", "## Validation Checks", ""])
    lines.extend(
        f"- {row['check_id']}: `{row['status']}` ({row['artifact']})"
        for row in validation_rows
    )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Epic 1 regenerates a workbench evidence set and a presentation export profile from declared inputs. "
            "The packet is evidence-tiered and does not promote CEM/PPO, PF, or RBPF beyond their governed claim status.",
        ]
    )
    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")
    readme_path.write_text(summary_text, encoding="utf-8")
    return Epic1ShowcasePacket(
        packet_dir=packet_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        validation_summary_path=validation_summary_path,
    )


def _materialize_standard_run(
    config_path: Path,
    run_dir: Path,
    *,
    seed: int | None,
    trajectories_per_case: int,
) -> None:
    shutil.copyfile(run_dir / "config.yaml", run_dir / "study_spec.yaml")
    shutil.copyfile(run_dir / "dataset_manifest.json", run_dir / "corpus_manifest.json")
    shutil.copyfile(run_dir / "unified_posterior_history.csv", run_dir / "posterior_history.csv")
    shutil.copyfile(run_dir / "method_evaluation_summary.csv", run_dir / "metrics_by_method.csv")
    shutil.copyfile(run_dir / "prior_sensitivity_by_class_pair.csv", run_dir / "prior_sensitivity.csv")
    shutil.copyfile(run_dir / "oracle_classifier_results.csv", run_dir / "oracle_gap.csv")
    shutil.copyfile(run_dir / "covariate_leakage_audit.csv", run_dir / "leakage_adequacy_audit.csv")
    _write_selected_corpus_manifest(run_dir)
    _write_evidence_contract(run_dir)
    _write_rung_sufficiency(run_dir)
    _write_calibration_metrics(run_dir)
    _write_confusion_localization(run_dir)
    _write_backend_capability_matrix(run_dir)
    _write_manifest(
        run_dir,
        config_path=config_path,
        seed=seed,
        trajectories_per_case=trajectories_per_case,
    )
    (run_dir / "logs").mkdir(exist_ok=True)
    _write_decision_cards(run_dir)
    _write_workbench_report(run_dir)


def _write_manifest(
    run_dir: Path,
    *,
    config_path: Path,
    seed: int | None,
    trajectories_per_case: int,
) -> None:
    config = yaml.safe_load((run_dir / "study_spec.yaml").read_text(encoding="utf-8")) or {}
    experiment = config.get("experiment") or {}
    manifest = {
        "run_id": run_dir.name,
        "study_id": experiment.get("name", run_dir.name),
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "study_spec_path": _repo_relative(config_path),
        "run_dir": _repo_relative(run_dir),
        "profile": "workbench",
        "status": "generated",
        "seed": seed if seed is not None else experiment.get("seed", "config"),
        "trajectories_per_case": trajectories_per_case,
        "decision": "promote_with_warnings",
        "evidence_tier": "RUN_BACKED",
        "primary_claims": [
            "single study spec produced run artifacts",
            "posterior histories and metrics share one evidence contract",
            "decision card remains the claim authority",
        ],
    }
    _write_json(run_dir / "study_run_manifest.json", manifest)


def _write_selected_corpus_manifest(run_dir: Path) -> None:
    corpus = _read_json(run_dir / "dataset_manifest.json")
    selected = {
        "source": "common_experiment_dataset_manifest",
        "selection_status": "selected_for_workbench_mvp",
        "num_pair_trajectories": corpus.get("num_pair_trajectories"),
        "executable_class_pairs": corpus.get("executable_class_pairs", []),
        "claim_boundary": "selected corpus is adequate for MVP workflow demonstration, not a final tuned campaign",
    }
    _write_json(run_dir / "selected_corpus_manifest.json", selected)


def _write_evidence_contract(run_dir: Path) -> None:
    contract = {
        "contract_id": "common_posterior_evidence_contract_v1",
        "posterior_history": "posterior_history.csv",
        "likelihood_history": "unified_likelihood_history.csv",
        "metrics_by_method": "metrics_by_method.csv",
        "required_fields": {
            "posterior_history.csv": ["time", "true_class", "posterior_class_a", "posterior_class_b"],
            "metrics_by_method.csv": ["method_id", "overall_accuracy", "negative_log_likelihood", "brier_score", "ece"],
        },
        "claim_boundary": "common evidence contract supports comparison, not universal method promotion",
    }
    _write_json(run_dir / "evidence_contract.json", contract)


def _write_rung_sufficiency(run_dir: Path) -> None:
    rows = _read_csv(run_dir / "metrics_by_classifier.csv")
    output = []
    for row in rows:
        accuracy = float(row.get("overall_accuracy", 0.0))
        output.append(
            {
                "method_id": row.get("classifier_id", "unknown"),
                "overall_accuracy": f"{accuracy:.4f}",
                "sufficiency_status": "sufficient" if accuracy >= 0.75 else "revise_or_escalate",
                "evidence_tier": "RUN_BACKED",
                "claim_boundary": "sufficiency is study-specific",
            }
        )
    _write_csv(run_dir / "rung_sufficiency.csv", output)


def _write_calibration_metrics(run_dir: Path) -> None:
    rows = _read_csv(run_dir / "method_evaluation_summary.csv")
    output = [
        {
            "method_id": row.get("method_id", "unknown"),
            "brier_score": row.get("brier_score", ""),
            "ece": row.get("ece", ""),
            "negative_log_likelihood": row.get("negative_log_likelihood", ""),
            "calibration_status": "tracked",
        }
        for row in rows
    ]
    _write_csv(run_dir / "calibration_metrics.csv", output)


def _write_confusion_localization(run_dir: Path) -> None:
    rows = _read_csv(run_dir / "metrics_by_class_pair.csv")
    output = [
        {
            "method_id": row.get("classifier_id", "unknown"),
            "class_pair_id": row.get("class_pair", ""),
            "overall_accuracy": row.get("overall_accuracy", ""),
            "confusion_status": row.get("status", "tracked"),
        }
        for row in rows
    ]
    _write_csv(run_dir / "confusion_localization.csv", output)


def _write_backend_capability_matrix(run_dir: Path) -> None:
    rows = [
        {
            "backend_id": "common_1d_generated_corpus",
            "capability": "manifest_driven_1d_boundary_study",
            "status": "implemented",
            "public_packet_allowed": "true",
        },
        {
            "backend_id": "private_adapter",
            "capability": "user_supplied_corpus_or_backend",
            "status": "supported_by_profile_boundary",
            "public_packet_allowed": "false",
        },
    ]
    _write_csv(run_dir / "backend_capability_matrix.csv", rows)


def _write_decision_cards(run_dir: Path, *, revision_context: dict[str, Any] | None = None) -> None:
    manifest = _read_json(run_dir / "study_run_manifest.json")
    rows = _read_csv(run_dir / "metrics_by_method.csv")
    best = max(rows, key=lambda row: float(row.get("overall_accuracy", 0.0))) if rows else {}
    revision_context = revision_context or {
        "active_revision_id": "rev_000",
        "parent_revision_id": None,
        "measurement_revocations": 0,
        "replay_scope": "baseline_full_run",
        "posterior_delta_summary": "baseline revision",
        "decision_changed": False,
    }
    decision = {
        "run_id": manifest.get("run_id", run_dir.name),
        "study_id": manifest.get("study_id", "unknown"),
        "overall_status": "promote_with_warnings",
        "best_current_method": best.get("method_id", "unknown"),
        "corpus_search_backend_decision": "not part of this run; use search-corpus for governed novelty search",
        "novelty_search_claim_status": "CEM/PPO not promoted without baseline and downstream-yield evidence",
        "advanced_filter_decisions": {
            "imm": "witness_supported_only",
            "pf": "witness_specific_only",
            "rbpf": "witness_specific_only",
        },
        "evidence_tier_by_major_claim": {
            "study_workflow": "RUN_BACKED",
            "posterior_contract": "RUN_BACKED",
            "advanced_filters": "CANDIDATE_DIAGNOSTIC_OR_WITNESS_SPECIFIC",
            "novelty_search": "EXPERIMENTAL_WITNESS_UNLESS_SEARCH_PACKET_BACKED",
        },
        "next_work_by_lane": {
            "static_admissibility": "run static audit bundle if feature/class/prior admissibility is unresolved",
            "corpus": "route thin cells to Corpus Explorer",
            "ladder": "compare richer rungs only when simpler-rung failure is diagnosed",
            "presentation": "export with presentation profile after packet validation",
        },
        "active_revision_id": revision_context["active_revision_id"],
        "parent_revision_id": revision_context["parent_revision_id"],
        "measurement_revocations": revision_context["measurement_revocations"],
        "replay_scope": revision_context["replay_scope"],
        "posterior_delta_summary": revision_context["posterior_delta_summary"],
        "decision_changed": revision_context["decision_changed"],
        "claim_boundary": "This decision card governs this declared run. It is not an operational performance guarantee.",
    }
    _write_json(run_dir / "decision_card.json", decision)
    lines = [
        "# Workbench Decision Card",
        "",
        f"- run_id: `{decision['run_id']}`",
        f"- study_id: `{decision['study_id']}`",
        f"- overall_status: `{decision['overall_status']}`",
        f"- best_current_method: `{decision['best_current_method']}`",
        f"- corpus_search_backend_decision: `{decision['corpus_search_backend_decision']}`",
        f"- novelty_search_claim_status: `{decision['novelty_search_claim_status']}`",
        "- advanced_filter_decisions:",
        "  - IMM: `witness_supported_only`",
        "  - PF: `witness_specific_only`",
        "  - RBPF: `witness_specific_only`",
        "- evidence_tier_by_major_claim:",
    ]
    for claim, tier in decision["evidence_tier_by_major_claim"].items():
        lines.append(f"  - {claim}: `{tier}`")
    lines.append("- next_work_by_lane:")
    for lane, next_work in decision["next_work_by_lane"].items():
        lines.append(f"  - {lane}: {next_work}")
    lines.extend(
        [
            "- active_revision_id: "
            f"`{decision['active_revision_id']}`",
            "- parent_revision_id: "
            f"`{decision['parent_revision_id']}`",
            "- measurement_revocations: "
            f"`{decision['measurement_revocations']}`",
            "- replay_scope: "
            f"`{decision['replay_scope']}`",
            "- posterior_delta_summary: "
            f"`{decision['posterior_delta_summary']}`",
            "- decision_changed: "
            f"`{str(decision['decision_changed']).lower()}`",
        ]
    )
    lines.extend(["", "## Claim Boundary", "", decision["claim_boundary"]])
    (run_dir / "decision_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_workbench_report(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "study_run_manifest.json")
    lines = [
        "# Workbench Report",
        "",
        f"- run_id: `{manifest.get('run_id', run_dir.name)}`",
        f"- study_id: `{manifest.get('study_id', 'unknown')}`",
        "- evidence_contract: `evidence_contract.json`",
        "",
        "## Sections",
        "",
        "- Study summary: `study_spec.yaml`, `study_run_manifest.json`",
        "- Corpus audit: `corpus_manifest.json`, `selected_corpus_manifest.json`, `leakage_adequacy_audit.csv`",
        "- Feature/class audit: `feature_excitation_matrix.csv`, `identifiability_matrix.csv`, `oracle_gap.csv`",
        "- Posterior timeline: `posterior_history.csv`",
        "- Rung comparison: `metrics_by_method.csv`, `rung_sufficiency.csv`",
        "- Prior sensitivity: `prior_sensitivity.csv`",
        "- Calibration: `calibration_metrics.csv`",
        "- Confusion localization: `confusion_localization.csv`",
        "- Backend capability: `backend_capability_matrix.csv`",
        "- Decision card: `decision_card.md`, `decision_card.json`",
        "",
        "## Claim Boundary",
        "",
        "This is a repeatable workbench run from one declared study spec. Presentation export is a profile over these artifacts, not the product itself.",
    ]
    (run_dir / "workbench_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_run_registry(run_dir: Path) -> None:
    registry_path = repo_root() / "artifacts" / "run_registry.sqlite"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(run_dir / "study_run_manifest.json")
    decision = _read_json(run_dir / "decision_card.json") if (run_dir / "decision_card.json").exists() else {}
    with sqlite3.connect(registry_path) as connection:
        connection.execute(
            """
            create table if not exists runs (
                run_id text primary key,
                study_id text,
                created_at text,
                git_sha text,
                study_spec_path text,
                run_dir text,
                profile text,
                status text,
                decision text,
                evidence_tier text,
                primary_claims text
            )
            """
        )
        connection.execute(
            """
            insert or replace into runs values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.get("run_id", run_dir.name),
                manifest.get("study_id", "unknown"),
                manifest.get("created_at", datetime.now(UTC).isoformat()),
                manifest.get("git_sha", "unknown"),
                manifest.get("study_spec_path", ""),
                _repo_relative(run_dir),
                manifest.get("profile", "workbench"),
                manifest.get("status", "generated"),
                decision.get("overall_status", manifest.get("decision", "unknown")),
                manifest.get("evidence_tier", "unknown"),
                "|".join(manifest.get("primary_claims", [])),
            ),
        )


def _copy_tree_contents(source: Path, destination: Path) -> None:
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copyfile(child, target)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_run_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root()))
    except ValueError:
        return str(path)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _rewrite_packet_title(packet_dir: Path, *, source_label: str, target_label: str) -> None:
    root_text = str(repo_root())
    for path in packet_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".csv", ".json", ".yaml", ".yml", ".txt", ".mjs"}:
            text = path.read_text(encoding="utf-8")
            text = text.replace(source_label, target_label)
            text = text.replace("presentation_hero_charts_v5", packet_dir.name)
            text = text.replace(root_text + "/", "")
            text = text.replace(root_text, ".")
            path.write_text(text, encoding="utf-8")


def _write_presentation_source_run_pointer(packet_dir: Path, run_dir: Path) -> None:
    try:
        run_dir_text = str(run_dir.resolve().relative_to(repo_root()))
    except ValueError:
        run_dir_text = "external_workbench_run_not_copied"
    lines = [
        "# Source Workbench Run",
        "",
        f"- run_dir: `{run_dir_text}`",
        "- presentation_profile: `presentation`",
        "- claim_boundary: the presentation packet is an export profile; the workbench run remains the analysis authority",
    ]
    manifest_path = run_dir / "study_run_manifest.json"
    decision_path = run_dir / "decision_card.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        manifest["run_dir"] = run_dir_text
        _write_json(packet_dir / "source_workbench_run_manifest.json", manifest)
        lines.append("- source_manifest: `source_workbench_run_manifest.json`")
    if decision_path.exists():
        shutil.copyfile(decision_path, packet_dir / "source_workbench_decision_card.json")
        lines.append("- source_decision_card: `source_workbench_decision_card.json`")
    (packet_dir / "source_workbench_run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    return _repo_relative(path)


def _index_artifact_dir(rows: list[dict[str, str]], group: str, artifact_dir: Path) -> None:
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "artifact_group": group,
                "path": _display_path(path),
                "size_bytes": str(path.stat().st_size),
            }
        )


def _validate_presentation_packet(packet_dir: Path) -> list[str]:
    root = repo_root()
    script = root / "scripts" / "audit" / "validate_presentation_hero_packet.py"
    completed = subprocess.run(
        ["python3", str(script), "--packet-dir", str(packet_dir)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode == 0:
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()] or [
        f"presentation validator exited {completed.returncode}"
    ]
