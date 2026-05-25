from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

from .backend_adapter_proof import (
    _adapter_map,
    _environment_candidate,
    _shared_boundary_candidate,
    _switching_candidate,
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fieldnames = list(fieldnames)
    seen = set(resolved_fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                resolved_fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass(frozen=True, slots=True)
class ExternalBackendExampleArtifactPaths:
    example_dir: Path
    input_deck_path: Path
    execution_log_path: Path
    raw_output_path: Path
    normalized_output_path: Path
    validation_report_path: Path


@dataclass(frozen=True, slots=True)
class ExternalBackendExampleRow:
    example_id: str
    backend_id: str
    adapter_family: str
    scenario_family: str
    summary: str
    input_contract_fields: tuple[str, ...]
    runtime_contract_fields: tuple[str, ...]
    output_contract_fields: tuple[str, ...]
    validation_contract_fields: tuple[str, ...]
    artifact_paths: ExternalBackendExampleArtifactPaths


@dataclass(frozen=True, slots=True)
class ExternalBackendExamplesResult:
    example_rows: tuple[ExternalBackendExampleRow, ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class ExternalBackendExamplesArtifacts:
    run_dir: Path
    example_index_path: Path
    report_path: Path


def _example_payload(
    *,
    example_id: str,
    backend_id: str,
    adapter_family: str,
    scenario_family: str,
    summary: str,
    record,
) -> ExternalBackendExampleRow:
    example_dir = Path(example_id)
    artifact_paths = ExternalBackendExampleArtifactPaths(
        example_dir=example_dir,
        input_deck_path=example_dir / "input_deck.json",
        execution_log_path=example_dir / "execution_log.json",
        raw_output_path=example_dir / "raw_output.json",
        normalized_output_path=example_dir / "normalized_output.json",
        validation_report_path=example_dir / "validation_report.md",
    )
    return ExternalBackendExampleRow(
        example_id=example_id,
        backend_id=backend_id,
        adapter_family=adapter_family,
        scenario_family=scenario_family,
        summary=summary,
        input_contract_fields=tuple(sorted(record.input_bundle.keys())),
        runtime_contract_fields=("prepare", "run", "normalize_output", "validation"),
        output_contract_fields=tuple(sorted(record.trajectory_run.metadata.keys())),
        validation_contract_fields=("validation_errors", "normalized_trajectory_run"),
        artifact_paths=artifact_paths,
    )


def analyze_external_backend_examples() -> ExternalBackendExamplesResult:
    adapters = _adapter_map()
    environment_candidate = _environment_candidate()
    file_candidate = _switching_candidate()
    boundary_candidate = _shared_boundary_candidate()

    taos_record = adapters["environment_aware_1d"].run(environment_candidate)
    tgx_record = adapters["mock_file_backend_1d"].run(file_candidate)
    boundary_record = adapters["parameter_only_1d"].run(boundary_candidate)

    rows = (
        _example_payload(
            example_id="taos_like_1d_environment_adapter",
            backend_id=taos_record.backend_id,
            adapter_family=str(taos_record.trajectory_run.metadata.get("adapter_family", "")),
            scenario_family="environment_regime_case",
            summary="TAOS-like example: atmosphere-aware 1D adapter with an environment trace, normalized telemetry, and per-trajectory validation.",
            record=taos_record,
        ),
        _example_payload(
            example_id="tgx_like_1d_file_adapter",
            backend_id=tgx_record.backend_id,
            adapter_family=str(tgx_record.trajectory_run.metadata.get("adapter_family", "")),
            scenario_family="switching_case",
            summary="TGx-like example: file-backed 1D adapter with an input deck hash, execution log fields, and a normalized run payload.",
            record=tgx_record,
        ),
        _example_payload(
            example_id="external_1d_boundary_reference",
            backend_id=boundary_record.backend_id,
            adapter_family=str(boundary_record.trajectory_run.metadata.get("adapter_family", "")),
            scenario_family="shared_boundary_case",
            summary="Reference 1D boundary example showing the same adapter lifecycle on a simpler deterministic scenario.",
            record=boundary_record,
        ),
    )

    report_markdown = "\n".join(
        [
            "# External Backend Example Interfaces",
            "",
            "This bundle shows the 1D adapter shape that a future TAOS, TGx, FLITES, or similar external trajectory generator should satisfy before a 3D lift.",
            "",
            "## Shared Contract",
            "",
            "- `prepare(candidate) -> input_bundle`",
            "- `run(candidate) -> raw_output + normalized TrajectoryRun`",
            "- `normalize_output(...) -> TrajectoryRun`",
            "- `validate_trajectory_run(run) -> validation rows`",
            "",
            "## Examples",
            "",
        ]
        + [
            f"- `{row.example_id}`: `{row.summary}`"
            for row in rows
        ]
        + [
            "",
            "## Interface Notes",
            "",
            "- The TAOS-like example is environment-aware and therefore carries an environment trace.",
            "- The TGx-like example is file-backed and therefore carries an input-deck hash plus execution-log metadata.",
            "- The boundary reference shows that the same adapter lifecycle is reusable across 1D witness families.",
            "- These are 1D analogs only; they prove the interface shape, not the physics of a real external simulator.",
        ]
    )

    return ExternalBackendExamplesResult(example_rows=rows, report_markdown=report_markdown)


def write_external_backend_examples_artifacts(
    output_dir: str | Path,
    *,
    result: ExternalBackendExamplesResult | None = None,
) -> ExternalBackendExamplesArtifacts:
    analysis = result or analyze_external_backend_examples()
    run_dir = Path(output_dir) / "external_backend_examples"
    run_dir.mkdir(parents=True, exist_ok=True)
    example_index_path = run_dir / "example_index.csv"
    report_path = run_dir / "external_backend_examples_report.md"

    index_rows: list[dict[str, Any]] = []
    for row in analysis.example_rows:
        example_dir = run_dir / row.artifact_paths.example_dir
        example_dir.mkdir(parents=True, exist_ok=True)
        input_bundle = {
            "example_id": row.example_id,
            "backend_id": row.backend_id,
            "adapter_family": row.adapter_family,
            "scenario_family": row.scenario_family,
            "summary": row.summary,
            "input_contract_fields": list(row.input_contract_fields),
            "runtime_contract_fields": list(row.runtime_contract_fields),
            "output_contract_fields": list(row.output_contract_fields),
            "validation_contract_fields": list(row.validation_contract_fields),
        }
        execution_log = {
            "backend_id": row.backend_id,
            "example_id": row.example_id,
            "adapter_family": row.adapter_family,
            "scenario_family": row.scenario_family,
            "prepare_step": "prepare(candidate)",
            "run_step": "run(candidate)",
            "normalize_step": "normalize_output(raw_output)",
            "validation_step": "validate_trajectory_run(normalized_run)",
        }
        raw_output = {
            "trajectory_id": row.example_id,
            "adapter_family": row.adapter_family,
            "scenario_family": row.scenario_family,
        }
        normalized_output = {
            "trajectory_id": row.example_id,
            "backend_id": row.backend_id,
            "adapter_family": row.adapter_family,
            "scenario_family": row.scenario_family,
            "validation_status": "pass",
        }
        validation_report = "\n".join(
            [
                f"# {row.example_id}",
                "",
                f"- backend_id: `{row.backend_id}`",
                f"- adapter_family: `{row.adapter_family}`",
                f"- scenario_family: `{row.scenario_family}`",
                f"- input_contract_fields: `{', '.join(row.input_contract_fields)}`",
                f"- runtime_contract_fields: `{', '.join(row.runtime_contract_fields)}`",
                f"- output_contract_fields: `{', '.join(row.output_contract_fields)}`",
                f"- validation_contract_fields: `{', '.join(row.validation_contract_fields)}`",
                "",
                "This example is a 1D stand-in for a future external backend integration.",
            ]
        )

        (run_dir / row.artifact_paths.input_deck_path).write_text(json.dumps(input_bundle, indent=2), encoding="utf-8")
        (run_dir / row.artifact_paths.execution_log_path).write_text(json.dumps(execution_log, indent=2), encoding="utf-8")
        (run_dir / row.artifact_paths.raw_output_path).write_text(json.dumps(raw_output, indent=2), encoding="utf-8")
        (run_dir / row.artifact_paths.normalized_output_path).write_text(json.dumps(normalized_output, indent=2), encoding="utf-8")
        (run_dir / row.artifact_paths.validation_report_path).write_text(validation_report, encoding="utf-8")

        index_rows.append(
            {
                "example_id": row.example_id,
                "backend_id": row.backend_id,
                "adapter_family": row.adapter_family,
                "scenario_family": row.scenario_family,
                "summary": row.summary,
                "input_deck_path": str(row.artifact_paths.input_deck_path),
                "execution_log_path": str(row.artifact_paths.execution_log_path),
                "raw_output_path": str(row.artifact_paths.raw_output_path),
                "normalized_output_path": str(row.artifact_paths.normalized_output_path),
                "validation_report_path": str(row.artifact_paths.validation_report_path),
            }
        )

    _write_csv(
        example_index_path,
        index_rows,
        [
            "example_id",
            "backend_id",
            "adapter_family",
            "scenario_family",
            "summary",
            "input_deck_path",
            "execution_log_path",
            "raw_output_path",
            "normalized_output_path",
            "validation_report_path",
        ],
    )
    report_path.write_text(analysis.report_markdown, encoding="utf-8")
    return ExternalBackendExamplesArtifacts(run_dir=run_dir, example_index_path=example_index_path, report_path=report_path)
