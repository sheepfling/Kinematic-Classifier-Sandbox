from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backend_adapter_proof_core import (
    _adapter_map,
    _environment_candidate,
    _shared_boundary_candidate,
    _switching_candidate,
)


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
