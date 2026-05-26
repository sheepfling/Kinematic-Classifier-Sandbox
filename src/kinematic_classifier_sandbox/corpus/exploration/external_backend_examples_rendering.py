from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from kinematic_classifier_sandbox.utils.io import _write_json, _write_text

from .external_backend_examples import (
    ExternalBackendExampleRow,
    analyze_external_backend_examples,
)

__all__ = ["ExternalBackendExamplesArtifacts", "write_external_backend_examples_artifacts"]


class ExternalBackendExamplesArtifacts:
    def __init__(self, run_dir: Path, example_index_path: Path, report_path: Path) -> None:
        self.run_dir = run_dir
        self.example_index_path = example_index_path
        self.report_path = report_path


def _to_serializable(example: ExternalBackendExampleRow) -> dict[str, Any]:
    artifact_paths = asdict(example.artifact_paths)
    return {
        "example_id": example.example_id,
        "backend_id": example.backend_id,
        "adapter_family": example.adapter_family,
        "scenario_family": example.scenario_family,
        "summary": example.summary,
        "input_contract_fields": list(example.input_contract_fields),
        "runtime_contract_fields": list(example.runtime_contract_fields),
        "output_contract_fields": list(example.output_contract_fields),
        "validation_contract_fields": list(example.validation_contract_fields),
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
    }


def write_external_backend_examples_artifacts(base_dir: str | Path) -> ExternalBackendExamplesArtifacts:
    run_dir = Path(base_dir) / "external_backend_examples"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_external_backend_examples()

    example_index_path = run_dir / "example_index.json"
    report_path = run_dir / "external_backend_examples_report.md"

    report_path.write_text(result.report_markdown, encoding="utf-8")
    index_payload = {
        "examples": [_to_serializable(row) for row in result.example_rows],
    }
    _write_json(example_index_path, index_payload)
    _write_text(
        run_dir / "README.md",
        "\n".join(
            [
                "# External Backend Examples",
                "",
                "This directory contains 1D external-adapter interface examples for future simulator integrations.",
            ]
        ),
    )

    return ExternalBackendExamplesArtifacts(
        run_dir=run_dir,
        example_index_path=example_index_path,
        report_path=report_path,
    )
