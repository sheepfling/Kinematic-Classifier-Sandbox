from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .study_candidate_protocol_utils import (
    _load_protocol_markdown,
    _study_candidate_schema,
    _validation_ladder_schema,
    _validation_summary,
)


@dataclass(frozen=True, slots=True)
class StudyCandidateProtocolResult:
    protocol_markdown: str
    study_candidate_schema: dict[str, object]
    validation_ladder_schema: dict[str, object]
    validation_summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class StudyCandidateProtocolArtifacts:
    run_dir: Path
    protocol_path: Path
    study_candidate_schema_path: Path
    validation_ladder_schema_path: Path
    validation_summary_path: Path


def analyze_study_candidate_protocol() -> StudyCandidateProtocolResult:
    protocol_markdown = _load_protocol_markdown()
    study_candidate_schema = _study_candidate_schema()
    validation_ladder_schema = _validation_ladder_schema()
    validation_summary = _validation_summary(
        study_candidate_schema=study_candidate_schema,
        validation_ladder_schema=validation_ladder_schema,
        protocol_markdown=protocol_markdown,
    )
    return StudyCandidateProtocolResult(
        protocol_markdown=protocol_markdown,
        study_candidate_schema=study_candidate_schema,
        validation_ladder_schema=validation_ladder_schema,
        validation_summary=validation_summary,
    )


def write_study_candidate_protocol_artifacts(
    output_dir: str | Path,
    *,
    result: StudyCandidateProtocolResult | None = None,
) -> StudyCandidateProtocolArtifacts:
    protocol = result or analyze_study_candidate_protocol()
    run_dir = Path(output_dir) / "study_candidate_generation"
    run_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = Path(output_dir) / "validation_ladder"
    validation_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir = Path(output_dir) / "protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    protocol_path = protocol_dir / "feature_class_classifier_analysis_protocol.md"
    study_candidate_schema_path = run_dir / "study_candidate_schema.json"
    validation_ladder_schema_path = validation_dir / "validation_ladder_schema.json"
    validation_summary_path = run_dir / "m18_validation_summary.json"

    protocol_path.write_text(protocol.protocol_markdown, encoding="utf-8")
    study_candidate_schema_path.write_text(json.dumps(protocol.study_candidate_schema, indent=2), encoding="utf-8")
    validation_ladder_schema_path.write_text(json.dumps(protocol.validation_ladder_schema, indent=2), encoding="utf-8")
    validation_summary_path.write_text(json.dumps(protocol.validation_summary, indent=2), encoding="utf-8")

    return StudyCandidateProtocolArtifacts(
        run_dir=run_dir,
        protocol_path=protocol_path,
        study_candidate_schema_path=study_candidate_schema_path,
        validation_ladder_schema_path=validation_ladder_schema_path,
        validation_summary_path=validation_summary_path,
    )
