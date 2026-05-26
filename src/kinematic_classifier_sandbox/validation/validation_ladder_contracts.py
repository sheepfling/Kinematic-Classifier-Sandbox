from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ValidationLadderResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_schema: dict[str, object]
    score_rows: tuple[dict[str, object], ...]
    decision_rows: tuple[dict[str, object], ...]
    report_markdown: str


class ValidationLadderArtifacts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    run_dir: Path
    schema_path: Path
    scores_path: Path
    decisions_path: Path
    report_path: Path


__all__ = ["ValidationLadderArtifacts", "ValidationLadderResult"]
