from __future__ import annotations

import json
from pathlib import Path

from ..utils.io import write_csv
from .validation_ladder_contracts import ValidationLadderArtifacts, ValidationLadderResult
from .validation_ladder_rendering import render_validation_ladder_report
from .validation_ladder_runner import analyze_validation_ladder


def write_validation_ladder_artifacts(
    output_dir: str | Path,
    *,
    result: ValidationLadderResult | None = None,
) -> ValidationLadderArtifacts:
    ladder = result or analyze_validation_ladder()
    run_dir = Path(output_dir) / "validation_ladder"
    run_dir.mkdir(parents=True, exist_ok=True)

    schema_path = run_dir / "validation_ladder_schema.json"
    scores_path = run_dir / "validation_ladder_scores.csv"
    decisions_path = run_dir / "validation_ladder_decisions.csv"
    report_path = run_dir / "validation_ladder_report.md"

    if ladder.score_rows:
        write_csv(scores_path, list(ladder.score_rows), list(ladder.score_rows[0].keys()))
    else:
        write_csv(scores_path, [], ["study_id", "level_id", "level_name", "status", "score", "evidence_summary"])
    if ladder.decision_rows:
        write_csv(decisions_path, list(ladder.decision_rows), list(ladder.decision_rows[0].keys()))
    else:
        write_csv(decisions_path, [], ["study_id", "final_decision", "decision_rationale"])
    schema_path.write_text(json.dumps(ladder.contract_schema, indent=2), encoding="utf-8")
    report_path.write_text(render_validation_ladder_report(ladder), encoding="utf-8")

    return ValidationLadderArtifacts(
        run_dir=run_dir,
        schema_path=schema_path,
        scores_path=scores_path,
        decisions_path=decisions_path,
        report_path=report_path,
    )


__all__ = ["write_validation_ladder_artifacts"]
