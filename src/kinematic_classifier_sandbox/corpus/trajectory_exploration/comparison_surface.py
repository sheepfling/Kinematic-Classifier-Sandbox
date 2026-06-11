from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ...utils.io import _write_text, write_csv


def write_comparison_summary_csv(
    run_dir: str | Path,
    rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    *,
    filename: str = "summary.csv",
    fieldnames: Sequence[str] | None = None,
) -> Path:
    output_path = Path(run_dir) / filename
    materialized_rows = list(rows)
    if materialized_rows:
        write_csv(output_path, materialized_rows, list(fieldnames or materialized_rows[0].keys()))
    else:
        output_path.write_text(",".join(fieldnames or ("study_id",)) + "\n", encoding="utf-8")
    return output_path


def write_comparison_markdown(run_dir: str | Path, markdown: str, *, filename: str = "report.md") -> Path:
    output_path = Path(run_dir) / filename
    _write_text(output_path, markdown)
    return output_path


def write_decision_card(run_dir: str | Path, markdown: str, *, filename: str = "decision_card.md") -> Path:
    output_path = Path(run_dir) / filename
    _write_text(output_path, markdown)
    return output_path


__all__ = [
    "write_comparison_markdown",
    "write_comparison_summary_csv",
    "write_decision_card",
]
