from __future__ import annotations

from collections.abc import Mapping, Sequence

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument


def _format_cell(value: object) -> str:
    return str(value)


def markdown_table_preview(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
    *,
    limit: int = 8,
    empty_message: str = "_No rows available._",
) -> str:
    if not rows:
        return empty_message
    visible_rows = rows[: max(limit, 1)]
    
    doc = MarkdownDocument()
    doc.table(
        list(columns),
        [tuple(row.get(column, "") for column in columns) for row in visible_rows]
    )
    return doc.text()
