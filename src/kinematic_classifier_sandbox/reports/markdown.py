from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from mdutils.mdutils import MdUtils


def _format_cell(value: object) -> str:
    return str(value)


@dataclass(frozen=True)
class MermaidEdge:
    source: str
    destination: str


@dataclass(frozen=True)
class MermaidNode:
    node_id: str
    label: str


@dataclass(frozen=True)
class MermaidFlow:
    nodes: tuple[MermaidNode, ...]
    edges: tuple[MermaidEdge, ...]
    orientation: str = "TD"


def build_mermaid_flow(flow: MermaidFlow) -> str:
    lines = [f"graph {flow.orientation}"]
    for node in flow.nodes:
        escaped = node.label.replace('"', "\\\"")
        lines.append(f'    {node.node_id}["{escaped}"]')
    for edge in flow.edges:
        lines.append(f"    {edge.source} --> {edge.destination}")
    return "\n".join(lines)


class MarkdownDocument:
    def __init__(self, title: str | None = None) -> None:
        self._title = "" if title is None else title
        self._md = MdUtils(file_name="", title=title or "")

    def heading(self, title: str, level: int = 1) -> None:
        self._md.new_header(level, title, add_table_of_contents="n")

    def paragraph(self, text: str) -> None:
        self._md.new_paragraph(text)

    def ordered_list(self, items: Iterable[str]) -> None:
        lines = [f"{index}. {item}" for index, item in enumerate(items, start=1)]
        self._md.new_paragraph("\n".join(lines))

    def bullet_list(self, items: Iterable[str]) -> None:
        self._md.new_list(list(items), marked_with="-")

    def table(self, headers: list[str], rows: list[tuple[Any, ...]]) -> None:
        text: list[str] = []
        for value in headers:
            text.append(_format_cell(value))
        for row in rows:
            for value in row:
                text.append(_format_cell(value))
        row_count = 1 + len(rows)
        self._md.new_table(columns=len(headers), rows=row_count, text=text)

    def markdown_link(self, target: str, label: str | None = None) -> str:
        visible = target if label is None else label
        return f"[{visible}]({target})"

    def inline_code(self, value: str | object) -> str:
        return f"`{value}`"

    def fence(self, content: str, language: str = "") -> None:
        label = language.strip()
        if label:
            self._md.new_paragraph(f"```{label}\n{content}\n```")
        else:
            self._md.new_paragraph(f"```\n{content}\n```")

    def mermaid(self, diagram: MermaidFlow | str) -> None:
        if isinstance(diagram, MermaidFlow):
            content = build_mermaid_flow(diagram)
        else:
            content = diagram
        self.fence(content, language="mermaid")

    def text(self) -> str:
        return self._md.get_md_text().rstrip()
