from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .catalog import METHOD_CATALOG, MethodEntry


def render_method_survey_markdown(entries: tuple[MethodEntry, ...] = METHOD_CATALOG) -> str:
    grouped: dict[str, list[MethodEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.family].append(entry)

    lines = [
        "# Kinematic Method Survey Summary",
        "",
        "This generated summary groups the initial sandbox method landscape by family.",
        "",
    ]
    for family in sorted(grouped):
        lines.append(f"## {family.replace('_', ' ').title()}")
        lines.append("")
        for entry in grouped[family]:
            strengths = "; ".join(entry.strengths)
            limits = "; ".join(entry.limits)
            use_cases = ", ".join(entry.typical_use_cases)
            inputs = ", ".join(entry.typical_inputs)
            lines.append(f"### {entry.name}")
            lines.append("")
            lines.append(f"- Style: `{entry.style}`")
            lines.append(f"- Typical inputs: {inputs}")
            lines.append(f"- Typical use cases: {use_cases}")
            lines.append(f"- Strengths: {strengths}")
            lines.append(f"- Limits: {limits}")
            lines.append("")
    lines.extend(
        [
            "## Current Recommended Sandbox Baseline",
            "",
            "The strongest initial model-based baseline is a Bayesian joint tracking and",
            "classification stack with a class-matched filter bank, IMM-style within-class",
            "mode switching, covariance-aware constraint likelihoods, optional aerodynamic",
            "parameter evidence, and explicit unknown-class handling.",
            "",
        ]
    )
    return "\n".join(lines)


def write_method_survey_artifact(output_dir: str | Path) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "method_survey_summary.md"
    output_path.write_text(render_method_survey_markdown(), encoding="utf-8")
    return output_path
