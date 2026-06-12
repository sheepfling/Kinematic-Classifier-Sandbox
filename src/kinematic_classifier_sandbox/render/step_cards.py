from __future__ import annotations

from pathlib import Path

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..tracing.filter_trace import FilterStepTrace


def render_step_card_markdown(traces: tuple[FilterStepTrace, ...] | list[FilterStepTrace], *, title: str | None = None) -> str:
    if not traces:
        raise ValueError("step card requires at least one trace row")
    rows = sorted(traces, key=lambda trace: trace.class_or_model)
    first = rows[0]
    doc = MarkdownDocument(title or f"Step Card: {first.trajectory_id}, t={first.time_index}")
    doc.heading("Measurement", level=2)
    doc.bullet_list(
        [
            f"`z_t`: `{first.measurement}`",
            f"true class: `{first.true_class}`",
            f"true mode: `{first.true_mode}`",
            f"time: `{first.time}`",
        ]
    )
    doc.heading("Prior Before Measurement", level=2)
    doc.table(
        ["model", "prior", "predicted"],
        [
            (
                row.class_or_model,
                "" if row.prior_probability is None else f"{row.prior_probability:.6g}",
                "" if row.predicted_probability is None else f"{row.predicted_probability:.6g}",
            )
            for row in rows
        ],
    )
    doc.heading("Prediction And Likelihood", level=2)
    doc.table(
        ["model", "innovation", "NIS", "log likelihood"],
        [
            (
                row.class_or_model,
                "" if row.innovation is None else str(row.innovation),
                "" if row.normalized_innovation_squared is None else f"{row.normalized_innovation_squared:.6g}",
                "" if row.log_likelihood is None else f"{row.log_likelihood:.6g}",
            )
            for row in rows
        ],
    )
    doc.heading("Posterior After Update", level=2)
    doc.table(
        ["model", "posterior"],
        [
            (
                row.class_or_model,
                "" if row.posterior_probability is None else f"{row.posterior_probability:.6g}",
            )
            for row in rows
        ],
    )
    winner = max(rows, key=lambda row: -1.0 if row.posterior_probability is None else row.posterior_probability)
    doc.heading("Interpretation", level=2)
    doc.paragraph(
        f"The posterior winner is `{winner.class_or_model}` with probability `{winner.posterior_probability}`. "
        "Inspect the prior and likelihood columns to separate transition pressure from measurement evidence."
    )
    return doc.text()


def write_step_card(path: str | Path, traces: tuple[FilterStepTrace, ...] | list[FilterStepTrace], *, title: str | None = None) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_step_card_markdown(traces, title=title), encoding="utf-8")
    return output_path
