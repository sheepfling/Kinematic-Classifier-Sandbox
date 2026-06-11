from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..advanced_filters.artifact_io import write_imm_artifacts
from ..advanced_filters.evaluation import (
    write_particle_filter_witness_artifacts,
    write_rbpf_witness_artifacts,
)
from ..advanced_filters.ou_witness import write_ornstein_uhlenbeck_witness_artifacts
from ..inference.kalman_filter_bank import write_kalman_bank_artifacts
from ..inference.transition_matrix.artifact_io import write_transition_benchmark_artifacts
from ..markdown_builder import MarkdownDocument
from ..registry.method_validation_os import analyze_method_validation_os
from ..utils.io import _write_json, _write_text, read_csv_rows, write_csv
from .trace_schema import filter_step_trace_schema


@dataclass(frozen=True, slots=True)
class FilterTracePacketSpec:
    method_id: str
    display_name: str
    trace_path: Path
    report_path: Path
    step_card_dir: Path | None
    intermediate_plot_dir: Path | None


@dataclass(frozen=True, slots=True)
class FilterTraceValidationResult:
    method_rows: tuple[dict[str, object], ...]
    requirement_rows: tuple[dict[str, object], ...]
    summary: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class FilterTraceValidationArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    method_trace_matrix_path: Path
    trace_requirement_matrix_path: Path
    schema_path: Path


def _materialize_default_trace_packets(output_root: Path) -> tuple[FilterTracePacketSpec, ...]:
    transition = write_transition_benchmark_artifacts(output_root)
    kalman = write_kalman_bank_artifacts(output_root)
    imm = write_imm_artifacts(output_root)
    particle_filter = write_particle_filter_witness_artifacts(output_root)
    rbpf = write_rbpf_witness_artifacts(output_root)
    ou = write_ornstein_uhlenbeck_witness_artifacts(output_root)
    return (
        FilterTracePacketSpec(
            method_id="static_mode_accumulator",
            display_name="Static mode accumulator",
            trace_path=transition.filter_step_trace_path,
            report_path=transition.report_path,
            step_card_dir=transition.step_card_dir,
            intermediate_plot_dir=transition.intermediate_plot_dir,
        ),
        FilterTracePacketSpec(
            method_id="transition_matrix_accumulator",
            display_name="Transition matrix accumulator",
            trace_path=transition.filter_step_trace_path,
            report_path=transition.report_path,
            step_card_dir=transition.step_card_dir,
            intermediate_plot_dir=transition.intermediate_plot_dir,
        ),
        FilterTracePacketSpec(
            method_id="kalman_bank",
            display_name="Kalman bank",
            trace_path=kalman.filter_step_trace_path,
            report_path=kalman.report_path,
            step_card_dir=kalman.step_card_dir,
            intermediate_plot_dir=kalman.intermediate_plot_dir,
        ),
        FilterTracePacketSpec(
            method_id="imm_v1",
            display_name="IMM",
            trace_path=imm.filter_step_trace_path,
            report_path=imm.report_path,
            step_card_dir=imm.step_card_dir,
            intermediate_plot_dir=imm.intermediate_plot_dir,
        ),
        FilterTracePacketSpec(
            method_id="particle_filter_bank_v1",
            display_name="Particle filter bank",
            trace_path=particle_filter.run_dir / "traces" / "filter_step_trace.csv",
            report_path=particle_filter.report_path,
            step_card_dir=None,
            intermediate_plot_dir=particle_filter.run_dir / "plots",
        ),
        FilterTracePacketSpec(
            method_id="rbpf_v1",
            display_name="RBPF",
            trace_path=rbpf.run_dir / "traces" / "filter_step_trace.csv",
            report_path=rbpf.report_path,
            step_card_dir=None,
            intermediate_plot_dir=rbpf.run_dir / "plots",
        ),
        FilterTracePacketSpec(
            method_id="ornstein_uhlenbeck_pf_v1",
            display_name="Ornstein-Uhlenbeck PF",
            trace_path=ou.run_dir / "traces" / "filter_step_trace.csv",
            report_path=ou.report_path,
            step_card_dir=None,
            intermediate_plot_dir=ou.run_dir / "plots",
        ),
    )


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _has_nonempty_cell(value: str | None) -> bool:
    return bool(str(value).strip()) if value is not None else False


def analyze_filter_trace_validation_packet(
    output_root: str | Path,
    *,
    materialize: bool = True,
) -> FilterTraceValidationResult:
    output_path = Path(output_root)
    specs = _materialize_default_trace_packets(output_path) if materialize else ()
    method_validation = analyze_method_validation_os()
    status_lookup = {row.method_id: row.current_status for row in method_validation.method_rows}
    validation_method_aliases = {
        "static_mode_accumulator": "pointwise",
        "transition_matrix_accumulator": "hmm_transition",
        "imm_v1": "imm",
        "particle_filter_bank_v1": "particle_filter",
        "rbpf_v1": "rbpf",
        "ornstein_uhlenbeck_pf_v1": "particle_filter",
    }

    method_rows: list[dict[str, object]] = []
    requirement_rows: list[dict[str, object]] = []
    for spec in specs:
        trace_rows = [
            row
            for row in read_csv_rows(spec.trace_path)
            if row.get("method_id") == spec.method_id
        ]
        row_count = len(trace_rows)
        has_prior = any(_parse_optional_float(row.get("prior_probability")) is not None for row in trace_rows)
        has_prediction = any(
            _parse_optional_float(row.get("predicted_probability")) is not None
            or _has_nonempty_cell(row.get("predicted_measurement"))
            or _has_nonempty_cell(row.get("predicted_state_mean"))
            for row in trace_rows
        )
        has_measurement = any(_has_nonempty_cell(row.get("measurement")) for row in trace_rows)
        has_innovation = any(
            _has_nonempty_cell(row.get("innovation"))
            or _parse_optional_float(row.get("normalized_innovation_squared")) is not None
            for row in trace_rows
        )
        has_likelihood = any(_parse_optional_float(row.get("log_likelihood")) is not None for row in trace_rows)
        has_posterior = any(_parse_optional_float(row.get("posterior_probability")) is not None for row in trace_rows)
        has_diagnostics = any(
            _parse_optional_float(row.get("posterior_entropy")) is not None
            or _parse_optional_float(row.get("effective_sample_size")) is not None
            or _has_nonempty_cell(row.get("is_resampled"))
            for row in trace_rows
        )
        has_step_cards = bool(spec.step_card_dir and spec.step_card_dir.exists() and any(spec.step_card_dir.glob("*.md")))
        has_intermediate_plots = bool(
            spec.intermediate_plot_dir
            and spec.intermediate_plot_dir.exists()
            and any(spec.intermediate_plot_dir.rglob("*.png"))
        )
        stage_rows = (
            ("prior", has_prior),
            ("prediction", has_prediction),
            ("measurement", has_measurement),
            ("innovation_or_residual", has_innovation),
            ("likelihood", has_likelihood),
            ("posterior", has_posterior),
            ("diagnostics", has_diagnostics),
            ("step_cards", has_step_cards),
            ("intermediate_plots", has_intermediate_plots),
        )
        for stage_name, status in stage_rows:
            requirement_rows.append(
                {
                    "method_id": spec.method_id,
                    "stage": stage_name,
                    "status": "yes" if status else "no",
                }
            )
        all_core_trace_fields = all(
            (
                has_measurement,
                has_likelihood,
                has_posterior,
                has_diagnostics,
            )
        )
        trace_status = "trace_validated" if all_core_trace_fields and has_intermediate_plots else "implemented"
        method_rows.append(
            {
                "method_id": spec.method_id,
                "display_name": spec.display_name,
                "row_count": row_count,
                "trace_path": str(spec.trace_path),
                "report_path": str(spec.report_path),
                "has_prior": "yes" if has_prior else "no",
                "has_prediction": "yes" if has_prediction else "no",
                "has_measurement": "yes" if has_measurement else "no",
                "has_innovation_or_residual": "yes" if has_innovation else "no",
                "has_likelihood": "yes" if has_likelihood else "no",
                "has_posterior": "yes" if has_posterior else "no",
                "has_diagnostics": "yes" if has_diagnostics else "no",
                "has_step_cards": "yes" if has_step_cards else "no",
                "has_intermediate_plots": "yes" if has_intermediate_plots else "no",
                "trace_status": trace_status,
                "method_validation_status": status_lookup.get(
                    validation_method_aliases.get(spec.method_id, spec.method_id),
                    "",
                ),
            }
        )

    summary = {
        "method_count": len(method_rows),
        "trace_validated_count": sum(row["trace_status"] == "trace_validated" for row in method_rows),
        "methods_with_step_cards": sum(row["has_step_cards"] == "yes" for row in method_rows),
        "methods_with_intermediate_plots": sum(row["has_intermediate_plots"] == "yes" for row in method_rows),
        "schema": filter_step_trace_schema(),
    }
    return FilterTraceValidationResult(
        method_rows=tuple(method_rows),
        requirement_rows=tuple(requirement_rows),
        summary=summary,
        report_markdown=_render_filter_trace_validation_report(method_rows, summary),
    )


def _render_filter_trace_validation_report(
    method_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> str:
    report = MarkdownDocument("Filter Trace Validation Packet")
    report.paragraph(
        "This packet validates the step-level evidence surfaces behind the classifier and filter ladder. It is intentionally narrower than a leaderboard: the question here is whether each method emits auditable prior, prediction, measurement, likelihood, posterior, and diagnostic traces."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Methods checked: `{summary['method_count']}`",
            f"Trace-validated methods: `{summary['trace_validated_count']}`",
            f"Methods with step cards: `{summary['methods_with_step_cards']}`",
            f"Methods with intermediate plots: `{summary['methods_with_intermediate_plots']}`",
        ]
    )
    report.heading("Status Rule", level=2)
    report.bullet_list(
        [
            "`implemented`: trace file exists but the full intermediate evidence packet is incomplete.",
            "`trace_validated`: trace file exposes the core update stages and has intermediate visual artifacts.",
            "`witness_supported` and `study_justified`: these remain higher-level method-validation statuses and should not be inferred from trace presence alone.",
        ]
    )
    report.heading("Method Matrix", level=2)
    report.table(
        [
            "Method",
            "Rows",
            "Trace Status",
            "Method Validation",
            "Prior",
            "Prediction",
            "Measurement",
            "Likelihood",
            "Posterior",
            "Diagnostics",
            "Step Cards",
        ],
        [
            (
                str(row["display_name"]),
                row["row_count"],
                row["trace_status"],
                row["method_validation_status"],
                row["has_prior"],
                row["has_prediction"],
                row["has_measurement"],
                row["has_likelihood"],
                row["has_posterior"],
                row["has_diagnostics"],
                row["has_step_cards"],
            )
            for row in method_rows
        ],
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "Trace validation proves that a method can explain its update steps mechanically.",
            "Trace validation does not by itself justify method promotion; use the method-validation and advanced-decision packets for that.",
            "The main consistency rule is now explicit: trace status, witness support, and study justification are separate layers.",
        ]
    )
    return report.text()


def write_filter_trace_validation_artifacts(
    output_dir: str | Path,
    *,
    materialize: bool = True,
) -> FilterTraceValidationArtifacts:
    output_root = Path(output_dir)
    result = analyze_filter_trace_validation_packet(output_root, materialize=materialize)
    run_dir = output_root / "filter_trace_validation_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "filter_trace_validation_report.md"
    summary_path = run_dir / "filter_trace_validation_summary.json"
    method_trace_matrix_path = run_dir / "method_trace_matrix.csv"
    trace_requirement_matrix_path = run_dir / "trace_requirement_matrix.csv"
    schema_path = run_dir / "filter_step_trace_schema.json"
    _write_text(report_path, result.report_markdown)
    _write_json(
        summary_path,
        {
            **result.summary,
            "schema": None,
        },
    )
    write_csv(method_trace_matrix_path, list(result.method_rows), list(result.method_rows[0]) if result.method_rows else [])
    write_csv(
        trace_requirement_matrix_path,
        list(result.requirement_rows),
        ["method_id", "stage", "status"],
    )
    schema_path.write_text(json.dumps(filter_step_trace_schema(), indent=2), encoding="utf-8")
    return FilterTraceValidationArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        method_trace_matrix_path=method_trace_matrix_path,
        trace_requirement_matrix_path=trace_requirement_matrix_path,
        schema_path=schema_path,
    )
