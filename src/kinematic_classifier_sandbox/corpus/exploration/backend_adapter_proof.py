from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from kinematic_classifier_sandbox.reports.markdown import (
    MarkdownDocument,
    MermaidEdge,
    MermaidFlow,
    MermaidNode,
)
from kinematic_classifier_sandbox.utils.io import write_csv

from ...utils.plotting import plt
from .backend_adapter_proof_core import (
    AdapterExecutionRecord,
    BackendCandidateSpec,
    EnvironmentAware1DAdapter,
    MockFileBackend1DAdapter,
    ParameterOnly1DAdapter,
    _adapter_map,
    _environment_candidate,
    _failing_candidates,
    _shared_boundary_candidate,
    _switching_candidate,
)

__all__ = [
    "AdapterExecutionRecord",
    "BackendAdapterProofArtifacts",
    "BackendAdapterProofResult",
    "BackendCandidateSpec",
    "EnvironmentAware1DAdapter",
    "MockFileBackend1DAdapter",
    "ParameterOnly1DAdapter",
    "_adapter_map",
    "_environment_candidate",
    "_failing_candidates",
    "_shared_boundary_candidate",
    "_switching_candidate",
    "analyze_backend_adapter_proof",
    "write_backend_adapter_proof_artifacts",
]


@dataclass(frozen=True, slots=True)
class BackendAdapterProofResult:
    backend_manifest: dict[str, Any]
    backend_run_rows: tuple[dict[str, Any], ...]
    equivalence_rows: tuple[dict[str, Any], ...]
    failure_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class BackendAdapterProofArtifacts:
    run_dir: Path
    backend_manifest_path: Path
    backend_run_examples_path: Path
    backend_output_equivalence_report_path: Path
    adapter_failure_cases_path: Path
    telemetry_comparison_png_path: Path
    failure_taxonomy_png_path: Path


def _run_row(record: AdapterExecutionRecord) -> dict[str, Any]:
    run = record.trajectory_run
    success = run.success and not record.validation_errors
    position = run.truth_state.get("position", ())
    velocity = run.truth_state.get("velocity", ())
    return {
        "backend_id": record.backend_id,
        "candidate_id": record.candidate_id,
        "scenario_id": run.scenario_id,
        "success": success,
        "failure_reason": run.failure_reason or "",
        "cache_key": record.cache_key,
        "cache_hit": record.cache_hit,
        "num_times": len(run.times),
        "position_final": position[-1] if position else "",
        "velocity_final": velocity[-1] if velocity else "",
        "event_count": len(run.events),
        "observation_fields": ",".join(sorted(run.observations.keys())),
        "environment_fields": ",".join(sorted(run.environment_trace.keys())),
        "validation_error_count": len(record.validation_errors),
    }


def _equivalence_rows(shared_records: tuple[AdapterExecutionRecord, ...]) -> tuple[dict[str, Any], ...]:
    baseline = shared_records[0].trajectory_run
    baseline_position = baseline.truth_state["position"]
    rows: list[dict[str, Any]] = []
    for record in shared_records[1:]:
        position = record.trajectory_run.truth_state["position"]
        velocity = record.trajectory_run.truth_state["velocity"]
        rows.append(
            {
                "baseline_backend_id": baseline.backend_id,
                "comparison_backend_id": record.backend_id,
                "scenario_id": baseline.scenario_id,
                "same_num_samples": len(position) == len(baseline_position),
                "max_position_delta": max(abs(a - b) for a, b in zip(position, baseline_position)),
                "final_position_delta": abs(position[-1] - baseline_position[-1]),
                "final_velocity": velocity[-1],
                "common_truth_fields": "position,velocity,acceleration",
                "common_observation_fields": ",".join(sorted(set(baseline.observations).intersection(record.trajectory_run.observations))),
            }
        )
    return tuple(rows)


def _render_telemetry_comparison_png(shared_records: tuple[AdapterExecutionRecord, ...]) -> bytes:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    for record in shared_records:
        run = record.trajectory_run
        axes[0].plot(run.times, run.truth_state["position"], marker="o", label=record.backend_id)
        axes[1].plot(run.times, run.truth_state["velocity"], marker="o", label=record.backend_id)
    axes[0].set_ylabel("Position")
    axes[1].set_ylabel("Velocity")
    axes[1].set_xlabel("Time")
    axes[0].set_title("Normalized Telemetry Comparison")
    axes[0].legend(fontsize=8)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_failure_taxonomy_png(failure_rows: tuple[dict[str, Any], ...]) -> bytes:
    counts: dict[str, int] = {}
    for row in failure_rows:
        reason = str(row["failure_reason"] or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    labels = list(counts.keys())
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.bar(labels, values, color="#ca5b4b")
    ax.set_ylabel("Count")
    ax.set_title("Adapter Failure Taxonomy")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def analyze_backend_adapter_proof() -> BackendAdapterProofResult:
    adapters = _adapter_map()
    shared_candidate = _shared_boundary_candidate()
    switching_candidate = _switching_candidate()
    environment_candidate = _environment_candidate()
    failures = _failing_candidates()

    shared_backend_ids = ("parameter_only_1d", "environment_aware_1d", "mock_file_backend_1d")
    shared_records = tuple(adapters[backend_id].run(shared_candidate) for backend_id in shared_backend_ids)
    repeated_cache_record = adapters["parameter_only_1d"].run(shared_candidate)
    switching_records = tuple(
        adapters[backend_id].run(switching_candidate)
        for backend_id in ("controlled_1d", "mock_file_backend_1d")
    )
    environment_record = adapters["environment_aware_1d"].run(environment_candidate)
    failure_records = (
        adapters["controlled_1d"].run(failures[0]),
        adapters["mock_file_backend_1d"].run(failures[1]),
    )

    all_records = list(shared_records) + [repeated_cache_record] + list(switching_records) + [environment_record] + list(failure_records)
    run_rows = [_run_row(record) for record in all_records]
    failure_rows = tuple(row for row in run_rows if not bool(row["success"]))
    equivalence_rows = _equivalence_rows(shared_records)

    cache_probe = {
        "candidate_id": shared_candidate.candidate_id,
        "backend_id": "parameter_only_1d",
        "first_cache_key": shared_records[0].cache_key,
        "second_cache_key": repeated_cache_record.cache_key,
        "second_run_cache_hit": repeated_cache_record.cache_hit,
        "stable_cache_key": shared_records[0].cache_key == repeated_cache_record.cache_key,
    }

    backend_manifest = {
        "proof_version": "m32_v1",
        "adapters": [
            {
                "backend_id": adapter.backend_id,
                "family": adapter.family,
                "runtime_class": adapter.definition.capabilities.runtime_class,
                "supports_environment": adapter.definition.capabilities.supports_environment,
                "supports_sequential_control": adapter.definition.capabilities.supports_sequential_control,
                "supported_scenario_families": [
                    scenario_family
                    for scenario_family in ("shared_boundary_case", "switching_case", "environment_regime_case", "file_backend_case")
                    if adapter.supports(
                        {
                            "shared_boundary_case": shared_candidate,
                            "switching_case": switching_candidate,
                            "environment_regime_case": environment_candidate,
                            "file_backend_case": failures[1],
                        }[scenario_family]
                    )
                ],
            }
            for adapter in adapters.values()
        ],
        "shared_compatible_scenario": {
            "candidate_id": shared_candidate.candidate_id,
            "backend_ids": list(shared_backend_ids),
            "common_truth_fields": ["position", "velocity", "acceleration"],
            "common_observation_fields": ["position"],
        },
        "cache_probe": cache_probe,
        "structured_failure_count": len(failure_rows),
    }

    report = MarkdownDocument()
    report.heading("Backend Adapter Proof", level=1)
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"adapters exercised: `{len(adapters)}`",
            f"shared scenario executed across compatible backends: `{', '.join(shared_backend_ids)}`",
            f"structured failures captured: `{len(failure_rows)}`",
            f"stable cache key proven: `{cache_probe['stable_cache_key']}`",
            f"second repeated execution served from cache: `{cache_probe['second_run_cache_hit']}`",
        ]
    )
    report.heading("Shared Compatible Scenario", level=2)
    report.table(
        ["Backend", "Success", "Samples", "Final Position", "Observation Fields"],
        [
            (
                f"`{record.backend_id}`",
                f"{record.trajectory_run.success}",
                f"{len(record.trajectory_run.times)}",
                f"`{record.trajectory_run.truth_state.get('position', ('',))[-1] if record.trajectory_run.truth_state.get('position') else ''}`",
                f"`{', '.join(sorted(record.trajectory_run.observations))}`",
            )
            for record in shared_records
        ],
    )
    report.heading("Adapter Flow", level=2)
    report.mermaid(
        MermaidFlow(
            nodes=(
                MermaidNode("A", "BackendCandidateSpec"),
                MermaidNode("B", "prepare(input_bundle)"),
                MermaidNode("C", "cache_key / cache lookup"),
                MermaidNode("D", "run simulator or reuse cache"),
                MermaidNode("E", "raw output"),
                MermaidNode("F", "normalize_output()"),
                MermaidNode("G", "TrajectoryRun"),
                MermaidNode("H", "validation + artifact rows"),
            ),
            edges=(
                MermaidEdge("A", "B"),
                MermaidEdge("B", "C"),
                MermaidEdge("C", "D"),
                MermaidEdge("D", "E"),
                MermaidEdge("E", "F"),
                MermaidEdge("F", "G"),
                MermaidEdge("G", "H"),
            ),
        )
    )
    report.heading("Output Equivalence", level=2)
    report.table(
        ["Baseline", "Comparison", "Same Samples", "Max Position Delta", "Common Observations"],
        [
            (
                f"`{row['baseline_backend_id']}`",
                f"`{row['comparison_backend_id']}`",
                f"`{row['same_num_samples']}`",
                f"`{row['max_position_delta']:.4f}`",
                f"`{row['common_observation_fields']}`",
            )
            for row in equivalence_rows
        ],
    )
    report.heading("Failure Cases", level=2)
    report.table(
        ["Backend", "Candidate", "Failure Reason"],
        [
            (f"`{row['backend_id']}`", f"`{row['candidate_id']}`", f"`{row['failure_reason']}`")
            for row in failure_rows
        ],
    )
    report.heading("Notes", level=2)
    report.bullet_list(
        [
            "This milestone proves adapter execution, normalization, cache-key stability, and structured failures with 1D backends only.",
            "The mock file backend remains synthetic, but it now follows the same prepare, run, normalize, and failure-capture flow expected of a future external simulator adapter.",
            "The shared boundary scenario is intentionally run across multiple backends so the proof covers execution equivalence rather than only schema compatibility.",
        ]
    )
    report_markdown = report.text()

    return BackendAdapterProofResult(
        backend_manifest=backend_manifest,
        backend_run_rows=tuple(run_rows),
        equivalence_rows=equivalence_rows,
        failure_rows=failure_rows,
        report_markdown=report_markdown,
    )


def write_backend_adapter_proof_artifacts(
    base_dir: str | Path,
    *,
    result: BackendAdapterProofResult | None = None,
) -> BackendAdapterProofArtifacts:
    run_dir = Path(base_dir) / "backend_adapter_proof"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_backend_adapter_proof()

    backend_manifest_path = run_dir / "backend_manifest.json"
    backend_run_examples_path = run_dir / "backend_run_examples.csv"
    backend_output_equivalence_report_path = run_dir / "backend_output_equivalence_report.md"
    adapter_failure_cases_path = run_dir / "adapter_failure_cases.csv"
    telemetry_comparison_png_path = run_dir / "normalized_telemetry_comparison.png"
    failure_taxonomy_png_path = run_dir / "adapter_failure_taxonomy.png"

    backend_manifest_path.write_text(json.dumps(payload.backend_manifest, indent=2), encoding="utf-8")
    backend_output_equivalence_report_path.write_text(payload.report_markdown, encoding="utf-8")

    run_fieldnames = list(payload.backend_run_rows[0].keys()) if payload.backend_run_rows else []
    write_csv(backend_run_examples_path, list(payload.backend_run_rows), run_fieldnames)

    failure_fieldnames = list(payload.failure_rows[0].keys()) if payload.failure_rows else run_fieldnames
    write_csv(adapter_failure_cases_path, list(payload.failure_rows), failure_fieldnames)

    adapters = _adapter_map()
    shared_candidate = _shared_boundary_candidate()
    shared_records = tuple(adapters[backend_id].run(shared_candidate) for backend_id in ("parameter_only_1d", "environment_aware_1d", "mock_file_backend_1d"))
    telemetry_comparison_png_path.write_bytes(_render_telemetry_comparison_png(shared_records))
    failure_taxonomy_png_path.write_bytes(_render_failure_taxonomy_png(payload.failure_rows))

    return BackendAdapterProofArtifacts(
        run_dir=run_dir,
        backend_manifest_path=backend_manifest_path,
        backend_run_examples_path=backend_run_examples_path,
        backend_output_equivalence_report_path=backend_output_equivalence_report_path,
        adapter_failure_cases_path=adapter_failure_cases_path,
        telemetry_comparison_png_path=telemetry_comparison_png_path,
        failure_taxonomy_png_path=failure_taxonomy_png_path,
    )
