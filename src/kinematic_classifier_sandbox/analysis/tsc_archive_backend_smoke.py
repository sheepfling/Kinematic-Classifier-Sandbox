from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..utils.io import write_csv


ARCHIVE_METHOD_FAMILIES = (
    "minirocket_family",
    "drcif_interval_forests",
    "dictionary_tde_family",
    "hive_cote",
)


@dataclass(frozen=True, slots=True)
class TSCArchiveBackendSmokeRow:
    method_family: str
    available: bool
    attempted: bool
    succeeded: bool
    timed_out: bool
    backend_name: str
    detail: str
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class TSCArchiveBackendSmokeResult:
    rows: tuple[TSCArchiveBackendSmokeRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class TSCArchiveBackendSmokeArtifacts:
    run_dir: Path
    row_summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path


def _probe_family(method_family: str, *, timeout_seconds: float) -> TSCArchiveBackendSmokeRow:
    probe_code = f"""
import json
from kinematic_classifier_sandbox.analysis.optional_external_backends import fit_archive_classifier_with_outcome
from kinematic_classifier_sandbox.analysis.common_dataset_comparison import generate_shared_dynamics_dataset

rows = tuple(
    trajectory
    for trajectory in generate_shared_dynamics_dataset(seed=1009, trajectories_per_case=1)
    if int(trajectory.trajectory_id.rsplit("_", 1)[-1]) < 4
)
outcome = fit_archive_classifier_with_outcome(
    {method_family!r},
    rows,
    class_names=("constant_velocity", "constant_acceleration"),
)
print(json.dumps({{
    "method_family": outcome.method_family,
    "available": outcome.availability.available,
    "attempted": outcome.attempted,
    "succeeded": outcome.succeeded,
    "backend_name": outcome.backend_name,
    "detail": outcome.detail,
}}))
""".strip()
    try:
        completed = subprocess.run(
            (sys.executable, "-c", probe_code),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return TSCArchiveBackendSmokeRow(
            method_family=method_family,
            available=True,
            attempted=True,
            succeeded=False,
            timed_out=True,
            backend_name="external_probe_timeout",
            detail="external_probe_timed_out",
            timeout_seconds=float(timeout_seconds),
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        return TSCArchiveBackendSmokeRow(
            method_family=method_family,
            available=True,
            attempted=True,
            succeeded=False,
            timed_out=False,
            backend_name="external_probe_failed",
            detail=f"external_probe_failed:{stderr.splitlines()[-1] if stderr else 'unknown_error'}",
            timeout_seconds=float(timeout_seconds),
        )
    payload = json.loads(completed.stdout.strip())
    return TSCArchiveBackendSmokeRow(
        method_family=str(payload["method_family"]),
        available=bool(payload["available"]),
        attempted=bool(payload["attempted"]),
        succeeded=bool(payload["succeeded"]),
        timed_out=False,
        backend_name=str(payload["backend_name"]),
        detail=str(payload["detail"]),
        timeout_seconds=float(timeout_seconds),
    )


def analyze_tsc_archive_backend_smoke(
    *,
    timeout_seconds: float = 20.0,
) -> TSCArchiveBackendSmokeResult:
    rows = tuple(_probe_family(method_family, timeout_seconds=timeout_seconds) for method_family in ARCHIVE_METHOD_FAMILIES)
    available_count = sum(1 for row in rows if row.available)
    attempted_count = sum(1 for row in rows if row.attempted)
    succeeded_count = sum(1 for row in rows if row.succeeded)
    timed_out_count = sum(1 for row in rows if row.timed_out)
    failed_count = sum(1 for row in rows if row.attempted and not row.succeeded and not row.timed_out)
    metrics: dict[str, float | int | str] = {
        "study_id": "tsc_archive_backend_smoke_v1",
        "family_count": len(rows),
        "available_family_count": available_count,
        "attempted_family_count": attempted_count,
        "succeeded_family_count": succeeded_count,
        "timed_out_family_count": timed_out_count,
        "failed_family_count": failed_count,
        "timeout_seconds": float(timeout_seconds),
        "integration_read": (
            "all_external_smoke_succeeded"
            if succeeded_count == len(rows)
            else "mixed_smoke_outcomes"
            if succeeded_count > 0 or attempted_count > 0
            else "no_external_backend_available"
        ),
    }
    return TSCArchiveBackendSmokeResult(rows=rows, metrics=metrics)


def write_tsc_archive_backend_smoke_artifacts(
    output_dir: str | Path,
    *,
    result: TSCArchiveBackendSmokeResult | None = None,
    timeout_seconds: float = 20.0,
) -> TSCArchiveBackendSmokeArtifacts:
    payload = result or analyze_tsc_archive_backend_smoke(timeout_seconds=timeout_seconds)
    run_dir = Path(output_dir) / "tsc_archive_backend_smoke_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    row_summary_path = run_dir / "backend_smoke_rows.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "tsc_archive_backend_smoke_report.md"
    decision_card_path = run_dir / "decision_card.md"

    write_csv(
        row_summary_path,
        [asdict(row) for row in payload.rows],
        list(TSCArchiveBackendSmokeRow.__dataclass_fields__.keys()),
    )
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report = MarkdownDocument("TSC Archive Backend Smoke")
    report.paragraph("This packet is the tiny capability probe for the generic time-series benchmark lane. It is designed to show whether optional external archive-family backends are unavailable, fail during tiny fit attempts, time out, or succeed on a minimal smoke surface.")
    report.table(
        ["Family", "Available", "Attempted", "Succeeded", "Timed out", "Backend", "Detail"],
        [
            (
                row.method_family,
                "yes" if row.available else "no",
                "yes" if row.attempted else "no",
                "yes" if row.succeeded else "no",
                "yes" if row.timed_out else "no",
                row.backend_name,
                row.detail,
            )
            for row in payload.rows
        ],
    )
    report.paragraph(f"Integration read: `{payload.metrics['integration_read']}`")
    report_path.write_text(report.text(), encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate family: `generic_time_series_benchmark_classifiers`",
        f"- Integration read: `{payload.metrics['integration_read']}`",
        "- Use: `backend-smoke evidence only; this does not replace the shared archive frontier or promote the family by itself`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    return TSCArchiveBackendSmokeArtifacts(
        run_dir=run_dir,
        row_summary_path=row_summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
    )


__all__ = [
    "TSCArchiveBackendSmokeArtifacts",
    "TSCArchiveBackendSmokeResult",
    "TSCArchiveBackendSmokeRow",
    "analyze_tsc_archive_backend_smoke",
    "write_tsc_archive_backend_smoke_artifacts",
]
