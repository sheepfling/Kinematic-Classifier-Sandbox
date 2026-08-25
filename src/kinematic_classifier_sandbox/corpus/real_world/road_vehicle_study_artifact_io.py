from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .projection import ProjectionResult
from .road_vehicle_study import RoadVehicleDurationStudy, RoadVehicleStudyResult


@dataclass(frozen=True, slots=True)
class RoadVehicleStudyArtifacts:
    output_dir: Path
    manifest_path: Path
    tracks_path: Path
    report_path: Path
    duration_dirs: tuple[Path, ...]
####


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
####


def _projection_metadata_rows(
    results: tuple[ProjectionResult, ...],
) -> list[dict[str, object]]:
    return [
        {
            "trajectory_id": result.metadata.trajectory_id,
            "window_id": result.metadata.window_id,
            "split_group_id": result.metadata.split_group_id,
            "dataset_id": result.metadata.dataset_id,
            "recording_id": result.metadata.recording_id,
            "run_id": result.metadata.run_id,
            "track_id": result.metadata.track_id,
            "source_asset_id": result.metadata.source_asset_id,
            "native_label": result.metadata.native_label,
            "normalized_class": result.metadata.normalized_class,
            "partition": (
                result.metadata.partition.value
                if result.metadata.partition is not None
                else ""
            ),
            "projection_kind": result.metadata.projection_kind.value,
            "source_start_time_s": result.metadata.source_start_time_s,
            "source_end_time_s": result.metadata.source_end_time_s,
            "sample_count": len(result.trajectory.times),
            "coordinate_frame": result.trajectory.coordinate_frame,
        }
        for result in results
    ]
####


def _write_duration_artifacts(
    study: RoadVehicleDurationStudy,
    *,
    output_dir: Path,
) -> Path:
    duration_token = f"{study.window_duration_s:g}".replace(".", "p")
    duration_dir = output_dir / f"duration_{duration_token}s"
    duration_dir.mkdir(parents=True, exist_ok=True)

    partition_by_group = {
        assignment.split_group_id: assignment.partition.value
        for assignment in study.split.assignments
    }
    _write_csv(
        duration_dir / "split_assignments.csv",
        [
            {
                "split_group_id": assignment.split_group_id,
                "normalized_class": assignment.normalized_class,
                "partition": assignment.partition.value,
                "hash_fraction": assignment.hash_fraction,
            }
            for assignment in study.split.assignments
        ],
        ("split_group_id", "normalized_class", "partition", "hash_fraction"),
    )
    _write_csv(
        duration_dir / "partition_summary.csv",
        [
            {
                "partition": row.partition.value,
                "normalized_class": row.normalized_class,
                "track_count": row.track_count,
                "source_duration_s": row.source_duration_s,
                "candidate_window_count": row.candidate_window_count,
                "accepted_window_count": row.accepted_window_count,
                "accepted_window_duration_s": row.accepted_window_duration_s,
                "rejected_low_coverage_count": row.rejected_low_coverage_count,
                "rejected_short_segment_count": row.rejected_short_segment_count,
            }
            for row in study.partition_summary
        ],
        (
            "partition",
            "normalized_class",
            "track_count",
            "source_duration_s",
            "candidate_window_count",
            "accepted_window_count",
            "accepted_window_duration_s",
            "rejected_low_coverage_count",
            "rejected_short_segment_count",
        ),
    )
    _write_csv(
        duration_dir / "windows.csv",
        [
            {
                "window_id": window.window_id,
                "split_group_id": window.split_group_id,
                "partition": partition_by_group[window.split_group_id],
                "segment_index": window.segment_index,
                "start_index": window.start_index,
                "end_index_exclusive": window.end_index_exclusive,
                "start_time_s": window.start_time_s,
                "end_time_s": window.end_time_s,
                "requested_duration_s": window.requested_duration_s,
                "sample_count": window.sample_count,
                "expected_sample_count": window.expected_sample_count,
                "coverage_fraction": window.coverage_fraction,
            }
            for window in study.windows
        ],
        (
            "window_id",
            "split_group_id",
            "partition",
            "segment_index",
            "start_index",
            "end_index_exclusive",
            "start_time_s",
            "end_time_s",
            "requested_duration_s",
            "sample_count",
            "expected_sample_count",
            "coverage_fraction",
        ),
    )
    projection_rows = _projection_metadata_rows(
        study.speed_profile + study.cumulative_path_length
    )
    _write_csv(
        duration_dir / "projection_metadata.csv",
        projection_rows,
        (
            "trajectory_id",
            "window_id",
            "split_group_id",
            "dataset_id",
            "recording_id",
            "run_id",
            "track_id",
            "source_asset_id",
            "native_label",
            "normalized_class",
            "partition",
            "projection_kind",
            "source_start_time_s",
            "source_end_time_s",
            "sample_count",
            "coordinate_frame",
        ),
    )
    return duration_dir
####


def _render_report(result: RoadVehicleStudyResult) -> str:
    lines = [
        "# Real-World Road-Vehicle Study Preparation",
        "",
        f"Pair: `{result.pair_spec.class_a}` vs `{result.pair_spec.class_b}`",
        f"Source tracks retained: {len(result.tracks)}",
        "",
        "This packet reports leakage-safe grouped splits and window coverage. "
        "It does not claim classifier performance.",
        "",
    ]
    for study in result.duration_studies:
        lines.extend(
            [
                f"## {study.window_duration_s:g}-second windows",
                "",
                f"Accepted windows: {len(study.windows)}",
                "",
                "| partition | class | tracks | source seconds | accepted windows | "
                "rejected low coverage |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in study.partition_summary:
            lines.append(
                "| "
                f"{row.partition.value} | {row.normalized_class} | "
                f"{row.track_count} | {row.source_duration_s:.1f} | "
                f"{row.accepted_window_count} | "
                f"{row.rejected_low_coverage_count} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
####


def write_road_vehicle_study_artifacts(
    result: RoadVehicleStudyResult,
    output_dir: str | Path,
) -> RoadVehicleStudyArtifacts:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "study_manifest.json"
    tracks_path = root / "tracks.csv"
    report_path = root / "report.md"

    manifest_path.write_text(
        json.dumps(
            {
                "pair_id": result.pair_spec.pair_id,
                "class_a": result.pair_spec.class_a,
                "class_b": result.pair_spec.class_b,
                "expected_difficulty": result.pair_spec.expected_difficulty,
                "track_count": len(result.tracks),
                "window_durations_s": [
                    study.window_duration_s
                    for study in result.duration_studies
                ],
                "projection_kinds": [
                    "speed_profile",
                    "cumulative_path_length",
                ],
                "claim_boundary": (
                    "Study preparation only; classifier performance is not established."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        tracks_path,
        [
            {
                "split_group_id": track.provenance.split_group_id,
                "dataset_id": track.provenance.dataset_id,
                "recording_id": track.provenance.recording_id,
                "run_id": track.provenance.run_id,
                "track_id": track.provenance.track_id,
                "native_label": track.labels.native_label,
                "normalized_class": track.labels.normalized_class,
                "source_duration_s": float(
                    track.timestamps_s[-1] - track.timestamps_s[0]
                ),
            }
            for track in result.tracks
        ],
        (
            "split_group_id",
            "dataset_id",
            "recording_id",
            "run_id",
            "track_id",
            "native_label",
            "normalized_class",
            "source_duration_s",
        ),
    )
    duration_dirs = tuple(
        _write_duration_artifacts(study, output_dir=root)
        for study in result.duration_studies
    )
    report_path.write_text(_render_report(result), encoding="utf-8")

    return RoadVehicleStudyArtifacts(
        output_dir=root,
        manifest_path=manifest_path,
        tracks_path=tracks_path,
        report_path=report_path,
        duration_dirs=duration_dirs,
    )
####


__all__ = [
    "RoadVehicleStudyArtifacts",
    "write_road_vehicle_study_artifacts",
]
