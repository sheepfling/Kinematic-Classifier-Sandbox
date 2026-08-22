from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..contracts import LabelEvidence, TrackLabels
from .tgsim_contracts import (
    DuplicateTimestampPolicy,
    TgsimFoggyBottomAdapterConfig,
    UnknownLabelPolicy,
    _TgsimRow,
)


_REQUIRED_COLUMNS = frozenset(
    {
        "id",
        "time",
        "xloc_kf",
        "yloc_kf",
        "lane_kf",
        "speed_kf_x",
        "speed_kf_y",
        "acceleration_kf_x",
        "acceleration_kf_y",
        "length_smoothed",
        "width_smoothed",
        "type_most_common",
    }
)
_OPTIONAL_RUN_COLUMNS = ("run_index", "run_id", "recording_run")


def _native_track_labels() -> dict[str, TrackLabels]:
    return {
        "0": TrackLabels(
            native_label="0",
            normalized_class="person",
            mobility_family="pedestrian",
            operating_domain="urban_ground",
            evidence=LabelEvidence.NATIVE,
        ),
        "1": TrackLabels(
            native_label="1",
            normalized_class="bicycle",
            mobility_family="human_powered_two_wheel",
            operating_domain="urban_road",
            evidence=LabelEvidence.NATIVE,
        ),
        "2": TrackLabels(
            native_label="2",
            normalized_class="scooter",
            mobility_family="two_wheel_micro_mobility",
            operating_domain="urban_road",
            evidence=LabelEvidence.NATIVE,
        ),
        "3": TrackLabels(
            native_label="3",
            normalized_class="passenger_car",
            mobility_family="conventional_steering",
            operating_domain="urban_road",
            evidence=LabelEvidence.NATIVE,
        ),
        "4": TrackLabels(
            native_label="4",
            normalized_class="automated_vehicle",
            mobility_family="conventional_steering",
            operating_domain="urban_road",
            evidence=LabelEvidence.NATIVE,
            notes=("Automation status is semantic context, not a distinct mobility family.",),
        ),
        "5": TrackLabels(
            native_label="5",
            normalized_class="motorcycle",
            mobility_family="powered_two_wheel",
            operating_domain="urban_road",
            evidence=LabelEvidence.NATIVE,
        ),
        "6": TrackLabels(
            native_label="6",
            normalized_class="bus",
            mobility_family="conventional_steering_heavy",
            operating_domain="urban_road",
            evidence=LabelEvidence.NATIVE,
        ),
        "7": TrackLabels(
            native_label="7",
            normalized_class="truck",
            mobility_family="conventional_steering_heavy",
            operating_domain="urban_road",
            evidence=LabelEvidence.NATIVE,
        ),
    }
####


_NATIVE_TRACK_LABELS = _native_track_labels()


def _normalize_column_name(value: str) -> str:
    normalized = value.lstrip("\ufeff").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")
####


def _normalize_headers(fieldnames: list[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("TGSIM CSV has no header row")
    normalized_to_source: dict[str, str] = {}
    for source_name in fieldnames:
        normalized_name = _normalize_column_name(source_name)
        if normalized_name in normalized_to_source:
            raise ValueError(
                "TGSIM CSV contains duplicate normalized header "
                f"{normalized_name!r}"
            )
        normalized_to_source[normalized_name] = source_name
    missing = sorted(_REQUIRED_COLUMNS.difference(normalized_to_source))
    if missing:
        raise ValueError(f"TGSIM CSV is missing required columns: {', '.join(missing)}")
    return normalized_to_source
####


def _normalized_row(
    raw_row: dict[str | None, str | None],
    normalized_headers: dict[str, str],
) -> dict[str, str]:
    row: dict[str, str] = {}
    for normalized_name, source_name in normalized_headers.items():
        raw_value = raw_row.get(source_name)
        row[normalized_name] = "" if raw_value is None else raw_value.strip()
    return row
####


def _required_text(row: dict[str, str], column_name: str) -> str:
    value = row.get(column_name, "").strip()
    if not value:
        raise ValueError(f"column {column_name!r} is blank")
    return value
####


def _finite_float(row: dict[str, str], column_name: str) -> float:
    raw_value = _required_text(row, column_name)
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"column {column_name!r} is not numeric: {raw_value!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"column {column_name!r} must be finite")
    return value
####


def _canonical_identifier(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("identifier is blank")
    try:
        numeric = float(stripped)
    except ValueError:
        return stripped
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return stripped
####


def _canonical_native_label(value: str) -> str:
    return _canonical_identifier(value).lower()
####


def _source_run_id(row: dict[str, str], *, fallback: str) -> str:
    for column_name in _OPTIONAL_RUN_COLUMNS:
        value = row.get(column_name, "").strip()
        if value:
            return _canonical_identifier(value)
    return fallback
####


def _parse_row(
    row: dict[str, str],
    *,
    source_row_number: int,
    fallback_run_id: str,
) -> _TgsimRow:
    return _TgsimRow(
        source_row_number=source_row_number,
        track_id=_canonical_identifier(_required_text(row, "id")),
        run_id=_source_run_id(row, fallback=fallback_run_id),
        time_s=_finite_float(row, "time"),
        x_m=_finite_float(row, "xloc_kf"),
        y_m=_finite_float(row, "yloc_kf"),
        lane_id=_finite_float(row, "lane_kf"),
        velocity_x_mps=_finite_float(row, "speed_kf_x"),
        velocity_y_mps=_finite_float(row, "speed_kf_y"),
        acceleration_x_mps2=_finite_float(row, "acceleration_kf_x"),
        acceleration_y_mps2=_finite_float(row, "acceleration_kf_y"),
        length_m=_finite_float(row, "length_smoothed"),
        width_m=_finite_float(row, "width_smoothed"),
        native_label=_canonical_native_label(_required_text(row, "type_most_common")),
    )
####


def _resolve_duplicates(
    rows: list[_TgsimRow],
    *,
    policy: DuplicateTimestampPolicy,
) -> tuple[list[_TgsimRow], int]:
    sorted_rows = sorted(rows, key=lambda item: (item.time_s, item.source_row_number))
    by_timestamp: dict[float, list[_TgsimRow]] = {}
    for row in sorted_rows:
        by_timestamp.setdefault(row.time_s, []).append(row)

    duplicate_count = sum(len(items) - 1 for items in by_timestamp.values())
    if duplicate_count and policy is DuplicateTimestampPolicy.ERROR:
        duplicate_times = tuple(
            timestamp for timestamp, items in by_timestamp.items() if len(items) > 1
        )
        raise ValueError(
            "duplicate timestamps in TGSIM track: "
            + ", ".join(f"{value:.12g}" for value in duplicate_times[:5])
        )

    resolved: list[_TgsimRow] = []
    for timestamp in sorted(by_timestamp):
        candidates = by_timestamp[timestamp]
        if policy is DuplicateTimestampPolicy.KEEP_LAST:
            resolved.append(candidates[-1])
        else:
            resolved.append(candidates[0])
    return resolved, duplicate_count
####


def _unknown_track_labels(native_label: str) -> TrackLabels:
    safe_label = re.sub(r"[^a-z0-9]+", "_", native_label.lower()).strip("_")
    normalized_class = f"unknown_tgsim_type_{safe_label or 'blank'}"
    return TrackLabels(
        native_label=native_label,
        normalized_class=normalized_class,
        mobility_family="unknown",
        operating_domain="urban_ground",
        evidence=LabelEvidence.WEAK,
        notes=("Native TGSIM label was not present in the verified 0-7 data dictionary.",),
    )
####


def _select_track_labels(
    rows: list[_TgsimRow],
    *,
    config: TgsimFoggyBottomAdapterConfig,
) -> tuple[TrackLabels | None, bool]:
    native_counts = Counter(row.native_label for row in rows)
    inconsistent = len(native_counts) > 1
    if inconsistent and config.require_consistent_track_labels:
        raise ValueError(
            "TGSIM track contains inconsistent native labels: "
            + ", ".join(sorted(native_counts))
        )
    native_label = native_counts.most_common(1)[0][0]
    labels = _NATIVE_TRACK_LABELS.get(native_label)
    if labels is not None:
        return labels, inconsistent
    if config.unknown_label_policy is UnknownLabelPolicy.ERROR:
        raise ValueError(f"unknown TGSIM native label: {native_label!r}")
    if config.unknown_label_policy is UnknownLabelPolicy.SKIP:
        return None, inconsistent
    return _unknown_track_labels(native_label), inconsistent
####


__all__ = []
