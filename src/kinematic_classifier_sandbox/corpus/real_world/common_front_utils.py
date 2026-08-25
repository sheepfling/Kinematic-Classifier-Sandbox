"""Small, domain-neutral helpers for validation-only common-front builders."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from .episode_contracts import AssetReference, QualityFinding, QualitySummary


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_asset(
    root: str | Path,
    relative_path: str | Path,
    payload: Any,
) -> AssetReference:
    relative = Path(relative_path)
    destination = Path(root) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    destination.write_bytes(encoded)
    return AssetReference(
        path=relative.as_posix(),
        media_type="application/json",
        sha256=sha256_bytes(encoded),
    )


def opaque_group_id(*, dataset_id: str, namespace: str, raw_value: str) -> str:
    material = f"{dataset_id}:{namespace}:{raw_value}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:24]


def iso_utc_from_unix(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()


def quality_summary_from_elapsed(
    elapsed_s: Sequence[float],
    *,
    findings: Iterable[QualityFinding] = (),
    disposition: str = "accept_with_findings",
) -> QualitySummary:
    if not elapsed_s:
        raise ValueError("quality summary requires at least one elapsed timestamp")
    deltas = tuple(right - left for left, right in zip(elapsed_s, elapsed_s[1:]))
    positive = tuple(delta for delta in deltas if delta > 0.0)
    return QualitySummary(
        disposition=disposition,
        sample_count=len(elapsed_s),
        duration_s=max(elapsed_s) - min(elapsed_s),
        median_sample_interval_s=float(median(positive)) if positive else 0.0,
        maximum_gap_s=max(positive, default=0.0),
        duplicate_timestamp_count=len(elapsed_s) - len(set(elapsed_s)),
        out_of_order_timestamp_count=sum(delta < 0.0 for delta in deltas),
        findings=tuple(findings),
    )


__all__ = [
    "iso_utc_from_unix",
    "opaque_group_id",
    "quality_summary_from_elapsed",
    "sha256_bytes",
    "sha256_file",
    "write_json_asset",
]
