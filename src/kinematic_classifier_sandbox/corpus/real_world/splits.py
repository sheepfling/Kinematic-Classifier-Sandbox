from __future__ import annotations

import hashlib
from enum import StrEnum
from math import floor
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import NormalizedTrack
from .windowing import TrackWindow


class DatasetPartition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
####


class SplitRatios(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    train: float = Field(default=0.7, gt=0.0, lt=1.0)
    validation: float = Field(default=0.15, ge=0.0, lt=1.0)
    test: float = Field(default=0.15, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-12:
            raise ValueError("split ratios must sum to one")
        return self
    ####
####


class GroupSplitPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: str = Field(default="real-world-land-vehicle-v1", min_length=1)
    ratios: SplitRatios = Field(default_factory=SplitRatios)
    stratify_by_class: bool = True
####


class SplitAssignment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split_group_id: str = Field(min_length=1)
    normalized_class: str = Field(min_length=1)
    partition: DatasetPartition
    hash_fraction: float = Field(ge=0.0, lt=1.0)
####


class SplitSummaryRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    partition: DatasetPartition
    normalized_class: str = Field(min_length=1)
    group_count: int = Field(ge=0)
    window_count: int = Field(ge=0)
####


class DatasetSplit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: GroupSplitPolicy
    assignments: tuple[SplitAssignment, ...]
    summary_rows: tuple[SplitSummaryRow, ...]

    @model_validator(mode="after")
    def validate_assignments(self) -> Self:
        group_ids = tuple(assignment.split_group_id for assignment in self.assignments)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("split groups must have exactly one assignment")
        return self
    ####

    def partition_for(self, split_group_id: str) -> DatasetPartition:
        for assignment in self.assignments:
            if assignment.split_group_id == split_group_id:
                return assignment.partition
        raise KeyError(split_group_id)
    ####
####


def _hash_fraction(
    *,
    split_group_id: str,
    normalized_class: str,
    policy: GroupSplitPolicy,
) -> float:
    class_component = normalized_class if policy.stratify_by_class else "*"
    payload = f"{policy.seed}|{class_component}|{split_group_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    numerator = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return numerator / float(1 << 64)
####


def _partition_counts(
    group_count: int,
    *,
    ratios: SplitRatios,
) -> dict[DatasetPartition, int]:
    ratio_by_partition = {
        DatasetPartition.TRAIN: ratios.train,
        DatasetPartition.VALIDATION: ratios.validation,
        DatasetPartition.TEST: ratios.test,
    }
    raw_counts = {
        partition: group_count * ratio
        for partition, ratio in ratio_by_partition.items()
    }
    counts = {
        partition: int(floor(raw_count))
        for partition, raw_count in raw_counts.items()
    }
    remaining = group_count - sum(counts.values())
    remainder_order = sorted(
        ratio_by_partition,
        key=lambda partition: (
            -(raw_counts[partition] - counts[partition]),
            tuple(DatasetPartition).index(partition),
        ),
    )
    for partition in remainder_order[:remaining]:
        counts[partition] += 1

    active = tuple(
        partition
        for partition, ratio in ratio_by_partition.items()
        if ratio > 0.0
    )
    if group_count >= len(active):
        for partition in active:
            if counts[partition] > 0:
                continue
            donors = sorted(
                (
                    candidate
                    for candidate in active
                    if counts[candidate] > 1
                ),
                key=lambda candidate: (
                    -counts[candidate],
                    tuple(DatasetPartition).index(candidate),
                ),
            )
            if not donors:
                break
            counts[donors[0]] -= 1
            counts[partition] += 1
    return counts
####


def _group_labels(tracks: tuple[NormalizedTrack, ...]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for track in tracks:
        split_group_id = track.provenance.split_group_id
        normalized_class = track.labels.normalized_class
        existing = labels.get(split_group_id)
        if existing is not None and existing != normalized_class:
            raise ValueError(
                f"split group {split_group_id!r} contains conflicting normalized classes"
            )
        labels[split_group_id] = normalized_class
    return labels
####


def _assignment_groups(
    labels_by_group: dict[str, str],
    *,
    policy: GroupSplitPolicy,
) -> tuple[SplitAssignment, ...]:
    buckets: dict[str, list[tuple[str, float]]] = {}
    for split_group_id, normalized_class in labels_by_group.items():
        bucket_key = normalized_class if policy.stratify_by_class else "*"
        fraction = _hash_fraction(
            split_group_id=split_group_id,
            normalized_class=normalized_class,
            policy=policy,
        )
        buckets.setdefault(bucket_key, []).append((split_group_id, fraction))

    assignments: list[SplitAssignment] = []
    for bucket_key in sorted(buckets):
        group_rows = sorted(buckets[bucket_key], key=lambda item: (item[1], item[0]))
        counts = _partition_counts(len(group_rows), ratios=policy.ratios)
        boundaries = (
            (DatasetPartition.TRAIN, counts[DatasetPartition.TRAIN]),
            (DatasetPartition.VALIDATION, counts[DatasetPartition.VALIDATION]),
            (DatasetPartition.TEST, counts[DatasetPartition.TEST]),
        )
        cursor = 0
        for partition, count in boundaries:
            for split_group_id, fraction in group_rows[cursor : cursor + count]:
                assignments.append(
                    SplitAssignment(
                        split_group_id=split_group_id,
                        normalized_class=labels_by_group[split_group_id],
                        partition=partition,
                        hash_fraction=fraction,
                    )
                )
            cursor += count

    return tuple(sorted(assignments, key=lambda item: item.split_group_id))
####


def _summary_rows(
    assignments: tuple[SplitAssignment, ...],
    *,
    windows: tuple[TrackWindow, ...],
) -> tuple[SplitSummaryRow, ...]:
    assignment_by_group = {
        assignment.split_group_id: assignment for assignment in assignments
    }
    window_counts: dict[str, int] = {split_group_id: 0 for split_group_id in assignment_by_group}
    for window in windows:
        if window.split_group_id not in assignment_by_group:
            raise ValueError(
                f"window {window.window_id!r} refers to an unassigned split group"
            )
        window_counts[window.split_group_id] += 1

    classes = sorted({assignment.normalized_class for assignment in assignments})
    rows: list[SplitSummaryRow] = []
    for partition in DatasetPartition:
        for normalized_class in classes:
            matching = tuple(
                assignment
                for assignment in assignments
                if assignment.partition is partition
                and assignment.normalized_class == normalized_class
            )
            rows.append(
                SplitSummaryRow(
                    partition=partition,
                    normalized_class=normalized_class,
                    group_count=len(matching),
                    window_count=sum(
                        window_counts[assignment.split_group_id]
                        for assignment in matching
                    ),
                )
            )
    return tuple(rows)
####


def assign_grouped_split(
    tracks: tuple[NormalizedTrack, ...],
    *,
    windows: tuple[TrackWindow, ...] = (),
    policy: GroupSplitPolicy | None = None,
) -> DatasetSplit:
    effective_policy = policy or GroupSplitPolicy()
    labels_by_group = _group_labels(tracks)
    assignments = _assignment_groups(labels_by_group, policy=effective_policy)
    return DatasetSplit(
        policy=effective_policy,
        assignments=assignments,
        summary_rows=_summary_rows(assignments, windows=windows),
    )
####


__all__ = [
    "DatasetPartition",
    "DatasetSplit",
    "GroupSplitPolicy",
    "SplitAssignment",
    "SplitRatios",
    "SplitSummaryRow",
    "assign_grouped_split",
]
