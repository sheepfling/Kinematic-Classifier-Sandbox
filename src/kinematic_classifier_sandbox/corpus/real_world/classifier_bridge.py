"""Bridge prepared Product 4 selections into the existing Product 2 ladder."""

from __future__ import annotations

import bisect
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from pydantic import Field, model_validator

from ...common_experiment.contracts import ExecutablePairSpec, ExecutableTrajectory
from .analysis_products import AnalysisProductId, AnalysisProductManifest
from .common_front_utils import sha256_file
from .episode_contracts import GroupingNamespace, StrictFrozenModel, TrajectoryEpisodeManifest
from .portfolio import (
    EpisodeSplitAssignment,
    SnapshotSplit,
    audit_split_assignments,
)


class RealWorldBridgeConfig(StrictFrozenModel):
    """Task and leakage policy for one Product 2 bridge evaluation."""

    pair_id: str
    class_a: str
    class_b: str
    target_label_namespace: str
    evaluation_split: SnapshotSplit = SnapshotSplit.TEST
    grouping_namespaces: tuple[GroupingNamespace, ...] = (
        GroupingNamespace.PHYSICAL_PLATFORM,
    )
    measurement_sigma_floor: float = 0.05
    max_episodes_per_group: int = Field(default=1, ge=1)

    def pair_spec(self) -> ExecutablePairSpec:
        return ExecutablePairSpec(
            pair_id=self.pair_id,
            class_a=self.class_a,
            class_b=self.class_b,
            expected_difficulty="real_world_bounded_holdout",
        )

    @model_validator(mode="after")
    def validate_policy(self) -> RealWorldBridgeConfig:
        if self.class_a == self.class_b:
            raise ValueError("real-world bridge classes must be distinct")
        if not self.grouping_namespaces:
            raise ValueError("real-world bridge requires at least one grouping namespace")
        if GroupingNamespace.PHYSICAL_PLATFORM not in self.grouping_namespaces:
            raise ValueError("real-world bridge must retain physical-platform grouping")
        if len(self.grouping_namespaces) != len(set(self.grouping_namespaces)):
            raise ValueError("real-world bridge grouping namespaces must be unique")
        return self
    ####
####


@dataclass(frozen=True, slots=True)
class BridgeTrajectory:
    episode_id: str
    source_dataset_id: str
    split: SnapshotSplit
    true_class: str
    trajectory: ExecutableTrajectory
####


@dataclass(frozen=True, slots=True)
class RealWorldBridgeSelection:
    config: RealWorldBridgeConfig
    pair_spec: ExecutablePairSpec
    trajectories_by_split: Mapping[SnapshotSplit, tuple[BridgeTrajectory, ...]]
    grouping_audit_passes: bool
    grouping_audit_issues: tuple[str, ...]

    def trajectories(self, split: SnapshotSplit) -> tuple[ExecutableTrajectory, ...]:
        return tuple(item.trajectory for item in self.trajectories_by_split.get(split, ()))

    def class_counts(self, split: SnapshotSplit) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for item in self.trajectories_by_split.get(split, ()):
            counts[item.true_class] = counts.get(item.true_class, 0) + 1
        return dict(sorted(counts.items()))
    ####
####


def _derivative(values: tuple[float, ...], times: tuple[float, ...]) -> tuple[float, ...]:
    if len(values) < 2:
        return tuple(0.0 for _ in values)
    slopes = tuple(
        (right - left) / max(right_time - left_time, 1e-12)
        for left, right, left_time, right_time in zip(
            values,
            values[1:],
            times,
            times[1:],
        )
    )
    return (slopes[0],) + tuple(
        (left + right) / 2.0 for left, right in zip(slopes, slopes[1:])
    ) + (slopes[-1],)
####


def _interpolate(times: tuple[float, ...], values: tuple[float, ...], target: float) -> float:
    if target <= times[0]:
        return values[0]
    if target >= times[-1]:
        return values[-1]
    right = bisect.bisect_right(times, target)
    left = right - 1
    fraction = (target - times[left]) / (times[right] - times[left])
    return values[left] + fraction * (values[right] - values[left])
####


def _empirical_reference_builder(
    train_trajectories: tuple[ExecutableTrajectory, ...],
) -> Callable[..., ExecutableTrajectory]:
    by_class: dict[str, tuple[ExecutableTrajectory, ...]] = {}
    for trajectory in train_trajectories:
        by_class.setdefault(trajectory.true_class, ())
        by_class[trajectory.true_class] += (trajectory,)
    if not by_class or any(not values for values in by_class.values()):
        raise ValueError("empirical Product 2 reference requires train trajectories for both classes")

    def build_reference(
        pair_spec: ExecutablePairSpec,
        class_name: str,
        scenario_id: str,
        times: tuple[float, ...],
    ) -> ExecutableTrajectory:
        references = by_class.get(class_name)
        if references is None:
            raise ValueError(f"no train reference trajectory for class: {class_name}")
        measurements = tuple(
            statistics.median(
                _interpolate(reference.times, reference.measurements, target)
                for reference in references
            )
            for target in times
        )
        velocity = _derivative(measurements, times)
        acceleration = _derivative(velocity, times)
        return ExecutableTrajectory(
            trajectory_id=f"reference:{pair_spec.pair_id}:{class_name}",
            class_pair_id=pair_spec.pair_id,
            class_a=pair_spec.class_a,
            class_b=pair_spec.class_b,
            true_class=class_name,
            scenario_id=scenario_id,
            seed=0,
            times=times,
            measurements=measurements,
            true_position=measurements,
            true_velocity=velocity,
            true_acceleration=acceleration,
            measurement_dim=1,
            coordinate_frame="scalar_projection",
        )

    return build_reference
####


def _empirical_measurement_sigma(
    train_trajectories: tuple[ExecutableTrajectory, ...],
    reference_builder: Callable[..., ExecutableTrajectory],
    pair_spec: ExecutablePairSpec,
    floor: float,
) -> Callable[[str], float]:
    residuals: list[float] = []
    for trajectory in train_trajectories:
        reference = reference_builder(
            pair_spec,
            trajectory.true_class,
            trajectory.scenario_id,
            trajectory.times,
        )
        residuals.extend(
            abs(measurement - expected)
            for measurement, expected in zip(
                trajectory.measurements,
                reference.measurements,
                strict=True,
            )
        )
    estimate = statistics.median(residuals) if residuals else floor
    sigma = max(floor, float(estimate))
    return lambda _scenario_id: sigma
####


def _load_classifier_asset(
    root: Path,
    *,
    episode: TrajectoryEpisodeManifest,
    asset_path: str,
    asset_sha256: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    path = root / asset_path
    if not path.is_file():
        raise ValueError(f"classifier asset is missing: {asset_path}")
    if sha256_file(path) != asset_sha256:
        raise ValueError(f"classifier asset hash mismatch: {asset_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"measurements", "timestamps_s"}:
        raise ValueError(
            "classifier asset must contain only timestamps_s and measurements: "
            f"{episode.episode_id}"
        )
    times = tuple(float(value) for value in payload["timestamps_s"])
    measurements = tuple(float(value) for value in payload["measurements"])
    if len(times) < 2 or len(times) != len(measurements):
        raise ValueError(f"classifier asset has invalid sample lengths: {episode.episode_id}")
    if any(not math.isfinite(value) for value in times + measurements):
        raise ValueError(f"classifier asset contains non-finite values: {episode.episode_id}")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(f"classifier asset timestamps are not strictly increasing: {episode.episode_id}")
    return times, measurements
####


def _trajectory_from_asset(
    *,
    episode: TrajectoryEpisodeManifest,
    times: tuple[float, ...],
    measurements: tuple[float, ...],
    target_label: str,
    pair_spec: ExecutablePairSpec,
    split: SnapshotSplit,
) -> BridgeTrajectory:
    velocity = _derivative(measurements, times)
    acceleration = _derivative(velocity, times)
    trajectory = ExecutableTrajectory(
        trajectory_id=episode.episode_id,
        class_pair_id=pair_spec.pair_id,
        class_a=pair_spec.class_a,
        class_b=pair_spec.class_b,
        true_class=target_label,
        scenario_id="real_world_prepared",
        seed=0,
        times=times,
        measurements=measurements,
        true_position=measurements,
        true_velocity=velocity,
        true_acceleration=acceleration,
        measurement_dim=1,
        coordinate_frame="scalar_projection",
    )
    return BridgeTrajectory(
        episode_id=episode.episode_id,
        source_dataset_id=episode.source_dataset_id,
        split=split,
        true_class=target_label,
        trajectory=trajectory,
    )
####


def build_real_world_bridge_selection(
    *,
    snapshot_root: str | Path,
    analysis_manifest: AnalysisProductManifest,
    episodes: tuple[TrajectoryEpisodeManifest, ...],
    assignments: tuple[EpisodeSplitAssignment, ...],
    config: RealWorldBridgeConfig,
) -> RealWorldBridgeSelection:
    """Load a classifier-only selection and keep its split/label boundary explicit."""

    if analysis_manifest.policy.product_id is not AnalysisProductId.CLASSIFIER_LADDER:
        raise ValueError("Product 2 bridge requires a classifier_ladder analysis manifest")
    if analysis_manifest.policy.target_label_namespace != config.target_label_namespace:
        raise ValueError("bridge target label namespace does not match analysis manifest")
    episodes_by_id = {episode.episode_id: episode for episode in episodes}
    assignment_by_id = {assignment.episode_id: assignment.split for assignment in assignments}
    selected: list[BridgeTrajectory] = []
    root = Path(snapshot_root)
    for selection in analysis_manifest.episodes:
        episode = episodes_by_id.get(selection.episode_id)
        if episode is None:
            raise ValueError(f"analysis manifest episode is not loaded: {selection.episode_id}")
        split = assignment_by_id.get(episode.episode_id)
        if split is None:
            raise ValueError(f"bridge episode has no split assignment: {episode.episode_id}")
        classifier = episode.classifier_trajectory_view
        if classifier is None or selection.classifier_asset != classifier.asset:
            raise ValueError(f"analysis manifest classifier asset mismatch: {episode.episode_id}")
        labels = tuple(
            label
            for label in episode.labels
            if label.namespace == config.target_label_namespace
        )
        if len(labels) != 1:
            raise ValueError(
                f"bridge requires exactly one target label per episode: {episode.episode_id}"
            )
        label = labels[0]
        if label.proxy:
            raise ValueError(f"bridge rejects proxy target labels: {episode.episode_id}")
        if label.value not in {config.class_a, config.class_b}:
            raise ValueError(f"bridge target label is outside pair: {episode.episode_id}")
        times, measurements = _load_classifier_asset(
            root,
            episode=episode,
            asset_path=selection.classifier_asset.path,
            asset_sha256=selection.classifier_asset.sha256,
        )
        selected.append(
            _trajectory_from_asset(
                episode=episode,
                times=times,
                measurements=measurements,
                target_label=label.value,
                pair_spec=config.pair_spec(),
                split=split,
            )
        )

    selected_ids = {item.episode_id for item in selected}
    selected_episodes = tuple(episodes_by_id[item.episode_id] for item in selected)
    selected_assignments = tuple(
        assignment for assignment in assignments if assignment.episode_id in selected_ids
    )
    grouping_audit = audit_split_assignments(
        selected_episodes,
        selected_assignments,
        grouping_namespaces=config.grouping_namespaces,
    )
    by_split: dict[SnapshotSplit, list[BridgeTrajectory]] = {
        split: [] for split in SnapshotSplit
    }
    for item in selected:
        by_split[item.split].append(item)
    if config.max_episodes_per_group == 1:
        selected_by_group: dict[tuple[SnapshotSplit, str], BridgeTrajectory] = {}
        for item in sorted(selected, key=lambda candidate: candidate.episode_id):
            episode = episodes_by_id[item.episode_id]
            physical_group = next(
                key.opaque_value
                for key in episode.grouping_keys
                if key.namespace is GroupingNamespace.PHYSICAL_PLATFORM
            )
            selected_by_group.setdefault((item.split, physical_group), item)
        by_split = {split: [] for split in SnapshotSplit}
        for item in selected_by_group.values():
            by_split[item.split].append(item)
    elif config.max_episodes_per_group > 1:
        selected_by_group_count: dict[tuple[SnapshotSplit, str], int] = {}
        capped: dict[SnapshotSplit, list[BridgeTrajectory]] = {
            split: [] for split in SnapshotSplit
        }
        for item in sorted(selected, key=lambda candidate: candidate.episode_id):
            episode = episodes_by_id[item.episode_id]
            physical_group = next(
                key.opaque_value
                for key in episode.grouping_keys
                if key.namespace is GroupingNamespace.PHYSICAL_PLATFORM
            )
            group_key = (item.split, physical_group)
            count = selected_by_group_count.get(group_key, 0)
            if count >= config.max_episodes_per_group:
                continue
            selected_by_group_count[group_key] = count + 1
            capped[item.split].append(item)
        by_split = capped
    for split in SnapshotSplit:
        if not by_split[split]:
            raise ValueError(f"bridge selection is missing {split.value} trajectories")
        classes = {item.true_class for item in by_split[split]}
        if classes != {config.class_a, config.class_b}:
            raise ValueError(f"bridge selection requires both classes in {split.value}")
    return RealWorldBridgeSelection(
        config=config,
        pair_spec=config.pair_spec(),
        trajectories_by_split={
            split: tuple(sorted(items, key=lambda item: item.episode_id))
            for split, items in by_split.items()
        },
        grouping_audit_passes=grouping_audit.passes,
        grouping_audit_issues=grouping_audit.issues,
    )


def build_empirical_product2_hooks(
    selection: RealWorldBridgeSelection,
) -> tuple[Callable[..., ExecutableTrajectory], Callable[[str], float]]:
    """Build Product 2 hooks from train only; validation/test remain held out."""

    train = selection.trajectories(SnapshotSplit.TRAIN)
    reference_builder = _empirical_reference_builder(train)
    measurement_sigma = _empirical_measurement_sigma(
        train,
        reference_builder,
        selection.pair_spec,
        selection.config.measurement_sigma_floor,
    )
    return reference_builder, measurement_sigma


__all__ = [
    "BridgeTrajectory",
    "RealWorldBridgeConfig",
    "RealWorldBridgeSelection",
    "build_empirical_product2_hooks",
    "build_real_world_bridge_selection",
]
