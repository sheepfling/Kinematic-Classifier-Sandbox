from __future__ import annotations

import pytest

from kinematic_classifier_sandbox.common_experiment.contracts import ExecutableTrajectory
from kinematic_classifier_sandbox.corpus.real_world.classifier_bridge import (
    BridgeTrajectory,
    RealWorldBridgeConfig,
    RealWorldBridgeSelection,
    build_empirical_product2_hooks,
)
from kinematic_classifier_sandbox.corpus.real_world.episode_contracts import GroupingNamespace
from kinematic_classifier_sandbox.corpus.real_world.portfolio import SnapshotSplit

pytestmark = pytest.mark.product4_classifier_ladder


def _trajectory(
    *,
    trajectory_id: str,
    true_class: str,
    measurements: tuple[float, ...],
) -> ExecutableTrajectory:
    times = tuple(float(index) for index in range(len(measurements)))
    return ExecutableTrajectory(
        trajectory_id=trajectory_id,
        class_pair_id="passenger_car_vs_truck",
        class_a="passenger_car",
        class_b="truck",
        true_class=true_class,
        scenario_id="real_world_prepared",
        seed=0,
        times=times,
        measurements=measurements,
        true_position=measurements,
        true_velocity=(0.0,) * len(measurements),
        true_acceleration=(0.0,) * len(measurements),
        measurement_dim=1,
        coordinate_frame="scalar_projection",
    )


def _bridge_item(
    *,
    episode_id: str,
    split: SnapshotSplit,
    true_class: str,
    measurements: tuple[float, ...],
) -> BridgeTrajectory:
    return BridgeTrajectory(
        episode_id=episode_id,
        source_dataset_id="tgsim_foggy_bottom",
        split=split,
        true_class=true_class,
        trajectory=_trajectory(
            trajectory_id=episode_id,
            true_class=true_class,
            measurements=measurements,
        ),
    )


def test_bridge_hooks_use_train_only_empirical_references() -> None:
    config = RealWorldBridgeConfig(
        pair_id="passenger_car_vs_truck",
        class_a="passenger_car",
        class_b="truck",
        target_label_namespace="platform_class",
        grouping_namespaces=(GroupingNamespace.PHYSICAL_PLATFORM,),
    )
    train = (
        _bridge_item(
            episode_id="train-car",
            split=SnapshotSplit.TRAIN,
            true_class="passenger_car",
            measurements=(1.0, 1.0, 1.0),
        ),
        _bridge_item(
            episode_id="train-truck",
            split=SnapshotSplit.TRAIN,
            true_class="truck",
            measurements=(3.0, 3.0, 3.0),
        ),
    )
    validation = (
        _bridge_item(
            episode_id="validation-car",
            split=SnapshotSplit.VALIDATION,
            true_class="passenger_car",
            measurements=(100.0, 100.0, 100.0),
        ),
        _bridge_item(
            episode_id="validation-truck",
            split=SnapshotSplit.VALIDATION,
            true_class="truck",
            measurements=(300.0, 300.0, 300.0),
        ),
    )
    test = (
        _bridge_item(
            episode_id="test-car",
            split=SnapshotSplit.TEST,
            true_class="passenger_car",
            measurements=(500.0, 500.0, 500.0),
        ),
        _bridge_item(
            episode_id="test-truck",
            split=SnapshotSplit.TEST,
            true_class="truck",
            measurements=(700.0, 700.0, 700.0),
        ),
    )
    selection = RealWorldBridgeSelection(
        config=config,
        pair_spec=config.pair_spec(),
        trajectories_by_split={
            SnapshotSplit.TRAIN: train,
            SnapshotSplit.VALIDATION: validation,
            SnapshotSplit.TEST: test,
        },
        grouping_audit_passes=True,
        grouping_audit_issues=(),
    )

    reference_builder, measurement_sigma = build_empirical_product2_hooks(selection)
    reference = reference_builder(
        config.pair_spec(),
        "passenger_car",
        "real_world_prepared",
        (0.0, 1.0, 2.0),
    )

    assert reference.measurements == (1.0, 1.0, 1.0)
    assert measurement_sigma("real_world_prepared") == config.measurement_sigma_floor
