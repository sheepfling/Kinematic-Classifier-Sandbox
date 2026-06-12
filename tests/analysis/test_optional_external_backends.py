from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy

from kinematic_classifier_sandbox.analysis import optional_external_backends


class OptionalExternalBackendEnvironmentTests(unittest.TestCase):
    def test_optional_backend_environment_sets_local_numba_cache_dir(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            optional_external_backends._ensure_optional_backend_environment()
            self.assertEqual(os.environ["NUMBA_DISABLE_JIT"], "1")
            self.assertTrue(os.environ["NUMBA_CACHE_DIR"])
            self.assertTrue(os.environ["NUMBA_CACHE_DIR"].startswith(tempfile.gettempdir()))

    def test_optional_backend_environment_preserves_existing_numba_cache_dir(self) -> None:
        with patch.dict(
            os.environ,
            {"NUMBA_CACHE_DIR": "/tmp/custom-numba-cache"},
            clear=True,
        ):
            optional_external_backends._ensure_optional_backend_environment()
            self.assertEqual(os.environ["NUMBA_DISABLE_JIT"], "1")
            self.assertEqual(os.environ["NUMBA_CACHE_DIR"], "/tmp/custom-numba-cache")

    def test_archive_adapter_predict_many_uses_single_probability_batch(self) -> None:
        class StubModel:
            def predict_proba(self, panel):
                self.last_shape = panel.shape
                return numpy.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=float)

        trajectories = (
            optional_external_backends.SharedDynamicsTrajectory(
                trajectory_id="t0",
                true_class="constant_velocity",
                scenario_name="easy",
                seed=0,
                times=(0.0, 1.0, 2.0),
                measurements=(0.0, 1.0, 2.0),
                true_position=(0.0, 1.0, 2.0),
                true_velocity=(1.0, 1.0, 1.0),
                true_acceleration=(0.0, 0.0, 0.0),
            ),
            optional_external_backends.SharedDynamicsTrajectory(
                trajectory_id="t1",
                true_class="constant_acceleration",
                scenario_name="easy",
                seed=1,
                times=(0.0, 1.0, 2.0),
                measurements=(0.0, 0.5, 2.0),
                true_position=(0.0, 0.5, 2.0),
                true_velocity=(0.0, 1.0, 2.0),
                true_acceleration=(1.0, 1.0, 1.0),
            ),
        )
        model = StubModel()
        adapter = optional_external_backends.ArchiveClassifierAdapter(
            method_family="stub",
            backend_name="stub",
            model=model,
            class_names=("constant_velocity", "constant_acceleration"),
            resample_length=16,
            panel_variant="normalized_position",
        )

        predictions = adapter.predict_many(trajectories)

        self.assertEqual(model.last_shape[0], 2)
        self.assertEqual(
            predictions,
            (
                ("constant_velocity", 0.8),
                ("constant_acceleration", 0.9),
            ),
        )

    def test_archive_adapter_reorders_probability_columns_to_match_class_names(self) -> None:
        class StubModel:
            classes_ = numpy.asarray(["constant_acceleration", "constant_velocity"], dtype=object)

            def predict_proba(self, panel):
                self.last_shape = panel.shape
                return numpy.asarray([[0.2, 0.8], [0.9, 0.1]], dtype=float)

        trajectories = (
            optional_external_backends.SharedDynamicsTrajectory(
                trajectory_id="t0",
                true_class="constant_velocity",
                scenario_name="easy",
                seed=0,
                times=(0.0, 1.0, 2.0),
                measurements=(0.0, 1.0, 2.0),
                true_position=(0.0, 1.0, 2.0),
                true_velocity=(1.0, 1.0, 1.0),
                true_acceleration=(0.0, 0.0, 0.0),
            ),
            optional_external_backends.SharedDynamicsTrajectory(
                trajectory_id="t1",
                true_class="constant_acceleration",
                scenario_name="easy",
                seed=1,
                times=(0.0, 1.0, 2.0),
                measurements=(0.0, 0.5, 2.0),
                true_position=(0.0, 0.5, 2.0),
                true_velocity=(0.0, 1.0, 2.0),
                true_acceleration=(1.0, 1.0, 1.0),
            ),
        )
        adapter = optional_external_backends.ArchiveClassifierAdapter(
            method_family="stub",
            backend_name="stub",
            model=StubModel(),
            class_names=("constant_velocity", "constant_acceleration"),
            resample_length=16,
            panel_variant="normalized_position",
        )

        predictions = adapter.predict_many(trajectories)

        self.assertEqual(
            predictions,
            (
                ("constant_velocity", 0.8),
                ("constant_acceleration", 0.9),
            ),
        )

    def test_archive_compatible_panel_is_batch_order_invariant(self) -> None:
        first = optional_external_backends.SharedDynamicsTrajectory(
            trajectory_id="t0",
            true_class="constant_velocity",
            scenario_name="easy",
            seed=0,
            times=(0.0, 1.0, 2.0),
            measurements=(0.0, 1.0, 2.0),
            true_position=(0.0, 1.0, 2.0),
            true_velocity=(1.0, 1.0, 1.0),
            true_acceleration=(0.0, 0.0, 0.0),
        )
        second = optional_external_backends.SharedDynamicsTrajectory(
            trajectory_id="t1",
            true_class="constant_acceleration",
            scenario_name="easy",
            seed=1,
            times=(0.0, 1.0, 2.0),
            measurements=(0.0, 0.5, 2.0),
            true_position=(0.0, 0.5, 2.0),
            true_velocity=(0.0, 1.0, 2.0),
            true_acceleration=(1.0, 1.0, 1.0),
        )

        forward = optional_external_backends._archive_compatible_panel((first, second), resample_length=16)
        reverse = optional_external_backends._archive_compatible_panel((second, first), resample_length=16)
        first_single = optional_external_backends._archive_compatible_panel((first,), resample_length=16)
        second_single = optional_external_backends._archive_compatible_panel((second,), resample_length=16)

        self.assertTrue(numpy.allclose(forward[0], first_single[0]))
        self.assertTrue(numpy.allclose(forward[1], second_single[0]))
        self.assertTrue(numpy.allclose(reverse[1], first_single[0]))
        self.assertTrue(numpy.allclose(reverse[0], second_single[0]))

    def test_archive_panel_variant_adds_expected_channels(self) -> None:
        trajectory = optional_external_backends.SharedDynamicsTrajectory(
            trajectory_id="t0",
            true_class="constant_velocity",
            scenario_name="easy",
            seed=0,
            times=(0.0, 1.0, 2.0, 3.0),
            measurements=(0.0, 1.0, 2.0, 3.0),
            true_position=(0.0, 1.0, 2.0, 3.0),
            true_velocity=(1.0, 1.0, 1.0, 1.0),
            true_acceleration=(0.0, 0.0, 0.0, 0.0),
        )

        position_panel = optional_external_backends._archive_compatible_panel(
            (trajectory,),
            resample_length=16,
            panel_variant="normalized_position",
        )
        pv_panel = optional_external_backends._archive_compatible_panel(
            (trajectory,),
            resample_length=16,
            panel_variant="normalized_position_velocity",
        )
        pva_panel = optional_external_backends._archive_compatible_panel(
            (trajectory,),
            resample_length=16,
            panel_variant="normalized_position_velocity_acceleration",
        )

        self.assertEqual(position_panel.shape[1], 1)
        self.assertEqual(pv_panel.shape[1], 2)
        self.assertEqual(pva_panel.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
