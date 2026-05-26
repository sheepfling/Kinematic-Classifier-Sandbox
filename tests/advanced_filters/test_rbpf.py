from __future__ import annotations

import unittest

import numpy as np

from kinematic_classifier_sandbox.advanced_filters.contracts import validate_advanced_filter_step
from kinematic_classifier_sandbox.advanced_filters.rbpf import RBPFConfig, RaoBlackwellizedParticleFilter
from kinematic_classifier_sandbox.advanced_filters.rbpf_models_1d import (
    default_mode_transition_matrix_1d,
    make_rbpf_1d_mode_models,
)


class RBPFTests(unittest.TestCase):
    def _make_filter(self) -> RaoBlackwellizedParticleFilter:
        particle_count = 128
        rbpf = RaoBlackwellizedParticleFilter(
            RBPFConfig(particle_count=particle_count, seed=19),
            make_rbpf_1d_mode_models(dt=1.0, measurement_std=0.2),
            default_mode_transition_matrix_1d(),
        )
        modes = np.zeros(particle_count, dtype=np.int64)
        means = np.zeros((particle_count, 3), dtype=np.float64)
        covariances = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], particle_count, axis=0)
        rbpf.reset("rbpf", modes, means, covariances)
        return rbpf

    def test_rbpf_mode_posterior_sums_to_one(self) -> None:
        rbpf = self._make_filter()
        step = rbpf.update(0.0, np.array([0.0], dtype=np.float64))
        validate_advanced_filter_step(step)
        self.assertAlmostEqual(sum(step.posterior_by_label.values()), 1.0)
        self.assertEqual(set(step.log_evidence_by_label), set(step.posterior_by_label))
        self.assertGreater(len(set(step.log_evidence_by_label.values())), 1)

    def test_rbpf_resampling_preserves_particle_count_and_covariances_are_psd(self) -> None:
        rbpf = self._make_filter()
        for index in range(3):
            rbpf.update(float(index), np.array([0.2 * index], dtype=np.float64))
        assert rbpf.state is not None
        self.assertEqual(len(rbpf.state.mode_indexes), rbpf.config.particle_count)
        for covariance in rbpf.state.covariances[:10]:
            self.assertTrue(np.allclose(covariance, covariance.T))
            self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(covariance))), -1.0e-8)

    def test_rbpf_latent_posterior_changes_after_event(self) -> None:
        rbpf = self._make_filter()
        first = rbpf.update(0.0, np.array([0.0], dtype=np.float64))
        for index in range(1, 6):
            latest = rbpf.update(float(index), np.array([0.4 * index * index], dtype=np.float64))
        self.assertNotEqual(first.predicted_label, "")
        self.assertGreater(max(latest.posterior_by_label.values()), 0.25)


if __name__ == "__main__":
    unittest.main()
