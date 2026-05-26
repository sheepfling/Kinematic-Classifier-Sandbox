from __future__ import annotations

import unittest

import numpy as np

from kinematic_classifier_sandbox.advanced_filters.models_1d import (
    constant_velocity_transition,
    make_initial_particles_1d,
    position_gaussian_log_likelihood,
)
from kinematic_classifier_sandbox.advanced_filters.particle_filter import (
    BootstrapParticleFilter,
    ParticleFilterConfig,
)
from kinematic_classifier_sandbox.advanced_filters.resampling import (
    effective_sample_size,
    systematic_resample,
)


class ParticleFilterTests(unittest.TestCase):
    def test_pf_weights_normalize_and_output_state(self) -> None:
        rng = np.random.default_rng(3)
        config = ParticleFilterConfig(particle_count=128, seed=5)
        pf = BootstrapParticleFilter(
            config,
            transition_fn=lambda particles, dt, gen: constant_velocity_transition(particles, dt, gen, process_std=0.01),
            log_likelihood_fn=lambda particles, obs: position_gaussian_log_likelihood(particles, obs, measurement_std=0.2),
        )
        pf.reset("pf", make_initial_particles_1d(128, 0.0, 0.2, 1.0, 0.1, rng))
        step = pf.update(0.0, np.array([0.0], dtype=np.float64))
        assert pf.state is not None
        self.assertAlmostEqual(float(np.sum(np.exp(pf.state.log_weights))), 1.0)
        self.assertEqual(step.state_mean.shape, (2,))
        self.assertEqual(step.state_covariance.shape, (2, 2))

    def test_pf_ess_formula(self) -> None:
        weights = np.array([0.5, 0.25, 0.25], dtype=np.float64)
        self.assertAlmostEqual(effective_sample_size(weights), 1.0 / (0.25 + 0.0625 + 0.0625))

    def test_pf_resampling_preserves_particle_count(self) -> None:
        weights = np.array([0.7, 0.2, 0.1], dtype=np.float64)
        indexes = systematic_resample(weights, np.random.default_rng(7))
        self.assertEqual(len(indexes), 3)
        self.assertTrue(np.all(indexes >= 0))
        self.assertTrue(np.all(indexes < 3))

    def test_pf_seed_reproducibility(self) -> None:
        def run_once() -> tuple[float, float]:
            rng = np.random.default_rng(9)
            pf = BootstrapParticleFilter(
                ParticleFilterConfig(particle_count=64, seed=11),
                transition_fn=lambda particles, dt, gen: constant_velocity_transition(particles, dt, gen, process_std=0.02),
                log_likelihood_fn=lambda particles, obs: position_gaussian_log_likelihood(particles, obs, measurement_std=0.2),
            )
            pf.reset("pf", make_initial_particles_1d(64, 0.0, 0.1, 1.0, 0.1, rng))
            step = pf.update(0.0, np.array([0.05], dtype=np.float64))
            return float(step.state_mean[0]), float(step.log_marginal_likelihood)

        self.assertEqual(run_once(), run_once())


if __name__ == "__main__":
    unittest.main()
