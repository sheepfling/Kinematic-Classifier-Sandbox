from __future__ import annotations

import unittest

import numpy as np

from kinematic_classifier_sandbox.advanced_filters.contracts import validate_advanced_filter_step
from kinematic_classifier_sandbox.advanced_filters.models_1d import (
    constant_velocity_transition,
    make_initial_particles_1d,
    nonlinear_drag_transition,
    position_gaussian_log_likelihood,
)
from kinematic_classifier_sandbox.advanced_filters.particle_filter import BootstrapParticleFilter, ParticleFilterConfig
from kinematic_classifier_sandbox.advanced_filters.particle_filter_bank import ParticleFilterBank


class ParticleFilterBankTests(unittest.TestCase):
    def test_pf_bank_posteriors_sum_to_one(self) -> None:
        rng = np.random.default_rng(13)
        filters = {
            "constant_velocity": BootstrapParticleFilter(
                ParticleFilterConfig(particle_count=96, seed=1),
                transition_fn=lambda particles, dt, gen: constant_velocity_transition(particles, dt, gen),
                log_likelihood_fn=lambda particles, obs: position_gaussian_log_likelihood(particles, obs, 0.2),
            ),
            "nonlinear_drag": BootstrapParticleFilter(
                ParticleFilterConfig(particle_count=96, seed=2),
                transition_fn=lambda particles, dt, gen: nonlinear_drag_transition(particles, dt, gen),
                log_likelihood_fn=lambda particles, obs: position_gaussian_log_likelihood(particles, obs, 0.2),
            ),
        }
        bank = ParticleFilterBank(filters)
        initial = {
            label: make_initial_particles_1d(96, 0.0, 0.2, 1.0, 0.2, rng)
            for label in filters
        }
        bank.reset("traj", initial)
        step = bank.update(0.0, np.array([0.0], dtype=np.float64))
        validate_advanced_filter_step(step)
        self.assertAlmostEqual(sum(step.posterior_by_label.values()), 1.0)
        self.assertEqual(step.confidence, max(step.posterior_by_label.values()))
        self.assertEqual(set(step.log_evidence_by_label), set(filters))


if __name__ == "__main__":
    unittest.main()
