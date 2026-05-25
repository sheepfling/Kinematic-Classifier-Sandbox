from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import AdvancedFilterStep
from .particle_filter import BootstrapParticleFilter
from .resampling import normalize_log_weights


@dataclass(slots=True)
class ParticleFilterBankState:
    trajectory_id: str
    label_ids: list[str]
    log_posterior: np.ndarray


def particle_filter_class_evidence_update(
    prior_log_posterior: np.ndarray,
    log_evidence_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    log_unnormalized = prior_log_posterior + log_evidence_values
    posterior_values, log_posterior, _ = normalize_log_weights(log_unnormalized)
    return posterior_values, log_posterior


class ParticleFilterBank:
    def __init__(
        self,
        filters_by_label: dict[str, BootstrapParticleFilter],
        prior_by_label: dict[str, float] | None = None,
        *,
        filter_id: str = "particle_filter_bank_v1",
    ) -> None:
        self.filter_id = filter_id
        self.filters_by_label = filters_by_label
        self.label_ids = list(filters_by_label)
        if prior_by_label is None:
            prior_by_label = {label: 1.0 / len(self.label_ids) for label in self.label_ids}
        self.initial_log_prior = np.array(
            [np.log(prior_by_label[label]) for label in self.label_ids],
            dtype=np.float64,
        )
        self.state: ParticleFilterBankState | None = None

    def reset(self, trajectory_id: str, initial_particles_by_label: dict[str, np.ndarray]) -> None:
        for label, particle_filter in self.filters_by_label.items():
            particle_filter.reset(trajectory_id=trajectory_id, initial_particles=initial_particles_by_label[label])
        self.state = ParticleFilterBankState(
            trajectory_id=trajectory_id,
            label_ids=self.label_ids,
            log_posterior=self.initial_log_prior.copy(),
        )

    def update(self, time: float, observation: np.ndarray) -> AdvancedFilterStep:
        if self.state is None:
            raise RuntimeError("ParticleFilterBank must be reset before update")
        log_evidence_values: list[float] = []
        ess_values: dict[str, float] = {}
        resampled_values: dict[str, bool] = {}
        for label in self.label_ids:
            step = self.filters_by_label[label].update(time=time, observation=observation)
            log_evidence_values.append(step.log_marginal_likelihood)
            ess_values[label] = step.ess
            resampled_values[label] = step.resampled
        posterior_values, log_posterior = particle_filter_class_evidence_update(
            self.state.log_posterior,
            np.asarray(log_evidence_values, dtype=np.float64),
        )
        self.state.log_posterior = log_posterior
        best_index = int(np.argmax(posterior_values))
        predicted_label = self.label_ids[best_index]
        return AdvancedFilterStep(
            trajectory_id=self.state.trajectory_id,
            time=float(time),
            filter_id=self.filter_id,
            predicted_label=predicted_label,
            confidence=float(posterior_values[best_index]),
            posterior_by_label={
                label: float(posterior_values[index])
                for index, label in enumerate(self.label_ids)
            },
            log_evidence_by_label={
                label: float(log_evidence_values[index])
                for index, label in enumerate(self.label_ids)
            },
            diagnostics={
                **{f"ess_{label}": ess_values[label] for label in self.label_ids},
                **{f"resampled_{label}": resampled_values[label] for label in self.label_ids},
            },
        )
