from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..utils.types import FloatArray

if TYPE_CHECKING:
    from .contracts import AdvancedFilterStep, AdvancedStateSummary

class AdvancedFilterBackend(Protocol):
    filter_id: str

    def reset(self, trajectory_id: str, initial_observation: FloatArray | None = None) -> None:
        ...

    def update(self, time: float, observation: FloatArray) -> AdvancedFilterStep:
        ...

    def state_summary(self) -> AdvancedStateSummary:
        ...


def validate_advanced_filter_step(step: AdvancedFilterStep, *, atol: float = 1.0e-6) -> None:
    if not step.posterior_by_label:
        raise ValueError("posterior_by_label must not be empty")
    posterior_sum = sum(step.posterior_by_label.values())
    if abs(posterior_sum - 1.0) > atol:
        raise ValueError(f"posterior probabilities must sum to 1.0, got {posterior_sum}")
    if step.predicted_label not in step.posterior_by_label:
        raise ValueError("predicted_label must be present in posterior_by_label")
    if step.log_evidence_by_label and set(step.log_evidence_by_label).difference(step.posterior_by_label):
        raise ValueError("log_evidence_by_label cannot contain labels outside posterior_by_label")
    expected_confidence = step.posterior_by_label[step.predicted_label]
    if abs(expected_confidence - step.confidence) > atol:
        raise ValueError("confidence must equal posterior of predicted_label")
