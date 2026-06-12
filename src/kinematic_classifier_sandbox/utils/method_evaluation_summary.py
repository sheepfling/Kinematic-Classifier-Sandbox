from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


METHOD_EVALUATION_SUMMARY_FIELDS = (
    "method_id",
    "study_surface",
    "evaluation_surface",
    "overall_accuracy",
    "negative_log_likelihood",
    "brier_score",
    "ece",
    "posterior_margin",
    "post_switch_accuracy",
    "switch_detection_delay",
    "runtime_seconds",
    "promotion_decision",
)


@dataclass(frozen=True, slots=True)
class PosteriorMetricSample:
    true_label: str
    predicted_label: str
    confidence: float
    posterior_by_label: Mapping[str, float]


def compute_multiclass_posterior_metrics(
    samples: Sequence[PosteriorMetricSample],
    *,
    bins: int = 10,
) -> dict[str, float]:
    eps = 1.0e-12
    if not samples:
        return {
            "overall_accuracy": 0.0,
            "negative_log_likelihood": 0.0,
            "brier_score": 0.0,
            "ece": 0.0,
            "posterior_margin": 0.0,
        }
    class_names = sorted(
        {
            label
            for sample in samples
            for label in (*sample.posterior_by_label.keys(), sample.true_label, sample.predicted_label)
        }
    )
    correct = 0
    nll_sum = 0.0
    brier_sum = 0.0
    margin_sum = 0.0
    bin_counts = [0] * bins
    bin_confidence = [0.0] * bins
    bin_accuracy = [0.0] * bins
    for sample in samples:
        posterior = _normalize_posterior(sample.posterior_by_label, class_names)
        probabilities = [posterior[name] for name in class_names]
        prob_true = posterior.get(sample.true_label, 0.0)
        correct_flag = 1.0 if sample.predicted_label == sample.true_label else 0.0
        correct += int(correct_flag)
        nll_sum += -math.log(max(prob_true, eps))
        brier_sum += sum(
            (posterior[name] - (1.0 if name == sample.true_label else 0.0)) ** 2
            for name in class_names
        )
        sorted_probs = sorted(probabilities, reverse=True)
        top1 = sorted_probs[0] if sorted_probs else 0.0
        top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin_sum += max(top1 - top2, 0.0)
        clipped_confidence = min(max(float(sample.confidence), 0.0), 1.0)
        bin_index = min(int(clipped_confidence * bins), bins - 1)
        bin_counts[bin_index] += 1
        bin_confidence[bin_index] += clipped_confidence
        bin_accuracy[bin_index] += correct_flag
    ece = 0.0
    total = len(samples)
    for count, confidence_sum, accuracy_sum in zip(bin_counts, bin_confidence, bin_accuracy, strict=True):
        if count == 0:
            continue
        ece += (count / total) * abs((confidence_sum / count) - (accuracy_sum / count))
    return {
        "overall_accuracy": correct / total,
        "negative_log_likelihood": nll_sum / total,
        "brier_score": brier_sum / total,
        "ece": ece,
        "posterior_margin": margin_sum / total,
    }


def build_method_evaluation_summary_row(
    *,
    method_id: str,
    study_surface: str,
    evaluation_surface: str,
    metrics: Mapping[str, float | int | str | None],
    post_switch_accuracy: float | None = None,
    switch_detection_delay: float | None = None,
    runtime_seconds: float | None = None,
    promotion_decision: str = "",
) -> dict[str, object]:
    return {
        "method_id": method_id,
        "study_surface": study_surface,
        "evaluation_surface": evaluation_surface,
        "overall_accuracy": float(metrics["overall_accuracy"]),
        "negative_log_likelihood": float(metrics["negative_log_likelihood"]),
        "brier_score": float(metrics["brier_score"]),
        "ece": float(metrics["ece"]),
        "posterior_margin": float(metrics["posterior_margin"]),
        "post_switch_accuracy": "" if post_switch_accuracy is None else float(post_switch_accuracy),
        "switch_detection_delay": "" if switch_detection_delay is None else float(switch_detection_delay),
        "runtime_seconds": "" if runtime_seconds is None else float(runtime_seconds),
        "promotion_decision": promotion_decision,
    }


def _normalize_posterior(
    posterior_by_label: Mapping[str, float],
    class_names: Sequence[str],
) -> dict[str, float]:
    total = sum(max(float(posterior_by_label.get(name, 0.0)), 0.0) for name in class_names)
    if total <= 0.0:
        uniform = 1.0 / max(len(class_names), 1)
        return {name: uniform for name in class_names}
    return {name: max(float(posterior_by_label.get(name, 0.0)), 0.0) / total for name in class_names}
