from __future__ import annotations

from ..utils.plotting import plt
from .adaptive_stress_types import AdaptiveStressCorpusResult

def _plot_posterior_timelines(result: AdaptiveStressCorpusResult):
    selected_modes = ("wrong_classification", "high_entropy", "transition_delay")
    payloads = []
    for mode in selected_modes:
        payload = next((item for item in result.posterior_trace_payloads if item["failure_mode"] == mode), None)
        if payload is not None:
            payloads.append(payload)
    fig, axes = plt.subplots(1, max(1, len(payloads)), figsize=(5.0 * max(1, len(payloads)), 4.0))
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax, payload in zip(axes, payloads):
        trace = payload["posterior_trace"]
        ax.plot([row["time"] for row in trace], [row["constant_velocity_probability"] for row in trace], label="P(CV)", color="#2563eb")
        ax.plot([row["time"] for row in trace], [row["constant_acceleration_probability"] for row in trace], label="P(CA/new mode)", color="#dc2626")
        ax.set_title(str(payload["failure_mode"]).replace("_", " "), loc="left", fontsize=11, fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Stress Case Posterior Timelines", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _plot_feature_traces(result: AdaptiveStressCorpusResult):
    selected_modes = ("raw_extrema_failure", "irregular_window_failure", "kalman_mismatch")
    payloads = []
    for mode in selected_modes:
        payload = next((item for item in result.feature_trace_payloads if item["failure_mode"] == mode), None)
        if payload is not None:
            payloads.append(payload)
    fig, axes = plt.subplots(1, max(1, len(payloads)), figsize=(5.0 * max(1, len(payloads)), 4.0))
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax, payload in zip(axes, payloads):
        ax.plot(payload["times"], payload["measurements"], marker="o", linewidth=1.5, color="#111827")
        ax.set_title(str(payload["failure_mode"]).replace("_", " "), loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("time")
        ax.set_ylabel("measurement")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Stress Case Feature Traces", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _plot_prior_flip_examples(result: AdaptiveStressCorpusResult):
    payload = next(iter(result.prior_flip_payloads), None)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if payload is not None:
        priors = [row[0] for row in payload["sweep"]]
        confidences = [row[2] for row in payload["sweep"]]
        predicted = [row[1] for row in payload["sweep"]]
        colors = ["#2563eb" if name == "constant_velocity" else "#dc2626" for name in predicted]
        ax.scatter(priors, confidences, c=colors, s=50)
        ax.plot(priors, confidences, color="#6b7280", alpha=0.5)
    ax.set_title("Prior Flip Examples", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("prior P(CV)")
    ax.set_ylabel("final confidence")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig

