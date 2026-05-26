from __future__ import annotations

from typing import Any

from kinematic_classifier_sandbox.utils.math import mean as _mean
from kinematic_classifier_sandbox.utils.plotting import figure_to_png_bytes
from kinematic_classifier_sandbox.utils.plotting import plt


def _render_posterior_plot(rows: tuple[dict[str, Any], ...]) -> bytes:
    selected = [row for row in rows if row["method_name"] == "sequential_bayes"][:24]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.plot([row["time"] for row in selected], [row["confidence"] for row in selected], marker="o", linewidth=1.0)
    ax.set_title("Posterior Confidence Trace Preview")
    ax.set_xlabel("Time")
    ax.set_ylabel("Confidence")
    fig.tight_layout()
    return figure_to_png_bytes(fig, dpi=180)


def _render_disagreement_plot(rows: tuple[dict[str, Any], ...]) -> bytes:
    labels = [str(row["trajectory_id"]).split("_", 1)[-1] for row in rows[:12]]
    values = [int(row["unique_prediction_count"]) for row in rows[:12]]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.bar(labels, values, color="#b56b4d")
    ax.set_title("Method Disagreement By Trajectory")
    ax.set_ylabel("Unique Final Predictions")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return figure_to_png_bytes(fig, dpi=180)


def _render_stress_plot(rows: tuple[dict[str, Any], ...]) -> bytes:
    methods = sorted({str(row["method_name"]) for row in rows})
    values = [_mean([float(row["measured_classifier_stress"]) for row in rows if row["method_name"] == method]) for method in methods]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(methods, values, color="#5c7ea5")
    ax.set_title("Measured Classifier Stress By Method")
    ax.set_ylabel("Mean Stress")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return figure_to_png_bytes(fig, dpi=180)
