from __future__ import annotations

from typing import Any

from kinematic_classifier_sandbox.utils.math import _mean

from ..utils.plotting import figure_to_png_bytes
from ..utils.plotting import plt


def render_corpus_classifier_scoring_report(
    *,
    record_count: int,
    candidate_score_rows: tuple[dict[str, Any], ...],
    posterior_rows: tuple[dict[str, Any], ...],
    disagreement_rows: tuple[dict[str, Any], ...],
) -> str:
    return "\n".join(
        [
            "# Corpus Classifier Scoring",
            "",
            "## Summary",
            f"- scored trajectories: `{record_count}`",
            f"- classifier result rows: `{len(candidate_score_rows)}`",
            f"- posterior history rows: `{len(posterior_rows)}`",
            f"- disagreement cases: `{sum(1 for row in disagreement_rows if row['has_disagreement'])}`",
            "",
            "## Methods",
            "- `pointwise`",
            "- `sequential_bayes`",
            "- `windowed_raw`",
            "- `windowed_robust`",
            "- `kalman_bank`",
            "",
            "## Notes",
            "- Classifier stress is now measured from real posterior outputs, margins, and errors rather than assigned from scenario family heuristics.",
            "- The class-model parameters are fit from the generated corpus slice so this milestone can score objective-driven trajectories without pretending the benchmark defaults apply unchanged.",
        ]
    )


def render_classifier_scoring_posterior_plot(rows: tuple[dict[str, Any], ...]) -> bytes:
    selected = [row for row in rows if row["method_name"] == "sequential_bayes"][:24]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.plot([row["time"] for row in selected], [row["confidence"] for row in selected], marker="o", linewidth=1.0)
    ax.set_title("Posterior Confidence Trace Preview")
    ax.set_xlabel("Time")
    ax.set_ylabel("Confidence")
    fig.tight_layout()
    return figure_to_png_bytes(fig, dpi=180)


def render_classifier_scoring_disagreement_plot(rows: tuple[dict[str, Any], ...]) -> bytes:
    labels = [str(row["trajectory_id"]).split("_", 1)[-1] for row in rows[:12]]
    values = [int(row["unique_prediction_count"]) for row in rows[:12]]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.bar(labels, values, color="#b56b4d")
    ax.set_title("Method Disagreement By Trajectory")
    ax.set_ylabel("Unique Final Predictions")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return figure_to_png_bytes(fig, dpi=180)


def render_classifier_scoring_stress_plot(rows: tuple[dict[str, Any], ...]) -> bytes:
    methods = sorted({str(row["method_name"]) for row in rows})
    values = [_mean([float(row["measured_classifier_stress"]) for row in rows if row["method_name"] == method]) for method in methods]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(methods, values, color="#5c7ea5")
    ax.set_title("Measured Classifier Stress By Method")
    ax.set_ylabel("Mean Stress")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return figure_to_png_bytes(fig, dpi=180)
