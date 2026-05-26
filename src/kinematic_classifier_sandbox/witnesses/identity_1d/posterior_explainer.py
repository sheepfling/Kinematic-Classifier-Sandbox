from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from ...markdown_builder import MarkdownDocument
from ...utils.plotting import plt
from .core import (
    IdentityBenchmarkResult,
    IdentityClassificationRun,
    SpeedScenario,
    run_identity_benchmark,
)


@dataclass(frozen=True, slots=True)
class IdentityPosteriorWalkthrough:
    title: str
    class_names: tuple[str, ...]
    scenario: SpeedScenario
    run: IdentityClassificationRun
    step_index: int
    measurement: float
    prior_weights: dict[str, float]
    posterior_weights: dict[str, float]
    log_terms: dict[str, dict[str, float]]


def _scenario_for_run(result: IdentityBenchmarkResult, run: IdentityClassificationRun) -> SpeedScenario:
    return next(scenario for scenario in result.scenarios if scenario.name == run.scenario_name)


def _predicted_class_weights(run: IdentityClassificationRun, step_index: int) -> dict[str, float]:
    if step_index == 0:
        class_names = tuple(run.steps[0].updated_class_weights)
        final_total = sum(run.final_weights.values())
        if final_total <= 0.0:
            return {class_name: 1.0 / len(class_names) for class_name in class_names}
        return {class_name: run.final_weights[class_name] / final_total for class_name in class_names}
    return run.steps[step_index - 1].updated_class_weights


def _build_walkthrough(
    result: IdentityBenchmarkResult,
    run: IdentityClassificationRun,
    *,
    title: str,
) -> IdentityPosteriorWalkthrough:
    class_names = tuple(spec.name for spec in result.class_specs)
    scenario = _scenario_for_run(result, run)
    if run.aggregate_map_class == run.expected_class:
        step_index = min(max(2, len(run.steps) // 3), len(run.steps) - 1)
    else:
        misclassified_indices = [index for index, step in enumerate(run.steps) if step.map_class != run.expected_class]
        step_index = misclassified_indices[min(len(misclassified_indices) // 2, len(misclassified_indices) - 1)] if misclassified_indices else len(run.steps) - 1
    step = run.steps[step_index]
    return IdentityPosteriorWalkthrough(
        title=title,
        class_names=class_names,
        scenario=scenario,
        run=run,
        step_index=step_index,
        measurement=scenario.speeds_obs_mph[step_index],
        prior_weights=_predicted_class_weights(run, step_index),
        posterior_weights=step.updated_class_weights,
        log_terms=step.log_likelihood_terms,
    )


def _select_success_walkthrough(result: IdentityBenchmarkResult) -> IdentityPosteriorWalkthrough:
    preferred_families = ("car_cruise", "horse_near_limit", "bike_cruise", "car_sprint")
    for family_name in preferred_families:
        for run in result.runs:
            if run.family_name == family_name and run.aggregate_map_class == run.expected_class and run.steps:
                return _build_walkthrough(result, run, title="Identity Posterior Walkthrough")
    run = next(run for run in result.runs if run.steps and run.aggregate_map_class == run.expected_class)
    return _build_walkthrough(result, run, title="Identity Posterior Walkthrough")


def _select_failure_walkthrough(result: IdentityBenchmarkResult) -> IdentityPosteriorWalkthrough:
    preferred_pairs = (
        ("bike_horse_border", "horse"),
        ("horse_car_border", "car"),
        ("car_push", "horse"),
    )
    for family_name, predicted_class in preferred_pairs:
        for run in result.runs:
            if run.family_name == family_name and run.aggregate_map_class == predicted_class and run.steps:
                return _build_walkthrough(result, run, title="Identity Posterior Failure Walkthrough")
    run = next(run for run in result.runs if run.steps and run.aggregate_map_class != run.expected_class)
    return _build_walkthrough(result, run, title="Identity Posterior Failure Walkthrough")


def _render_walkthrough_markdown(walkthrough: IdentityPosteriorWalkthrough) -> str:
    step = walkthrough.run.steps[walkthrough.step_index]
    report = MarkdownDocument(walkthrough.title)
    report.paragraph(
        "This artifact illustrates the class-posterior update used in the identity speed benchmark "
        "for one concrete speed measurement step."
    )
    report.heading("Measurement Model", level=2)
    report.bullet_list(
        [
            "Measurement: `z_t` is the observed scalar speed sample in mph.",
            "Each class uses a speed-shape term around its cruise regime and a soft upper-speed validity term around its feasible envelope.",
            "A short history term compares the running mean speed against each class signature.",
        ]
    )
    report.heading("Bayesian Update", level=2)
    report.bullet_list(
        [
            "Score the current observed speed with a Gaussian class-shape likelihood.",
            "Apply a soft speed-validity likelihood using Gaussian CDF mass below the class speed limit.",
            "Add a history-shape term on the running mean speed.",
            (
                "Add a short-window within-class mode term so fast-pack bikes, near-limit horses, "
                "and pushed cars are not forced into one static cruise template."
            ),
            "Add a temporal-dynamics term from recent speed differences.",
            "Update class weights with `p(s_i | z_1:t) ∝ p(s_i | z_1:t-1) * L_i(z_t)` and normalize across classes.",
        ]
    )
    report.heading("Worked Example", level=2)
    report.bullet_list(
        [
            f"True class: `{walkthrough.run.expected_class}`",
            f"Scenario family: `{walkthrough.run.family_name}`",
            f"Scenario: `{walkthrough.run.scenario_name}`",
            f"Aggregate predicted class: `{walkthrough.run.aggregate_map_class}`",
            f"Step index: `{walkthrough.step_index + 1}` of `{len(walkthrough.run.steps)}` posterior updates",
            f"Observed speed `z_t`: `{walkthrough.measurement:.3f} mph`",
            f"MAP class after update: `{step.map_class}`",
        ]
    )
    report.heading("Composite Log-Likelihood Terms by Class", level=2)
    report.table(
        [
            "Class",
            "Prior",
            "speed_shape",
            "speed_validity",
            "history_shape",
            "mode_shape",
            "dynamics_shape",
            "total",
            "Posterior",
        ],
        [
            (
                f"`{class_name}`",
                f"{walkthrough.prior_weights[class_name]:.3f}",
                f"{walkthrough.log_terms[class_name].get('speed_shape', 0.0):.3f}",
                f"{walkthrough.log_terms[class_name].get('speed_validity', 0.0):.3f}",
                f"{walkthrough.log_terms[class_name].get('history_shape', 0.0):.3f}",
                f"{walkthrough.log_terms[class_name].get('mode_shape', 0.0):.3f}",
                f"{walkthrough.log_terms[class_name].get('dynamics_shape', 0.0):.3f}",
                f"{walkthrough.log_terms[class_name].get('total', 0.0):.3f}",
                f"{walkthrough.posterior_weights[class_name]:.3f}",
            )
            for class_name in walkthrough.class_names
        ],
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "`speed_shape` is the Gaussian log likelihood of the current speed under the class cruise regime.",
            "`speed_validity` is a soft envelope term from Gaussian CDF mass below the class speed cap, including any class margin.",
            "`history_shape` compares the running mean speed against the class cruise signature.",
            "`mode_shape` is a within-class submode term on the recent speed window.",
            "`dynamics_shape` scores recent speed-difference behavior.",
        ]
    )
    report.heading("Posterior Formula Used in Practice", level=2)
    report.paragraph(
        "`log w_t(s_i) = log w_{t-1}(s_i) + log_speed_shape_i + 1.4 * log_speed_validity_i + 0.45 * log_history_shape_i + log_mode_shape_i + log_dynamics_shape_i - log Z_t`"
    )
    report.paragraph("where `log Z_t` is the across-class normalizer from `logsumexp`.")
    return report.text()


def render_identity_posterior_explainer_markdown(result: IdentityBenchmarkResult) -> str:
    return _render_walkthrough_markdown(_select_success_walkthrough(result))


def render_identity_posterior_failure_markdown(result: IdentityBenchmarkResult) -> str:
    return _render_walkthrough_markdown(_select_failure_walkthrough(result))


def _build_posterior_explainer_figure(walkthrough: IdentityPosteriorWalkthrough):
    class_names = list(walkthrough.class_names)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    prior_ax, score_ax, posterior_ax = axes
    x = list(range(len(class_names)))

    prior_ax.bar(x, [walkthrough.prior_weights[name] for name in class_names], color="#2563eb")
    prior_ax.set_title("Prior Class Weights", loc="left", fontsize=12, fontweight="bold")
    prior_ax.set_xticks(x, class_names, rotation=45, ha="right")
    prior_ax.set_ylim(0.0, 1.0)
    prior_ax.grid(True, axis="y", alpha=0.25)

    speed_shape_scores = [walkthrough.log_terms[name].get("speed_shape", 0.0) for name in class_names]
    extra_scores = [
        walkthrough.log_terms[name].get("speed_validity", 0.0)
        + walkthrough.log_terms[name].get("history_shape", 0.0)
        + walkthrough.log_terms[name].get("mode_shape", 0.0)
        + walkthrough.log_terms[name].get("dynamics_shape", 0.0)
        for name in class_names
    ]
    score_ax.bar(x, speed_shape_scores, color="#0f766e", label="speed shape")
    score_ax.bar(x, extra_scores, bottom=speed_shape_scores, color="#f59e0b", label="validity + history")
    score_ax.set_title("Composite Log Score", loc="left", fontsize=12, fontweight="bold")
    score_ax.set_xticks(x, class_names, rotation=45, ha="right")
    score_ax.grid(True, axis="y", alpha=0.25)
    score_ax.legend(frameon=False)

    posterior_ax.bar(x, [walkthrough.posterior_weights[name] for name in class_names], color="#7c3aed")
    posterior_ax.set_title("Posterior Class Weights", loc="left", fontsize=12, fontweight="bold")
    posterior_ax.set_xticks(x, class_names, rotation=45, ha="right")
    posterior_ax.set_ylim(0.0, 1.0)
    posterior_ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"{walkthrough.title}: {walkthrough.run.scenario_name} step {walkthrough.step_index + 1}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def render_identity_posterior_explainer_png_bytes(result: IdentityBenchmarkResult) -> bytes:
    fig = _build_posterior_explainer_figure(_select_success_walkthrough(result))
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_identity_posterior_failure_png_bytes(result: IdentityBenchmarkResult) -> bytes:
    fig = _build_posterior_explainer_figure(_select_failure_walkthrough(result))
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_identity_posterior_comparison_markdown(result: IdentityBenchmarkResult) -> str:
    success = _select_success_walkthrough(result)
    failure = _select_failure_walkthrough(result)
    report = MarkdownDocument("Identity Posterior Comparison")
    report.paragraph(
        "This artifact compares one successful identity posterior update against one confused update "
        "using the same score decomposition."
    )
    report.heading("Cases", level=2)
    report.bullet_list(
        [
            (
                f"Success case: `{success.run.scenario_name}` with true class `{success.run.expected_class}` "
                f"and aggregate prediction `{success.run.aggregate_map_class}` at step `{success.step_index + 1}`"
            ),
            (
                f"Failure case: `{failure.run.scenario_name}` with true class `{failure.run.expected_class}` "
                f"and aggregate prediction `{failure.run.aggregate_map_class}` at step `{failure.step_index + 1}`"
            ),
        ]
    )
    report.heading("Side-by-Side Posterior Terms", level=2)
    report.table(
        [
            "Class",
            "Success prior",
            "Success total",
            "Success posterior",
            "Failure prior",
            "Failure total",
            "Failure posterior",
        ],
        [
            (
                f"`{class_name}`",
                f"{success.prior_weights[class_name]:.3f}",
                f"{success.log_terms[class_name].get('total', 0.0):.3f}",
                f"{success.posterior_weights[class_name]:.3f}",
                f"{failure.prior_weights[class_name]:.3f}",
                f"{failure.log_terms[class_name].get('total', 0.0):.3f}",
                f"{failure.posterior_weights[class_name]:.3f}",
            )
            for class_name in success.class_names
        ],
    )

    def margin_text(walkthrough: IdentityPosteriorWalkthrough) -> str:
        ranked = sorted(walkthrough.posterior_weights.items(), key=lambda item: item[1], reverse=True)
        return f"{ranked[0][0]} over {ranked[1][0]} by {ranked[0][1] - ranked[1][1]:.3f}"

    report.heading("Decision Margins", level=2)
    report.bullet_list([f"Success margin: {margin_text(success)}", f"Failure margin: {margin_text(failure)}"])

    report.heading("Largest Score Shifts Between Cases", level=2)
    report.table(
        ["Class", "Delta total (success - failure)", "Delta posterior (success - failure)"],
        [
            (
                f"`{class_name}`",
                f"{success.log_terms[class_name].get('total', 0.0) - failure.log_terms[class_name].get('total', 0.0):.3f}",
                f"{success.posterior_weights[class_name] - failure.posterior_weights[class_name]:.3f}",
            )
            for class_name in success.class_names
        ],
    )
    return report.text()


def _build_identity_posterior_comparison_figure(result: IdentityBenchmarkResult):
    success = _select_success_walkthrough(result)
    failure = _select_failure_walkthrough(result)
    class_names = list(success.class_names)
    x = list(range(len(class_names)))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    score_ax, posterior_ax = axes

    success_totals = [success.log_terms[name].get("total", 0.0) for name in class_names]
    failure_totals = [failure.log_terms[name].get("total", 0.0) for name in class_names]
    score_ax.bar([value - width / 2 for value in x], success_totals, width=width, color="#0f766e", label="success")
    score_ax.bar([value + width / 2 for value in x], failure_totals, width=width, color="#b91c1c", label="failure")
    score_ax.set_title("Composite Log Totals", loc="left", fontsize=12, fontweight="bold")
    score_ax.set_xticks(x, class_names, rotation=45, ha="right")
    score_ax.grid(True, axis="y", alpha=0.25)
    score_ax.legend(frameon=False)

    success_posteriors = [success.posterior_weights[name] for name in class_names]
    failure_posteriors = [failure.posterior_weights[name] for name in class_names]
    posterior_ax.bar([value - width / 2 for value in x], success_posteriors, width=width, color="#2563eb", label="success")
    posterior_ax.bar([value + width / 2 for value in x], failure_posteriors, width=width, color="#7c3aed", label="failure")
    posterior_ax.set_title("Posterior Weights", loc="left", fontsize=12, fontweight="bold")
    posterior_ax.set_xticks(x, class_names, rotation=45, ha="right")
    posterior_ax.set_ylim(0.0, 1.0)
    posterior_ax.grid(True, axis="y", alpha=0.25)
    posterior_ax.legend(frameon=False)

    fig.suptitle("Identity Posterior Success vs Failure Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def render_identity_posterior_comparison_png_bytes(result: IdentityBenchmarkResult) -> bytes:
    fig = _build_identity_posterior_comparison_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_identity_posterior_margin_trace_markdown(result: IdentityBenchmarkResult) -> str:
    failure = _select_failure_walkthrough(result)
    true_class = failure.run.expected_class
    predicted_class = failure.run.aggregate_map_class
    report = MarkdownDocument("Identity Posterior Margin Trace")
    report.paragraph(
        "This artifact traces the margin between the true class and the winning confused class across time for one failure run."
    )
    report.heading("Case", level=2)
    report.bullet_list(
        [
            f"Scenario family: `{failure.run.family_name}`",
            f"Scenario: `{failure.run.scenario_name}`",
            f"True class: `{true_class}`",
            f"Aggregate predicted class: `{predicted_class}`",
        ]
    )
    report.heading("Margin Definition", level=2)
    report.bullet_list(
        [
            f"Posterior margin: `p({true_class} | z_1:t) - p({predicted_class} | z_1:t)`",
            f"Score margin: `log_total({true_class}) - log_total({predicted_class})`",
        ]
    )
    report.heading("Stepwise Margins", level=2)
    rows: list[tuple[str, ...]] = []
    for step_index, step in enumerate(failure.run.steps):
        true_terms = step.log_likelihood_terms[true_class]
        predicted_terms = step.log_likelihood_terms[predicted_class]
        posterior_margin = step.updated_class_weights[true_class] - step.updated_class_weights[predicted_class]
        score_margin = true_terms["total"] - predicted_terms["total"]
        component_deltas = {
            "speed_shape": true_terms.get("speed_shape", 0.0) - predicted_terms.get("speed_shape", 0.0),
            "speed_validity": true_terms.get("speed_validity", 0.0) - predicted_terms.get("speed_validity", 0.0),
            "history_shape": true_terms.get("history_shape", 0.0) - predicted_terms.get("history_shape", 0.0),
            "mode_shape": true_terms.get("mode_shape", 0.0) - predicted_terms.get("mode_shape", 0.0),
            "dynamics_shape": true_terms.get("dynamics_shape", 0.0) - predicted_terms.get("dynamics_shape", 0.0),
        }
        dominant_negative_term = min(component_deltas.items(), key=lambda item: item[1])[0]
        rows.append(
            (
                f"{step_index + 1}",
                f"{step.observed_speed_mph:.3f}",
                f"{step.updated_class_weights[true_class]:.3f}",
                f"{step.updated_class_weights[predicted_class]:.3f}",
                f"{posterior_margin:.3f}",
                f"{score_margin:.3f}",
                dominant_negative_term,
            )
        )
    report.table(
        [
            "Step",
            "z_t (mph)",
            "True posterior",
            "Predicted posterior",
            "Posterior margin",
            "Score margin",
            "Dominant negative term",
        ],
        rows,
    )
    crossover_index = next(
        (
            index + 1
            for index, step in enumerate(failure.run.steps)
            if step.updated_class_weights[predicted_class] > step.updated_class_weights[true_class]
        ),
        None,
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            (
                f"First posterior crossover step: `{crossover_index}`"
                if crossover_index is not None
                else "No crossover found"
            ),
            "The trace is most useful for seeing whether the wrong class wins because of the instantaneous speed shape, the upper-speed validity gate, or the running-mean history term.",
        ]
    )
    return report.text()


def _build_identity_posterior_margin_trace_figure(result: IdentityBenchmarkResult):
    failure = _select_failure_walkthrough(result)
    true_class = failure.run.expected_class
    predicted_class = failure.run.aggregate_map_class
    steps = list(range(1, len(failure.run.steps) + 1))
    measurements = [step.observed_speed_mph for step in failure.run.steps]
    posterior_margins = [
        step.updated_class_weights[true_class] - step.updated_class_weights[predicted_class]
        for step in failure.run.steps
    ]
    score_margins = [
        step.log_likelihood_terms[true_class]["total"] - step.log_likelihood_terms[predicted_class]["total"]
        for step in failure.run.steps
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True)
    measurement_ax, posterior_ax, score_ax = axes

    measurement_ax.plot(steps, measurements, color="#111827", linewidth=2.0)
    measurement_ax.set_title("Observed Measurement z_t", loc="left", fontsize=12, fontweight="bold")
    measurement_ax.set_ylabel("mph")
    measurement_ax.grid(True, alpha=0.25)

    posterior_ax.plot(steps, posterior_margins, color="#2563eb", linewidth=2.2)
    posterior_ax.axhline(0.0, color="#991b1b", linestyle="--", linewidth=1.0)
    posterior_ax.set_title(f"Posterior Margin: {true_class} - {predicted_class}", loc="left", fontsize=12, fontweight="bold")
    posterior_ax.set_ylabel("probability")
    posterior_ax.grid(True, alpha=0.25)

    score_ax.plot(steps, score_margins, color="#7c3aed", linewidth=2.2)
    score_ax.axhline(0.0, color="#991b1b", linestyle="--", linewidth=1.0)
    score_ax.set_title(f"Composite Score Margin: {true_class} - {predicted_class}", loc="left", fontsize=12, fontweight="bold")
    score_ax.set_xlabel("step")
    score_ax.set_ylabel("log-score")
    score_ax.grid(True, alpha=0.25)

    fig.suptitle(f"Identity Posterior Margin Trace: {failure.run.scenario_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def render_identity_posterior_margin_trace_png_bytes(result: IdentityBenchmarkResult) -> bytes:
    fig = _build_identity_posterior_margin_trace_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_identity_posterior_explainer_artifacts(
    output_dir: str | Path,
    *,
    result: IdentityBenchmarkResult | None = None,
) -> tuple[Path, Path]:
    benchmark_result = result or run_identity_benchmark()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "identity_1d_posterior_walkthrough.md"
    png_path = output_root / "identity_1d_posterior_walkthrough.png"
    markdown_path.write_text(render_identity_posterior_explainer_markdown(benchmark_result), encoding="utf-8")
    png_path.write_bytes(render_identity_posterior_explainer_png_bytes(benchmark_result))
    return markdown_path, png_path


def write_identity_posterior_failure_artifacts(
    output_dir: str | Path,
    *,
    result: IdentityBenchmarkResult | None = None,
) -> tuple[Path, Path]:
    benchmark_result = result or run_identity_benchmark()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "identity_1d_posterior_failure_walkthrough.md"
    png_path = output_root / "identity_1d_posterior_failure_walkthrough.png"
    markdown_path.write_text(render_identity_posterior_failure_markdown(benchmark_result), encoding="utf-8")
    png_path.write_bytes(render_identity_posterior_failure_png_bytes(benchmark_result))
    return markdown_path, png_path


def write_identity_posterior_comparison_artifacts(
    output_dir: str | Path,
    *,
    result: IdentityBenchmarkResult | None = None,
) -> tuple[Path, Path]:
    benchmark_result = result or run_identity_benchmark()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "identity_1d_posterior_comparison.md"
    png_path = output_root / "identity_1d_posterior_comparison.png"
    markdown_path.write_text(render_identity_posterior_comparison_markdown(benchmark_result), encoding="utf-8")
    png_path.write_bytes(render_identity_posterior_comparison_png_bytes(benchmark_result))
    return markdown_path, png_path


def write_identity_posterior_margin_trace_artifacts(
    output_dir: str | Path,
    *,
    result: IdentityBenchmarkResult | None = None,
) -> tuple[Path, Path]:
    benchmark_result = result or run_identity_benchmark()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "identity_1d_posterior_margin_trace.md"
    png_path = output_root / "identity_1d_posterior_margin_trace.png"
    markdown_path.write_text(render_identity_posterior_margin_trace_markdown(benchmark_result), encoding="utf-8")
    png_path.write_bytes(render_identity_posterior_margin_trace_png_bytes(benchmark_result))
    return markdown_path, png_path
