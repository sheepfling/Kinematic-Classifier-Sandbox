from __future__ import annotations

from math import log
from typing import NamedTuple

from ...markdown_builder import MarkdownDocument
from .contracts import (
    SwitchingModeSpec,
    SwitchingScenario,
    TransitionBenchmarkResult,
    TransitionRun,
)
from ...witnesses.benchmarks.transition_matrix_runner import (
    _emission_term_breakdown,
    default_switching_mode_specs,
    default_transition_matrix,
)


class TransitionWalkthroughSelection(NamedTuple):
    scenario: SwitchingScenario
    static_run: TransitionRun
    transition_run: TransitionRun


def render_transition_benchmark_report(result: TransitionBenchmarkResult) -> str:
    scenario_lines = []
    for static_run, transition_run, kalman_run in zip(result.static_runs[:9], result.transition_runs[:9], result.kalman_runs[:9]):
        scenario_lines.append(
            (
                static_run.scenario_name,
                f"{static_run.accuracy:.3f}",
                f"{transition_run.accuracy:.3f}",
                f"{kalman_run.accuracy:.3f}",
                f"{static_run.post_switch_accuracy:.3f}",
                f"{transition_run.post_switch_accuracy:.3f}",
                f"{kalman_run.post_switch_accuracy:.3f}",
            )
        )
    report = MarkdownDocument("Transition-Matrix Accumulator")
    report.paragraph(
        "Milestone 16 comparison between a static mode accumulator, a transition-matrix accumulator, and a Kalman mode bank on explicit switching scenarios."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Scenarios: {result.summary.num_scenarios}",
            f"Static accuracy: {result.summary.static_accuracy:.3f}",
            f"Transition accuracy: {result.summary.transition_accuracy:.3f}",
            f"Kalman accuracy: {result.summary.kalman_accuracy:.3f}",
            f"Static post-switch accuracy: {result.summary.static_post_switch_accuracy:.3f}",
            f"Transition post-switch accuracy: {result.summary.transition_post_switch_accuracy:.3f}",
            f"Kalman post-switch accuracy: {result.summary.kalman_post_switch_accuracy:.3f}",
            f"Scenarios improved post-switch: {result.summary.improved_scenarios}",
        ]
    )
    report.heading("Scenario Comparison", level=2)
    report.table(
        [
            "scenario_name",
            "static_accuracy",
            "transition_accuracy",
            "kalman_accuracy",
            "static_post_switch",
            "transition_post_switch",
            "kalman_post_switch",
        ],
        scenario_lines,
    )
    report.heading("Notes", level=2)
    report.bullet_list(
        [
            "Both methods use the same local emission model over derived speed and acceleration.",
            "The only difference is whether prior mass can move through an explicit transition matrix.",
            "The Kalman mode bank adds innovation-likelihood behavior from per-mode kinematic filters without transition mixing.",
            "This isolates the value of transition dynamics before moving to IMM.",
        ]
    )
    return report.text()


def _select_transition_walkthrough(result: TransitionBenchmarkResult) -> TransitionWalkthroughSelection:
    preferred_names = ("constant_velocity_then_braking", "constant_velocity_then_maneuver", "stationary_then_moving")
    for preferred_name in preferred_names:
        for scenario, static_run, transition_run in zip(result.scenarios, result.static_runs, result.transition_runs):
            if scenario.scenario_name == preferred_name:
                return TransitionWalkthroughSelection(
                    scenario=scenario,
                    static_run=static_run,
                    transition_run=transition_run,
                )
    return TransitionWalkthroughSelection(
        scenario=result.scenarios[0],
        static_run=result.static_runs[0],
        transition_run=result.transition_runs[0],
    )


def render_transition_numeric_walkthrough_markdown(
    result: TransitionBenchmarkResult,
    *,
    specs: tuple[SwitchingModeSpec, ...] | None = None,
    transition_matrix: dict[str, dict[str, float]] | None = None,
) -> str:
    selected_specs = specs or default_switching_mode_specs()
    transition = transition_matrix or default_transition_matrix()
    selection = _select_transition_walkthrough(result)
    scenario = selection.scenario
    static_run = selection.static_run
    transition_run = selection.transition_run
    switch_index = next((index for index, step in enumerate(transition_run.steps) if step.true_mode != transition_run.steps[0].true_mode), len(transition_run.steps) - 1)
    start_index = max(0, switch_index - 1)
    end_index = min(len(transition_run.steps), switch_index + 2)
    selected_steps = transition_run.steps[start_index:end_index]
    switch_step = transition_run.steps[switch_index]
    previous_posterior = transition_run.steps[switch_index - 1].posterior_weights if switch_index > 0 else transition_run.steps[switch_index].prior_weights
    switched_mode = switch_step.true_mode
    contribution_rows = []
    total_prior = 0.0
    for source_mode, source_weight in previous_posterior.items():
        contribution = source_weight * transition[source_mode][switched_mode]
        total_prior += contribution
        contribution_rows.append((f"`{source_mode}`", f"{source_weight:.3f}", f"{transition[source_mode][switched_mode]:.3f}", f"{contribution:.3f}"))

    report = MarkdownDocument("Transition-Matrix Numeric Walkthrough")
    report.paragraph("This worked example uses a real benchmark run from `inference/transition_matrix_accumulator.py` and shows the full transition-aware recursion on a short switching trajectory.")
    report.heading("Selected trajectory", level=2)
    report.bullet_list(
        [
            f"Scenario: `{scenario.scenario_name}`",
            f"Trajectory: `{scenario.trajectory_id}`",
            f"True mode sequence starts as `{transition_run.steps[0].true_mode}` and switches to `{switched_mode}` at step `{switch_index}`",
            f"Static post-switch accuracy: `{static_run.post_switch_accuracy:.3f}`",
            f"Transition-matrix post-switch accuracy: `{transition_run.post_switch_accuracy:.3f}`",
        ]
    )
    report.heading("Transition propagation at the first switched step", level=2)
    report.paragraph(f"For the switched target mode `{switched_mode}`, the propagated prior is")
    report.fence(rf"\bar{{p}}_t({switched_mode}) = \sum_s p_{{t-1}}(s)\,T_{{s,{switched_mode}}}", language="tex")
    report.table(["source mode", "previous posterior", "transition probability", "contribution"], contribution_rows + [("**total**", "", "", f"**{total_prior:.3f}**")])

    for step in selected_steps:
        report.heading(f"Step `{step.step}` at time `{step.time:.3f}`", level=2)
        report.bullet_list(
            [
                f"Measurement: `{step.measurement:.3f}`",
                f"Estimated speed: `{step.estimated_speed:.3f}`",
                f"Estimated acceleration: `{step.estimated_accel:.3f}`",
                f"True mode: `{step.true_mode}`",
                f"Predicted mode: `{step.predicted_mode}` with confidence `{step.confidence:.3f}`",
            ]
        )
        step_rows = []
        for spec in selected_specs:
            terms = _emission_term_breakdown(spec, speed=step.estimated_speed, accel=step.estimated_accel)
            prior = step.prior_weights[spec.name]
            log_prior = log(max(prior, 1e-12))
            log_numerator = log_prior + terms["emission_total"]
            posterior = step.posterior_weights[spec.name]
            step_rows.append((f"`{spec.name}`", f"{prior:.3f}", f"{log_prior:.3f}", f"{terms['speed_term']:.3f}", f"{terms['accel_term']:.3f}", f"{terms['abs_accel_term']:.3f}", f"{terms['emission_total']:.3f}", f"{log_numerator:.3f}", f"{posterior:.3f}"))
        report.table(["mode", "propagated prior", "log prior", "speed term", "accel term", "abs-accel term", "emission total", "log numerator", "posterior"], step_rows)
        report.paragraph("The transition-aware update at this step is")
        report.fence("\n".join([r"\log \tilde{p}_t(s) = \log \bar{p}_t(s) + \log E_t(s),", r"\qquad", r"p_t(s) = \frac{\exp(\log \tilde{p}_t(s))}{\sum_j \exp(\log \tilde{p}_t(j))}."]), language="tex")

    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "The static accumulator and transition-matrix accumulator use the same emission model; the difference is only the prior propagation step.",
            f"On this trajectory, the switched target mode `{switched_mode}` gets nontrivial prior mass before the emission term is applied because the transition matrix allows probability to move from the pre-switch mode family.",
            "That is the concrete mechanism by which the transition-aware accumulator improves post-switch behavior before the repo needs a full IMM implementation.",
        ]
    )
    return report.text()
