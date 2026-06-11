from __future__ import annotations

import math
import time as wall_time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy.random as random
from numpy import arange, float64

from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import write_comparison_summary_csv
from kinematic_classifier_sandbox.markdown_builder import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import plt

from .surface import AdvancedFilterSurface


def _normal_pdf(x: float, mean_value: float, variance: float) -> float:
    variance = max(float(variance), 1.0e-12)
    return math.exp(-0.5 * ((float(x) - float(mean_value)) ** 2) / variance) / math.sqrt(2.0 * math.pi * variance)


@dataclass(frozen=True, slots=True)
class HSMMDurationTruthRow:
    time: float
    truth_mode: str
    observation: float


@dataclass(frozen=True, slots=True)
class HSMMDurationPosteriorRow:
    method_id: str
    time: float
    mode: str
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class HSMMChainPosteriorRow:
    time: float
    chain_state: str
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class HSMMDurationStateRow:
    time: float
    truth_mode: str
    observation: float
    hmm_coast_posterior: float
    hmm_maneuver_posterior: float
    hsmm_coast_posterior: float
    hsmm_maneuver_posterior: float
    hmm_predicted_mode: str
    hsmm_predicted_mode: str


@dataclass(frozen=True, slots=True)
class HSMMDurationWitnessResult:
    truth_rows: tuple[HSMMDurationTruthRow, ...]
    posterior_rows: tuple[HSMMDurationPosteriorRow, ...]
    chain_rows: tuple[HSMMChainPosteriorRow, ...]
    state_rows: tuple[HSMMDurationStateRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class HSMMDurationWitnessArtifacts:
    run_dir: Path
    truth_path: Path
    posterior_history_path: Path
    chain_posterior_path: Path
    state_estimate_history_path: Path
    summary_path: Path
    metrics_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _transition_normalize(matrix: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in matrix:
        row_sum = max(sum(row), 1.0e-15)
        normalized.append([float(value / row_sum) for value in row])
    return normalized


def _switch_delay(predicted_modes: list[str], truth_modes: list[str], target_index: int, target_mode: str) -> int:
    for index in range(target_index, len(predicted_modes)):
        if predicted_modes[index] == target_mode:
            return index - target_index
    return len(predicted_modes) - target_index


def analyze_hsmm_duration_limited_witness(*, seed: int = 503) -> HSMMDurationWitnessResult:
    rng = random.default_rng(seed)
    times = tuple(float(time_value) for time_value in arange(0.0, 5.25, 0.25, dtype=float64))
    truth_modes = ["coast"] * 8 + ["maneuver"] * 5 + ["coast"] * 8
    assert len(truth_modes) == len(times)
    observations = [
        float((0.0 if mode == "coast" else 0.9) + rng.normal(0.0, 0.55))
        for mode in truth_modes
    ]

    truth_rows = tuple(
        HSMMDurationTruthRow(time=time_value, truth_mode=truth_mode, observation=observation)
        for time_value, truth_mode, observation in zip(times, truth_modes, observations, strict=True)
    )

    start = wall_time.perf_counter()
    emission_variance = 0.55**2

    hmm_modes = ("coast", "maneuver")
    hmm_transition = _transition_normalize([[0.90, 0.10], [0.10, 0.90]])
    hmm_prior = [0.85, 0.15]
    hmm_state = list(hmm_prior)

    hsmm_states = (
        "coast",
        "maneuver_d4_s1",
        "maneuver_d4_s2",
        "maneuver_d4_s3",
        "maneuver_d4_s4",
        "maneuver_d5_s1",
        "maneuver_d5_s2",
        "maneuver_d5_s3",
        "maneuver_d5_s4",
        "maneuver_d5_s5",
        "maneuver_d6_s1",
        "maneuver_d6_s2",
        "maneuver_d6_s3",
        "maneuver_d6_s4",
        "maneuver_d6_s5",
        "maneuver_d6_s6",
    )
    hsmm_index = {name: idx for idx, name in enumerate(hsmm_states)}
    hsmm_prior = [0.0] * len(hsmm_states)
    hsmm_prior[hsmm_index["coast"]] = 1.0
    hsmm_transition = [[0.0 for _ in hsmm_states] for _ in hsmm_states]
    coast = hsmm_index["coast"]
    hsmm_transition[coast][coast] = 0.88
    hsmm_transition[coast][hsmm_index["maneuver_d4_s1"]] = 0.04
    hsmm_transition[coast][hsmm_index["maneuver_d5_s1"]] = 0.04
    hsmm_transition[coast][hsmm_index["maneuver_d6_s1"]] = 0.04
    for duration in (4, 5, 6):
        for step in range(1, duration):
            hsmm_transition[hsmm_index[f"maneuver_d{duration}_s{step}"]][hsmm_index[f"maneuver_d{duration}_s{step + 1}"]] = 1.0
        hsmm_transition[hsmm_index[f"maneuver_d{duration}_s{duration}"]][coast] = 1.0
    hsmm_state = list(hsmm_prior)

    posterior_rows: list[HSMMDurationPosteriorRow] = []
    chain_rows: list[HSMMChainPosteriorRow] = []
    state_rows: list[HSMMDurationStateRow] = []
    hmm_predicted_modes: list[str] = []
    hsmm_predicted_modes: list[str] = []

    for time_value, truth_mode, observation in zip(times, truth_modes, observations, strict=True):
        hmm_predicted = [
            sum(hmm_state[source] * hmm_transition[source][destination] for source in range(len(hmm_modes)))
            for destination in range(len(hmm_modes))
        ]
        hmm_likelihood = [
            _normal_pdf(observation, 0.0 if mode == "coast" else 0.9, emission_variance)
            for mode in hmm_modes
        ]
        hmm_unnormalized = [predicted * likelihood for predicted, likelihood in zip(hmm_predicted, hmm_likelihood, strict=True)]
        hmm_norm = max(sum(hmm_unnormalized), 1.0e-15)
        hmm_state = [float(value / hmm_norm) for value in hmm_unnormalized]

        hsmm_predicted = [
            sum(hsmm_state[source] * hsmm_transition[source][destination] for source in range(len(hsmm_states)))
            for destination in range(len(hsmm_states))
        ]
        hsmm_likelihood = [
            _normal_pdf(observation, 0.0 if state_name == "coast" else 0.9, emission_variance)
            for state_name in hsmm_states
        ]
        hsmm_unnormalized = [predicted * likelihood for predicted, likelihood in zip(hsmm_predicted, hsmm_likelihood, strict=True)]
        hsmm_norm = max(sum(hsmm_unnormalized), 1.0e-15)
        hsmm_state = [float(value / hsmm_norm) for value in hsmm_unnormalized]

        hmm_coast = float(hmm_state[0])
        hmm_maneuver = float(hmm_state[1])
        hsmm_coast = float(hsmm_state[coast])
        hsmm_maneuver = float(sum(hsmm_state[idx] for idx in range(len(hsmm_states)) if idx != coast))
        hmm_predicted_mode = "coast" if hmm_coast >= hmm_maneuver else "maneuver"
        hsmm_predicted_mode = "coast" if hsmm_coast >= hsmm_maneuver else "maneuver"
        hmm_predicted_modes.append(hmm_predicted_mode)
        hsmm_predicted_modes.append(hsmm_predicted_mode)

        posterior_rows.extend(
            (
                HSMMDurationPosteriorRow("hmm_transition_v1", time_value, "coast", hmm_coast),
                HSMMDurationPosteriorRow("hmm_transition_v1", time_value, "maneuver", hmm_maneuver),
                HSMMDurationPosteriorRow("hsmm_duration_v1", time_value, "coast", hsmm_coast),
                HSMMDurationPosteriorRow("hsmm_duration_v1", time_value, "maneuver", hsmm_maneuver),
            )
        )
        for state_name, probability in zip(hsmm_states, hsmm_state, strict=True):
            chain_rows.append(HSMMChainPosteriorRow(time_value, state_name, float(probability)))
        state_rows.append(
            HSMMDurationStateRow(
                time=time_value,
                truth_mode=truth_mode,
                observation=observation,
                hmm_coast_posterior=hmm_coast,
                hmm_maneuver_posterior=hmm_maneuver,
                hsmm_coast_posterior=hsmm_coast,
                hsmm_maneuver_posterior=hsmm_maneuver,
                hmm_predicted_mode=hmm_predicted_mode,
                hsmm_predicted_mode=hsmm_predicted_mode,
            )
        )

    runtime_seconds = wall_time.perf_counter() - start
    truth_maneuver_mask = [1.0 if mode == "maneuver" else 0.0 for mode in truth_modes]
    hmm_accuracy = float(sum(pred == truth for pred, truth in zip(hmm_predicted_modes, truth_modes, strict=True)) / len(truth_modes))
    hsmm_accuracy = float(sum(pred == truth for pred, truth in zip(hsmm_predicted_modes, truth_modes, strict=True)) / len(truth_modes))
    onset_index = truth_modes.index("maneuver")
    offset_index = onset_index + 5
    hmm_onset_delay = _switch_delay(hmm_predicted_modes, truth_modes, onset_index, "maneuver")
    hsmm_onset_delay = _switch_delay(hsmm_predicted_modes, truth_modes, onset_index, "maneuver")
    hmm_offset_delay = _switch_delay(hmm_predicted_modes, truth_modes, offset_index, "coast")
    hsmm_offset_delay = _switch_delay(hsmm_predicted_modes, truth_modes, offset_index, "coast")
    hmm_maneuver_brier = float(sum((row.hmm_maneuver_posterior - truth) ** 2 for row, truth in zip(state_rows, truth_maneuver_mask, strict=True)) / len(state_rows))
    hsmm_maneuver_brier = float(sum((row.hsmm_maneuver_posterior - truth) ** 2 for row, truth in zip(state_rows, truth_maneuver_mask, strict=True)) / len(state_rows))
    promotion_decision = (
        "promote_hsmm_for_duration_limited_maneuver"
        if hsmm_accuracy > hmm_accuracy
        and hsmm_maneuver_brier < hmm_maneuver_brier
        and hsmm_offset_delay < hmm_offset_delay
        else "revise_hsmm_duration_witness"
    )
    metrics = {
        "study_id": "hsmm_duration_limited_maneuver_v1",
        "seed": seed,
        "step_count": len(times),
        "hmm_mode_accuracy": hmm_accuracy,
        "hsmm_mode_accuracy": hsmm_accuracy,
        "hmm_maneuver_brier": hmm_maneuver_brier,
        "hsmm_maneuver_brier": hsmm_maneuver_brier,
        "hmm_onset_delay_steps": hmm_onset_delay,
        "hsmm_onset_delay_steps": hsmm_onset_delay,
        "hmm_offset_delay_steps": hmm_offset_delay,
        "hsmm_offset_delay_steps": hsmm_offset_delay,
        "runtime_seconds": runtime_seconds,
        "promotion_decision": promotion_decision,
    }
    return HSMMDurationWitnessResult(
        truth_rows=truth_rows,
        posterior_rows=tuple(posterior_rows),
        chain_rows=tuple(chain_rows),
        state_rows=tuple(state_rows),
        metrics=metrics,
    )


def write_hsmm_duration_limited_witness_artifacts(
    output_dir: str | Path,
    *,
    result: HSMMDurationWitnessResult | None = None,
) -> HSMMDurationWitnessArtifacts:
    analysis = result or analyze_hsmm_duration_limited_witness()
    run_dir = Path(output_dir) / "hsmm_duration_limited_maneuver_v1"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    truth_path = run_dir / "truth_history.csv"
    posterior_path = run_dir / "posterior_history.csv"
    chain_path = run_dir / "duration_chain_posterior.csv"
    state_path = run_dir / "state_estimate_history.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    decision_card_path = run_dir / "decision_card.md"
    posterior_plot_path = plot_dir / "posterior_timeline.png"
    chain_plot_path = plot_dir / "duration_chain_heatmap.png"
    recovery_plot_path = plot_dir / "exit_recovery_panel.png"

    write_csv(truth_path, [asdict(row) for row in analysis.truth_rows], ["time", "truth_mode", "observation"])
    write_csv(posterior_path, [asdict(row) for row in analysis.posterior_rows], ["method_id", "time", "mode", "posterior_probability"])
    write_csv(chain_path, [asdict(row) for row in analysis.chain_rows], ["time", "chain_state", "posterior_probability"])
    write_csv(
        state_path,
        [asdict(row) for row in analysis.state_rows],
        [
            "time",
            "truth_mode",
            "observation",
            "hmm_coast_posterior",
            "hmm_maneuver_posterior",
            "hsmm_coast_posterior",
            "hsmm_maneuver_posterior",
            "hmm_predicted_mode",
            "hsmm_predicted_mode",
        ],
    )
    write_comparison_summary_csv(summary_path, [analysis.metrics])
    write_csv(metrics_path, [analysis.metrics], list(analysis.metrics))
    decision_card_path.write_text(_render_hsmm_decision_card(analysis), encoding="utf-8")
    _write_hsmm_plots(analysis, posterior_plot_path, chain_plot_path, recovery_plot_path)
    return HSMMDurationWitnessArtifacts(
        run_dir=run_dir,
        truth_path=truth_path,
        posterior_history_path=posterior_path,
        chain_posterior_path=chain_path,
        state_estimate_history_path=state_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        decision_card_path=decision_card_path,
        plot_paths=(posterior_plot_path, chain_plot_path, recovery_plot_path),
    )


def hsmm_duration_limited_witness_surface() -> AdvancedFilterSurface[HSMMDurationWitnessResult, HSMMDurationWitnessArtifacts]:
    return AdvancedFilterSurface(
        study_id="hsmm_duration_limited_maneuver_v1",
        run=analyze_hsmm_duration_limited_witness,
        write_artifacts=write_hsmm_duration_limited_witness_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.metrics_path),
            str(artifacts.decision_card_path),
        ),
        metadata={
            "study_kind": "duration_witness",
            "problem_family": "duration_limited_maneuver",
        },
    )


def _render_hsmm_decision_card(result: HSMMDurationWitnessResult) -> str:
    report = MarkdownDocument("HSMM Duration-Limited Maneuver Witness")
    report.paragraph(
        "This witness asks whether explicit maneuver duration structure explains failures that a geometric-duration HMM smooths poorly."
    )
    report.bullet_list(
        [
            f"HMM accuracy: `{result.metrics['hmm_mode_accuracy']}`",
            f"HSMM accuracy: `{result.metrics['hsmm_mode_accuracy']}`",
            f"HMM maneuver Brier: `{result.metrics['hmm_maneuver_brier']}`",
            f"HSMM maneuver Brier: `{result.metrics['hsmm_maneuver_brier']}`",
            f"HMM onset delay: `{result.metrics['hmm_onset_delay_steps']}`",
            f"HSMM onset delay: `{result.metrics['hsmm_onset_delay_steps']}`",
            f"HMM offset delay: `{result.metrics['hmm_offset_delay_steps']}`",
            f"HSMM offset delay: `{result.metrics['hsmm_offset_delay_steps']}`",
            f"Decision: `{result.metrics['promotion_decision']}`",
        ]
    )
    report.paragraph(
        "Interpretation: if explicit duration shortens exit lag and improves posterior quality on a fixed-duration maneuver witness, HSMM earns the duration blocker rung before BOCPD or richer latent-structure models."
    )
    return report.text()


def _write_hsmm_plots(
    result: HSMMDurationWitnessResult,
    posterior_plot_path: Path,
    chain_plot_path: Path,
    recovery_plot_path: Path,
) -> None:
    times = [row.time for row in result.state_rows]
    hmm_maneuver = [row.hmm_maneuver_posterior for row in result.state_rows]
    hsmm_maneuver = [row.hsmm_maneuver_posterior for row in result.state_rows]
    truth = [1.0 if row.truth_mode == "maneuver" else 0.0 for row in result.state_rows]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(times, truth, label="truth maneuver", color="black", linewidth=1.2)
    ax.plot(times, hmm_maneuver, label="HMM maneuver posterior")
    ax.plot(times, hsmm_maneuver, label="HSMM maneuver posterior")
    ax.set_title("Posterior timeline")
    ax.set_xlabel("time")
    ax.set_ylabel("maneuver posterior")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(posterior_plot_path)
    plt.close(fig)

    ordered_states = sorted({row.chain_state for row in result.chain_rows})
    matrix = []
    for state_name in ordered_states:
        row_values = []
        for time_value in times:
            probability = next(
                chain_row.posterior_probability
                for chain_row in result.chain_rows
                if chain_row.time == time_value and chain_row.chain_state == state_name
            )
            row_values.append(probability)
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(ordered_states)))
    ax.set_yticklabels(ordered_states, fontsize=7)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels([f"{time_value:.2f}" for time_value in times], rotation=90, fontsize=6)
    ax.set_title("Duration chain heatmap")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(chain_plot_path)
    plt.close(fig)

    onset_index = next(index for index, row in enumerate(result.state_rows) if row.truth_mode == "maneuver")
    offset_index = onset_index + 5
    window = range(max(offset_index - 3, 0), min(offset_index + 5, len(result.state_rows)))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot([times[index] for index in window], [truth[index] for index in window], label="truth maneuver", color="black", linewidth=1.2)
    ax.plot([times[index] for index in window], [hmm_maneuver[index] for index in window], label="HMM")
    ax.plot([times[index] for index in window], [hsmm_maneuver[index] for index in window], label="HSMM")
    ax.set_title("Exit recovery panel")
    ax.set_xlabel("time")
    ax.set_ylabel("maneuver posterior")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(recovery_plot_path)
    plt.close(fig)
