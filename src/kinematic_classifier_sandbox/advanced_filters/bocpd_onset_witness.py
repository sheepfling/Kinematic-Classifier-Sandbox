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
class BOCPDTruthRow:
    time: float
    truth_mode: str
    observation: float


@dataclass(frozen=True, slots=True)
class BOCPDPosteriorRow:
    method_id: str
    time: float
    label: str
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class BOCPDOnsetPosteriorRow:
    time: float
    onset_step: int
    posterior_probability: float


@dataclass(frozen=True, slots=True)
class BOCPDStateRow:
    time: float
    truth_mode: str
    observation: float
    hmm_maneuver_posterior: float
    hsmm_maneuver_posterior: float
    bocpd_maneuver_posterior: float
    bocpd_changepoint_probability: float
    hmm_predicted_mode: str
    hsmm_predicted_mode: str
    bocpd_predicted_mode: str


@dataclass(frozen=True, slots=True)
class BOCPDOnsetWitnessResult:
    truth_rows: tuple[BOCPDTruthRow, ...]
    posterior_rows: tuple[BOCPDPosteriorRow, ...]
    onset_rows: tuple[BOCPDOnsetPosteriorRow, ...]
    state_rows: tuple[BOCPDStateRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class BOCPDOnsetWitnessArtifacts:
    run_dir: Path
    truth_path: Path
    posterior_history_path: Path
    onset_posterior_path: Path
    state_estimate_history_path: Path
    summary_path: Path
    metrics_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _switch_delay(predicted_modes: list[str], truth_modes: list[str], target_index: int, target_mode: str) -> int:
    for index in range(target_index, len(predicted_modes)):
        if predicted_modes[index] == target_mode:
            return index - target_index
    return len(predicted_modes) - target_index


def analyze_bocpd_unknown_onset_witness(*, seed: int = 607) -> BOCPDOnsetWitnessResult:
    rng = random.default_rng(seed)
    times = tuple(float(time_value) for time_value in arange(0.0, 5.5, 0.25, dtype=float64))
    onset_index = 9
    truth_modes = ["coast"] * onset_index + ["maneuver"] * (len(times) - onset_index)
    observations = [
        float((0.0 if mode == "coast" else 0.9) + rng.normal(0.0, 0.50))
        for mode in truth_modes
    ]
    truth_rows = tuple(
        BOCPDTruthRow(time=time_value, truth_mode=truth_mode, observation=observation)
        for time_value, truth_mode, observation in zip(times, truth_modes, observations, strict=True)
    )

    start = wall_time.perf_counter()
    emission_variance = 0.50**2

    hmm_modes = ("coast", "maneuver")
    hmm_transition = ((0.92, 0.08), (0.05, 0.95))
    hmm_state = [0.90, 0.10]

    hsmm_states = ("coast", "maneuver_d3_s1", "maneuver_d3_s2", "maneuver_d3_s3", "maneuver_d5_s1", "maneuver_d5_s2", "maneuver_d5_s3", "maneuver_d5_s4", "maneuver_d5_s5")
    hsmm_index = {name: idx for idx, name in enumerate(hsmm_states)}
    coast = hsmm_index["coast"]
    hsmm_state = [0.0] * len(hsmm_states)
    hsmm_state[coast] = 1.0
    hsmm_transition = [[0.0 for _ in hsmm_states] for _ in hsmm_states]
    hsmm_transition[coast][coast] = 0.90
    hsmm_transition[coast][hsmm_index["maneuver_d3_s1"]] = 0.05
    hsmm_transition[coast][hsmm_index["maneuver_d5_s1"]] = 0.05
    for step in range(1, 3):
        hsmm_transition[hsmm_index[f"maneuver_d3_s{step}"]][hsmm_index[f"maneuver_d3_s{step + 1}"]] = 1.0
    hsmm_transition[hsmm_index["maneuver_d3_s3"]][coast] = 1.0
    for step in range(1, 5):
        hsmm_transition[hsmm_index[f"maneuver_d5_s{step}"]][hsmm_index[f"maneuver_d5_s{step + 1}"]] = 1.0
    hsmm_transition[hsmm_index["maneuver_d5_s5"]][coast] = 1.0

    hazard = 0.08
    onset_weights = [1.0]  # "no onset yet"

    posterior_rows: list[BOCPDPosteriorRow] = []
    onset_rows: list[BOCPDOnsetPosteriorRow] = []
    state_rows: list[BOCPDStateRow] = []
    hmm_predicted_modes: list[str] = []
    hsmm_predicted_modes: list[str] = []
    bocpd_predicted_modes: list[str] = []

    for step_index, (time_value, truth_mode, observation) in enumerate(zip(times, truth_modes, observations, strict=True)):
        hmm_predicted = [
            sum(hmm_state[source] * hmm_transition[source][destination] for source in range(len(hmm_modes)))
            for destination in range(len(hmm_modes))
        ]
        hmm_likelihood = [_normal_pdf(observation, 0.0 if mode == "coast" else 0.9, emission_variance) for mode in hmm_modes]
        hmm_state = [predicted * likelihood for predicted, likelihood in zip(hmm_predicted, hmm_likelihood, strict=True)]
        hmm_norm = max(sum(hmm_state), 1.0e-15)
        hmm_state = [float(value / hmm_norm) for value in hmm_state]

        hsmm_predicted = [
            sum(hsmm_state[source] * hsmm_transition[source][destination] for source in range(len(hsmm_states)))
            for destination in range(len(hsmm_states))
        ]
        hsmm_likelihood = [_normal_pdf(observation, 0.0 if state_name == "coast" else 0.9, emission_variance) for state_name in hsmm_states]
        hsmm_state = [predicted * likelihood for predicted, likelihood in zip(hsmm_predicted, hsmm_likelihood, strict=True)]
        hsmm_norm = max(sum(hsmm_state), 1.0e-15)
        hsmm_state = [float(value / hsmm_norm) for value in hsmm_state]

        next_weights = [0.0] * (step_index + 2)  # 0 = no onset yet, 1..t+1 = onset at prior step
        if onset_weights:
            no_onset_weight = onset_weights[0]
            next_weights[0] += no_onset_weight * (1.0 - hazard) * _normal_pdf(observation, 0.0, emission_variance)
            next_weights[step_index + 1] += no_onset_weight * hazard * _normal_pdf(observation, 0.9, emission_variance)
        for onset_step in range(1, len(onset_weights)):
            next_weights[onset_step] += onset_weights[onset_step] * _normal_pdf(observation, 0.9, emission_variance)
        onset_norm = max(sum(next_weights), 1.0e-15)
        onset_weights = [float(value / onset_norm) for value in next_weights]

        hmm_maneuver = float(hmm_state[1])
        hsmm_maneuver = float(sum(hsmm_state[idx] for idx in range(len(hsmm_states)) if idx != coast))
        bocpd_maneuver = float(sum(onset_weights[1:]))
        bocpd_cp = float(onset_weights[step_index + 1])
        hmm_mode = "maneuver" if hmm_maneuver > 0.5 else "coast"
        hsmm_mode = "maneuver" if hsmm_maneuver > 0.5 else "coast"
        bocpd_mode = "maneuver" if bocpd_maneuver > 0.5 else "coast"
        hmm_predicted_modes.append(hmm_mode)
        hsmm_predicted_modes.append(hsmm_mode)
        bocpd_predicted_modes.append(bocpd_mode)

        posterior_rows.extend(
            (
                BOCPDPosteriorRow("hmm_transition_v1", time_value, "coast", float(hmm_state[0])),
                BOCPDPosteriorRow("hmm_transition_v1", time_value, "maneuver", hmm_maneuver),
                BOCPDPosteriorRow("hsmm_duration_v1", time_value, "coast", float(hsmm_state[coast])),
                BOCPDPosteriorRow("hsmm_duration_v1", time_value, "maneuver", hsmm_maneuver),
                BOCPDPosteriorRow("bocpd_v1", time_value, "coast", 1.0 - bocpd_maneuver),
                BOCPDPosteriorRow("bocpd_v1", time_value, "maneuver", bocpd_maneuver),
            )
        )
        for onset_step, probability in enumerate(onset_weights):
            onset_rows.append(
                BOCPDOnsetPosteriorRow(
                    time=time_value,
                    onset_step=onset_step - 1,  # -1 means no onset yet
                    posterior_probability=float(probability),
                )
            )
        state_rows.append(
            BOCPDStateRow(
                time=time_value,
                truth_mode=truth_mode,
                observation=observation,
                hmm_maneuver_posterior=hmm_maneuver,
                hsmm_maneuver_posterior=hsmm_maneuver,
                bocpd_maneuver_posterior=bocpd_maneuver,
                bocpd_changepoint_probability=bocpd_cp,
                hmm_predicted_mode=hmm_mode,
                hsmm_predicted_mode=hsmm_mode,
                bocpd_predicted_mode=bocpd_mode,
            )
        )

    runtime_seconds = wall_time.perf_counter() - start
    truth_maneuver_mask = [1.0 if mode == "maneuver" else 0.0 for mode in truth_modes]
    hmm_accuracy = float(sum(pred == truth for pred, truth in zip(hmm_predicted_modes, truth_modes, strict=True)) / len(truth_modes))
    hsmm_accuracy = float(sum(pred == truth for pred, truth in zip(hsmm_predicted_modes, truth_modes, strict=True)) / len(truth_modes))
    bocpd_accuracy = float(sum(pred == truth for pred, truth in zip(bocpd_predicted_modes, truth_modes, strict=True)) / len(truth_modes))
    hmm_brier = float(sum((row.hmm_maneuver_posterior - truth) ** 2 for row, truth in zip(state_rows, truth_maneuver_mask, strict=True)) / len(state_rows))
    hsmm_brier = float(sum((row.hsmm_maneuver_posterior - truth) ** 2 for row, truth in zip(state_rows, truth_maneuver_mask, strict=True)) / len(state_rows))
    bocpd_brier = float(sum((row.bocpd_maneuver_posterior - truth) ** 2 for row, truth in zip(state_rows, truth_maneuver_mask, strict=True)) / len(state_rows))
    hmm_onset_delay = _switch_delay(hmm_predicted_modes, truth_modes, onset_index, "maneuver")
    hsmm_onset_delay = _switch_delay(hsmm_predicted_modes, truth_modes, onset_index, "maneuver")
    bocpd_onset_delay = _switch_delay(bocpd_predicted_modes, truth_modes, onset_index, "maneuver")
    changepoint_peak_index = max(range(len(state_rows)), key=lambda idx: state_rows[idx].bocpd_changepoint_probability)
    final_onset_rows = [row for row in onset_rows if row.time == times[-1]]
    final_map_onset_step = max(final_onset_rows, key=lambda row: row.posterior_probability).onset_step
    final_truth_window_mass = float(
        sum(
            row.posterior_probability
            for row in final_onset_rows
            if row.onset_step >= onset_index - 1 and row.onset_step <= onset_index + 2
        )
    )
    promotion_decision = (
        "promote_bocpd_for_unknown_maneuver_onset"
        if bocpd_accuracy >= max(hmm_accuracy, hsmm_accuracy)
        and bocpd_brier < min(hmm_brier, hsmm_brier)
        and bocpd_onset_delay <= min(hmm_onset_delay, hsmm_onset_delay)
        and abs(final_map_onset_step - onset_index) <= 2
        and final_truth_window_mass >= 0.40
        else "revise_bocpd_onset_witness"
    )
    metrics = {
        "study_id": "bocpd_unknown_maneuver_onset_v1",
        "seed": seed,
        "step_count": len(times),
        "hmm_mode_accuracy": hmm_accuracy,
        "hsmm_mode_accuracy": hsmm_accuracy,
        "bocpd_mode_accuracy": bocpd_accuracy,
        "hmm_maneuver_brier": hmm_brier,
        "hsmm_maneuver_brier": hsmm_brier,
        "bocpd_maneuver_brier": bocpd_brier,
        "hmm_onset_delay_steps": hmm_onset_delay,
        "hsmm_onset_delay_steps": hsmm_onset_delay,
        "bocpd_onset_delay_steps": bocpd_onset_delay,
        "bocpd_changepoint_peak_index": changepoint_peak_index,
        "bocpd_final_map_onset_step": final_map_onset_step,
        "bocpd_final_truth_window_mass": final_truth_window_mass,
        "truth_onset_index": onset_index,
        "runtime_seconds": runtime_seconds,
        "promotion_decision": promotion_decision,
    }
    return BOCPDOnsetWitnessResult(
        truth_rows=truth_rows,
        posterior_rows=tuple(posterior_rows),
        onset_rows=tuple(onset_rows),
        state_rows=tuple(state_rows),
        metrics=metrics,
    )


def write_bocpd_unknown_onset_witness_artifacts(
    output_dir: str | Path,
    *,
    result: BOCPDOnsetWitnessResult | None = None,
) -> BOCPDOnsetWitnessArtifacts:
    analysis = result or analyze_bocpd_unknown_onset_witness()
    run_dir = Path(output_dir) / "bocpd_unknown_maneuver_onset_v1"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    truth_path = run_dir / "truth_history.csv"
    posterior_path = run_dir / "posterior_history.csv"
    onset_path = run_dir / "onset_posterior.csv"
    state_path = run_dir / "state_estimate_history.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    decision_card_path = run_dir / "decision_card.md"
    posterior_plot_path = plot_dir / "posterior_timeline.png"
    changepoint_plot_path = plot_dir / "changepoint_probability_timeline.png"
    onset_heatmap_path = plot_dir / "onset_posterior_heatmap.png"

    write_csv(truth_path, [asdict(row) for row in analysis.truth_rows], ["time", "truth_mode", "observation"])
    write_csv(posterior_path, [asdict(row) for row in analysis.posterior_rows], ["method_id", "time", "label", "posterior_probability"])
    write_csv(onset_path, [asdict(row) for row in analysis.onset_rows], ["time", "onset_step", "posterior_probability"])
    write_csv(
        state_path,
        [asdict(row) for row in analysis.state_rows],
        [
            "time",
            "truth_mode",
            "observation",
            "hmm_maneuver_posterior",
            "hsmm_maneuver_posterior",
            "bocpd_maneuver_posterior",
            "bocpd_changepoint_probability",
            "hmm_predicted_mode",
            "hsmm_predicted_mode",
            "bocpd_predicted_mode",
        ],
    )
    write_comparison_summary_csv(summary_path, [analysis.metrics])
    write_csv(metrics_path, [analysis.metrics], list(analysis.metrics))
    decision_card_path.write_text(_render_bocpd_decision_card(analysis), encoding="utf-8")
    _write_bocpd_plots(analysis, posterior_plot_path, changepoint_plot_path, onset_heatmap_path)
    return BOCPDOnsetWitnessArtifacts(
        run_dir=run_dir,
        truth_path=truth_path,
        posterior_history_path=posterior_path,
        onset_posterior_path=onset_path,
        state_estimate_history_path=state_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        decision_card_path=decision_card_path,
        plot_paths=(posterior_plot_path, changepoint_plot_path, onset_heatmap_path),
    )


def bocpd_unknown_onset_witness_surface() -> AdvancedFilterSurface[BOCPDOnsetWitnessResult, BOCPDOnsetWitnessArtifacts]:
    return AdvancedFilterSurface(
        study_id="bocpd_unknown_maneuver_onset_v1",
        run=analyze_bocpd_unknown_onset_witness,
        write_artifacts=write_bocpd_unknown_onset_witness_artifacts,
        describe_artifacts=lambda artifacts: (
            str(artifacts.run_dir),
            str(artifacts.metrics_path),
            str(artifacts.decision_card_path),
        ),
        metadata={
            "study_kind": "changepoint_witness",
            "problem_family": "unknown_maneuver_onset",
        },
    )


def _render_bocpd_decision_card(result: BOCPDOnsetWitnessResult) -> str:
    report = MarkdownDocument("BOCPD Unknown Maneuver Onset Witness")
    report.paragraph(
        "This witness asks whether explicit changepoint reasoning improves unknown maneuver-onset recovery beyond HMM and duration-model smoothing."
    )
    report.bullet_list(
        [
            f"HMM accuracy: `{result.metrics['hmm_mode_accuracy']}`",
            f"HSMM accuracy: `{result.metrics['hsmm_mode_accuracy']}`",
            f"BOCPD accuracy: `{result.metrics['bocpd_mode_accuracy']}`",
            f"HMM Brier: `{result.metrics['hmm_maneuver_brier']}`",
            f"HSMM Brier: `{result.metrics['hsmm_maneuver_brier']}`",
            f"BOCPD Brier: `{result.metrics['bocpd_maneuver_brier']}`",
            f"BOCPD onset delay: `{result.metrics['bocpd_onset_delay_steps']}`",
            f"BOCPD changepoint peak index: `{result.metrics['bocpd_changepoint_peak_index']}`",
            f"BOCPD final MAP onset step: `{result.metrics['bocpd_final_map_onset_step']}`",
            f"BOCPD final truth-window mass: `{result.metrics['bocpd_final_truth_window_mass']}`",
            f"Truth onset index: `{result.metrics['truth_onset_index']}`",
            f"Decision: `{result.metrics['promotion_decision']}`",
        ]
    )
    report.paragraph(
        "Interpretation: if changepoint posterior mass localizes onset more cleanly than HMM/HSMM maneuver posteriors, BOCPD earns the unknown-onset blocker rung."
    )
    return report.text()


def _write_bocpd_plots(
    result: BOCPDOnsetWitnessResult,
    posterior_plot_path: Path,
    changepoint_plot_path: Path,
    onset_heatmap_path: Path,
) -> None:
    times = [row.time for row in result.state_rows]
    truth = [1.0 if row.truth_mode == "maneuver" else 0.0 for row in result.state_rows]
    hmm = [row.hmm_maneuver_posterior for row in result.state_rows]
    hsmm = [row.hsmm_maneuver_posterior for row in result.state_rows]
    bocpd = [row.bocpd_maneuver_posterior for row in result.state_rows]
    cp = [row.bocpd_changepoint_probability for row in result.state_rows]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(times, truth, label="truth maneuver", color="black", linewidth=1.2)
    ax.plot(times, hmm, label="HMM")
    ax.plot(times, hsmm, label="HSMM")
    ax.plot(times, bocpd, label="BOCPD")
    ax.set_title("Posterior timeline")
    ax.set_xlabel("time")
    ax.set_ylabel("maneuver posterior")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(posterior_plot_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(times, cp, label="BOCPD changepoint probability")
    ax.set_title("Changepoint probability timeline")
    ax.set_xlabel("time")
    ax.set_ylabel("probability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(changepoint_plot_path)
    plt.close(fig)

    onset_labels = sorted({row.onset_step for row in result.onset_rows})
    matrix = []
    for onset_step in onset_labels:
        matrix.append(
            [
                next(
                    (
                        onset_row.posterior_probability
                        for onset_row in result.onset_rows
                        if onset_row.time == time_value and onset_row.onset_step == onset_step
                    ),
                    0.0,
                )
                for time_value in times
            ]
        )
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(onset_labels)))
    ax.set_yticklabels([str(label) for label in onset_labels], fontsize=7)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels([f"{time_value:.2f}" for time_value in times], rotation=90, fontsize=6)
    ax.set_title("Onset posterior heatmap")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(onset_heatmap_path)
    plt.close(fig)
