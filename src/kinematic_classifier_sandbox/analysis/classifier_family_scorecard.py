from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.registry.algorithm_coverage_matrix import (
    ALGORITHM_COVERAGE_MATRIX,
    AlgorithmCoverageEntry,
)
from kinematic_classifier_sandbox.registry.method_validation_os import (
    MethodSpec,
    analyze_method_validation_os,
)
from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..utils.io import write_csv
from ..utils.plotting import _figure_to_png, plt


@dataclass(frozen=True, slots=True)
class FamilyScorecardSpec:
    method_id: str
    display_name: str
    public_family: str
    algorithm_coverage_method_id: str | None
    method_validation_id: str | None
    validation_ladder_classifier_ids: tuple[str, ...]
    complexity_band: str
    complexity_score: float
    capability_added: str
    expected_win_condition: str
    expected_failure_mode: str
    evidence_type: str
    assumptions: str
    three_d_relevance: str
    latent_events: bool
    mode_mixing: bool


@dataclass(frozen=True, slots=True)
class FamilyCapabilityRow:
    method_id: str
    display_name: str
    public_family: str
    history_accumulation: str
    dynamics_evidence: str
    switching_logic: str
    mode_mixing: str
    nonlinear_state: str
    nongaussian_state: str
    latent_events: str
    interpretability: str
    calibration: str
    online_inference: str
    lift_relevance_3d: str
    current_status: str


@dataclass(frozen=True, slots=True)
class CeilingEfficiencyRow:
    method_id: str
    display_name: str
    public_family: str
    complexity_band: str
    complexity_score: float
    ladder_pair_count: int
    mean_classifier_accuracy: float | None
    mean_epic1_oracle_proxy: float | None
    mean_best_pair_oracle_proxy: float | None
    mean_fraction_of_proxy_captured_capped: float | None
    mean_proxy_gap: float | None
    proxy_exceeded_fraction: float | None
    ceiling_status: str
    evidence_note: str


@dataclass(frozen=True, slots=True)
class FamilySummaryRow:
    public_family: str
    method_count: int
    witness_supported_or_better: int
    study_justified_or_better: int
    capability_added: str
    expected_win_condition: str
    mean_fraction_of_proxy_captured_capped: float | None
    ceiling_alignment_status: str
    three_d_relevance: str


@dataclass(frozen=True, slots=True)
class ClassifierFamilyScorecardResult:
    atlas_specs: tuple[FamilyScorecardSpec, ...]
    capability_rows: tuple[FamilyCapabilityRow, ...]
    ceiling_rows: tuple[CeilingEfficiencyRow, ...]
    family_summary_rows: tuple[FamilySummaryRow, ...]
    metrics: dict[str, object]
    atlas_markdown: str
    report_markdown: str


@dataclass(frozen=True, slots=True)
class ClassifierFamilyScorecardArtifacts:
    run_dir: Path
    capability_matrix_path: Path
    ceiling_efficiency_path: Path
    family_summary_path: Path
    atlas_path: Path
    report_path: Path
    summary_path: Path
    plot_paths: tuple[Path, ...]


SCORECARD_SPECS: tuple[FamilyScorecardSpec, ...] = (
    FamilyScorecardSpec(
        method_id="pointwise",
        display_name="Pointwise",
        public_family="interpretable_kinematic",
        algorithm_coverage_method_id="pointwise_likelihood",
        method_validation_id="pointwise",
        validation_ladder_classifier_ids=("pointwise",),
        complexity_band="low",
        complexity_score=1.0,
        capability_added="Instantaneous local evidence without temporal state.",
        expected_win_condition="Easy separable classes where late-time evidence alone is sufficient.",
        expected_failure_mode="Posterior flicker, local ambiguity, and hidden temporal ordering.",
        evidence_type="Per-step likelihood from current observation only.",
        assumptions="Current observation is sufficiently informative without history.",
        three_d_relevance="medium",
        latent_events=False,
        mode_mixing=False,
    ),
    FamilyScorecardSpec(
        method_id="windowed",
        display_name="Windowed",
        public_family="interpretable_kinematic",
        algorithm_coverage_method_id="windowed_feature_likelihood",
        method_validation_id="windowed",
        validation_ladder_classifier_ids=("windowed_raw_extrema", "windowed_robust_extrema", "windowed_shape_features"),
        complexity_band="low_medium",
        complexity_score=1.8,
        capability_added="Short-horizon temporal summaries and robust local history.",
        expected_win_condition="Timing-sensitive or outlier-stressed classes where local summaries matter.",
        expected_failure_mode="Long-range ordering, latent switches, and matched-endpoint dynamics.",
        evidence_type="Window statistics and short shape summaries.",
        assumptions="Short local context carries most of the usable signal.",
        three_d_relevance="medium",
        latent_events=False,
        mode_mixing=False,
    ),
    FamilyScorecardSpec(
        method_id="sequential_bayes",
        display_name="Sequential Bayes",
        public_family="sequential_evidence",
        algorithm_coverage_method_id="sequential_bayes",
        method_validation_id=None,
        validation_ladder_classifier_ids=("bayes_accumulator",),
        complexity_band="medium",
        complexity_score=2.4,
        capability_added="Recursive accumulation of evidence through time.",
        expected_win_condition="Weak repeated evidence where history accumulation resolves ambiguity.",
        expected_failure_mode="Mode switching and dynamics mismatch when persistence is not enough.",
        evidence_type="Recursive prior-likelihood-posterior update.",
        assumptions="Per-step evidence is conditionally useful and should be accumulated over time.",
        three_d_relevance="high",
        latent_events=False,
        mode_mixing=False,
    ),
    FamilyScorecardSpec(
        method_id="kalman_bank",
        display_name="Kalman Bank",
        public_family="physics_aware_inference",
        algorithm_coverage_method_id="kalman_bank",
        method_validation_id="kalman_bank",
        validation_ladder_classifier_ids=("kalman_bank",),
        complexity_band="medium_high",
        complexity_score=3.0,
        capability_added="Dynamics-aware innovation evidence and state uncertainty.",
        expected_win_condition="Matched-endpoint or dynamics-separable cases where residuals matter more than local features.",
        expected_failure_mode="Switching, nonlinear ambiguity, or non-Gaussian state structure.",
        evidence_type="Innovation likelihoods from class-conditioned linear-Gaussian models.",
        assumptions="Candidate dynamics are approximately linear-Gaussian and class-conditioned.",
        three_d_relevance="high",
        latent_events=False,
        mode_mixing=False,
    ),
    FamilyScorecardSpec(
        method_id="transition_matrix",
        display_name="Transition Matrix",
        public_family="physics_aware_inference",
        algorithm_coverage_method_id="transition_hmm",
        method_validation_id="hmm_transition",
        validation_ladder_classifier_ids=(),
        complexity_band="medium_high",
        complexity_score=3.4,
        capability_added="Switch persistence and explicit regime-transition logic.",
        expected_win_condition="Switching witnesses where label persistence and transition constraints reduce flicker.",
        expected_failure_mode="Continuous-state mode mixing or latent event timing beyond a label Markov chain.",
        evidence_type="Transition-aware posterior propagation with per-step emissions.",
        assumptions="Regime switches are well described by a Markov transition prior over labels.",
        three_d_relevance="high",
        latent_events=False,
        mode_mixing=False,
    ),
    FamilyScorecardSpec(
        method_id="imm",
        display_name="IMM",
        public_family="physics_aware_inference",
        algorithm_coverage_method_id="imm",
        method_validation_id="imm",
        validation_ladder_classifier_ids=(),
        complexity_band="high",
        complexity_score=4.1,
        capability_added="Mode-conditioned state mixing across switching dynamics.",
        expected_win_condition="Switching dynamic regimes where transition-only smoothing is insufficient.",
        expected_failure_mode="Strongly nonlinear, non-Gaussian, or large latent mode spaces.",
        evidence_type="Mixed model-conditioned state estimates and mode posteriors.",
        assumptions="Switching dynamics remain within a tractable linear-Gaussian multi-model bank.",
        three_d_relevance="high",
        latent_events=False,
        mode_mixing=True,
    ),
    FamilyScorecardSpec(
        method_id="particle_filter",
        display_name="Particle Filter",
        public_family="physics_aware_inference",
        algorithm_coverage_method_id="particle_filter",
        method_validation_id="particle_filter",
        validation_ladder_classifier_ids=(),
        complexity_band="very_high",
        complexity_score=4.8,
        capability_added="Weighted-sample approximation of nonlinear or non-Gaussian posteriors.",
        expected_win_condition="Witnesses with multimodal, nonlinear, or heavy-tailed state evidence.",
        expected_failure_mode="Compute-heavy approximate inference without diagnosed representational need.",
        evidence_type="Particle cloud, ESS, and sample-based posterior approximation.",
        assumptions="Sample-based filtering is needed because Gaussian-family approximations are inadequate.",
        three_d_relevance="high",
        latent_events=True,
        mode_mixing=True,
    ),
    FamilyScorecardSpec(
        method_id="rbpf",
        display_name="RBPF",
        public_family="physics_aware_inference",
        algorithm_coverage_method_id="rbpf",
        method_validation_id="rbpf",
        validation_ladder_classifier_ids=(),
        complexity_band="very_high",
        complexity_score=5.0,
        capability_added="Sampled latent structure plus analytic conditional state filtering.",
        expected_win_condition="Latent event timing or mode-path witnesses with tractable conditional dynamics.",
        expected_failure_mode="No exploitable conditional structure or no compute-normalized win over PF.",
        evidence_type="Particles over latent variables with conditional Kalman-style subfilters.",
        assumptions="The hard latent part can be sampled while the conditional state remains analytically filterable.",
        three_d_relevance="high",
        latent_events=True,
        mode_mixing=True,
    ),
    FamilyScorecardSpec(
        method_id="shapelet",
        display_name="Shapelets",
        public_family="interpretable_kinematic",
        algorithm_coverage_method_id="shapelet_family",
        method_validation_id="shapelet",
        validation_ladder_classifier_ids=(),
        complexity_band="medium",
        complexity_score=2.6,
        capability_added="Localized motif evidence for short discriminative maneuvers.",
        expected_win_condition="Witnesses where brief local patterns carry the class signal.",
        expected_failure_mode="Global state ambiguity and long-horizon uncertainty accumulation.",
        evidence_type="Distance-to-motif or subsequence activation evidence.",
        assumptions="Classes differ by short local kinematic motifs rather than only aggregate trends.",
        three_d_relevance="medium",
        latent_events=False,
        mode_mixing=False,
    ),
    FamilyScorecardSpec(
        method_id="feature_boosting",
        display_name="Feature Boosting",
        public_family="interpretable_kinematic",
        algorithm_coverage_method_id="gradient_boosted_features",
        method_validation_id="gradient_boosted_features",
        validation_ladder_classifier_ids=(),
        complexity_band="medium",
        complexity_score=2.3,
        capability_added="Nonlinear decision boundaries over engineered features.",
        expected_win_condition="Tabular feature surfaces with nonlinear interactions but still interpretable-ish inputs.",
        expected_failure_mode="Hidden temporal ordering, latent switching, and state uncertainty.",
        evidence_type="Supervised nonlinear feature interactions over engineered kinematic summaries.",
        assumptions="The key signal is exposed in engineered feature tables.",
        three_d_relevance="high",
        latent_events=False,
        mode_mixing=False,
    ),
    FamilyScorecardSpec(
        method_id="minirocket_family",
        display_name="MiniRocket Family",
        public_family="generic_tsc",
        algorithm_coverage_method_id="rocket_family",
        method_validation_id="minirocket_family",
        validation_ladder_classifier_ids=(),
        complexity_band="high",
        complexity_score=3.6,
        capability_added="Fast convolutional archive-style time-series classification.",
        expected_win_condition="Strong generic TSC baseline pressure on handcrafted features and physics baselines.",
        expected_failure_mode="Reduced interpretability and incomplete ceiling alignment to Epic 1 witnesses.",
        evidence_type="Random or fixed convolutional feature transforms with lightweight classifier head.",
        assumptions="Generic time-series representations can capture useful class structure without explicit physics.",
        three_d_relevance="high",
        latent_events=False,
        mode_mixing=False,
    ),
    FamilyScorecardSpec(
        method_id="drcif_interval_forests",
        display_name="DrCIF",
        public_family="generic_tsc",
        algorithm_coverage_method_id="drcif_interval_forests",
        method_validation_id="drcif_interval_forests",
        validation_ladder_classifier_ids=(),
        complexity_band="high",
        complexity_score=3.8,
        capability_added="Interval-feature ensemble evidence for archive-style TSC.",
        expected_win_condition="Class differences that are best expressed through learned interval summaries.",
        expected_failure_mode="Parity-only witness evidence and unstable method-level promotion support.",
        evidence_type="Interval forest ensemble over multi-resolution time-series summaries.",
        assumptions="Useful class evidence is encoded in interval-based summaries rather than explicit physical state models.",
        three_d_relevance="high",
        latent_events=False,
        mode_mixing=False,
    ),
    FamilyScorecardSpec(
        method_id="dictionary_tde_family",
        display_name="Dictionary Family",
        public_family="generic_tsc",
        algorithm_coverage_method_id="dictionary_tde_family",
        method_validation_id="dictionary_tde_family",
        validation_ladder_classifier_ids=(),
        complexity_band="high",
        complexity_score=3.7,
        capability_added="Symbolic bag-of-pattern evidence over discretized subsequences.",
        expected_win_condition="Motif-like or phrase-like time-series patterns that survive discretization.",
        expected_failure_mode="Lower interpretability than direct physics evidence and limited online readiness.",
        evidence_type="Dictionary counts and symbolic subsequence statistics.",
        assumptions="Discretized local words preserve the important class signal.",
        three_d_relevance="medium",
        latent_events=False,
        mode_mixing=False,
    ),
    FamilyScorecardSpec(
        method_id="hive_cote",
        display_name="HIVE-COTE",
        public_family="generic_tsc",
        algorithm_coverage_method_id="hive_cote",
        method_validation_id="hive_cote",
        validation_ladder_classifier_ids=(),
        complexity_band="very_high",
        complexity_score=4.4,
        capability_added="Heterogeneous ensemble ceiling over multiple TSC representations.",
        expected_win_condition="Offline benchmark ceiling checks where raw accuracy matters more than interpretability or latency.",
        expected_failure_mode="High compute cost, reduced operational transparency, and warning-heavy diagnostics.",
        evidence_type="Ensemble over multiple archive-style time-series representations.",
        assumptions="A heterogeneous representation mix can recover signal missed by single TSC families.",
        three_d_relevance="high",
        latent_events=False,
        mode_mixing=False,
    ),
    FamilyScorecardSpec(
        method_id="tcn_inceptiontime",
        display_name="TCN / InceptionTime",
        public_family="learned_sequence_embedding",
        algorithm_coverage_method_id="tcn_inceptiontime",
        method_validation_id="tcn",
        validation_ladder_classifier_ids=(),
        complexity_band="very_high",
        complexity_score=4.3,
        capability_added="Trained neural sequence filters over raw or multi-channel trajectories.",
        expected_win_condition="Larger data settings where learned temporal features outperform handcrafted or archive baselines.",
        expected_failure_mode="Data hunger, reduced interpretability, and bounded robustness evidence only.",
        evidence_type="Supervised neural sequence encoder over trajectory channels.",
        assumptions="Training data is sufficient to learn temporal features that generalize beyond handcrafted summaries.",
        three_d_relevance="high",
        latent_events=False,
        mode_mixing=False,
    ),
    FamilyScorecardSpec(
        method_id="ts2vec",
        display_name="TS2Vec",
        public_family="learned_sequence_embedding",
        algorithm_coverage_method_id="ts2vec_family",
        method_validation_id="ts2vec",
        validation_ladder_classifier_ids=(),
        complexity_band="very_high",
        complexity_score=4.0,
        capability_added="Reusable self-supervised embeddings for downstream classification.",
        expected_win_condition="Low-label or transfer settings where representation reuse matters.",
        expected_failure_mode="Ceiling alignment and broader benchmark breadth still bounded.",
        evidence_type="Contrastive embedding space followed by downstream probe classifier.",
        assumptions="Unlabeled temporal structure can be learned and reused for class discrimination.",
        three_d_relevance="high",
        latent_events=False,
        mode_mixing=False,
    ),
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _coverage_map() -> dict[str, AlgorithmCoverageEntry]:
    return {entry.method_id: entry for entry in ALGORITHM_COVERAGE_MATRIX}


def _method_validation_map() -> dict[str, MethodSpec]:
    result = analyze_method_validation_os()
    return {row.method_id: row for row in result.method_rows}


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


def _ceiling_rows_by_classifier_id() -> dict[str, list[dict[str, str]]]:
    rows = _read_csv_rows(Path("artifacts/showcase/tables/validation_ladder_scores.csv"))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["classifier_id"], []).append(row)
    return grouped


def _parse_oracle_value(summary: str, *, prefix: str) -> float | None:
    for chunk in summary.split(","):
        item = chunk.strip()
        if item.startswith(prefix):
            try:
                return float(item.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _build_capability_row(
    spec: FamilyScorecardSpec,
    coverage_row: AlgorithmCoverageEntry | None,
    validation_row: MethodSpec | None,
) -> FamilyCapabilityRow:
    return FamilyCapabilityRow(
        method_id=spec.method_id,
        display_name=spec.display_name,
        public_family=spec.public_family,
        history_accumulation=_bool_text(bool(coverage_row and (coverage_row.posterior_native or coverage_row.online))),
        dynamics_evidence=_bool_text(bool(coverage_row and coverage_row.uses_dynamics)),
        switching_logic=_bool_text(bool(coverage_row and coverage_row.handles_switching)),
        mode_mixing=_bool_text(spec.mode_mixing),
        nonlinear_state=_bool_text(bool(coverage_row and coverage_row.handles_nonlinearity)),
        nongaussian_state=_bool_text(bool(coverage_row and coverage_row.handles_nongaussianity)),
        latent_events=_bool_text(spec.latent_events),
        interpretability=_bool_text(bool(coverage_row and coverage_row.interpretable)),
        calibration=_bool_text(bool(coverage_row and coverage_row.calibratable)),
        online_inference=_bool_text(bool(coverage_row and coverage_row.online)),
        lift_relevance_3d=spec.three_d_relevance,
        current_status=validation_row.current_status if validation_row is not None else (coverage_row.status if coverage_row is not None else "unknown"),
    )


def _build_ceiling_row(
    spec: FamilyScorecardSpec,
    ladder_rows: dict[str, list[dict[str, str]]],
) -> CeilingEfficiencyRow:
    selected: list[dict[str, str]] = []
    for classifier_id in spec.validation_ladder_classifier_ids:
        selected.extend(
            row for row in ladder_rows.get(classifier_id, ())
            if row["level_name"] in {"oracle_separability", "classifier_performance"}
        )
    if not selected:
        return CeilingEfficiencyRow(
            method_id=spec.method_id,
            display_name=spec.display_name,
            public_family=spec.public_family,
            complexity_band=spec.complexity_band,
            complexity_score=spec.complexity_score,
            ladder_pair_count=0,
            mean_classifier_accuracy=None,
            mean_epic1_oracle_proxy=None,
            mean_best_pair_oracle_proxy=None,
            mean_fraction_of_proxy_captured_capped=None,
            mean_proxy_gap=None,
            proxy_exceeded_fraction=None,
            ceiling_status="no_named_ceiling_alignment",
            evidence_note="Current Epic 1 ceiling proxy is not yet aligned to this family's named witness surfaces.",
        )

    pair_rows: dict[tuple[str, str], dict[str, float]] = {}
    for row in selected:
        key = (row["class_pair_id"], row["classifier_id"])
        pair_entry = pair_rows.setdefault(key, {})
        if row["level_name"] == "oracle_separability":
            pair_entry["oracle_accuracy"] = _parse_oracle_value(row["evidence_summary"], prefix="oracle_accuracy") or float(row["score"])
        elif row["level_name"] == "classifier_performance":
            pair_entry["classifier_accuracy"] = _parse_oracle_value(row["evidence_summary"], prefix="accuracy") or float(row["score"])
            if "oracle_gap=" in row["evidence_summary"]:
                pair_entry["oracle_gap"] = _parse_oracle_value(row["evidence_summary"], prefix="oracle_gap") or 0.0

    comparable_rows = [row for row in pair_rows.values() if "oracle_accuracy" in row and "classifier_accuracy" in row]
    if not comparable_rows:
        return CeilingEfficiencyRow(
            method_id=spec.method_id,
            display_name=spec.display_name,
            public_family=spec.public_family,
            complexity_band=spec.complexity_band,
            complexity_score=spec.complexity_score,
            ladder_pair_count=0,
            mean_classifier_accuracy=None,
            mean_epic1_oracle_proxy=None,
            mean_best_pair_oracle_proxy=None,
            mean_fraction_of_proxy_captured_capped=None,
            mean_proxy_gap=None,
            proxy_exceeded_fraction=None,
            ceiling_status="named_ladder_rows_missing_pair_alignment",
            evidence_note="Validation ladder rows exist, but no aligned oracle-performance pairs were found.",
        )

    classifier_values = [row["classifier_accuracy"] for row in comparable_rows]
    oracle_values = [row["oracle_accuracy"] for row in comparable_rows]
    fractions = [min(1.0, row["classifier_accuracy"] / max(row["oracle_accuracy"], 1.0e-12)) for row in comparable_rows]
    proxy_gaps = [row["classifier_accuracy"] - row["oracle_accuracy"] for row in comparable_rows]
    proxy_exceeded_fraction = sum(1.0 for gap in proxy_gaps if gap > 0.0) / len(proxy_gaps)
    ceiling_status = "proxy_exceeded_static_ceiling" if proxy_exceeded_fraction > 0.0 else "aligned_to_epic1_proxy"
    evidence_note = (
        "Dynamic or temporal performance exceeds the current static Epic 1 oracle proxy on part of this surface; the fraction is capped at 1.0 and should be treated as a lower-fidelity ceiling alignment."
        if ceiling_status == "proxy_exceeded_static_ceiling"
        else "Current ladder rows remain within the Epic 1 proxy envelope."
    )
    return CeilingEfficiencyRow(
        method_id=spec.method_id,
        display_name=spec.display_name,
        public_family=spec.public_family,
        complexity_band=spec.complexity_band,
        complexity_score=spec.complexity_score,
        ladder_pair_count=len(comparable_rows),
        mean_classifier_accuracy=sum(classifier_values) / len(classifier_values),
        mean_epic1_oracle_proxy=sum(oracle_values) / len(oracle_values),
        mean_best_pair_oracle_proxy=sum(oracle_values) / len(oracle_values),
        mean_fraction_of_proxy_captured_capped=sum(fractions) / len(fractions),
        mean_proxy_gap=sum(proxy_gaps) / len(proxy_gaps),
        proxy_exceeded_fraction=proxy_exceeded_fraction,
        ceiling_status=ceiling_status,
        evidence_note=evidence_note,
    )


def _build_family_summary_rows(
    specs: tuple[FamilyScorecardSpec, ...],
    validation_map: dict[str, MethodSpec],
    ceiling_rows: tuple[CeilingEfficiencyRow, ...],
) -> tuple[FamilySummaryRow, ...]:
    by_family: dict[str, list[FamilyScorecardSpec]] = {}
    for spec in specs:
        by_family.setdefault(spec.public_family, []).append(spec)
    ceiling_map = {row.method_id: row for row in ceiling_rows}
    rows: list[FamilySummaryRow] = []
    for family_id, family_specs in by_family.items():
        validation_rows = [validation_map[spec.method_validation_id] for spec in family_specs if spec.method_validation_id and spec.method_validation_id in validation_map]
        supported_count = sum(1 for row in validation_rows if row.current_status in {"witness_supported", "study_justified", "generalized"})
        justified_count = sum(1 for row in validation_rows if row.current_status in {"study_justified", "generalized"})
        ceiling_values = [
            ceiling_map[spec.method_id].mean_fraction_of_proxy_captured_capped
            for spec in family_specs
            if ceiling_map[spec.method_id].mean_fraction_of_proxy_captured_capped is not None
        ]
        ceiling_statuses = {ceiling_map[spec.method_id].ceiling_status for spec in family_specs}
        if not ceiling_values:
            alignment = "no_family_ceiling_alignment"
        elif "proxy_exceeded_static_ceiling" in ceiling_statuses:
            alignment = "static_proxy_understates_temporal_family"
        else:
            alignment = "family_aligned_to_epic1_proxy"
        rows.append(
            FamilySummaryRow(
                public_family=family_id,
                method_count=len(family_specs),
                witness_supported_or_better=supported_count,
                study_justified_or_better=justified_count,
                capability_added=family_specs[0].capability_added,
                expected_win_condition=family_specs[0].expected_win_condition,
                mean_fraction_of_proxy_captured_capped=(sum(value for value in ceiling_values if value is not None) / len(ceiling_values)) if ceiling_values else None,
                ceiling_alignment_status=alignment,
                three_d_relevance=max((spec.three_d_relevance for spec in family_specs), key=lambda value: {"low": 0, "medium": 1, "high": 2}.get(value, 0)),
            )
        )
    return tuple(rows)


def _render_atlas_markdown(
    specs: tuple[FamilyScorecardSpec, ...],
    validation_map: dict[str, MethodSpec],
) -> str:
    lines = ["# Classifier Family Atlas", ""]
    for spec in specs:
        current_status = validation_map[spec.method_validation_id].current_status if spec.method_validation_id and spec.method_validation_id in validation_map else "not_scored_in_method_validation_os"
        lines.extend(
            [
                f"## {spec.display_name}",
                "",
                f"- Family: `{spec.public_family}`",
                f"- Assumptions: {spec.assumptions}",
                f"- Evidence Type: {spec.evidence_type}",
                f"- Expected Strengths: {spec.expected_win_condition}",
                f"- Expected Failure Modes: {spec.expected_failure_mode}",
                f"- Complexity: `{spec.complexity_band}`",
                f"- Current Status: `{current_status}`",
                f"- 3D Relevance: `{spec.three_d_relevance}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _render_efficiency_plot(rows: tuple[CeilingEfficiencyRow, ...]):
    plot_rows = [row for row in rows if row.mean_fraction_of_proxy_captured_capped is not None]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for row in plot_rows:
        color = {
            "interpretable_kinematic": "#2563eb",
            "sequential_evidence": "#0f766e",
            "physics_aware_inference": "#7c3aed",
            "generic_tsc": "#d97706",
            "learned_sequence_embedding": "#dc2626",
        }.get(row.public_family, "#6b7280")
        ax.scatter(row.complexity_score, row.mean_fraction_of_proxy_captured_capped, color=color, s=58)
        ax.text(row.complexity_score + 0.03, row.mean_fraction_of_proxy_captured_capped + 0.01, row.display_name, fontsize=8)
    ax.axhline(0.5, color="#9ca3af", linestyle="--", linewidth=1.0, label="random")
    ax.axhline(1.0, color="#111827", linestyle=":", linewidth=1.1, label="Epic 1 proxy ceiling")
    ax.set_xlim(0.8, 5.3)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("model complexity")
    ax.set_ylabel("fraction of Epic 1 proxy captured (capped)")
    ax.set_title("Classifier Efficiency vs Epic 1 Proxy Ceiling", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def analyze_classifier_family_scorecard() -> ClassifierFamilyScorecardResult:
    coverage_map = _coverage_map()
    validation_map = _method_validation_map()
    ladder_rows = _ceiling_rows_by_classifier_id()

    capability_rows = tuple(
        _build_capability_row(
            spec,
            coverage_map.get(spec.algorithm_coverage_method_id) if spec.algorithm_coverage_method_id is not None else None,
            validation_map.get(spec.method_validation_id) if spec.method_validation_id is not None else None,
        )
        for spec in SCORECARD_SPECS
    )
    ceiling_rows = tuple(_build_ceiling_row(spec, ladder_rows) for spec in SCORECARD_SPECS)
    family_summary_rows = _build_family_summary_rows(SCORECARD_SPECS, validation_map, ceiling_rows)
    atlas_markdown = _render_atlas_markdown(SCORECARD_SPECS, validation_map)

    report = MarkdownDocument("Classifier Family Scorecard")
    report.paragraph(
        "This surface is the missing Epic 2 scorecard layer: it answers what capability each family adds, when it should win, and how well the current repo aligns that family to the current Epic 1 ceiling proxy."
    )
    report.heading("Family Summary", level=2)
    report.table(
        ["Family", "Methods", "Witness-supported+", "Study-justified+", "Mean capped proxy fraction", "Ceiling alignment"],
        [
            (
                row.public_family,
                row.method_count,
                row.witness_supported_or_better,
                row.study_justified_or_better,
                "" if row.mean_fraction_of_proxy_captured_capped is None else f"{row.mean_fraction_of_proxy_captured_capped:.3f}",
                row.ceiling_alignment_status,
            )
            for row in family_summary_rows
        ],
    )
    report.heading("Current Read", level=2)
    report.bullet_list(
        [
            "Capability coverage is now explicit per classifier family rather than only implied by the ladder.",
            "Ceiling-relative efficiency currently has direct alignment only for the families already represented in the validation ladder score surface.",
            "Several temporal or dynamic methods exceed the current static Epic 1 proxy on the bounded ladder surface; those rows are flagged as proxy-exceeded rather than being treated as clean ceiling capture.",
            "Archive and learned families are now visible in the scorecard, but most still need named ceiling-aligned witness packets before the Epic 1-to-Epic 2 ceiling story is fully strong.",
        ]
    )
    metrics = {
        "study_id": "classifier_family_scorecard_v1",
        "method_count": len(SCORECARD_SPECS),
        "family_count": len({spec.public_family for spec in SCORECARD_SPECS}),
        "ceiling_aligned_method_count": sum(1 for row in ceiling_rows if row.mean_fraction_of_proxy_captured_capped is not None),
        "proxy_exceeded_method_count": sum(1 for row in ceiling_rows if row.ceiling_status == "proxy_exceeded_static_ceiling"),
        "public_family_completion_read": "scorecard_added_bounded_ceiling_alignment",
    }
    return ClassifierFamilyScorecardResult(
        atlas_specs=SCORECARD_SPECS,
        capability_rows=capability_rows,
        ceiling_rows=ceiling_rows,
        family_summary_rows=family_summary_rows,
        metrics=metrics,
        atlas_markdown=atlas_markdown,
        report_markdown=report.text(),
    )


def write_classifier_family_scorecard_artifacts(
    output_dir: str | Path,
    *,
    result: ClassifierFamilyScorecardResult | None = None,
) -> ClassifierFamilyScorecardArtifacts:
    payload = result or analyze_classifier_family_scorecard()
    run_dir = Path(output_dir) / "classifier_family_scorecard_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    capability_matrix_path = run_dir / "capability_matrix.csv"
    ceiling_efficiency_path = run_dir / "ceiling_efficiency.csv"
    family_summary_path = run_dir / "family_summary.csv"
    atlas_path = run_dir / "classifier_family_atlas.md"
    report_path = run_dir / "classifier_family_scorecard_report.md"
    summary_path = run_dir / "summary.json"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    efficiency_plot_path = plots_dir / "classifier_efficiency_vs_epic1_proxy_ceiling.png"

    write_csv(capability_matrix_path, [asdict(row) for row in payload.capability_rows], list(FamilyCapabilityRow.__dataclass_fields__.keys()))
    write_csv(ceiling_efficiency_path, [asdict(row) for row in payload.ceiling_rows], list(CeilingEfficiencyRow.__dataclass_fields__.keys()))
    write_csv(family_summary_path, [asdict(row) for row in payload.family_summary_rows], list(FamilySummaryRow.__dataclass_fields__.keys()))
    atlas_path.write_text(payload.atlas_markdown, encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    summary_path.write_text(json.dumps(payload.metrics, indent=2), encoding="utf-8")
    efficiency_plot_path.write_bytes(_figure_to_png(_render_efficiency_plot(payload.ceiling_rows)))

    return ClassifierFamilyScorecardArtifacts(
        run_dir=run_dir,
        capability_matrix_path=capability_matrix_path,
        ceiling_efficiency_path=ceiling_efficiency_path,
        family_summary_path=family_summary_path,
        atlas_path=atlas_path,
        report_path=report_path,
        summary_path=summary_path,
        plot_paths=(efficiency_plot_path,),
    )


__all__ = [
    "ClassifierFamilyScorecardArtifacts",
    "ClassifierFamilyScorecardResult",
    "CeilingEfficiencyRow",
    "FamilyCapabilityRow",
    "FamilyScorecardSpec",
    "FamilySummaryRow",
    "analyze_classifier_family_scorecard",
    "write_classifier_family_scorecard_artifacts",
]
