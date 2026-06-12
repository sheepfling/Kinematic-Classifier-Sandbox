from __future__ import annotations

import csv
import json
import os
import shutil
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kcs-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _bootstrap import bootstrap_repo


ROOT = bootstrap_repo(configure_runtime=True)
PACKET_DIR = ROOT / "artifacts" / "packets" / "classifier_ladder_mvp"
FIGURE_DIR = PACKET_DIR / "figures"
HERO_FIGURES = ROOT / "artifacts" / "presentation_hero_charts_v5" / "figures"
RUNG = ROOT / "artifacts" / "rung_sufficiency"
ADVANCED = ROOT / "artifacts" / "advanced_filter_comparison_v1"
TRACE = ROOT / "artifacts" / "filter_trace_validation_v1"

BASE_FIGURES = [
    "06_posterior_timeline_witness.png",
    "07_rung_sufficiency_map.png",
    "10_advanced_filter_gate_matrix.png",
    "10b_imm_switching_shine_witness.png",
    "10c_pf_nonlinear_nongaussian_shine_witness.png",
    "10d_rbpf_latent_event_shine_witness.png",
    "10e_advanced_filter_sweet_spot_matrix.png",
    "10f_advanced_filter_showcase_summary.png",
    "11_witness_coverage_matrix.png",
    "15_prior_sensitivity_surface.png",
    "16_calibration_reliability.png",
    "17_oracle_gap_bridge.png",
    "19_confusion_localization_matrix.png",
]

NEW_FIGURES = [
    "06c_capability_ladder.png",
    "10g_method_capability_matrix.png",
    "10h_advanced_inference_architecture_map.png",
    "10i_filter_promotion_criteria.png",
]

METHODS = [
    {
        "method_id": "pointwise",
        "display_name": "Pointwise",
        "capability": "local evidence",
        "architectural_role": "instantaneous likelihood baseline",
        "evaluated": "yes",
        "architecturally_exercised": "yes",
        "promotion_status": "promoted_when_sufficient",
        "future_3d_role": "local feature likelihoods remain the first gate",
    },
    {
        "method_id": "windowed",
        "display_name": "Windowed",
        "capability": "local temporal evidence",
        "architectural_role": "short-horizon shape and extrema evidence",
        "evaluated": "yes",
        "architecturally_exercised": "yes",
        "promotion_status": "promoted_when_sufficient",
        "future_3d_role": "short-window kinematic features before full state inference",
    },
    {
        "method_id": "sequential_bayes",
        "display_name": "Sequential Bayes",
        "capability": "history accumulation",
        "architectural_role": "recursive posterior memory",
        "evaluated": "yes",
        "architecturally_exercised": "yes",
        "promotion_status": "promoted_when_sufficient",
        "future_3d_role": "posterior accumulation over richer tracklet histories",
    },
    {
        "method_id": "kalman_bank",
        "display_name": "Kalman Bank",
        "capability": "dynamic residual evidence",
        "architectural_role": "state prediction and innovation likelihood",
        "evaluated": "yes",
        "architecturally_exercised": "yes",
        "promotion_status": "promoted_when_dynamics_are_excited",
        "future_3d_role": "PVA state prediction and residual scoring",
    },
    {
        "method_id": "transition_matrix",
        "display_name": "Transition Matrix",
        "capability": "switching logic",
        "architectural_role": "explicit mode persistence and transition prior",
        "evaluated": "yes",
        "architecturally_exercised": "yes",
        "promotion_status": "promoted_for_label_switching_witnesses",
        "future_3d_role": "discrete mode evolution before mode-mixed state inference",
    },
    {
        "method_id": "imm_v1",
        "display_name": "IMM",
        "capability": "mode mixing",
        "architectural_role": "multiple dynamic models with mixed state estimates",
        "evaluated": "yes",
        "architecturally_exercised": "witness_supported",
        "promotion_status": "witness_specific",
        "future_3d_role": "mode-conditioned 3D dynamics and maneuver models",
    },
    {
        "method_id": "particle_filter_bank_v1",
        "display_name": "PF / GSF",
        "capability": "non-Gaussian posterior reasoning",
        "architectural_role": "sampled or mixture posterior support",
        "evaluated": "yes",
        "architecturally_exercised": "prototype_plus_witness",
        "promotion_status": "witness_specific",
        "future_3d_role": "range/bearing ambiguity, occlusion, nonlinear measurement support",
    },
    {
        "method_id": "rbpf_v1",
        "display_name": "RBPF",
        "capability": "latent-structure inference",
        "architectural_role": "sampled discrete path with conditional continuous state",
        "evaluated": "yes",
        "architecturally_exercised": "prototype_plus_witness",
        "promotion_status": "witness_specific",
        "future_3d_role": "hidden maneuver onset, command state, and hybrid mode/state inference",
    },
]

CAPABILITIES = [
    "local evidence",
    "local temporal evidence",
    "history accumulation",
    "dynamic residual evidence",
    "switching logic",
    "mode mixing",
    "non-Gaussian posterior reasoning",
    "latent-structure inference",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def copy_base_figures() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in BASE_FIGURES:
        source = HERO_FIGURES / name
        if source.exists():
            shutil.copy2(source, FIGURE_DIR / name)


def write_tables() -> None:
    rows = [
        {
            "rank": index,
            **method,
        }
        for index, method in enumerate(METHODS)
    ]
    write_csv(
        PACKET_DIR / "evidence_capability_ladder.csv",
        rows,
        [
            "rank",
            "method_id",
            "display_name",
            "capability",
            "architectural_role",
            "evaluated",
            "architecturally_exercised",
            "promotion_status",
            "future_3d_role",
        ],
    )

    matrix_rows = []
    for method in METHODS:
        for capability in CAPABILITIES:
            matrix_rows.append(
                {
                    "method_id": method["method_id"],
                    "display_name": method["display_name"],
                    "capability": capability,
                    "has_capability": "yes" if method["capability"] == capability else "no",
                }
            )
    write_csv(
        PACKET_DIR / "method_capability_matrix.csv",
        matrix_rows,
        ["method_id", "display_name", "capability", "has_capability"],
    )

    gate_rows = {row["method_id"]: row for row in read_csv(ADVANCED / "advanced_method_gate_matrix.csv")}
    trace_rows = {row["method_id"]: row for row in read_csv(TRACE / "method_trace_matrix.csv")}
    criteria_rows = []
    for method in METHODS:
        method_id = method["method_id"]
        gate = gate_rows.get(method_id, {})
        trace = trace_rows.get(method_id, {})
        criteria_rows.append(
            {
                "method_id": method_id,
                "display_name": method["display_name"],
                "evaluated": method["evaluated"],
                "architecturally_exercised": method["architecturally_exercised"],
                "trace_status": trace.get("trace_status", "not_required_for_basic_rung"),
                "witness_exists": gate.get("witness_exists", "basic_ladder_witness"),
                "promoted": "yes" if method["promotion_status"].startswith("promoted") else "witness_specific",
                "promotion_status": method["promotion_status"],
                "promotion_condition": promotion_condition(method_id),
            }
        )
    write_csv(
        PACKET_DIR / "filter_promotion_criteria.csv",
        criteria_rows,
        [
            "method_id",
            "display_name",
            "evaluated",
            "architecturally_exercised",
            "trace_status",
            "witness_exists",
            "promoted",
            "promotion_status",
            "promotion_condition",
        ],
    )

    architecture_rows = [
        {
            "from_node": "Pointwise",
            "to_node": "Sequential",
            "edge": "local likelihoods accumulate into posterior history",
        },
        {
            "from_node": "Sequential",
            "to_node": "Kalman Bank",
            "edge": "history becomes state prediction and residual evidence",
        },
        {
            "from_node": "Kalman Bank",
            "to_node": "Transition",
            "edge": "state evidence is paired with explicit switching priors",
        },
        {
            "from_node": "Transition",
            "to_node": "IMM",
            "edge": "switching logic becomes mode-mixed dynamic state inference",
        },
        {
            "from_node": "IMM",
            "to_node": "PF / GSF",
            "edge": "mode/state inference extends to nonlinear or non-Gaussian posterior shape",
        },
        {
            "from_node": "PF / GSF",
            "to_node": "RBPF",
            "edge": "sampled posterior support is split into latent path plus conditional state",
        },
    ]
    write_csv(PACKET_DIR / "advanced_inference_architecture_map.csv", architecture_rows, ["from_node", "to_node", "edge"])


def promotion_condition(method_id: str) -> str:
    if method_id == "imm_v1":
        return "promote when switching dynamics require mode mixing beyond transition logic"
    if method_id == "particle_filter_bank_v1":
        return "promote when Gaussian approximations collapse under nonlinear or multimodal posterior evidence"
    if method_id == "rbpf_v1":
        return "promote when latent event or mode-path structure benefits from sampled discrete paths plus conditional state estimates"
    if method_id == "kalman_bank":
        return "promote when dynamic residual likelihood is the evidence source"
    if method_id == "transition_matrix":
        return "promote when explicit switching priors fix static-class assumptions"
    return "promote when this is the simplest sufficient evidence capability"


def render_capability_ladder() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axis("off")
    x_positions = [i for i in range(len(METHODS))]
    y = 0.55
    colors = ["#2E86AB", "#2E86AB", "#1B998B", "#1B998B", "#D99C00", "#E67E22", "#6C5CE7", "#C0392B"]
    for index, method in enumerate(METHODS):
        x = x_positions[index]
        ax.scatter([x], [y], s=1200, color=colors[index], zorder=3)
        ax.text(x, y, str(index), ha="center", va="center", color="white", fontsize=14, fontweight="bold")
        ax.text(x, y - 0.16, method["display_name"], ha="center", va="top", fontsize=10, fontweight="bold")
        ax.text(x, y - 0.31, method["capability"], ha="center", va="top", fontsize=8, wrap=True)
        if index < len(METHODS) - 1:
            ax.annotate("", xy=(x + 0.72, y), xytext=(x + 0.28, y), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#5D6D7E"})
    ax.text(
        0,
        0.95,
        "Capability ladder, not winner ladder",
        ha="left",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#17202A",
    )
    ax.text(
        0,
        0.84,
        "Each rung adds representational capability under one evidence/posterior contract.",
        ha="left",
        va="center",
        fontsize=11,
        color="#5D6D7E",
    )
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.set_ylim(0.05, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06c_capability_ladder.png", dpi=180)
    plt.close(fig)


def render_method_capability_matrix() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    values = [[1 if method["capability"] == capability else 0 for capability in CAPABILITIES] for method in METHODS]
    ax.imshow(values, cmap=matplotlib.colors.ListedColormap(["#F7F9F9", "#2E86AB"]), aspect="auto")
    ax.set_xticks(range(len(CAPABILITIES)), CAPABILITIES, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(METHODS)), [method["display_name"] for method in METHODS], fontsize=9)
    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            if value:
                ax.text(col_index, row_index, "x", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.set_title("Method Capability Matrix", loc="left", fontsize=16, fontweight="bold")
    ax.text(
        -0.5,
        -1.0,
        "Advanced filters are exercised for new representational capability, not ranked as universal winners.",
        fontsize=10,
        color="#5D6D7E",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "10g_method_capability_matrix.png", dpi=180)
    plt.close(fig)


def render_architecture_map() -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    nodes = {
        "Pointwise": (0.5, 0.92),
        "Sequential": (0.5, 0.78),
        "Kalman Bank": (0.5, 0.64),
        "Transition": (0.26, 0.48),
        "IMM": (0.74, 0.48),
        "PF / GSF": (0.50, 0.30),
        "RBPF": (0.50, 0.14),
    }
    labels = {
        "Pointwise": "local evidence",
        "Sequential": "history accumulation",
        "Kalman Bank": "dynamic residuals",
        "Transition": "switching logic",
        "IMM": "mode mixing",
        "PF / GSF": "non-Gaussian posterior",
        "RBPF": "latent structure",
    }
    edges = [
        ("Pointwise", "Sequential"),
        ("Sequential", "Kalman Bank"),
        ("Kalman Bank", "Transition"),
        ("Kalman Bank", "IMM"),
        ("Transition", "IMM"),
        ("IMM", "PF / GSF"),
        ("PF / GSF", "RBPF"),
    ]
    for source, target in edges:
        ax.annotate(
            "",
            xy=nodes[target],
            xytext=nodes[source],
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#5D6D7E"},
        )
    for node, (x, y) in nodes.items():
        ax.add_patch(plt.Rectangle((x - 0.12, y - 0.045), 0.24, 0.09, color="#F7F9F9", ec="#2E86AB", lw=2))
        ax.text(x, y + 0.012, node, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x, y - 0.022, labels[node], ha="center", va="center", fontsize=8, color="#5D6D7E")
    ax.text(0.02, 0.99, "Advanced Inference Architecture Map", fontsize=17, fontweight="bold", va="top")
    ax.text(
        0.02,
        0.94,
        "1D witnesses prove the route and trace contract; richer 3D dynamics excite the advanced methods.",
        fontsize=10,
        color="#5D6D7E",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "10h_advanced_inference_architecture_map.png", dpi=180)
    plt.close(fig)


def render_promotion_criteria() -> None:
    rows = read_csv(PACKET_DIR / "filter_promotion_criteria.csv")
    columns = ["evaluated", "architecturally_exercised", "trace_status", "promoted"]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.axis("off")
    table_data = []
    for row in rows:
        table_data.append(
            [
                row["display_name"],
                row["evaluated"],
                row["architecturally_exercised"],
                row["trace_status"],
                row["promoted"],
            ]
        )
    table = ax.table(
        cellText=table_data,
        colLabels=["Method", "Evaluated", "Exercised", "Trace", "Promoted"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#17202A")
        elif col == 2 and "witness" in cell.get_text().get_text():
            cell.set_facecolor("#FDEBD0")
        elif col == 4 and cell.get_text().get_text() == "witness_specific":
            cell.set_facecolor("#FCF3CF")
        elif col > 0:
            cell.set_facecolor("#F7F9F9")
    ax.set_title("Filter Promotion Criteria", loc="left", fontsize=16, fontweight="bold")
    ax.text(
        0,
        0.04,
        "Architecturally exercised means the route and contract work; promotion still requires a matching excited scenario.",
        transform=ax.transAxes,
        fontsize=10,
        color="#5D6D7E",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "10i_filter_promotion_criteria.png", dpi=180)
    plt.close(fig)


def write_reports() -> None:
    write_text(
        PACKET_DIR / "README.md",
        """
        # Epic 2: Evidence Capability Ladder

        Core question: What evidence capabilities exist, how are they evaluated,
        and under what conditions should each rung be promoted?

        Decision field: `epic_2_evidence_capability_ladder`
        Status: `capability_architecture_exercised_with_witness_scope_limits`

        The repository contains a unified evidence contract capable of hosting
        increasingly sophisticated inference backends as study complexity grows.
        The 1D witnesses prove the evaluation machinery and integration path.
        Richer 3D dynamics are where algorithm usefulness should be fully excited.

        ## New Hero Figures

        - `06c_capability_ladder.png`: capability progression, not winner ranking.
        - `10g_method_capability_matrix.png`: method-to-capability mapping.
        - `10h_advanced_inference_architecture_map.png`: route from local evidence to IMM/PF/RBPF.
        - `10i_filter_promotion_criteria.png`: evaluated vs architecturally exercised vs promoted.

        ## Packet Tables

        - `evidence_capability_ladder.csv`
        - `method_capability_matrix.csv`
        - `filter_promotion_criteria.csv`
        - `advanced_inference_architecture_map.csv`

        ## Interpretation

        The ladder is not a claim that more complex methods always win. Pointwise,
        windowed, sequential, Kalman, transition, IMM, PF/GSF, and RBPF methods
        are evidence capabilities under one posterior contract. IMM proves mode
        mixing. PF/GSF proves nonlinear and non-Gaussian posterior support. RBPF
        proves latent discrete state support with conditional continuous-state
        inference.
        """,
    )
    write_text(
        PACKET_DIR / "decision_card.md",
        """
        # Epic 2: Evidence Capability Ladder Decision Card

        ```yaml
        epic_2_evidence_capability_ladder:
          status: capability_architecture_exercised_with_witness_scope_limits
          thesis: >
            The repository contains a unified evidence/posterior contract capable
            of hosting increasingly sophisticated inference backends as study
            complexity grows.
          goals:
            - prove shared evidence/posterior contract
            - demonstrate capability progression across the ladder
            - show evaluation machinery can promote or defer methods
            - architecturally exercise IMM/PF/RBPF
            - define the future 3D path where advanced methods become fully excited
          status_vocabulary:
            - evaluated
            - applicable
            - architecturally_exercised
            - witness_supported
            - promoted
            - deferred
          advanced_methods:
            IMM:
              capability: mode mixing and multi-model state estimation
              current_status: witness_supported
              future_3d_role: mode-conditioned 3D dynamics
            PF_GSF:
              capability: nonlinear, non-Gaussian, and multimodal posterior reasoning
              current_status: prototype_plus_witness
              future_3d_role: sensor geometry, range/bearing ambiguity, occlusion
            RBPF:
              capability: latent discrete state plus conditional continuous inference
              current_status: prototype_plus_witness
              future_3d_role: hidden maneuver onset and command-state inference
          decision:
            selected_action:
              - present Epic 2 as a capability ladder
              - stop treating 1D as the final proof of advanced-algorithm usefulness
              - use 1D witnesses to prove promotion machinery and trace contracts
              - route 3D lift work toward scenarios that excite IMM/PF/RBPF assumptions
        ```
        """,
    )
    write_text(
        PACKET_DIR / "claim_boundary.md",
        """
        # Epic 2 Claim Boundary

        The advanced-filter claim is architectural, not a toy accuracy claim.

        Current 1D witnesses prove that IMM, PF/GSF, RBPF, and related advanced
        filters can be hosted under the same evidence/posterior contract, traced,
        compared, and promoted or deferred by named conditions.

        They do not prove that every advanced method is globally better on every
        corpus. Full usefulness should be judged on richer 3D scenarios that
        excite mode uncertainty, nonlinear measurements, multimodal posterior
        shape, occlusion, and latent maneuver state.
        """,
    )


def write_figure_manifest() -> None:
    rows = [
        {
            "figure_id": figure.removesuffix(".png"),
            "file": f"figures/{figure}",
            "role": "base_epic_2_chart" if figure in BASE_FIGURES else "capability_architecture_chart",
        }
        for figure in [*BASE_FIGURES, *NEW_FIGURES]
        if (FIGURE_DIR / figure).exists()
    ]
    write_csv(PACKET_DIR / "figure_manifest.csv", rows, ["figure_id", "file", "role"])


def write_manifest() -> None:
    write_text(
        PACKET_DIR / "packet_manifest.json",
        json.dumps(
            {
                "packet_id": "classifier_ladder_mvp",
                "epic": "Epic 2",
                "title": "Evidence Capability Ladder",
                "status": "capability_architecture_exercised_with_witness_scope_limits",
                "new_figures": NEW_FIGURES,
            },
            indent=2,
        ),
    )


def validate_packet() -> None:
    required = [
        "README.md",
        "decision_card.md",
        "claim_boundary.md",
        "evidence_capability_ladder.csv",
        "method_capability_matrix.csv",
        "filter_promotion_criteria.csv",
        "advanced_inference_architecture_map.csv",
        "packet_manifest.json",
        *[f"figures/{figure}" for figure in NEW_FIGURES],
    ]
    missing = [path for path in required if not (PACKET_DIR / path).exists()]
    if missing:
        raise RuntimeError(f"missing classifier capability packet files: {missing}")
    criteria = read_csv(PACKET_DIR / "filter_promotion_criteria.csv")
    advanced = [row for row in criteria if row["method_id"] in {"imm_v1", "particle_filter_bank_v1", "rbpf_v1"}]
    if any(row["architecturally_exercised"] not in {"witness_supported", "prototype_plus_witness"} for row in advanced):
        raise RuntimeError("advanced methods must be marked architecturally exercised")
    if any(row["trace_status"] != "trace_validated" for row in advanced):
        raise RuntimeError("advanced methods must have trace-validated evidence")


def main() -> int:
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    copy_base_figures()
    write_tables()
    render_capability_ladder()
    render_method_capability_matrix()
    render_architecture_map()
    render_promotion_criteria()
    write_reports()
    write_figure_manifest()
    write_manifest()
    validate_packet()
    print(PACKET_DIR.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
