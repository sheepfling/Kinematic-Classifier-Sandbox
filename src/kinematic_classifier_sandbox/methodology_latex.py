from __future__ import annotations

from dataclasses import dataclass
import csv
import os
import shutil
import subprocess
from pathlib import Path

from .advanced_filter_decision import analyze_advanced_filter_decision
from .bayesian_walkthroughs import analyze_bayesian_walkthroughs
from .transition_matrix_accumulator import run_transition_benchmark
from .validation_ladder import analyze_validation_ladder


ROOT = Path(__file__).resolve().parents[2]
DOCS_LATEX_DIR = ROOT / "docs" / "latex"
DOCS_TABLES_DIR = DOCS_LATEX_DIR / "tables"
DOCS_FIGURES_DIR = DOCS_LATEX_DIR / "figures"
SOURCE_TEX_PATH = DOCS_LATEX_DIR / "kinematic_classifier_methodology.tex"


@dataclass(frozen=True, slots=True)
class MethodologyLatexResult:
    toy_problem_rows: tuple[dict[str, object], ...]
    algorithm_ladder_rows: tuple[dict[str, object], ...]
    bayesian_table_rows: tuple[dict[str, object], ...]
    methodology_tex: str
    corpus_synthesis_algorithm_tex: str
    algorithm_ladder_table_tex: str
    bayesian_update_walkthrough_table_tex: str
    toy_problem_summary_table_tex: str
    study_candidate_generation_algorithm_tex: str


@dataclass(frozen=True, slots=True)
class MethodologyLatexArtifacts:
    run_dir: Path
    source_tex_path: Path
    artifact_tex_path: Path
    pdf_path: Path | None
    algorithm_ladder_csv_path: Path
    toy_problem_summary_csv_path: Path
    corpus_synthesis_algorithm_path: Path
    algorithm_ladder_table_path: Path
    bayesian_update_walkthrough_table_path: Path
    toy_problem_summary_table_path: Path
    study_candidate_generation_algorithm_path: Path


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _tabularx_table(
    *,
    caption: str,
    label: str,
    columns: list[tuple[str, str]],
    rows: list[dict[str, object]],
) -> str:
    colspec = "".join(spec for _, spec in columns)
    header = " & ".join(_latex_escape(name) for name, _ in columns) + r" \\"
    body = "\n".join(
        " & ".join(_latex_escape(row.get(name, "")) for name, _ in columns) + r" \\"
        for row in rows
    )
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{_latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabularx}}{{\textwidth}}{{{colspec}}}",
            r"\toprule",
            header,
            r"\midrule",
            body,
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ]
    )


def _study_candidate_generation_algorithm_tex() -> str:
    return "\n".join(
        [
            r"\begin{enumerate}",
            r"  \item Load the current class-pair manifest, feature-set manifest, classifier manifest, and corpus objectives.",
            r"  \item Enumerate study candidates over $(\text{class pair}, \text{feature set}, \text{classifier}, \text{prior})$.",
            r"  \item Score each candidate statically for feature-class compatibility, oracle separability, corpus coverage, dimensional transfer, and double-counting risk.",
            r"  \item Reuse executed common-study evidence where available to attach classifier accuracy, prior sensitivity, and robustness signals.",
            r"  \item Emit \texttt{promote}, \texttt{revise}, \texttt{reject}, or \texttt{defer} according to the validation ladder rather than ad hoc manual judgment.",
            r"\end{enumerate}",
        ]
    )


def _corpus_synthesis_algorithm_tex() -> str:
    return "\n".join(
        [
            r"\begin{enumerate}",
            r"  \item Load class definitions, scenario templates, and corpus adequacy objectives.",
            r"  \item Sample candidate corpus parameters over balance, difficulty, noise, outliers, and boundary emphasis.",
            r"  \item Generate a candidate corpus and extract the active feature sets.",
            r"  \item Audit adequacy, leakage, feature excitation, and quick oracle/classifier behavior.",
            r"  \item Score the corpus on balance, boundary coverage, feature excitation, difficulty diversity, and leakage penalties.",
            r"  \item Preserve both the selected candidate and the rejected or Pareto-front alternatives as evidence.",
            r"\end{enumerate}",
        ]
    )


def _source_tex() -> str:
    return "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{amsmath, amssymb}",
            r"\usepackage{booktabs}",
            r"\usepackage{enumitem}",
            r"\usepackage{graphicx}",
            r"\usepackage{hyperref}",
            r"\usepackage{tabularx}",
            r"\usepackage{xcolor}",
            r"\graphicspath{{figures/}}",
            r"",
            r"\title{Kinematic Classifier Methodology: Automated Study Design, Corpus Synthesis, and Evidence-Grounded Promotion}",
            r"\author{kinematic-classifier-sandbox}",
            r"\date{\today}",
            r"",
            r"\begin{document}",
            r"",
            r"\maketitle",
            r"",
            r"\begin{abstract}",
            r"This document describes the repository as a methodology stack rather than only as a 1D classifier benchmark. The core contribution is an automated loop over corpus design, feature-set and class-pair analysis, classifier-family evaluation, Bayesian evidence inspection, validation-ladder promotion, and dimensional-transfer auditing. The current 1D studies serve as witness problems proving the methodology layers while also defining the gates that must be satisfied before more advanced filtering or a true 3D transition is justified.",
            r"\end{abstract}",
            r"",
            r"\section{Problem Formulation}",
            r"The active repository goal is not only to maximize one benchmark score, but to prove that a study candidate can be declared, screened, executed, audited, and promoted through a reusable methodology surface. A study candidate combines a corpus, feature set, class set, classifier family, prior specification, and optional filtering backend.",
            r"",
            r"\section{Generic Study Object}",
            r"The machine-readable study candidate schema and validation-ladder schema are implemented in the repository and exercised by the current artifact generation path. The workflow is checklist-driven: static compatibility, corpus adequacy, feature separability, oracle separability, classifier performance, posterior quality, prior sensitivity, robustness, dimensional transfer, and final promotion decision.",
            r"",
            r"\section{Bayesian Evidence Model}",
            r"The repository treats posterior updating as a reusable contract. For class label $c$ and observation history $y_{1:k}$, the recursive update is",
            r"\begin{equation}",
            r"p_k(c) = \frac{p(y_k \mid c)\, p_{k-1}(c)}{\sum_{c'} p(y_k \mid c')\, p_{k-1}(c')}.",
            r"\end{equation}",
            r"In log form,",
            r"\begin{equation}",
            r"\log p_k(c) = \log p_{k-1}(c) + \log p(y_k \mid c) - \log Z_k,",
            r"\end{equation}",
            r"and in the two-class case the log-odds recursion becomes",
            r"\begin{equation}",
            r"\log \frac{p_k(a)}{p_k(b)} = \log \frac{p_{k-1}(a)}{p_{k-1}(b)} + \log \frac{p(y_k \mid a)}{p(y_k \mid b)}.",
            r"\end{equation}",
            r"This is why the walkthrough artifacts emphasize Bayes factors, prior sweeps, and flip thresholds rather than only endpoint accuracy.",
            r"",
            r"\section{Feature Taxonomy}",
            r"The methodology distinguishes instantaneous, windowed, cumulative, robust-summary, derivative-based, and model-residual evidence. Instantaneous features look the most like fresh evidence. Windowed and robust features are summary evidence. Cumulative features can double-count prior signal if they are treated as independent measurements. Model-residual features compare observations against class-conditioned dynamics and are therefore the natural bridge toward filtering backends.",
            r"",
            r"\section{Witness Problems}",
            r"\input{tables/toy_problem_summary_table.tex}",
            r"",
            r"\section{Algorithm Ladder}",
            r"\input{tables/algorithm_ladder_table.tex}",
            r"",
            r"\section{Bayesian Update Walkthrough}",
            r"The current repository emits representative Bayesian walkthroughs from promoted and revised study candidates. These use real common-study trajectories and expose prior odds, class-score increments, posterior odds, prior sweeps, and flip thresholds. The step table below is a compact numeric witness of that process.",
            r"\input{tables/bayesian_update_walkthrough_table.tex}",
            r"",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=0.9\textwidth]{prior_sweep_examples.png}",
            r"\caption{Representative prior-sweep examples from the Bayesian walkthrough suite.}",
            r"\end{figure}",
            r"",
            r"\section{Corpus Adequacy-Driven Corpus Synthesis}",
            r"The corpus layer is treated as an optimization problem over balance, boundary coverage, feature excitation, leakage control, difficulty diversity, and nontriviality. The current automation does not simply audit the default corpus; it generates and ranks alternatives.",
            r"\input{tables/corpus_synthesis_algorithm.tex}",
            r"",
            r"\section{Automated Study Proposal and Promotion}",
            r"The repository now automatically generates study candidates from declared manifests and uses a validation ladder to issue \texttt{promote}, \texttt{revise}, \texttt{reject}, and \texttt{defer} decisions. The algorithmic skeleton used by the current implementation is:",
            r"\input{tables/study_candidate_generation_algorithm.tex}",
            r"",
            r"\section{Corpus Synthesis and Adequacy}",
            r"Corpus design is treated as an optimization problem over balance, boundary coverage, feature excitation, leakage control, difficulty diversity, and nontriviality. Candidate corpora are generated, scored, compared, and filtered through a Pareto-style view instead of being tuned only by anecdotal benchmark inspection.",
            r"",
            r"\section{Feature and Class-Pair Analysis}",
            r"The methodology separates static identifiability from end-to-end classifier success. Feature-space overlap, oracle separability, pair difficulty, and duration/noise sensitivity are all treated as first-class evidence surfaces. This helps distinguish feature failure, corpus failure, and classifier failure.",
            r"",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=0.9\textwidth]{class_confusability_heatmap.png}",
            r"\caption{Feature-space confusability remains pair-specific and motivates targeted feature or corpus work instead of one global leaderboard conclusion.}",
            r"\end{figure}",
            r"",
            r"\section{Filtering Taxonomy and Advanced-Method Gates}",
            r"The current ladder proves the utility of pointwise, windowed, sequential Bayes, Kalman-bank, and transition-matrix evidence surfaces. It also records explicit no-go decisions for IMM, PF, and RBPF unless switching-mode or nonlinear evidence demands them. In the current decision reports, transition-aware accumulation is justified for switching witness problems, IMM remains decision-gated, PF remains decision-gated, and RBPF is only meaningful if future vector studies expose conditionally tractable mixed discrete/continuous structure.",
            r"",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=0.9\textwidth]{transition_matrix_diagnostics.png}",
            r"\caption{Transition-matrix accumulation improves on static accumulation for switching trajectories and acts as the bridge evidence before considering IMM.}",
            r"\end{figure}",
            r"",
            r"\section{Results Summary}",
            r"The current artifact stack proves the methodology process: study candidates can be declared, screened statically, evaluated against generated corpora, explained through posterior and prior-sensitivity surfaces, and assigned promotion decisions. It does not yet prove that the current synthetic corpus is final or fully adequate for every future class family.",
            r"",
            r"\section{3D Transition Status}",
            r"The current methodology layer distinguishes three statuses: \texttt{dimension\_agnostic}, \texttt{adapter\_compatible}, and \texttt{rewrite\_required}. Contracts, shared evaluation, and study promotion are already dimension-aware. Scalar feature extraction and scalar state-space filtering remain rewrite-required, so current 3D readiness is methodological rather than dynamical.",
            r"",
            r"\section{Limitations and Next Work}",
            r"The current common-study evidence still uses a class-score proxy for some walkthroughs, not a pure additive sensor likelihood decomposition for every family. The 1D studies are witness problems rather than a final deployment corpus. The next work after this paper bundle is a proof-oriented showcase refresh that organizes the packet around explicit claims and evidence, followed by true vector corpus, vector feature, and vector filter implementations for a full 3D lift.",
            r"",
            r"\end{document}",
        ]
    )


def analyze_methodology_latex(
    *,
    seed: int = 7,
    trajectories_per_case: int = 6,
) -> MethodologyLatexResult:
    validation = analyze_validation_ladder(seed=seed, trajectories_per_case=trajectories_per_case)
    bayes = analyze_bayesian_walkthroughs(seed=seed, trajectories_per_case=trajectories_per_case)
    transition = run_transition_benchmark(seed=seed)
    advanced = analyze_advanced_filter_decision()

    toy_problem_rows = [
        {
            "toy_problem_id": "pointwise_easy_overlap",
            "purpose": "Lower-bound instantaneous classifier baseline",
            "classes": "stationary vs constant_velocity",
            "features": "instantaneous",
            "classifiers": "pointwise",
            "priors": "uniform/mild/strong",
            "what_it_proves": "A minimal likelihood-only baseline exists and can be audited for prior fragility.",
            "key_artifacts": "pointwise_baseline, prior_sensitivity_pointwise_v1",
            "known_limitations": "Cannot exploit history or dynamics.",
        },
        {
            "toy_problem_id": "windowed_outlier_witness",
            "purpose": "Show robust extrema beating raw extrema under outliers",
            "classes": "constant_velocity vs constant_acceleration",
            "features": "raw_extrema, robust_extrema",
            "classifiers": "windowed_raw_extrema, windowed_robust_extrema",
            "priors": "uniform",
            "what_it_proves": "Feature design changes class stability under corrupted observations.",
            "key_artifacts": "windowed_baseline, common_dataset_comparison_v1",
            "known_limitations": "Feature contributions are not independent Bayes terms.",
        },
        {
            "toy_problem_id": "sequential_history_help",
            "purpose": "Demonstrate that sequential accumulation improves over pointwise evidence",
            "classes": "stationary vs constant_velocity; constant_velocity vs braking",
            "features": "instantaneous",
            "classifiers": "bayes_accumulator",
            "priors": "uniform/mild/strong",
            "what_it_proves": "History can improve classification and exposes prior sensitivity explicitly.",
            "key_artifacts": "bayes_accumulator, monte_carlo_accumulator, bayesian_walkthroughs",
            "known_limitations": "Still static-class unless switching logic is added.",
        },
        {
            "toy_problem_id": "kalman_endpoint_match",
            "purpose": "Show model-based filtering on matched-endpoint irregular tracks",
            "classes": "constant_velocity vs constant_acceleration",
            "features": "model_residuals",
            "classifiers": "kalman_bank",
            "priors": "uniform",
            "what_it_proves": "Dynamics-aware evidence helps when endpoint-only reasoning fails.",
            "key_artifacts": "kalman_filter_bank, common_dataset_comparison_v1",
            "known_limitations": "Position-only sensing remains weak on short noisy horizons.",
        },
        {
            "toy_problem_id": "transition_switching_bridge",
            "purpose": "Bridge from static accumulation to explicit switching dynamics",
            "classes": "constant_velocity vs braking-style mode changes",
            "features": "derived speed/acceleration evidence",
            "classifiers": "static_accumulator, transition_matrix",
            "priors": "uniform",
            "what_it_proves": "Switching structure can help before IMM is justified.",
            "key_artifacts": "transition_matrix_accumulator_v1, advanced_filter_decision_v1",
            "known_limitations": "Not yet a full IMM or nonlinear filter.",
        },
    ]

    decision_lookup = {str(row["study_id"]): str(row["final_decision"]) for row in validation.decision_rows}
    algorithm_ladder_rows = [
        {
            "level": 1,
            "algorithm": "pointwise",
            "new_capability": "Instantaneous class likelihood baseline",
            "assumption_added": "Single-step evidence suffices",
            "failure_mode_addressed": "No baseline for ambiguity",
            "toy_problem_evidence": "pointwise_easy_overlap",
            "promotion_status": decision_lookup.get("instantaneous_stationary_vs_constant_velocity_pointwise_uniform", "n/a"),
        },
        {
            "level": 2,
            "algorithm": "windowed",
            "new_capability": "History-derived engineered features",
            "assumption_added": "Short-window summaries capture local dynamics",
            "failure_mode_addressed": "Pointwise noise sensitivity",
            "toy_problem_evidence": "windowed_outlier_witness",
            "promotion_status": decision_lookup.get("robust_extrema_stationary_vs_constant_velocity_windowed_robust_extrema_uniform", "n/a"),
        },
        {
            "level": 3,
            "algorithm": "sequential_bayes",
            "new_capability": "Recursive posterior accumulation",
            "assumption_added": "Per-step evidence can be accumulated coherently",
            "failure_mode_addressed": "History ignored by pointwise baseline",
            "toy_problem_evidence": "sequential_history_help",
            "promotion_status": decision_lookup.get("instantaneous_stationary_vs_constant_velocity_bayes_accumulator_uniform", "n/a"),
        },
        {
            "level": 4,
            "algorithm": "kalman_bank",
            "new_capability": "Model-based innovation evidence",
            "assumption_added": "Class-specific dynamics are meaningful",
            "failure_mode_addressed": "Endpoint ambiguity under irregular timing",
            "toy_problem_evidence": "kalman_endpoint_match",
            "promotion_status": decision_lookup.get("model_residuals_stationary_vs_constant_velocity_kalman_bank_uniform", "n/a"),
        },
        {
            "level": 5,
            "algorithm": "transition_matrix",
            "new_capability": "Explicit mode switching before IMM",
            "assumption_added": "Simple transition dynamics approximate switching",
            "failure_mode_addressed": "Static accumulator under switching trajectories",
            "toy_problem_evidence": "transition_switching_bridge",
            "promotion_status": "pass" if transition.summary.transition_accuracy > transition.summary.static_accuracy else "revise",
        },
        {
            "level": 6,
            "algorithm": "advanced_filter_gate",
            "new_capability": "Evidence-based go/no-go decision for IMM/PF",
            "assumption_added": "Advanced methods require explicit failure evidence",
            "failure_mode_addressed": "Premature complexity escalation",
            "toy_problem_evidence": "advanced_filter_decision_v1",
            "promotion_status": "defer" if not advanced.imm_justified and not advanced.particle_filter_justified else "promote",
        },
    ]

    promoted_step_rows = [
        row for row in bayes.bayesian_step_rows if str(row.get("example_type", "")) == "trajectory_walkthrough"
    ]
    bayesian_table_rows = [
        {
            "time": float(row["time"]),
            "prior_class_a": round(float(row["prior_a"]), 3),
            "incremental_log_bayes_factor_ab_proxy": round(float(row["log_bayes_factor_ab"]), 3),
            "posterior_class_a": round(float(row["posterior_a"]), 3),
            "predicted_class": str(row["predicted_class"]),
            "confidence": round(max(float(row["posterior_a"]), float(row["posterior_b"])), 3),
        }
        for row in promoted_step_rows[:6]
    ]

    algorithm_ladder_table_tex = _tabularx_table(
        caption="Algorithm ladder proof summary.",
        label="tab:algorithm_ladder",
        columns=[
            ("level", "c"),
            ("algorithm", "l"),
            ("new_capability", "X"),
            ("failure_mode_addressed", "X"),
            ("promotion_status", "c"),
        ],
        rows=algorithm_ladder_rows,
    )
    bayesian_update_walkthrough_table_tex = _tabularx_table(
        caption="Representative Bayesian walkthrough steps from a promoted study candidate.",
        label="tab:bayes_walkthrough",
        columns=[
            ("time", "c"),
            ("prior_class_a", "c"),
            ("incremental_log_bayes_factor_ab_proxy", "c"),
            ("posterior_class_a", "c"),
            ("predicted_class", "l"),
            ("confidence", "c"),
        ],
        rows=bayesian_table_rows,
    )
    toy_problem_summary_table_tex = _tabularx_table(
        caption="Witness problems used to prove distinct methodology layers.",
        label="tab:witness_problems",
        columns=[
            ("toy_problem_id", "l"),
            ("purpose", "X"),
            ("what_it_proves", "X"),
            ("known_limitations", "X"),
        ],
        rows=toy_problem_rows,
    )

    return MethodologyLatexResult(
        toy_problem_rows=tuple(toy_problem_rows),
        algorithm_ladder_rows=tuple(algorithm_ladder_rows),
        bayesian_table_rows=tuple(bayesian_table_rows),
        methodology_tex=_source_tex(),
        corpus_synthesis_algorithm_tex=_corpus_synthesis_algorithm_tex(),
        algorithm_ladder_table_tex=algorithm_ladder_table_tex,
        bayesian_update_walkthrough_table_tex=bayesian_update_walkthrough_table_tex,
        toy_problem_summary_table_tex=toy_problem_summary_table_tex,
        study_candidate_generation_algorithm_tex=_study_candidate_generation_algorithm_tex(),
    )


def _copy_figure(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _compile_pdf(run_dir: Path, tex_path: Path) -> Path | None:
    if shutil.which("latexmk") is None:
        return None
    env = dict(**os.environ)
    env.setdefault("TEXMFVAR", str(run_dir / ".texmf-var"))
    env.setdefault("TEXMFCONFIG", str(run_dir / ".texmf-config"))
    env.setdefault("TEXMFHOME", str(run_dir / ".texmf-home"))
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-outdir=.",
            tex_path.name,
        ],
        check=True,
        cwd=run_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pdf_path = run_dir / "kinematic_classifier_methodology.pdf"
    return pdf_path if pdf_path.exists() else None


def write_methodology_latex_artifacts(
    output_dir: str | Path,
    *,
    result: MethodologyLatexResult | None = None,
    build_pdf: bool = True,
) -> MethodologyLatexArtifacts:
    latex = result or analyze_methodology_latex()
    run_dir = Path(output_dir) / "latex"
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    DOCS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    source_tex_path = SOURCE_TEX_PATH
    artifact_tex_path = run_dir / "kinematic_classifier_methodology.tex"
    algorithm_ladder_csv_path = run_dir / "algorithm_ladder_proof.csv"
    toy_problem_summary_csv_path = run_dir / "toy_problem_summary.csv"
    corpus_synthesis_algorithm_path = tables_dir / "corpus_synthesis_algorithm.tex"
    algorithm_ladder_table_path = tables_dir / "algorithm_ladder_table.tex"
    bayesian_update_walkthrough_table_path = tables_dir / "bayesian_update_walkthrough_table.tex"
    toy_problem_summary_table_path = tables_dir / "toy_problem_summary_table.tex"
    study_candidate_generation_algorithm_path = tables_dir / "study_candidate_generation_algorithm.tex"

    _write_text(source_tex_path, latex.methodology_tex)
    _write_text(artifact_tex_path, latex.methodology_tex)
    _write_text(DOCS_TABLES_DIR / "algorithm_ladder_table.tex", latex.algorithm_ladder_table_tex)
    _write_text(DOCS_TABLES_DIR / "bayesian_update_walkthrough_table.tex", latex.bayesian_update_walkthrough_table_tex)
    _write_text(DOCS_TABLES_DIR / "corpus_synthesis_algorithm.tex", latex.corpus_synthesis_algorithm_tex)
    _write_text(DOCS_TABLES_DIR / "toy_problem_summary_table.tex", latex.toy_problem_summary_table_tex)
    _write_text(DOCS_TABLES_DIR / "study_candidate_generation_algorithm.tex", latex.study_candidate_generation_algorithm_tex)
    _write_text(algorithm_ladder_table_path, latex.algorithm_ladder_table_tex)
    _write_text(bayesian_update_walkthrough_table_path, latex.bayesian_update_walkthrough_table_tex)
    _write_text(corpus_synthesis_algorithm_path, latex.corpus_synthesis_algorithm_tex)
    _write_text(toy_problem_summary_table_path, latex.toy_problem_summary_table_tex)
    _write_text(study_candidate_generation_algorithm_path, latex.study_candidate_generation_algorithm_tex)

    _write_csv(algorithm_ladder_csv_path, list(latex.algorithm_ladder_rows), list(latex.algorithm_ladder_rows[0].keys()))
    _write_csv(toy_problem_summary_csv_path, list(latex.toy_problem_rows), list(latex.toy_problem_rows[0].keys()))

    figure_sources = {
        "prior_sweep_examples.png": ROOT / "artifacts" / "bayesian_walkthroughs" / "prior_sweep_examples.png",
        "class_confusability_heatmap.png": ROOT / "artifacts" / "feature_analysis_v1" / "class_confusability_heatmap.png",
        "transition_matrix_diagnostics.png": ROOT / "artifacts" / "transition_matrix_accumulator_v1" / "transition_matrix_diagnostics.png",
    }
    for filename, source in figure_sources.items():
        if source.exists():
            _copy_figure(source, figures_dir / filename)
            _copy_figure(source, DOCS_FIGURES_DIR / filename)

    pdf_path = _compile_pdf(run_dir, artifact_tex_path) if build_pdf else None

    return MethodologyLatexArtifacts(
        run_dir=run_dir,
        source_tex_path=source_tex_path,
        artifact_tex_path=artifact_tex_path,
        pdf_path=pdf_path,
        algorithm_ladder_csv_path=algorithm_ladder_csv_path,
        toy_problem_summary_csv_path=toy_problem_summary_csv_path,
        corpus_synthesis_algorithm_path=corpus_synthesis_algorithm_path,
        algorithm_ladder_table_path=algorithm_ladder_table_path,
        bayesian_update_walkthrough_table_path=bayesian_update_walkthrough_table_path,
        toy_problem_summary_table_path=toy_problem_summary_table_path,
        study_candidate_generation_algorithm_path=study_candidate_generation_algorithm_path,
    )
