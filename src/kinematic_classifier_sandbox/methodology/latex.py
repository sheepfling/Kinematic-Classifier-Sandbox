from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import _write_json, _write_text, write_csv
from kinematic_classifier_sandbox.utils.runtime import repo_root

from ..inference.transition_matrix_accumulator import run_transition_benchmark
from .context import MethodologyExecutionContext, build_methodology_execution_context
from ..validation.advanced_filter_decision import analyze_advanced_filter_decision
from ..validation.validation_ladder import analyze_validation_ladder
from ..witnesses.toy_1d.bayesian_walkthroughs import analyze_bayesian_walkthroughs

ROOT = repo_root()
DOCS_LATEX_DIR = ROOT / "docs" / "latex"
DOCS_MATH_DIR = ROOT / "docs" / "math"
SYMBOL_GLOSSARY_PATH = DOCS_MATH_DIR / "symbol_glossary.tex"
DOCS_TABLES_DIR = DOCS_LATEX_DIR / "tables"
DOCS_FIGURES_DIR = DOCS_LATEX_DIR / "figures"
SOURCE_TEX_PATH = DOCS_LATEX_DIR / "kinematic_classifier_methodology.tex"
SECTION_SYMBOL_HEADING = r"\paragraph{Section Symbols.}"
SECTION_SYMBOL_BLOCK = r"\sectionsymbols{"


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


@dataclass(frozen=True, slots=True)
class SectionSymbolCoverage:
    required_section_titles: tuple[str, ...]
    covered_section_titles: tuple[str, ...]
    missing_section_titles: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return not self.missing_section_titles


@dataclass(frozen=True, slots=True)
class SectionSymbolAudit:
    section_title: str
    global_symbols: tuple[str, ...]
    declared_symbols: tuple[str, ...]
    used_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]

    @property
    def has_gaps(self) -> bool:
        return bool(self.missing_symbols)


@dataclass(frozen=True, slots=True)
class MethodologySectionSymbolAuditArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    rows_path: Path
    section_coverage_path: Path


IGNORED_SYMBOL_FAMILIES = frozenset(
    {
        "a",
        "b",
        "c",
        "d",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "m",
        "p",
        "P",
        "q",
        "r",
        "s",
        "S",
        "t",
        "w",
        "x",
        "y",
        "z",
        r"\arg",
        r"\begin",
        r"\cdot",
        r"\end",
        r"\exp",
        r"\frac",
        r"\in",
        r"\land",
        r"\left",
        r"\log",
        r"\mathcalN",
        r"\mapsto",
        r"\max",
        r"\mid",
        r"\neg",
        r"\not",
        r"\operatorname",
        r"\propto",
        r"\Rightarrow",
        r"\right",
        r"\rightarrow",
        r"\sim",
        r"\sum",
        r"\text",
    }
)

LATIN_TOKEN_PATTERN = re.compile(
    r"(?<!\\)\b[A-Za-z](?:_\{[^{}]+\}|_[A-Za-z0-9])?(?:\^\{[^{}]+\}|\^[A-Za-z0-9])?(?:\([^)]*\))?"
)
COMMAND_TOKEN_PATTERN = re.compile(
    r"\\(?:mathcal|mathbb|hat|bar|tilde|Delta|Gamma|Lambda|Pi|Sigma|Omega|alpha|beta|gamma|delta|epsilon|eta|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|varphi|psi|omega|ell)"
    r"(?:\{[^{}]+\}|[A-Za-z])?(?:_\{[^{}]+\}|_[A-Za-z0-9])?(?:\^\{[^{}]+\}|\^[A-Za-z0-9])?(?:\([^)]*\))?"
)




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
    columns: list[tuple[str, str] | tuple[str, str, str]],
    rows: list[dict[str, object]],
) -> str:
    normalized: list[tuple[str, str, str]] = []
    for column in columns:
        if len(column) == 2:
            key, spec = column
            normalized.append((key, _latex_escape(key), spec))
        else:
            key, header, spec = column
            normalized.append((key, header, spec))
    colspec = "".join(spec for _, _, spec in normalized)
    header = " & ".join(header_text for _, header_text, _ in normalized) + r" \\"
    body = "\n".join(
        " & ".join(_latex_escape(row.get(name, "")) for name, _, _ in normalized) + r" \\"
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
    return SOURCE_TEX_PATH.read_text(encoding="utf-8")


def _symbol_glossary_tex() -> str:
    return SYMBOL_GLOSSARY_PATH.read_text(encoding="utf-8")


def _find_matching_brace(text: str, open_brace_index: int) -> int:
    depth = 0
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unbalanced braces in LaTeX text")


def _iter_stage_sections(methodology_tex: str) -> tuple[tuple[str, str], ...]:
    section_pattern = re.compile(r"\\section\{([^}]*)\}")
    matches = list(section_pattern.finditer(methodology_tex))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        if not title.startswith("Stage "):
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(methodology_tex)
        sections.append((title, methodology_tex[start:end]))
    return tuple(sections)


def _extract_section_symbol_block(section_body: str) -> tuple[str, str]:
    block_start = section_body.find(SECTION_SYMBOL_BLOCK)
    if block_start < 0:
        return "", section_body
    open_brace_index = block_start + len(SECTION_SYMBOL_BLOCK) - 1
    close_brace_index = _find_matching_brace(section_body, open_brace_index)
    block = section_body[block_start : close_brace_index + 1]
    section_without_block = section_body[:block_start] + section_body[close_brace_index + 1 :]
    return block, section_without_block


def _extract_declared_symbols(symbol_block: str) -> tuple[str, ...]:
    declared_symbols: list[str] = []
    marker = r"\sectionsymbol{"
    start = 0
    while True:
        marker_index = symbol_block.find(marker, start)
        if marker_index < 0:
            break
        open_brace_index = marker_index + len(marker) - 1
        close_brace_index = _find_matching_brace(symbol_block, open_brace_index)
        declared_symbols.append(symbol_block[open_brace_index + 1 : close_brace_index].strip())
        start = close_brace_index + 1
    return tuple(symbol for symbol in declared_symbols if symbol)


def _extract_global_glossary_symbols(glossary_tex: str) -> tuple[str, ...]:
    matches = re.findall(r"(\\\(.+?\\\))\s*&", glossary_tex, flags=re.DOTALL)
    return tuple(symbol.strip() for symbol in matches if symbol.strip())


def _extract_math_chunks(section_body: str) -> tuple[str, ...]:
    chunks: list[str] = []
    patterns = (
        re.compile(r"\\\((.+?)\\\)", re.DOTALL),
        re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
        re.compile(
            r"\\begin\{(?:equation|equation\*|align|align\*)\}(.+?)\\end\{(?:equation|equation\*|align|align\*)\}",
            re.DOTALL,
        ),
    )
    for pattern in patterns:
        chunks.extend(match.group(1) for match in pattern.finditer(section_body))
    return tuple(chunks)


def _normalize_symbol_family(symbol: str) -> str:
    normalized = " ".join(symbol.split())
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = re.sub(r"_[A-Za-z0-9,+\-|:*]+", "_*", normalized)
    normalized = re.sub(r"\^[A-Za-z0-9,+\-|:*]+", "", normalized)
    normalized = re.sub(r"\([^)]*\)", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.strip(",.;:")


def _extract_symbol_families(text: str) -> tuple[str, ...]:
    candidates = list(COMMAND_TOKEN_PATTERN.findall(text))
    candidates.extend(LATIN_TOKEN_PATTERN.findall(text))
    families = set()
    for token in candidates:
        if len(token) == 1 and token.isalpha():
            continue
        family = _normalize_symbol_family(token)
        if not family or family in IGNORED_SYMBOL_FAMILIES:
            continue
        families.add(family)
    return tuple(sorted(families))


def analyze_section_symbol_coverage(methodology_tex: str) -> SectionSymbolCoverage:
    required_section_titles: list[str] = []
    covered_section_titles: list[str] = []
    missing_section_titles: list[str] = []

    for title, body in _iter_stage_sections(methodology_tex):
        required_section_titles.append(title)
        opening_window = body[:1500]
        has_symbol_block = (
            SECTION_SYMBOL_HEADING in opening_window
            and SECTION_SYMBOL_BLOCK in opening_window
        )
        if has_symbol_block:
            covered_section_titles.append(title)
        else:
            missing_section_titles.append(title)

    return SectionSymbolCoverage(
        required_section_titles=tuple(required_section_titles),
        covered_section_titles=tuple(covered_section_titles),
        missing_section_titles=tuple(missing_section_titles),
    )


def analyze_section_symbol_audits(methodology_tex: str) -> tuple[SectionSymbolAudit, ...]:
    global_symbols = _extract_global_glossary_symbols(_symbol_glossary_tex())
    global_families = set()
    for global_symbol in global_symbols:
        global_families.update(_extract_symbol_families(global_symbol))

    audits: list[SectionSymbolAudit] = []
    for title, body in _iter_stage_sections(methodology_tex):
        symbol_block, body_without_symbol_block = _extract_section_symbol_block(body)
        declared_symbols = _extract_declared_symbols(symbol_block)
        declared_families = set()
        for declared_symbol in declared_symbols:
            declared_families.update(_extract_symbol_families(declared_symbol))

        used_families = set()
        for chunk in _extract_math_chunks(body_without_symbol_block):
            used_families.update(_extract_symbol_families(chunk))

        audits.append(
            SectionSymbolAudit(
                section_title=title,
                global_symbols=global_symbols,
                declared_symbols=declared_symbols,
                used_symbols=tuple(sorted(used_families)),
                missing_symbols=tuple(sorted(used_families - declared_families - global_families)),
            )
        )
    return tuple(audits)


def summarize_section_symbol_audits(
    audits: tuple[SectionSymbolAudit, ...],
) -> dict[str, object]:
    sections_with_gaps = [audit for audit in audits if audit.has_gaps]
    clean_sections = [audit for audit in audits if not audit.has_gaps]
    return {
        "section_count": len(audits),
        "sections_with_gaps": len(sections_with_gaps),
        "clean_sections": len(clean_sections),
        "max_missing_symbols": max((len(audit.missing_symbols) for audit in audits), default=0),
        "worst_sections": [
            {
                "section_title": audit.section_title,
                "missing_symbol_count": len(audit.missing_symbols),
                "missing_symbols": list(audit.missing_symbols[:12]),
            }
            for audit in sorted(sections_with_gaps, key=lambda audit: len(audit.missing_symbols), reverse=True)[:5]
        ],
    }


def _methodology_build_status(*, build_pdf: bool) -> dict[str, object]:
    latexmk_available = shutil.which("latexmk") is not None
    if build_pdf and latexmk_available:
        status = "pdf_build_enabled"
    elif build_pdf and not latexmk_available:
        status = "pdf_build_unavailable"
    else:
        status = "source_export_only"
    return {
        "status": status,
        "build_pdf_requested": build_pdf,
        "latexmk_available": latexmk_available,
        "junior_rerun_command": "python -m kinematic_classifier_sandbox methodology-section-symbol-audit",
        "methodology_export_command": "python scripts/export_artifacts.py",
        "tracked_latex_byproducts_required": False,
    }


def _section_symbol_audit_report(
    audits: tuple[SectionSymbolAudit, ...],
    summary: dict[str, object],
    *,
    coverage: SectionSymbolCoverage,
    build_status: dict[str, object],
) -> str:
    lines = [
        "# Methodology Section Symbol Audit",
        "",
        "This artifact scans each `Stage ...` section in the methodology LaTeX document,",
        "compares the local `Section Symbols` declarations against later math usage,",
        "and reports likely undeclared symbol families.",
        "",
        "## Methodology Packet Status",
        "",
        f"- Build status: `{build_status['status']}`",
        f"- PDF build requested: `{build_status['build_pdf_requested']}`",
        f"- `latexmk` available: `{build_status['latexmk_available']}`",
        f"- Junior rerun command: `{build_status['junior_rerun_command']}`",
        f"- Packet export command: `{build_status['methodology_export_command']}`",
        f"- Tracked LaTeX byproducts required: `{build_status['tracked_latex_byproducts_required']}`",
        "",
        "## Section Coverage",
        "",
        f"- Required stage sections: {len(coverage.required_section_titles)}",
        f"- Covered stage sections: {len(coverage.covered_section_titles)}",
        f"- Missing section-symbol blocks: {len(coverage.missing_section_titles)}",
        "",
        f"- Sections scanned: {summary['section_count']}",
        f"- Sections with likely gaps: {summary['sections_with_gaps']}",
        f"- Clean sections: {summary['clean_sections']}",
        f"- Largest gap count in one section: {summary['max_missing_symbols']}",
        "",
        "| Section | Declared | Used | Missing | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for audit in audits:
        lines.append(
            f"| `{audit.section_title}` | {len(audit.declared_symbols)} | {len(audit.used_symbols)} | "
            f"{len(audit.missing_symbols)} | {'gap' if audit.has_gaps else 'clean'} |"
        )
    lines.append("")
    for audit in audits:
        if not audit.has_gaps:
            continue
        preview = ", ".join(f"`{symbol}`" for symbol in audit.missing_symbols[:20])
        lines.append(f"## {audit.section_title}")
        lines.append("")
        lines.append(f"- Declared symbols: {len(audit.declared_symbols)}")
        lines.append(f"- Used symbol families: {len(audit.used_symbols)}")
        lines.append(f"- Likely missing families: {len(audit.missing_symbols)}")
        lines.append(f"- Preview: {preview if preview else '_none_'}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_methodology_section_symbol_audit_artifacts(
    output_dir: str | Path,
    *,
    methodology_tex: str | None = None,
    build_pdf: bool = False,
) -> MethodologySectionSymbolAuditArtifacts:
    run_dir = Path(output_dir) / "methodology_section_symbol_audit"
    report_path = run_dir / "methodology_section_symbol_audit.md"
    summary_path = run_dir / "methodology_section_symbol_audit_summary.json"
    rows_path = run_dir / "methodology_section_symbol_audit_rows.csv"
    section_coverage_path = run_dir / "methodology_section_coverage.json"

    tex = methodology_tex if methodology_tex is not None else _source_tex()
    coverage = analyze_section_symbol_coverage(tex)
    audits = analyze_section_symbol_audits(tex)
    summary = summarize_section_symbol_audits(audits)
    build_status = _methodology_build_status(build_pdf=build_pdf)
    row_dicts = [
        {
            "section_title": audit.section_title,
            "declared_symbol_count": len(audit.declared_symbols),
            "used_symbol_count": len(audit.used_symbols),
            "missing_symbol_count": len(audit.missing_symbols),
            "declared_symbols": "; ".join(audit.declared_symbols),
            "used_symbols": "; ".join(audit.used_symbols),
            "missing_symbols": "; ".join(audit.missing_symbols),
            "status": "gap" if audit.has_gaps else "clean",
        }
        for audit in audits
    ]
    _write_text(
        report_path,
        _section_symbol_audit_report(
            audits,
            summary,
            coverage=coverage,
            build_status=build_status,
        ),
    )
    _write_json(
        summary_path,
        {
            **summary,
            "section_coverage_complete": coverage.is_complete,
            "missing_section_titles": list(coverage.missing_section_titles),
            "build_status": build_status,
        },
    )
    _write_json(
        section_coverage_path,
        {
            "required_section_titles": list(coverage.required_section_titles),
            "covered_section_titles": list(coverage.covered_section_titles),
            "missing_section_titles": list(coverage.missing_section_titles),
            "is_complete": coverage.is_complete,
        },
    )
    write_csv(
        rows_path,
        row_dicts,
        [
            "section_title",
            "declared_symbol_count",
            "used_symbol_count",
            "missing_symbol_count",
            "declared_symbols",
            "used_symbols",
            "missing_symbols",
            "status",
        ],
    )
    return MethodologySectionSymbolAuditArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        rows_path=rows_path,
        section_coverage_path=section_coverage_path,
    )


def analyze_methodology_latex(
    *,
    seed: int = 7,
    trajectories_per_case: int = 6,
    methodology_context: MethodologyExecutionContext | None = None,
    use_cache: bool = True,
) -> MethodologyLatexResult:
    context = methodology_context or build_methodology_execution_context(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        use_cache=use_cache,
    )
    validation = context.validation_result
    bayes = analyze_bayesian_walkthroughs(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        common_result=context.common_result,
    )
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
        {
            "toy_problem_id": "nonlinear_drag_particle_filter",
            "purpose": "Show nonlinear/non-Gaussian filtering beyond linear-Gaussian dynamics",
            "classes": "constant_velocity vs nonlinear_drag",
            "features": "particle evidence over latent state trajectories",
            "classifiers": "particle_filter_bank",
            "priors": "uniform",
            "what_it_proves": "Particle evidence can absorb drag and outlier behavior through sampled state histories.",
            "key_artifacts": "particle_filter_v1, advanced_filter_comparison_v1",
            "known_limitations": "Witness is targeted and does not replace simpler rungs on easy regimes.",
        },
        {
            "toy_problem_id": "ornstein_uhlenbeck_mean_reversion",
            "purpose": "Demonstrate mean-reverting stochastic dynamics as a concrete SDE-style witness",
            "classes": "constant_velocity vs mean_reverting_velocity",
            "features": "particle evidence over mean-reverting velocity state",
            "classifiers": "ornstein_uhlenbeck_pf_v1",
            "priors": "uniform",
            "what_it_proves": "The advanced branch can represent pull-back-to-class behavior with a typed witness rather than only prose.",
            "key_artifacts": "ornstein_uhlenbeck_witness_v1",
            "known_limitations": "This is a 1D witness, not a general continuous-time SDE framework.",
        },
        {
            "toy_problem_id": "latent_maneuver_onset_rbpf",
            "purpose": "Separate sampled latent mode timing from conditional continuous state filtering",
            "classes": "coast vs maneuver onset",
            "features": "sampled mode path plus conditional Kalman state",
            "classifiers": "rbpf",
            "priors": "uniform",
            "what_it_proves": "RBPF can carry discrete latent timing while retaining analytic conditional state updates.",
            "key_artifacts": "rbpf_v1, advanced_filter_comparison_v1",
            "known_limitations": "Current witness is still 1D and mode-structured.",
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
            "algorithm": "imm",
            "new_capability": "Switching-aware state mixing with shared posterior output",
            "assumption_added": "Mode-conditioned state distributions must be mixed, not only class mass",
            "failure_mode_addressed": "Transition-only switching evidence saturates",
            "toy_problem_evidence": "advanced_filter_decision_v1, imm_filter_v1",
            "promotion_status": "defer" if not advanced.imm_justified else "promote",
        },
        {
            "level": 7,
            "algorithm": "particle_filter_bank",
            "new_capability": "Sampled nonlinear and non-Gaussian state evidence",
            "assumption_added": "State posteriors need particle support rather than only Gaussian summaries",
            "failure_mode_addressed": "Linear-Gaussian baselines fail under drag, outliers, or mean reversion",
            "toy_problem_evidence": "particle_filter_v1, ornstein_uhlenbeck_witness_v1",
            "promotion_status": "promote",
        },
        {
            "level": 8,
            "algorithm": "rbpf",
            "new_capability": "Sampled latent mode path with conditional analytic state filtering",
            "assumption_added": "The latent structure splits into sampled discrete hypotheses plus tractable continuous state",
            "failure_mode_addressed": "Pure PF wastes structure on mixed discrete/continuous latent problems",
            "toy_problem_evidence": "rbpf_v1",
            "promotion_status": "promote",
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
            ("time", r"$t_k$", "c"),
            ("prior_class_a", r"$p_{k-1}(a)$", "c"),
            ("incremental_log_bayes_factor_ab_proxy", r"$\Delta \lambda_k^{ab}$", "c"),
            ("posterior_class_a", r"$p_k(a)$", "c"),
            ("predicted_class", r"$\hat{c}_k$", "l"),
            ("confidence", r"$\max_c p_k(c)$", "c"),
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


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


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
    methodology_context: MethodologyExecutionContext | None = None,
    build_pdf: bool = True,
    artifact_mode: str = "full",
) -> MethodologyLatexArtifacts:
    if artifact_mode == "fast":
        build_pdf = False
    latex = result or analyze_methodology_latex(methodology_context=methodology_context)
    run_dir = Path(output_dir) / "latex"
    figures_dir = run_dir / "figures"
    math_dir = run_dir / "math"
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
    artifact_tex = latex.methodology_tex.replace("../math/", "math/")
    _write_text(artifact_tex_path, artifact_tex)
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

    write_csv(algorithm_ladder_csv_path, list(latex.algorithm_ladder_rows), list(latex.algorithm_ladder_rows[0].keys()))
    write_csv(toy_problem_summary_csv_path, list(latex.toy_problem_rows), list(latex.toy_problem_rows[0].keys()))

    figure_sources = {
        "prior_sweep_examples.png": ROOT / "artifacts" / "bayesian_walkthroughs" / "prior_sweep_examples.png",
        "class_confusability_heatmap.png": ROOT / "artifacts" / "feature_analysis_v1" / "class_confusability_heatmap.png",
        "transition_matrix_diagnostics.png": ROOT / "artifacts" / "transition_matrix_accumulator_v1" / "transition_matrix_diagnostics.png",
    }
    for filename, source in figure_sources.items():
        if source.exists():
            _copy_figure(source, figures_dir / filename)
            _copy_figure(source, DOCS_FIGURES_DIR / filename)
    if DOCS_MATH_DIR.exists():
        _copy_tree(DOCS_MATH_DIR, math_dir)

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
