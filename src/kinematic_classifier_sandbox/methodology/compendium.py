from __future__ import annotations

from kinematic_classifier_sandbox.markdown_builder import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import _write_text
from kinematic_classifier_sandbox.utils.runtime import repo_root

from dataclasses import dataclass
from pathlib import Path


ROOT = repo_root()
DOCS_SURVEYS_DIR = ROOT / "docs" / "surveys"


@dataclass(frozen=True, slots=True)
class MethodologyCompendiumResult:
    markdown: str


@dataclass(frozen=True, slots=True)
class MethodologyCompendiumArtifacts:
    run_dir: Path
    source_markdown_path: Path
    artifact_markdown_path: Path


SURVEY_SPECS: tuple[dict[str, object], ...] = (
    {
        "slug": "posterior_update_math",
        "title": "Posterior Update Math",
        "source": DOCS_SURVEYS_DIR / "posterior_update_math.md",
        "artifact": "artifacts/posterior_update_math.pdf",
    },
    {
        "slug": "methodology_evaluation_framework",
        "title": "Methodology Evaluation Framework",
        "source": DOCS_SURVEYS_DIR / "methodology_evaluation_framework.md",
        "artifact": "artifacts/methodology_evaluation_framework.pdf",
    },
    {
        "slug": "classifier_ladder_and_contracts",
        "title": "Classifier Ladder and Contracts",
        "source": DOCS_SURVEYS_DIR / "classifier_ladder_and_contracts.md",
        "artifact": "artifacts/classifier_ladder_and_contracts.pdf",
    },
    {
        "slug": "corpus_generation_and_search",
        "title": "Corpus Generation and Search",
        "source": DOCS_SURVEYS_DIR / "corpus_generation_and_search.md",
        "artifact": "artifacts/corpus_generation_and_search.pdf",
    },
    {
        "slug": "dimensional_lift_and_advanced_filter_gates",
        "title": "Dimensional Lift and Advanced Filter Gates",
        "source": DOCS_SURVEYS_DIR / "dimensional_lift_and_advanced_filter_gates.md",
        "artifact": "artifacts/dimensional_lift_and_advanced_filter_gates.pdf",
    },
)


def _strip_top_title(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    return "\n".join(lines).rstrip()



def analyze_methodology_compendium() -> MethodologyCompendiumResult:
    report = MarkdownDocument()
    report.heading("Kinematic Classifier Methodology Compendium")
    report.paragraph("This document combines the current survey notes into one reference file.")
    report.paragraph(
        "Use it when you want the full methodology stack in one place rather than reading "
        "the survey notes separately."
    )
    report.paragraph(
        "For a shorter narrative entry point, start with "
        f"{report.markdown_link('/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/artifacts/latex/kinematic_classifier_methodology.pdf', 'artifacts/latex/kinematic_classifier_methodology.pdf')}."
    )
    report.paragraph("This compendium is the long-form reference companion to that paper.")
    report.heading("Included Documents", level=2)
    included_entries: list[str] = []
    for index, spec in enumerate(SURVEY_SPECS, start=1):
        title = str(spec["title"])
        source = Path(spec["source"])
        artifact = str(spec["artifact"])
        included_entries.append(
            f"{report.markdown_link(str(source), label=title)} with rendered companion {report.markdown_link(f'/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/{artifact}', label=f'`{artifact}`')}."
        )
    report.ordered_list(included_entries)
    for index, spec in enumerate(SURVEY_SPECS, start=1):
        title = str(spec["title"])
        source = Path(spec["source"])
        body = _strip_top_title(source.read_text(encoding="utf-8"))
        report.heading(f"Part {index}. {title}", level=2)
        report.paragraph(f"Source: [{source.name}]({source})")
        report.paragraph(body)
    return MethodologyCompendiumResult(markdown=report.text())


def write_methodology_compendium_artifacts(
    output_dir: str | Path,
    *,
    result: MethodologyCompendiumResult | None = None,
) -> MethodologyCompendiumArtifacts:
    compendium = result or analyze_methodology_compendium()
    run_dir = Path(output_dir)
    source_markdown_path = DOCS_SURVEYS_DIR / "methodology_compendium.md"
    artifact_markdown_path = run_dir / "methodology_compendium.md"
    _write_text(source_markdown_path, compendium.markdown)
    _write_text(artifact_markdown_path, compendium.markdown)
    return MethodologyCompendiumArtifacts(
        run_dir=run_dir,
        source_markdown_path=source_markdown_path,
        artifact_markdown_path=artifact_markdown_path,
    )
