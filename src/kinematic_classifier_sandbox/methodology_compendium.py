from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def analyze_methodology_compendium() -> MethodologyCompendiumResult:
    parts: list[str] = [
        "# Kinematic Classifier Methodology Compendium",
        "",
        "This document combines the current survey notes into one reference file.",
        "Use it when you want the full methodology stack in one place rather",
        "than reading the survey notes separately.",
        "",
        "For a shorter narrative entry point, start with",
        "[`artifacts/latex/kinematic_classifier_methodology.pdf`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/artifacts/latex/kinematic_classifier_methodology.pdf).",
        "This compendium is the long-form reference companion to that paper.",
        "",
        "## Included Documents",
        "",
    ]
    for index, spec in enumerate(SURVEY_SPECS, start=1):
        title = str(spec["title"])
        source = Path(spec["source"])
        artifact = str(spec["artifact"])
        parts.append(
            f"{index}. [{title}]({source})"
            f" with rendered companion [`{artifact}`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/{artifact})."
        )
    for index, spec in enumerate(SURVEY_SPECS, start=1):
        title = str(spec["title"])
        source = Path(spec["source"])
        body = _strip_top_title(source.read_text(encoding="utf-8"))
        parts.extend(
            [
                "",
                f"## Part {index}. {title}",
                "",
                f"Source: [{source.name}]({source})",
                "",
                body,
            ]
        )
    return MethodologyCompendiumResult(markdown="\n".join(parts).rstrip() + "\n")


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
