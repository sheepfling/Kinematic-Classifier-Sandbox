from __future__ import annotations

import json
from pathlib import Path

from .gym_types import CorpusGymArtifacts, CorpusGymContractResult


def render_corpus_gym_numeric_walkthrough_markdown(
    result: CorpusGymContractResult | None = None,
) -> str:
    if result is None:
        from .gym import analyze_corpus_gym_contract

        result = analyze_corpus_gym_contract()
    contract = result
    episode = contract.example_episode
    reward = episode.reward
    lines = [
        "# Corpus Gym Numeric Walkthrough",
        "",
        "This walkthrough shows how the search-facing environment computes a reward on one concrete episode.",
        "",
        f"- target_id: `{episode.target.target_id}`",
        f"- target_type: `{episode.target.target_type}`",
        f"- generated_class: `{episode.trajectory.true_class}`",
        f"- generated_tier: `{episode.trajectory.generator_parameters.get('tier')}`",
        "",
        "## Reward Components",
        "",
        f"- `class_validity (C)`: `{reward.class_validity:.3f}`",
        f"- `feature_excitation (E)`: `{reward.feature_excitation:.3f}`",
        f"- `coverage_gain (G)`: `{reward.coverage_gain:.3f}`",
        f"- `boundary_closeness (B)`: `{reward.boundary_closeness:.3f}`",
        f"- `classifier_stress (S)`: `{reward.classifier_stress:.3f}`",
        f"- `prior_sensitivity (P)`: `{reward.prior_sensitivity:.3f}`",
        f"- `leakage_penalty (L)`: `{reward.leakage_penalty:.3f}`",
        f"- `physical_invalidity_penalty (I)`: `{reward.physical_invalidity_penalty:.3f}`",
        f"- `total_utility`: `{reward.total_utility:.3f}`",
        "",
        "## Interpretation",
        "",
        "- This example is high utility because it is valid, feature-matching, and",
        "  moderately stressful without paying much leakage or invalidity cost.",
        "- The walkthrough is the Gym-side analogue of the transition and corpus",
        "  autodevelopment numeric examples: it exposes how a search-facing reward",
        "  is actually computed on one concrete episode.",
    ]
    return "\n".join(lines)


def write_corpus_gym_artifacts(
    output_dir: str | Path,
    *,
    result: CorpusGymContractResult | None = None,
) -> CorpusGymArtifacts:
    if result is None:
        from .gym import analyze_corpus_gym_contract

        result = analyze_corpus_gym_contract()
    contract = result
    run_dir = Path(output_dir) / "corpus_gym"
    run_dir.mkdir(parents=True, exist_ok=True)
    environment_contract_path = run_dir / "environment_contract.json"
    example_targets_path = run_dir / "example_targets.json"
    report_path = run_dir / "corpus_gym_report.md"
    numeric_walkthrough_path = run_dir / "corpus_gym_numeric_walkthrough.md"
    environment_contract_path.write_text(json.dumps(contract.environment_contract, indent=2), encoding="utf-8")
    example_targets_path.write_text(json.dumps({"targets": list(contract.example_targets)}, indent=2), encoding="utf-8")
    report_path.write_text(contract.report_markdown, encoding="utf-8")
    numeric_walkthrough_path.write_text(render_corpus_gym_numeric_walkthrough_markdown(contract), encoding="utf-8")
    return CorpusGymArtifacts(
        run_dir=run_dir,
        environment_contract_path=environment_contract_path,
        example_targets_path=example_targets_path,
        report_path=report_path,
        numeric_walkthrough_path=numeric_walkthrough_path,
    )
