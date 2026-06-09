from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

from .feature_analysis import (
    FeatureSpec,
    load_feature_registry,
    load_feature_set_manifest,
    resolve_feature_names,
)


@dataclass(frozen=True, slots=True)
class GenericFeatureTaxonomyResult:
    taxonomy_rows: tuple[dict[str, object], ...]
    feature_set_rows: tuple[dict[str, object], ...]
    sensitivity_rows: tuple[dict[str, object], ...]
    dependency_rows: tuple[dict[str, object], ...]
    transfer_report_markdown: str
    validation_results: dict[str, object]


@dataclass(frozen=True, slots=True)
class GenericFeatureTaxonomyArtifacts:
    run_dir: Path
    taxonomy_path: Path
    feature_sets_path: Path
    sensitivity_matrix_path: Path
    dependency_matrix_path: Path
    transfer_report_path: Path
    validation_results_path: Path


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _taxonomy_rows(registry: dict[str, FeatureSpec]) -> tuple[dict[str, object], ...]:
    rows = []
    for name, spec in sorted(registry.items()):
        rows.append(
            {
                "name": name,
                "group": spec.group,
                "role": spec.role,
                "description": spec.description,
                "history_behavior": spec.history_behavior,
                "geometry_assumption": spec.geometry_assumption,
                "dimensional_transfer": spec.dimensional_transfer,
                "dependency_tags": list(spec.dependency_tags),
                "sensitivity_tags": list(spec.sensitivity_tags),
                "default_excitation_thresholds": list(spec.default_excitation_thresholds),
            }
        )
    return tuple(rows)


def _feature_set_rows(
    registry: dict[str, FeatureSpec],
    manifest: dict[str, dict[str, object]],
) -> tuple[dict[str, object], ...]:
    rows = []
    for feature_set_id, entry in sorted(manifest.items()):
        feature_names = resolve_feature_names(feature_set=feature_set_id, manifest=manifest)
        groups = sorted({registry[name].group for name in feature_names})
        roles = sorted({registry[name].role for name in feature_names})
        transfer_modes = sorted({registry[name].dimensional_transfer for name in feature_names})
        rows.append(
            {
                "feature_set_id": feature_set_id,
                "description": str(entry.get("description", "")),
                "history_behavior": str(entry.get("history_behavior", "unknown")),
                "num_features": len(feature_names),
                "feature_names": list(feature_names),
                "groups": groups,
                "roles": roles,
                "transfer_modes": transfer_modes,
            }
        )
    return tuple(rows)


def _sensitivity_rows(registry: dict[str, FeatureSpec]) -> tuple[dict[str, object], ...]:
    rows = []
    for name, spec in sorted(registry.items()):
        tags = set(spec.sensitivity_tags)
        rows.append(
            {
                "feature_name": name,
                "history_behavior": spec.history_behavior,
                "duration_sensitive": "duration_sensitive" in tags,
                "sample_count_sensitive": "sample_count_sensitive" in tags,
                "sampling_rate_sensitive": "sampling_rate_sensitive" in tags,
                "sampling_irregularity_sensitive": "sampling_irregularity_sensitive" in tags,
                "sampling_gap_sensitive": "sampling_gap_sensitive" in tags,
                "noise_sensitive": "noise_sensitive" in tags,
                "outlier_sensitive": "outlier_sensitive" in tags,
                "window_definition_sensitive": "window_definition_sensitive" in tags,
                "dt_sensitive": "dt_sensitive" in tags,
            }
        )
    return tuple(rows)


def _dependency_rows(registry: dict[str, FeatureSpec]) -> tuple[dict[str, object], ...]:
    dependency_vocab = sorted(
        {
            dependency
            for spec in registry.values()
            for dependency in spec.dependency_tags
        }
    )
    rows = []
    for name, spec in sorted(registry.items()):
        row: dict[str, object] = {"feature_name": name}
        for dependency in dependency_vocab:
            row[dependency] = dependency in spec.dependency_tags
        rows.append(row)
    return tuple(rows)


def _validation_results(
    registry: dict[str, FeatureSpec],
    manifest: dict[str, dict[str, object]],
) -> dict[str, object]:
    missing_metadata = []
    cumulative_mislabeled = []
    scalar_only = []
    vector_compatible = []
    for name, spec in sorted(registry.items()):
        if not all(
            [
                spec.group,
                spec.role,
                spec.description,
                spec.history_behavior,
                spec.geometry_assumption,
                spec.dimensional_transfer,
                spec.dependency_tags,
                spec.sensitivity_tags,
            ]
        ):
            missing_metadata.append(name)
        if any(tag in spec.sensitivity_tags for tag in ("duration_sensitive", "sample_count_sensitive")) and spec.history_behavior == "instantaneous":
            cumulative_mislabeled.append(name)
        if "scalar" in spec.geometry_assumption:
            scalar_only.append(name)
        else:
            vector_compatible.append(name)

    tag_selection_examples = {
        "sampling_features": resolve_feature_names(required_tags=("sampling",)),
        "outlier_features": resolve_feature_names(required_tags=("outlier_sensitive",)),
        "vector_compatible_timing": resolve_feature_names(required_tags=("vector_compatible", "timing")),
    }
    unknown_sets = [
        feature_set_id
        for feature_set_id in manifest
        if not resolve_feature_names(feature_set=feature_set_id, manifest=manifest)
    ]
    return {
        "all_features_have_metadata": not missing_metadata,
        "missing_metadata": missing_metadata,
        "cumulative_features_labeled": not cumulative_mislabeled,
        "cumulative_mislabeled": cumulative_mislabeled,
        "scalar_only_features": scalar_only,
        "vector_compatible_features": vector_compatible,
        "feature_set_tag_selection_examples": tag_selection_examples,
        "all_feature_sets_resolve": not unknown_sets,
        "unresolved_feature_sets": unknown_sets,
        "overall_status": "pass" if not missing_metadata and not cumulative_mislabeled and not unknown_sets else "fail",
    }


def render_generic_feature_taxonomy_report(
    *,
    taxonomy_rows: tuple[dict[str, object], ...],
    feature_set_rows: tuple[dict[str, object], ...],
    validation_results: dict[str, object],
) -> str:
    taxonomy_lines = "\n".join(
        f"| {row['name']} | {row['role']} | {row['history_behavior']} | {row['geometry_assumption']} | {row['dimensional_transfer']} |"
        for row in taxonomy_rows
    )
    feature_set_lines = "\n".join(
        f"| {row['feature_set_id']} | {row['history_behavior']} | {row['num_features']} | {', '.join(row['roles'])} |"
        for row in feature_set_rows
    )
    tag_examples = validation_results["feature_set_tag_selection_examples"]
    return "\n".join(
        [
            "# Generic Feature Taxonomy",
            "",
            "This artifact proves that feature machinery is driven by generic metadata rather than only by named 1D feature bundles.",
            "",
            "## Validation Summary",
            "",
            f"- Overall status: `{validation_results['overall_status']}`",
            f"- All features have metadata: `{validation_results['all_features_have_metadata']}`",
            f"- Cumulative/history labels complete: `{validation_results['cumulative_features_labeled']}`",
            f"- All named feature sets resolve: `{validation_results['all_feature_sets_resolve']}`",
            "",
            "## Feature Taxonomy",
            "",
            "| feature_name | role | history_behavior | geometry_assumption | dimensional_transfer |",
            "| --- | --- | --- | --- | --- |",
            taxonomy_lines,
            "",
            "## Feature Sets",
            "",
            "| feature_set_id | history_behavior | num_features | roles |",
            "| --- | --- | ---: | --- |",
            feature_set_lines,
            "",
            "## Tag Selection Examples",
            "",
            f"- `sampling`: `{', '.join(tag_examples['sampling_features'])}`",
            f"- `outlier_sensitive`: `{', '.join(tag_examples['outlier_features'])}`",
            f"- `vector_compatible + timing`: `{', '.join(tag_examples['vector_compatible_timing'])}`",
            "",
            "## Notes",
            "",
            "- `geometry_assumption` distinguishes dimension-agnostic timing features from scalar-axis kinematic features that need a vector policy in 3D.",
            "- `dimensional_transfer` makes the 3D lift constraint explicit per feature instead of hiding it in implementation details.",
            "- `sensitivity_tags` and `dependency_tags` are queryable so feature bundles can be selected by semantics, not only by hardcoded names.",
        ]
    )


def analyze_generic_feature_taxonomy() -> GenericFeatureTaxonomyResult:
    registry = load_feature_registry()
    manifest = load_feature_set_manifest()
    taxonomy_rows = _taxonomy_rows(registry)
    feature_set_rows = _feature_set_rows(registry, manifest)
    sensitivity_rows = _sensitivity_rows(registry)
    dependency_rows = _dependency_rows(registry)
    validation_results = _validation_results(registry, manifest)
    transfer_report_markdown = render_generic_feature_taxonomy_report(
        taxonomy_rows=taxonomy_rows,
        feature_set_rows=feature_set_rows,
        validation_results=validation_results,
    )
    return GenericFeatureTaxonomyResult(
        taxonomy_rows=taxonomy_rows,
        feature_set_rows=feature_set_rows,
        sensitivity_rows=sensitivity_rows,
        dependency_rows=dependency_rows,
        transfer_report_markdown=transfer_report_markdown,
        validation_results=validation_results,
    )


def write_generic_feature_taxonomy_artifacts(
    output_dir: str | Path,
    *,
    result: GenericFeatureTaxonomyResult | None = None,
) -> GenericFeatureTaxonomyArtifacts:
    taxonomy = result or analyze_generic_feature_taxonomy()
    run_dir = Path(output_dir) / "feature_taxonomy"
    run_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_path = run_dir / "feature_taxonomy.json"
    feature_sets_path = run_dir / "feature_sets.json"
    sensitivity_matrix_path = run_dir / "feature_sensitivity_matrix.csv"
    dependency_matrix_path = run_dir / "feature_dependency_matrix.csv"
    transfer_report_path = run_dir / "feature_transfer_report.md"
    validation_results_path = run_dir / "validation_results.json"

    taxonomy_path.write_text(json.dumps(list(taxonomy.taxonomy_rows), indent=2), encoding="utf-8")
    feature_sets_path.write_text(json.dumps(list(taxonomy.feature_set_rows), indent=2), encoding="utf-8")
    _write_csv(
        sensitivity_matrix_path,
        list(taxonomy.sensitivity_rows),
        [
            "feature_name",
            "history_behavior",
            "duration_sensitive",
            "sample_count_sensitive",
            "sampling_rate_sensitive",
            "sampling_irregularity_sensitive",
            "sampling_gap_sensitive",
            "noise_sensitive",
            "outlier_sensitive",
            "window_definition_sensitive",
            "dt_sensitive",
        ],
    )
    dependency_fieldnames = sorted({key for row in taxonomy.dependency_rows for key in row})
    _write_csv(
        dependency_matrix_path,
        list(taxonomy.dependency_rows),
        dependency_fieldnames,
    )
    transfer_report_path.write_text(taxonomy.transfer_report_markdown, encoding="utf-8")
    validation_results_path.write_text(json.dumps(taxonomy.validation_results, indent=2), encoding="utf-8")

    return GenericFeatureTaxonomyArtifacts(
        run_dir=run_dir,
        taxonomy_path=taxonomy_path,
        feature_sets_path=feature_sets_path,
        sensitivity_matrix_path=sensitivity_matrix_path,
        dependency_matrix_path=dependency_matrix_path,
        transfer_report_path=transfer_report_path,
        validation_results_path=validation_results_path,
    )
