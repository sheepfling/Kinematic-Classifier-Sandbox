from __future__ import annotations

import csv
from pathlib import Path

from kinematic_classifier_sandbox.analysis.static_feature_class_prior_audit_contracts import (
    StaticAuditFeatureSchemaEntry,
    StaticAuditSample,
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value `{value}`")


def _parse_tags(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    text = str(value).strip()
    if not text:
        return ()
    normalized = text.replace("|", ",")
    return tuple(tag.strip() for tag in normalized.split(",") if tag.strip())


def load_feature_schema_entries(
    path: str | Path | None,
) -> tuple[StaticAuditFeatureSchemaEntry, ...]:
    if path is None:
        return ()
    rows = _read_csv_rows(Path(path))
    entries: list[StaticAuditFeatureSchemaEntry] = []
    for row in rows:
        name = str(row.get("feature_name", "")).strip()
        if not name:
            raise ValueError(f"feature schema row in `{path}` is missing `feature_name`")
        entries.append(
            StaticAuditFeatureSchemaEntry(
                name=name,
                provenance_tags=_parse_tags(row.get("provenance_tags")),
                online_available=_parse_bool(row.get("online_available"), default=True),
                label_rule_overlap=_parse_bool(row.get("label_rule_overlap"), default=False),
            )
        )
    return tuple(entries)


def load_class_names(path: str | Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    rows = _read_csv_rows(Path(path))
    class_names = tuple(str(row.get("class_name", "")).strip() for row in rows if str(row.get("class_name", "")).strip())
    if not class_names:
        raise ValueError(f"class schema `{path}` does not declare any `class_name` rows")
    return class_names


def load_samples(
    path: str | Path,
    *,
    feature_names: tuple[str, ...] = (),
) -> tuple[StaticAuditSample, ...]:
    rows = _read_csv_rows(Path(path))
    if not rows:
        raise ValueError(f"sample table `{path}` is empty")
    resolved_feature_names = feature_names or tuple(
        column
        for column in rows[0].keys()
        if column not in {"sample_id", "true_class"}
    )
    if not resolved_feature_names:
        raise ValueError(f"sample table `{path}` does not declare any feature columns")
    samples: list[StaticAuditSample] = []
    for row_index, row in enumerate(rows, start=1):
        true_class = str(row.get("true_class", "")).strip()
        if not true_class:
            raise ValueError(f"sample table `{path}` row {row_index} is missing `true_class`")
        feature_values: dict[str, float] = {}
        for feature_name in resolved_feature_names:
            raw_value = row.get(feature_name)
            if raw_value in {None, ""}:
                raise ValueError(
                    f"sample table `{path}` row {row_index} is missing feature `{feature_name}`"
                )
            try:
                feature_values[feature_name] = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"sample table `{path}` row {row_index} has non-numeric `{feature_name}` value `{raw_value}`"
                ) from exc
        samples.append(
            StaticAuditSample(
                true_class=true_class,
                sample_id=str(row.get("sample_id", "")).strip(),
                feature_values=feature_values,
            )
        )
    return tuple(samples)


def load_static_audit_bundle(
    *,
    sample_table_path: str | Path,
    feature_schema_path: str | Path | None = None,
    class_schema_path: str | Path | None = None,
    feature_names: tuple[str, ...] = (),
) -> tuple[
    tuple[StaticAuditSample, ...],
    tuple[StaticAuditFeatureSchemaEntry, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    feature_schema = load_feature_schema_entries(feature_schema_path)
    class_names = load_class_names(class_schema_path)
    resolved_feature_names = feature_names or tuple(entry.name for entry in feature_schema)
    samples = load_samples(sample_table_path, feature_names=resolved_feature_names)
    sample_class_names = tuple(sorted({sample.true_class for sample in samples}))
    if class_names and tuple(sorted(class_names)) != sample_class_names:
        raise ValueError(
            "declared class schema does not match sample classes: "
            f"declared={tuple(sorted(class_names))} samples={sample_class_names}"
        )
    resolved_feature_names = resolved_feature_names or tuple(samples[0].feature_values.keys())
    if feature_schema:
        schema_feature_names = tuple(entry.name for entry in feature_schema)
        if schema_feature_names != resolved_feature_names:
            raise ValueError(
                "feature schema ordering does not match configured feature names: "
                f"schema={schema_feature_names} configured={resolved_feature_names}"
            )
    return samples, feature_schema, sample_class_names, resolved_feature_names
