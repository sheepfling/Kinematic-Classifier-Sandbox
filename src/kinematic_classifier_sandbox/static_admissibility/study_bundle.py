from __future__ import annotations

import csv
from pathlib import Path

from kinematic_classifier_sandbox.analysis.static_feature_class_prior_audit_contracts import (
    StaticAuditClassFeatureExpectation,
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


def _parse_optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid numeric schema value `{value}`") from exc


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
                semantic_group=str(row.get("semantic_group", "") or "").strip(),
                units=str(row.get("units", "") or "").strip(),
                aggregation=str(row.get("aggregation", "") or "").strip(),
                dimension=str(row.get("dimension", "") or "").strip(),
                measurement_resolution=_parse_optional_float(row.get("measurement_resolution")),
                threshold_operator=str(row.get("threshold_operator", "") or "").strip(),
                threshold_value=_parse_optional_float(row.get("threshold_value")),
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
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"class schema `{path}` contains duplicate `class_name` rows")
    return class_names


def load_class_feature_expectations(
    path: str | Path | None,
    *,
    class_names: tuple[str, ...] = (),
    feature_names: tuple[str, ...] = (),
) -> tuple[StaticAuditClassFeatureExpectation, ...]:
    if path is None:
        return ()
    rows = _read_csv_rows(Path(path))
    declared_classes = set(class_names)
    declared_features = set(feature_names)
    expectations: list[StaticAuditClassFeatureExpectation] = []
    seen: set[tuple[str, str]] = set()
    for row_index, row in enumerate(rows, start=1):
        class_name = str(row.get("class_name", "")).strip()
        feature_name = str(row.get("feature_name", "")).strip()
        if not class_name or not feature_name:
            raise ValueError(
                f"class feature signature row {row_index} in `{path}` requires `class_name` and `feature_name`"
            )
        if declared_classes and class_name not in declared_classes:
            raise ValueError(
                f"class feature signature row {row_index} names undeclared class `{class_name}`"
            )
        if declared_features and feature_name not in declared_features:
            raise ValueError(
                f"class feature signature row {row_index} names undeclared feature `{feature_name}`"
            )
        key = (class_name, feature_name)
        if key in seen:
            raise ValueError(f"class feature signature `{path}` contains duplicate row for {class_name}/{feature_name}")
        seen.add(key)
        expected_std = _parse_optional_float(row.get("expected_std"))
        if expected_std is not None and expected_std < 0.0:
            raise ValueError(f"class feature signature row {row_index} has negative `expected_std`")
        expectations.append(
            StaticAuditClassFeatureExpectation(
                class_name=class_name,
                feature_name=feature_name,
                expected_mean=_parse_optional_float(row.get("expected_mean")),
                expected_std=expected_std,
                source=str(row.get("source", "") or "").strip(),
            )
        )
    return tuple(expectations)


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
    allow_unobserved_classes: bool = False,
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
    resolved_class_names = sample_class_names
    if class_names:
        declared_class_names = tuple(sorted(class_names))
        undeclared_sample_classes = tuple(sorted(set(sample_class_names) - set(declared_class_names)))
        if undeclared_sample_classes:
            raise ValueError(
                "sample table contains classes missing from the declared class schema: "
                f"undeclared={undeclared_sample_classes} declared={declared_class_names}"
            )
        if not allow_unobserved_classes and declared_class_names != sample_class_names:
            raise ValueError(
                "declared class schema does not match sample classes: "
                f"declared={declared_class_names} samples={sample_class_names}"
            )
        resolved_class_names = declared_class_names if allow_unobserved_classes else sample_class_names
    resolved_feature_names = resolved_feature_names or tuple(samples[0].feature_values.keys())
    if feature_schema:
        schema_feature_names = tuple(entry.name for entry in feature_schema)
        if schema_feature_names != resolved_feature_names:
            raise ValueError(
                "feature schema ordering does not match configured feature names: "
                f"schema={schema_feature_names} configured={resolved_feature_names}"
            )
    return samples, feature_schema, resolved_class_names, resolved_feature_names
