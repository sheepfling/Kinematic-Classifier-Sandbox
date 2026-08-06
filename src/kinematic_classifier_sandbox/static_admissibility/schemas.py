from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class StaticAdmissibilityInputBundle:
    config_path: Path
    sample_table_path: Path
    feature_schema_path: Path | None = None
    class_schema_path: Path | None = None
    class_feature_signature_path: Path | None = None
    feature_names: tuple[str, ...] = ()
    declared_dimension: str = ""
    allow_unobserved_classes: bool = False


@dataclass(frozen=True, slots=True)
class StaticAdmissibilityConfig:
    study_id: str = "common_1d_static_admissibility_mvp"
    seed: int = 7
    trajectories_per_class: int = 5
    priors: dict[str, float] | None = None
    source_config_path: Path | None = None
    input_bundle: StaticAdmissibilityInputBundle | None = None


@dataclass(frozen=True, slots=True)
class StaticAdmissibilityPacket:
    packet_dir: Path
    readme_path: Path
    decision_card_path: Path
    static_audit_report_path: Path
    static_audit_decision_card_path: Path
    figure_manifest_path: Path
    lane_proof_matrix_path: Path
    contact_sheet_path: Path


def _resolve_config_relative_path(config_path: Path, raw_value: object) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("static admissibility input bundle paths must be non-empty strings")
    raw_path = Path(raw_value)
    if raw_path.is_absolute():
        return raw_path
    return (config_path.parent / raw_path).resolve()


def _parse_config_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value `{value}` for allow_unobserved_classes")


def load_static_admissibility_config(path: str | Path | None) -> StaticAdmissibilityConfig:
    if path is None:
        return StaticAdmissibilityConfig()
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    static = raw.get("static_admissibility", raw)
    input_bundle = None
    raw_input_bundle = static.get("input_bundle")
    if raw_input_bundle is not None:
        if not isinstance(raw_input_bundle, dict):
            raise ValueError("static_admissibility.input_bundle must be a mapping")
        input_bundle = StaticAdmissibilityInputBundle(
            config_path=config_path.resolve(),
            sample_table_path=_resolve_config_relative_path(
                config_path,
                raw_input_bundle.get("sample_table"),
            ),
            feature_schema_path=None
            if raw_input_bundle.get("feature_schema") in {None, ""}
            else _resolve_config_relative_path(config_path, raw_input_bundle.get("feature_schema")),
            class_schema_path=None
            if raw_input_bundle.get("class_schema") in {None, ""}
            else _resolve_config_relative_path(config_path, raw_input_bundle.get("class_schema")),
            class_feature_signature_path=None
            if raw_input_bundle.get("class_feature_signature") in {None, ""}
            else _resolve_config_relative_path(
                config_path,
                raw_input_bundle.get("class_feature_signature"),
            ),
            feature_names=tuple(str(name) for name in raw_input_bundle.get("feature_names", ())),
            declared_dimension=str(raw_input_bundle.get("dimension", "")),
            allow_unobserved_classes=_parse_config_bool(
                raw_input_bundle.get("allow_unobserved_classes"),
            ),
        )
    return StaticAdmissibilityConfig(
        study_id=str(static.get("study_id", "common_1d_static_admissibility_mvp")),
        seed=int(static.get("seed", 7)),
        trajectories_per_class=int(static.get("trajectories_per_class", 5)),
        priors=None if static.get("priors") is None else dict(static["priors"]),
        source_config_path=config_path.resolve(),
        input_bundle=input_bundle,
    )
