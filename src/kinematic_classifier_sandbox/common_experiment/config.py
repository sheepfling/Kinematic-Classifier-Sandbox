from __future__ import annotations

from pathlib import Path

import yaml

from ..utils.runtime import repo_root
from .adapters import (
    ExecutablePairSpec,
    build_pair_specs,
    generate_boundary_pair_dataset,
    generate_pair_dataset,
)
from .config_models import CommonExperimentConfigModel
from .contracts import CommonExperimentConfig, CommonStudyAdapter

ROOT = repo_root()
EXPERIMENT_DIR = ROOT / "experiments" / "common_1d_classifier_study"
CONFIG_PATH = EXPERIMENT_DIR / "common_experiment_config.yaml"
FEATURE_SET_PATH = EXPERIMENT_DIR / "feature_sets.json"
CLASS_PAIR_PATH = EXPERIMENT_DIR / "class_pair_manifest.json"
CLASSIFIER_MANIFEST_PATH = EXPERIMENT_DIR / "classifier_manifest.json"
BOUNDARY_EXPERIMENT_DIR = ROOT / "experiments" / "common_1d_boundary_study"


def load_common_experiment_config(config_path: str | Path | None = None) -> CommonExperimentConfig:
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = CommonExperimentConfigModel.model_validate(payload)
    return CommonExperimentConfig(
        experiment_name=model.experiment.name,
        study_adapter_id=model.experiment.study_adapter_id,
        config_path=path,
        output_dir_name=Path(model.experiment.output_dir).name,
        dataset_generator_id=model.dataset.generator,
        declared_class_pairs=tuple(model.dataset.class_pairs),
        output_filenames=model.outputs.model_dump(),
        feature_sets_path=ROOT / model.feature_sets.manifest_path if model.feature_sets.manifest_path else FEATURE_SET_PATH,
        class_pair_manifest_path=ROOT / model.class_pairs.manifest_path if model.class_pairs.manifest_path else CLASS_PAIR_PATH,
        classifier_manifest_path=ROOT / model.classifiers.manifest_path if model.classifiers.manifest_path else CLASSIFIER_MANIFEST_PATH,
    )


def list_common_studies() -> tuple[CommonStudyAdapter, ...]:
    return (
        CommonStudyAdapter(
            study_id="common_1d_classifier_study",
            description="Manifest-driven 1D common experiment study.",
            pair_spec_builder=_parse_executable_pair_specs,
            trajectory_generator=generate_pair_dataset,
        ),
        CommonStudyAdapter(
            study_id="common_1d_boundary_study",
            description="Boundary-focused 1D common experiment study with harder scenarios only.",
            pair_spec_builder=_parse_executable_pair_specs,
            trajectory_generator=generate_boundary_pair_dataset,
        ),
    )


def resolve_common_study_adapter(study_id: str | CommonExperimentConfig) -> CommonStudyAdapter:
    resolved = study_id.study_adapter_id if isinstance(study_id, CommonExperimentConfig) else study_id
    for adapter in list_common_studies():
        if adapter.study_id == resolved:
            return adapter
    raise KeyError(f"unknown common study: {resolved}")


def _parse_executable_pair_specs(config: CommonExperimentConfig) -> tuple[ExecutablePairSpec, ...]:
    return build_pair_specs(
        declared_class_pairs=config.declared_class_pairs,
        class_pair_manifest_path=config.class_pair_manifest_path,
    )
