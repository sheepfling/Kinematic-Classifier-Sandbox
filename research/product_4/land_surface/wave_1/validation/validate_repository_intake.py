from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


EXPECTED_SOURCE_IDS = {
    "amazon_precision_gnss",
    "epfl_pneuma",
    "fhwa_tgsim_foggy_bottom",
    "rwth_highd",
    "thi_revsted",
}

PROHIBITED_FIXTURE_FILES = {
    "uahl_revsted_adma_source_extract.json",
    "uahl_revsted_adma_source_native_samples.json",
    "uahl_revsted_adma_analysis_samples.json",
    "uahl_revsted_adma_tiny_fixture.json",
}

PROHIBITED_DATA_SUFFIXES = {
    ".7z",
    ".arrow",
    ".avro",
    ".bin",
    ".csv",
    ".feather",
    ".h5",
    ".hdf5",
    ".npz",
    ".parquet",
    ".tar",
    ".tgz",
    ".zip",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    ####
####


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
####


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
####


def source_ids(directory: Path) -> set[str]:
    identifiers: list[str] = []
    for path in sorted(directory.glob("*.yaml")):
        payload = load_yaml(path)
        require(isinstance(payload, dict), f"Expected mapping in {path}")
        identifier = payload.get("source_dataset_id")
        require(isinstance(identifier, str), f"Missing source_dataset_id in {path}")
        require(identifier != "", f"Empty source_dataset_id in {path}")
        identifiers.append(identifier)
    ####

    unique_identifiers = set(identifiers)
    require(
        len(unique_identifiers) == len(identifiers),
        f"Duplicate source_dataset_id in {directory}",
    )
    return unique_identifiers
####


def registry_source_ids(entries: list[Any]) -> set[str]:
    require(
        len(entries) == len(EXPECTED_SOURCE_IDS),
        "Registry entry count does not match the expected portfolio",
    )

    identifiers: list[str] = []
    for index, entry in enumerate(entries):
        require(isinstance(entry, dict), f"Registry entry {index} must be a mapping")
        identifier = entry.get("source_dataset_id")
        require(
            isinstance(identifier, str),
            f"Registry entry {index} is missing a string source_dataset_id",
        )
        require(identifier != "", f"Registry entry {index} has an empty source_dataset_id")
        identifiers.append(identifier)
    ####

    unique_identifiers = set(identifiers)
    require(
        len(unique_identifiers) == len(identifiers),
        "Registry contains duplicate source_dataset_id values",
    )
    return unique_identifiers
####


def validate_highd_rights(root: Path) -> None:
    source_card = load_yaml(root / "source_cards/rwth_highd.yaml")
    require(isinstance(source_card, dict), "highD source card must be a mapping")
    access = source_card.get("access")
    require(isinstance(access, dict), "highD access metadata must be a mapping")
    require(
        access.get("redistribution_allowed") is False,
        "highD raw redistribution must remain disabled",
    )
    require(
        access.get("derived_data_redistribution_allowed") is False,
        "highD derived-data redistribution must remain disabled",
    )
####


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    yaml_paths = sorted(root.rglob("*.yaml"))
    json_paths = sorted(root.rglob("*.json"))

    for path in yaml_paths:
        load_yaml(path)
    ####
    for path in json_paths:
        load_json(path)
    ####

    source_card_ids = source_ids(root / "source_cards")
    scorecard_ids = source_ids(root / "scorecards")
    require(source_card_ids == EXPECTED_SOURCE_IDS, "Source-card portfolio mismatch")
    require(scorecard_ids == EXPECTED_SOURCE_IDS, "Scorecard portfolio mismatch")

    registry = load_yaml(root / "registry_updates/source_registry_patch.yaml")
    require(isinstance(registry, dict), "Registry patch must be a mapping")
    entries = registry.get("entries")
    require(isinstance(entries, list), "Registry entries must be a list")
    registry_ids = registry_source_ids(entries)
    require(registry_ids == EXPECTED_SOURCE_IDS, "Registry source portfolio mismatch")

    validate_highd_rights(root)

    status = load_yaml(root / "agent_status.yaml")
    require(isinstance(status, dict), "Agent status must be a mapping")
    require(
        status.get("current_evidence_state") == "access_verified",
        "LAND evidence state must remain access_verified",
    )

    summary = load_json(root / "validation/validation_summary.json")
    require(isinstance(summary, dict), "Validation summary must be a mapping")
    require(
        summary.get("overall_land_evidence_state") == "access_verified",
        "Validation summary overstates LAND evidence",
    )
    require(summary.get("g2_fixture_validated") is False, "Product 4 G2 must remain open")
    require(
        summary.get("authoritative_common_front_validation_run") is False,
        "Authoritative validation must not be claimed",
    )
    require(
        summary.get("repository_fixture_bytes_committed") is False,
        "Restricted fixture bytes must remain absent",
    )

    fixture_directory = root / "fixtures"
    committed_fixture_assets = [
        path
        for path in fixture_directory.iterdir()
        if path.is_file() and path.name != "README.md"
    ]
    require(not committed_fixture_assets, "Fixture directory contains restricted assets")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        ####
        require(path.name not in PROHIBITED_FIXTURE_FILES, f"Prohibited fixture: {path}")
        require(
            path.suffix.lower() not in PROHIBITED_DATA_SUFFIXES,
            f"Raw or binary data container committed: {path}",
        )
        require(path.stat().st_size <= 262_144, f"Research file exceeds 256 KiB: {path}")
    ####

    result = {
        "validator": "land-wave1-repository-intake-v0.2",
        "status": "pass",
        "yaml_files_parsed": len(yaml_paths),
        "json_files_parsed": len(json_paths),
        "source_ids": sorted(source_card_ids),
        "registry_source_count": len(entries),
        "committed_fixture_asset_count": len(committed_fixture_assets),
        "g2_fixture_validated": False,
        "overall_land_evidence_state": "access_verified",
        "checks": [
            "structured files parse",
            "source-card and scorecard IDs are complete and unique",
            "registry entries are mappings with complete unique source IDs",
            "highD raw and derived redistribution remain disabled",
            "evidence state remains access_verified",
            "Product 4 G2 remains open",
            "authoritative validation is not claimed",
            "restricted fixture assets are absent",
            "raw and large binary containers are absent",
        ],
    }
    output = root / "validation/repository_intake_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0
####


if __name__ == "__main__":
    raise SystemExit(main())
####
