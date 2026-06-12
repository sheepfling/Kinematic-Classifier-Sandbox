from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class StaticAdmissibilityConfig:
    study_id: str = "common_1d_static_admissibility_mvp"
    seed: int = 7
    trajectories_per_class: int = 5
    priors: dict[str, float] | None = None


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


def load_static_admissibility_config(path: str | Path | None) -> StaticAdmissibilityConfig:
    if path is None:
        return StaticAdmissibilityConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    static = raw.get("static_admissibility", raw)
    return StaticAdmissibilityConfig(
        study_id=str(static.get("study_id", "common_1d_static_admissibility_mvp")),
        seed=int(static.get("seed", 7)),
        trajectories_per_class=int(static.get("trajectories_per_class", 5)),
        priors=None if static.get("priors") is None else dict(static["priors"]),
    )

