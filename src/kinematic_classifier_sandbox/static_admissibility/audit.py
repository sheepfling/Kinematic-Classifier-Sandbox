from __future__ import annotations

from pathlib import Path

from .io import write_static_admissibility_packet
from .schemas import StaticAdmissibilityPacket, load_static_admissibility_config


def run_static_admissibility_audit(
    config_path: str | Path | None,
    output_dir: str | Path,
) -> StaticAdmissibilityPacket:
    config = load_static_admissibility_config(config_path)
    return write_static_admissibility_packet(output_dir, config=config)

