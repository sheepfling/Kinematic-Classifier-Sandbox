from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adapters.tgsim import TgsimFoggyBottomAdapter
from .adapters.tgsim_contracts import TgsimFoggyBottomAdapterConfig, TgsimLoadResult
from .road_vehicle_study import (
    RoadVehicleStudyConfig,
    RoadVehicleStudyResult,
    build_road_vehicle_study,
)
from .road_vehicle_study_artifact_io import (
    RoadVehicleStudyArtifacts,
    write_road_vehicle_study_artifacts,
)


@dataclass(frozen=True, slots=True)
class TgsimRoadVehicleStudyRun:
    load_result: TgsimLoadResult
    study_result: RoadVehicleStudyResult
    artifacts: RoadVehicleStudyArtifacts
####


def run_tgsim_road_vehicle_study(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    adapter_config: TgsimFoggyBottomAdapterConfig | None = None,
    study_config: RoadVehicleStudyConfig | None = None,
) -> TgsimRoadVehicleStudyRun:
    adapter = TgsimFoggyBottomAdapter(config=adapter_config)
    load_result = adapter.load_file(input_csv)
    study_result = build_road_vehicle_study(
        load_result.tracks,
        config=study_config,
    )
    artifacts = write_road_vehicle_study_artifacts(study_result, output_dir)
    return TgsimRoadVehicleStudyRun(
        load_result=load_result,
        study_result=study_result,
        artifacts=artifacts,
    )
####


__all__ = [
    "TgsimRoadVehicleStudyRun",
    "run_tgsim_road_vehicle_study",
]
