from __future__ import annotations

from datetime import date

from .contracts import DatasetManifest, SourceAsset


TGSIM_FOGGY_BOTTOM_ADAPTER_ID = "tgsim_foggy_bottom_csv"
TGSIM_FOGGY_BOTTOM_ADAPTER_VERSION = "0.1.0"
TGSIM_FOGGY_BOTTOM_DATASET_ID = "tgsim_foggy_bottom"
TGSIM_FOGGY_BOTTOM_COORDINATE_FRAME = "tgsim_foggy_bottom_reference_image_xy_m"


def build_tgsim_foggy_bottom_manifest(*, accessed_on: date) -> DatasetManifest:
    access_date = accessed_on.isoformat()
    return DatasetManifest(
        dataset_id=TGSIM_FOGGY_BOTTOM_DATASET_ID,
        title="Third Generation Simulation Data (TGSIM) Foggy Bottom Trajectories",
        version="catalog-2026-01-20",
        publisher="U.S. Department of Transportation Federal Highway Administration",
        citation=(
            "U.S. Department of Transportation Federal Highway Administration. (2024). "
            "Third Generation Simulation Data (TGSIM) Foggy Bottom Trajectories. "
            "[Dataset]. Provided by ITS DataHub through Data.transportation.gov. "
            f"Accessed {access_date}. DOI: 10.21949/1404230."
        ),
        doi="10.21949/1404230",
        license_id="us-public-domain",
        license_url="http://www.usa.gov/publicdomain/label/1.0/",
        landing_page_url="https://data.transportation.gov/d/brzy-6zfh",
        accessed_on=accessed_on,
        adapter_id=TGSIM_FOGGY_BOTTOM_ADAPTER_ID,
        adapter_version=TGSIM_FOGGY_BOTTOM_ADAPTER_VERSION,
        coordinate_frame=TGSIM_FOGGY_BOTTOM_COORDINATE_FRAME,
        nominal_sample_interval_s=0.1,
        source_assets=(
            SourceAsset(
                asset_id="trajectory_csv",
                title="TGSIM-Foggy Bottom-Data.csv",
                download_url=(
                    "https://data.transportation.gov/api/v3/views/brzy-6zfh/"
                    "export.csv?accessType=DOWNLOAD"
                ),
                media_type="text/csv",
            ),
        ),
        notes=(
            "Raw trajectory data are not redistributed by this repository.",
            "The source data dictionary reports 0.1-second samples and separate x/y "
            "velocity and acceleration components.",
            "Width, length, and lane/region identifiers are audit-only covariates for "
            "the initial kinematics-only classifier study.",
        ),
    )
####


__all__ = [
    "TGSIM_FOGGY_BOTTOM_ADAPTER_ID",
    "TGSIM_FOGGY_BOTTOM_ADAPTER_VERSION",
    "TGSIM_FOGGY_BOTTOM_COORDINATE_FRAME",
    "TGSIM_FOGGY_BOTTOM_DATASET_ID",
    "build_tgsim_foggy_bottom_manifest",
]
