"""Run Product 4 test groups in isolated worker processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PRODUCT4_GROUPS = (
    "product4_common",
    "product4_cross_domain",
    "product4_source_audit",
    "product4_kinematic_analysis",
    "product4_classifier_ladder",
    "product4_land_surface",
    "product4_sea_surface",
    "product4_sea_subsurface",
    "product4_air_atmospheric",
    "product4_space_near",
    "product4_space_orbital",
)


def _run_group(arguments: tuple[str, Path, tuple[str, ...]]) -> tuple[str, int, str]:
    marker, root, extra_args = arguments
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-m",
        marker,
        *extra_args,
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return marker, completed.returncode, completed.stdout + completed.stderr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Product 4 common, cross-domain, and lane tests in parallel."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(len(PRODUCT4_GROUPS), max(1, os.cpu_count() or 1)),
        help="Number of isolated pytest worker processes.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to every worker pytest invocation.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    if arguments.workers < 1:
        raise SystemExit("--workers must be at least 1")
    root = Path(__file__).resolve().parents[2]
    jobs = tuple((marker, root, tuple(arguments.pytest_args)) for marker in PRODUCT4_GROUPS)
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        results = tuple(executor.map(_run_group, jobs))

    failed = False
    for marker, returncode, output in results:
        print(f"=== {marker} ===")
        print(output, end="" if output.endswith("\n") else "\n")
        if returncode != 0:
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
