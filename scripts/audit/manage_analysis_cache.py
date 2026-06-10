#!/usr/bin/env python3
from __future__ import annotations

import sys

from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)

from kinematic_classifier_sandbox.__main__ import main as package_main


def main(argv: list[str] | None = None) -> int:
    return package_main(["analysis-cache", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
