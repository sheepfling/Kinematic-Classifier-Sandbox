from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        str(src) if not existing_pythonpath else f"{src}{os.pathsep}{existing_pythonpath}"
    )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from kinematic_classifier_sandbox.analysis.inspection_bundle import (
        write_abstract_inspection_artifacts,
    )

    artifacts = write_abstract_inspection_artifacts(
        root / "artifacts",
        seed=7,
        trajectories_per_class=5,
        n_components=3,
    )
    print(artifacts.run_dir)
    print(artifacts.index_path)
    print(artifacts.machine_summary_path)
    print(artifacts.summary_table_path)
    print(artifacts.summary_chart_path)
    print(artifacts.class_pair_summary_table_path)
    print(artifacts.class_pair_summary_chart_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
