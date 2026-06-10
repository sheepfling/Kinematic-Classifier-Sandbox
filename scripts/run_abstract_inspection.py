from __future__ import annotations

from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



def main() -> int:
    from kinematic_classifier_sandbox.analysis.inspection_bundle import (
        write_abstract_inspection_artifacts,
    )

    artifacts = write_abstract_inspection_artifacts(
        ROOT / "artifacts",
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
