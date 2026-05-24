from __future__ import annotations

import json
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

    from kinematic_classifier_sandbox import recommend_feature_set, recommend_hardest_class_pair

    summary_path = root / "artifacts" / "abstract_inspection_v1" / "abstract_inspection_summary.json"
    if not summary_path.exists():
        print("missing abstract inspection summary; run scripts/run_abstract_inspection.py first", file=sys.stderr)
        return 1

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    feature_set = recommend_feature_set(payload)
    class_pair = recommend_hardest_class_pair(payload)

    print(f"summary: {summary_path}")
    print(
        "recommended feature set: "
        f"{feature_set['feature_set']} "
        f"(status={feature_set['feature_set_status']}, "
        f"avg_pairwise_auc={float(feature_set['avg_pairwise_auc']):.3f}, "
        f"avg_overlap={float(feature_set['avg_overlap']):.3f})"
    )
    print(f"top features: {feature_set['top_features']}")
    print(
        "hardest class pair: "
        f"{class_pair['class_pair']} "
        f"(pairwise_auc={float(class_pair['pairwise_auc']):.3f}, "
        f"overlap={float(class_pair['overlap_estimate']):.3f}, "
        f"mahalanobis={float(class_pair['mahalanobis_distance']):.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
