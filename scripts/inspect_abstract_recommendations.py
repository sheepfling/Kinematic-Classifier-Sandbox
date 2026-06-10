from __future__ import annotations

import json
import sys

from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



def main() -> int:
    from kinematic_classifier_sandbox.analysis.inspection_bundle import (
        recommend_feature_set,
        recommend_hardest_class_pair,
    )

    summary_path = ROOT / "artifacts" / "abstract_inspection_v1" / "abstract_inspection_summary.json"
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
