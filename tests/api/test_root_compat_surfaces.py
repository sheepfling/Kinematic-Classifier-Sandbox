from __future__ import annotations

import unittest

from kinematic_classifier_sandbox import common_dataset_comparison
from kinematic_classifier_sandbox import dimensional_lift_audit
from kinematic_classifier_sandbox import feature_analysis
from kinematic_classifier_sandbox import inspection_bundle
from kinematic_classifier_sandbox import pca_analysis
from kinematic_classifier_sandbox import technique_comparison
from kinematic_classifier_sandbox.analysis import common_dataset_comparison as grouped_common_dataset_comparison
from kinematic_classifier_sandbox.analysis import dimensional_lift_audit as grouped_dimensional_lift_audit
from kinematic_classifier_sandbox.analysis import feature_analysis as grouped_feature_analysis
from kinematic_classifier_sandbox.analysis import inspection_bundle as grouped_inspection_bundle
from kinematic_classifier_sandbox.analysis import pca_analysis as grouped_pca_analysis
from kinematic_classifier_sandbox.validation import technique_comparison as grouped_technique_comparison


class RootCompatibilitySurfaceTests(unittest.TestCase):
    def test_root_modules_forward_to_grouped_implementations(self) -> None:
        self.assertIs(feature_analysis.analyze_feature_datasets, grouped_feature_analysis.analyze_feature_datasets)
        self.assertIs(feature_analysis.write_feature_analysis_artifacts, grouped_feature_analysis.write_feature_analysis_artifacts)
        self.assertIs(pca_analysis.analyze_feature_pca, grouped_pca_analysis.analyze_feature_pca)
        self.assertIs(dimensional_lift_audit.analyze_dimensional_lift_audit, grouped_dimensional_lift_audit.analyze_dimensional_lift_audit)
        self.assertIs(common_dataset_comparison.analyze_common_dataset_comparison, grouped_common_dataset_comparison.analyze_common_dataset_comparison)
        self.assertIs(technique_comparison.analyze_technique_comparison, grouped_technique_comparison.analyze_technique_comparison)
        self.assertIs(inspection_bundle.recommend_feature_set, grouped_inspection_bundle.recommend_feature_set)


if __name__ == "__main__":
    unittest.main()
