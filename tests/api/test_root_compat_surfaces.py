from __future__ import annotations

import unittest

from kinematic_classifier_sandbox import common_experiment_harness
from kinematic_classifier_sandbox.common_experiment import analysis as grouped_common_experiment_analysis
from kinematic_classifier_sandbox.common_experiment import artifact_io as grouped_common_experiment_artifact_io
from kinematic_classifier_sandbox.common_experiment import config as grouped_common_experiment_config


class RootCompatibilitySurfaceTests(unittest.TestCase):
    def test_common_experiment_harness_is_a_root_compat_shim(self) -> None:
        self.assertIs(common_experiment_harness.load_common_experiment_config, grouped_common_experiment_config.load_common_experiment_config)
        self.assertIs(common_experiment_harness.analyze_common_experiment, grouped_common_experiment_analysis.analyze_common_experiment)
        self.assertIs(common_experiment_harness.write_common_experiment_artifacts, grouped_common_experiment_artifact_io.write_common_experiment_artifacts)


if __name__ == "__main__":
    unittest.main()
