from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from kinematic_classifier_sandbox.corpus.policy import load_corpus_policy_spec
from kinematic_classifier_sandbox.methodology.cached_analysis import (
    cached_common_experiment_analysis,
    common_experiment_cache_key,
    study_candidate_generation_cache_key,
)
from kinematic_classifier_sandbox.methodology.context import (
    build_methodology_execution_context,
)


class CachedAnalysisTests(unittest.TestCase):
    def test_common_experiment_cache_reuses_pickled_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("KINEMATIC_CLASSIFIER_RUNTIME_ROOT")
            os.environ["KINEMATIC_CLASSIFIER_RUNTIME_ROOT"] = temp_dir
            try:
                first = cached_common_experiment_analysis(seed=7, trajectories_per_case=2, use_cache=True)
                with patch(
                    "kinematic_classifier_sandbox.methodology.cached_analysis.analyze_common_experiment",
                    side_effect=AssertionError("cache should satisfy second call"),
                ):
                    second = cached_common_experiment_analysis(seed=7, trajectories_per_case=2, use_cache=True)
            finally:
                if previous is None:
                    os.environ.pop("KINEMATIC_CLASSIFIER_RUNTIME_ROOT", None)
                else:
                    os.environ["KINEMATIC_CLASSIFIER_RUNTIME_ROOT"] = previous
        self.assertEqual(first.summary, second.summary)

    def test_cache_keys_change_with_seed_and_config(self) -> None:
        key_a = common_experiment_cache_key(seed=7, trajectories_per_case=2)
        key_b = common_experiment_cache_key(seed=8, trajectories_per_case=2)
        self.assertNotEqual(key_a, key_b)

        from kinematic_classifier_sandbox.common_experiment.config import CONFIG_PATH

        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir) / "common_experiment_config_copy.yaml"
            copied.write_text(CONFIG_PATH.read_text(encoding="utf-8") + "\n# cache key test\n", encoding="utf-8")
            key_c = common_experiment_cache_key(config_path=copied, seed=7, trajectories_per_case=2)
        self.assertNotEqual(key_a, key_c)

    def test_study_generation_cache_key_changes_with_policy(self) -> None:
        baseline = load_corpus_policy_spec()
        modified = replace(baseline, policy_id=baseline.policy_id + "_alt")
        self.assertNotEqual(
            study_candidate_generation_cache_key(seed=7, trajectories_per_case=2, policy=baseline),
            study_candidate_generation_cache_key(seed=7, trajectories_per_case=2, policy=modified),
        )

    def test_methodology_execution_context_reuses_process_local_cache(self) -> None:
        first = build_methodology_execution_context(seed=7, trajectories_per_case=2, use_cache=True)
        with patch(
            "kinematic_classifier_sandbox.methodology.context.cached_common_experiment_analysis",
            side_effect=AssertionError("process-local cache should satisfy second call"),
        ):
            second = build_methodology_execution_context(seed=7, trajectories_per_case=2, use_cache=True)
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
