from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.utils.analysis_cache import (
    analysis_cache_namespace_root,
    clear_analysis_cache,
    describe_analysis_cache_stats,
    describe_analysis_cache,
    load_or_compute_pickled,
    reset_analysis_cache_stats,
    stable_cache_key,
)
from kinematic_classifier_sandbox.utils.runtime import runtime_root


class AnalysisCacheTests(unittest.TestCase):
    def test_describe_and_clear_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("KINEMATIC_CLASSIFIER_RUNTIME_ROOT")
            os.environ["KINEMATIC_CLASSIFIER_RUNTIME_ROOT"] = temp_dir
            try:
                key = stable_cache_key("demo_cache", {"value": 1})
                reset_analysis_cache_stats()
                self.assertEqual(
                    load_or_compute_pickled(
                        namespace="demo_cache",
                        cache_key=key,
                        compute=lambda: {"value": 1},
                        metadata={"value": 1},
                    ),
                    {"value": 1},
                )
                self.assertEqual(
                    load_or_compute_pickled(
                        namespace="demo_cache",
                        cache_key=key,
                        compute=lambda: {"value": 999},
                    ),
                    {"value": 1},
                )
                stats = describe_analysis_cache_stats(namespace="demo_cache")
                self.assertEqual(stats["miss"], 1)
                self.assertEqual(stats["hit"], 1)
                summary = describe_analysis_cache(namespace="demo_cache")
                self.assertEqual(summary["namespace_count"], 1)
                self.assertEqual(summary["entry_count"], 1)
                cleared = clear_analysis_cache(namespace="demo_cache")
                self.assertEqual(cleared["cleared_namespace"], "demo_cache")
                self.assertEqual(cleared["cleared_entry_count"], 1)
                self.assertFalse((runtime_root() / "analysis_cache" / "demo_cache").exists())
            finally:
                if previous is None:
                    os.environ.pop("KINEMATIC_CLASSIFIER_RUNTIME_ROOT", None)
                else:
                    os.environ["KINEMATIC_CLASSIFIER_RUNTIME_ROOT"] = previous

    def test_corrupt_pickle_is_removed_and_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("KINEMATIC_CLASSIFIER_RUNTIME_ROOT")
            os.environ["KINEMATIC_CLASSIFIER_RUNTIME_ROOT"] = temp_dir
            try:
                namespace = "demo_cache"
                key = stable_cache_key(namespace, {"value": 2})
                cache_dir = analysis_cache_namespace_root(namespace)
                pickle_path = cache_dir / f"{key}.pkl"
                metadata_path = cache_dir / f"{key}.json"
                pickle_path.write_bytes(b"not a pickle")
                metadata_path.write_text("{}", encoding="utf-8")
                result = load_or_compute_pickled(
                    namespace=namespace,
                    cache_key=key,
                    compute=lambda: {"value": 2},
                )
                self.assertEqual(result, {"value": 2})
                self.assertNotEqual(pickle_path.read_bytes(), b"not a pickle")
            finally:
                if previous is None:
                    os.environ.pop("KINEMATIC_CLASSIFIER_RUNTIME_ROOT", None)
                else:
                    os.environ["KINEMATIC_CLASSIFIER_RUNTIME_ROOT"] = previous

    def test_invalid_namespace_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            stable_cache_key("../bad", {"value": 1})


if __name__ == "__main__":
    unittest.main()
