from __future__ import annotations

import unittest

from kinematic_classifier_sandbox.utils.categorical import bucket2, bucket_thresholds, status_score


class UtilsCategoricalTests(unittest.TestCase):
    def test_status_score_defaults_and_overrides(self) -> None:
        self.assertEqual(status_score("green"), 1.0)
        self.assertEqual(status_score("yellow"), 0.5)
        self.assertEqual(status_score("red"), 0.0)
        self.assertEqual(status_score("yellow", yellow=0.6, red=0.2), 0.6)
        self.assertEqual(status_score("missing", default=-1.0), -1.0)

    def test_bucket_helpers(self) -> None:
        self.assertEqual(bucket2(1.0, 2.0, 4.0), "low")
        self.assertEqual(bucket2(3.0, 2.0, 4.0), "medium")
        self.assertEqual(bucket2(5.0, 2.0, 4.0), "high")
        self.assertEqual(bucket_thresholds(0.7, (0.2, 0.8), ("small", "mid", "large")), "mid")


if __name__ == "__main__":
    unittest.main()
