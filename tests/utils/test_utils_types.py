from __future__ import annotations

import unittest

from kinematic_classifier_sandbox.utils.types import (
    FloatArray,
    IntArray,
    LogLikelihoodFn,
    TransitionFn,
)


class UtilsTypesTests(unittest.TestCase):
    def test_shared_aliases_are_exported(self) -> None:
        self.assertIsNotNone(FloatArray)
        self.assertIsNotNone(IntArray)
        self.assertIsNotNone(TransitionFn)
        self.assertIsNotNone(LogLikelihoodFn)


if __name__ == "__main__":
    unittest.main()
