from __future__ import annotations

import unittest

from kinematic_classifier_sandbox import corpus
from kinematic_classifier_sandbox.corpus.adequacy_artifact_io import write_corpus_adequacy_artifacts
from kinematic_classifier_sandbox.corpus.adequacy_audit import analyze_corpus_adequacy
from kinematic_classifier_sandbox.corpus.coverage_artifact_io import write_coverage_report_artifacts
from kinematic_classifier_sandbox.corpus.coverage_report import analyze_coverage_report
from kinematic_classifier_sandbox.corpus.selected_generated_corpus import analyze_selected_generated_corpus
from kinematic_classifier_sandbox.corpus.selected_generated_corpus_artifact_io import (
    write_selected_generated_corpus_artifacts,
)


class CorpusPackageSurfaceTests(unittest.TestCase):
    def test_package_initializer_is_intentionally_minimal(self) -> None:
        self.assertEqual(corpus.__all__, [])
        self.assertFalse(hasattr(corpus, "analyze_corpus_adequacy"))
        self.assertFalse(hasattr(corpus, "write_coverage_report_artifacts"))

    def test_core_entrypoints_are_available_from_concrete_modules(self) -> None:
        self.assertTrue(callable(analyze_corpus_adequacy))
        self.assertTrue(callable(write_corpus_adequacy_artifacts))
        self.assertTrue(callable(analyze_coverage_report))
        self.assertTrue(callable(write_coverage_report_artifacts))
        self.assertTrue(callable(analyze_selected_generated_corpus))
        self.assertTrue(callable(write_selected_generated_corpus_artifacts))


if __name__ == "__main__":
    unittest.main()
