from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class NewStudyWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.study_path = cls.root / "experiments" / "new_study_workflow_demo" / "new_study_workflow_demo.yaml"

    def test_templates_and_docs_exist(self) -> None:
        required = (
            self.root / "docs" / "workflows" / "new_study_user_guide.md",
            self.root / "docs" / "workflows" / "new_study_checklist.md",
            self.root / "templates" / "study_candidate.yaml",
            self.root / "templates" / "class_manifest.csv",
            self.root / "templates" / "feature_manifest.csv",
            self.root / "templates" / "prior_manifest.csv",
            self.root / "templates" / "corpus_objective.yaml",
            self.study_path,
        )
        for path in required:
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 0, path)

    def test_end_to_end_workflow_writes_expected_artifacts(self) -> None:
        script_path = self.root / "scripts" / "workflows" / "evaluate_and_package.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                ["python3", str(script_path), "--study", str(self.study_path), "--output-dir", temp_dir],
                check=True,
                cwd=self.root,
                env={
                    **os.environ,
                    "PYTHONPATH": str(self.root / "src"),
                    "PYTHONPYCACHEPREFIX": "/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache",
                },
            )
            workflow_root = Path(temp_dir) / "new_study_workflow_demo"
            expected = (
                workflow_root / "00_study_declaration" / "study_candidate.yaml",
                workflow_root / "01_feature_class_analysis" / "pairwise_auc.csv",
                workflow_root / "01_feature_class_analysis" / "feature_redundancy_matrix.csv",
                workflow_root / "02_corpus_generation" / "candidate_scores.csv",
                workflow_root / "03_corpus_audit" / "class_validity_scores.csv",
                workflow_root / "03_corpus_audit" / "corpus_decision_gate.json",
                workflow_root / "04_ladder_evaluation" / "posterior_history_by_method.csv",
                workflow_root / "04_ladder_evaluation" / "sufficiency_matrix.csv",
                workflow_root / "04b_confidence" / "study_confidence_summary.json",
                workflow_root / "04b_confidence" / "confidence_dashboard.png",
                workflow_root / "05_report" / "study_report.md",
                workflow_root / "05_report" / "decision_card.md",
                workflow_root / "05_report" / "visual_gallery.md",
                workflow_root / "index.md",
            )
            for path in expected:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)


if __name__ == "__main__":
    unittest.main()
