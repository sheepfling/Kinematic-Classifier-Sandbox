from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


class WorkbenchMvpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.study = cls.root / "experiments" / "common_1d_classifier_study" / "common_experiment_config.yaml"

    def _env(self) -> dict[str, str]:
        return {
            **os.environ,
            "PYTHONPATH": str(self.root / "src"),
            "PYTHONPYCACHEPREFIX": "/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache",
        }

    def test_claim_and_visualization_registries_exist(self) -> None:
        for path in (
            self.root / "docs" / "story" / "claim_registry.yaml",
            self.root / "docs" / "story" / "visualization_registry.yaml",
            self.root / "docs" / "story" / "epics" / "01_epic1_exit_criteria.md",
            self.root / "docs" / "workflows" / "epic1_showcase_regeneration.md",
        ):
            self.assertTrue(path.exists(), path)
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix == ".yaml" else None
            if payload is not None:
                self.assertIn("registry_id", payload)

    def test_templates_exist(self) -> None:
        for name in (
            "basic_classifier_study.yaml",
            "corpus_search_study.yaml",
            "advanced_filter_witness.yaml",
            "private_work_study.yaml",
            "presentation_export.yaml",
        ):
            path = self.root / "experiments" / "templates" / name
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 0, path)

    def test_validate_study_cli(self) -> None:
        result = subprocess.run(
            [
                "python3",
                "-m",
                "kinematic_classifier_sandbox",
                "validate-study",
                str(self.study),
            ],
            cwd=self.root,
            env=self._env(),
            text=True,
            check=True,
            capture_output=True,
        )
        self.assertIn("PASS:", result.stdout)

    def test_run_analyze_export_validate_workbench_cli(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp_dir:
            run_dir = Path(temp_dir) / "interview_demo"
            subprocess.run(
                [
                    "python3",
                    "-m",
                    "kinematic_classifier_sandbox",
                    "run-study",
                    str(self.study),
                    "--output-dir",
                    str(run_dir),
                    "--trajectories-per-case",
                    "2",
                ],
                cwd=self.root,
                env=self._env(),
                check=True,
            )
            for filename in (
                "study_spec.yaml",
                "study_run_manifest.json",
                "evidence_contract.json",
                "posterior_history.csv",
                "metrics_by_method.csv",
                "decision_card.md",
                "decision_card.json",
                "workbench_report.md",
            ):
                self.assertTrue((run_dir / filename).exists(), filename)

            subprocess.run(
                [
                    "python3",
                    "-m",
                    "kinematic_classifier_sandbox",
                    "analyze-run",
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=self.root,
                env=self._env(),
                check=True,
            )
            packet_dir = Path(temp_dir) / "workbench_packet"
            subprocess.run(
                [
                    "python3",
                    "-m",
                    "kinematic_classifier_sandbox",
                    "export-packet",
                    "--profile",
                    "workbench",
                    "--run-dir",
                    str(run_dir),
                    "--output-dir",
                    str(packet_dir),
                ],
                cwd=self.root,
                env=self._env(),
                check=True,
            )
            subprocess.run(
                [
                    "python3",
                    "-m",
                    "kinematic_classifier_sandbox",
                    "validate-packet",
                    "--profile",
                    "workbench",
                    "--packet-dir",
                    str(packet_dir),
                ],
                cwd=self.root,
                env=self._env(),
                check=True,
            )

    def test_search_corpus_cli_writes_governed_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp_dir:
            output_dir = Path(temp_dir) / "corpus_search"
            subprocess.run(
                [
                    "python3",
                    "-m",
                    "kinematic_classifier_sandbox",
                    "search-corpus",
                    "experiments/templates/corpus_search_study.yaml",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=self.root,
                env=self._env(),
                check=True,
            )
            for filename in (
                "corpus_search_manifest.json",
                "backend_comparison.csv",
                "downstream_diagnostic_yield.csv",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)

    def test_build_epic1_showcase_smoke_cli(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp_dir:
            output_dir = Path(temp_dir) / "epic1_showcase"
            subprocess.run(
                [
                    "python3",
                    "-m",
                    "kinematic_classifier_sandbox",
                    "build-epic1-showcase",
                    "--output-dir",
                    str(output_dir),
                    "--skip-static",
                    "--skip-presentation",
                    "--trajectories-per-case",
                    "2",
                ],
                cwd=self.root,
                env=self._env(),
                check=True,
            )
            for filename in (
                "README.md",
                "regeneration_summary.md",
                "epic1_showcase_manifest.json",
                "validation_summary.csv",
                "artifact_index.csv",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            for subdir in ("workbench_run", "workbench_packet", "corpus_search"):
                self.assertTrue((output_dir / subdir).is_dir(), subdir)
            summary = (output_dir / "regeneration_summary.md").read_text(encoding="utf-8")
            self.assertIn("validate_workbench_run: `pass`", summary)
            self.assertIn("validate_presentation_packet: `skipped`", summary)


if __name__ == "__main__":
    unittest.main()
