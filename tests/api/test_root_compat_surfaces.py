from __future__ import annotations

import unittest

from kinematic_classifier_sandbox import (
    common_experiment_harness,
    contracts,
    milestone_runs,
    milestones,
    repo_story,
)
from kinematic_classifier_sandbox.common_experiment import (
    analysis as grouped_common_experiment_analysis,
)
from kinematic_classifier_sandbox.common_experiment import (
    artifact_io as grouped_common_experiment_artifact_io,
)
from kinematic_classifier_sandbox.common_experiment import (
    config as grouped_common_experiment_config,
)
from kinematic_classifier_sandbox.schema import artifacts as schema_artifacts
from kinematic_classifier_sandbox.schema import milestone0 as schema_milestone0
from kinematic_classifier_sandbox.story import repo_story as grouped_repo_story


class RootCompatibilitySurfaceTests(unittest.TestCase):
    def test_common_experiment_harness_is_a_root_compat_shim(self) -> None:
        self.assertIs(common_experiment_harness.load_common_experiment_config, grouped_common_experiment_config.load_common_experiment_config)
        self.assertIs(common_experiment_harness.analyze_common_experiment, grouped_common_experiment_analysis.analyze_common_experiment)
        self.assertIs(common_experiment_harness.write_common_experiment_artifacts, grouped_common_experiment_artifact_io.write_common_experiment_artifacts)

    def test_contracts_is_a_root_compat_shim(self) -> None:
        self.assertIs(contracts.TrajectoryArtifact, schema_artifacts.TrajectoryArtifact)
        self.assertIs(contracts.validate_classifier_output_artifact, schema_artifacts.validate_classifier_output_artifact)
        self.assertIs(contracts.Milestone0SampleArtifacts, schema_milestone0.Milestone0SampleArtifacts)
        self.assertIs(contracts.write_milestone0_sample_run_artifacts, schema_milestone0.write_milestone0_sample_run_artifacts)

    def test_repo_story_is_a_root_compat_shim(self) -> None:
        self.assertIs(repo_story.RepoStoryArtifacts, grouped_repo_story.RepoStoryArtifacts)
        self.assertIs(repo_story.render_repo_story_index, grouped_repo_story.render_repo_story_index)
        self.assertIs(repo_story.write_repo_story_artifacts, grouped_repo_story.write_repo_story_artifacts)
        self.assertEqual(repo_story.WITNESSES, grouped_repo_story.WITNESSES)

    def test_milestones_is_a_root_compat_shim(self) -> None:
        self.assertIs(milestones.MilestoneEntry, milestone_runs.MilestoneEntry)
        self.assertIs(milestones.MilestoneRunResult, milestone_runs.MilestoneRunResult)
        self.assertIs(milestones.list_milestones, milestone_runs.list_milestones)
        self.assertIs(milestones.resolve_milestone_ids, milestone_runs.resolve_milestone_ids)
        self.assertIs(milestones.run_milestones, milestone_runs.run_milestones)


if __name__ == "__main__":
    unittest.main()
