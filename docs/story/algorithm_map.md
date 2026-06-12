# Algorithm Map

The repo keeps two distinct method surfaces on purpose:

1. The proof ladder in [algorithm_ladder.md](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/story/algorithm_ladder.md), which is the disciplined escalation path for methods justified by witnesses.
2. The broader atlas in [docs/methods/algorithm_atlas.md](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/methods/algorithm_atlas.md), which records the wider family map and current claim boundaries.

Use this page as the short orientation layer:

- the proof ladder is for promotion decisions
- the atlas is for method-family coverage
- the backend registry in `src/kinematic_classifier_sandbox/corpus/trajectory_exploration/backend_registry.py` is for generator scaffolding
- the coverage matrix in `src/kinematic_classifier_sandbox/registry/algorithm_coverage_matrix.py` is for the public tracked method families

The sequential-control generator lane is now first-class in the repo story:
PPO, SAC, and TD3 all have real frontier packets, shared comparison baselines,
and explicit claim boundaries rather than roadmap-only mentions.

The rule stays the same: a method only gets promoted when a named witness shows why the simpler rung fails for the problem at hand.
