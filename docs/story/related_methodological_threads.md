# Related Methodological Threads

This repository is not inventing each ingredient from scratch. It integrates a set of established methodological threads into one kinematic-classification workbench.

## Thread Map

| Thread | What we borrow | What we do not claim |
| --- | --- | --- |
| Active learning and experimental design | Search for informative trajectories, uncertainty gaps, and coverage holes rather than generating more data blindly. | A new active-learning algorithm. |
| Dataset documentation and model reporting | Structured corpus manifests, intended-use statements, limitations, and model-quality summaries. | That one report can replace corpus governance. |
| Simulation-based falsification and adaptive stress testing | Search a simulator for failure, fragility, and near-failure witnesses. | That every failure search is globally optimal. |
| Quality-diversity and MAP-Elites | Preserve elites across class, tier, backend, and feature cells instead of one scalar winner. | A mature MAP-Elites implementation on every axis. |
| Calibration and posterior-quality evaluation | Measure NLL, Brier, ECE, entropy, confident-error rate, and prior fragility. | That accuracy alone is enough. |
| Feature separability and oracle diagnostics | Distinguish feature insufficiency from classifier insufficiency and corpus bias. | That AUC or overlap alone determines the final answer. |
| Bayesian filtering and switching state inference | Reuse one posterior/evidence contract across pointwise, windowed, Kalman, IMM, PF, and RBPF families. | That one filter family is universally optimal. |
| Benchmark and test-suite construction | Use witness problems to prove each methodology layer before 3D lift. | That the current witness suite is the final benchmark. |

## Repo Mapping

The integration path is:

1. `feature/class geometry`
2. `corpus objective and search`
3. `adequacy and leakage`
4. `evidence providers`
5. `posterior quality`
6. `promotion decision`

That flow is mirrored in the synthesis paper and in the layer map:

- [Repo story](00_repo_story.md)
- [Methodology map](01_methodology_map.md)
- [Methodology synthesis paper](../latex/kinematic_classifier_methodology.tex)
- [Corpus generation and search](../surveys/corpus_generation_and_search.md)
- [Classifier ladder and contracts](../surveys/classifier_ladder_and_contracts.md)
- [Methodology evaluation framework](../surveys/methodology_evaluation_framework.md)

## Why This Is Worth Saying Explicitly

The repo is best understood as an integration of established research threads into a single methodology workflow for kinematic classification studies. The novel part is the composition:

- corpus exploration is tied to evidence geometry,
- evidence geometry is tied to posterior quality,
- posterior quality is tied to promotion decisions,
- and the 1D witness suite is used to prove each layer before 3D lift.

That integration is the claim the repository is making.
