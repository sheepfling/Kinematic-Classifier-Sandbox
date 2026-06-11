# Documentation Front Door

This repo is a methodology workbench for kinematic classification studies. Start with the canonical story docs before diving into surveys or generated artifacts.

## Canonical Reading Order

1. [Repo story](story/00_repo_story.md)
2. [Methodology map](story/01_methodology_map.md)
3. [Algorithm map](story/algorithm_map.md)
4. [Corpus Explorer](story/corpus_explorer.md)
5. [Composite methodology compendium](latex/kinematic_classifier_compendium.tex)
6. [Workflow and math walkthrough](latex/kinematic_classifier_workflow.tex)
7. [Methodology synthesis paper source](latex/kinematic_classifier_methodology.tex)
8. [Classifier ladder math](latex/classifier_ladder_math.tex)
9. [Corpus search math](latex/corpus_search_math.tex)
10. [Classifier ladder and contracts](surveys/classifier_ladder_and_contracts.md)
11. [Corpus generation and search](surveys/corpus_generation_and_search.md)
12. [Methodology evaluation framework](surveys/methodology_evaluation_framework.md)
13. [Witness problem index](witnesses/index.md)
14. [Artifact index](../artifacts/repo_story/artifact_index.md)
15. [Claim evidence matrix](story/claim_evidence_matrix.md)

## Core Terms

- Study Candidate Evaluator: evaluates `s = (D, f, C, m, pi, b)`.
- Static Feature/Class/Prior Audit: checks whether `(f, C, pi)` is separable, informative, prior-robust, leak-free, and decisionable before corpus generation or classifier escalation.
- Corpus Explorer: generates, searches, validates, scores, and selects corpora.
- Algorithm Map: tracks promoted ladder rungs separately from benchmark, neural, calibration, and roadmap lanes.
- Classifier/Filter Ladder: builds comparable evidence providers.
- Evaluation/Promotion Layer: assigns promote, revise, reject, or defer.
- 1D Witness Suite: controlled proofs of methodology layers before 3D transition.
