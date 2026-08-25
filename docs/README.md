# Documentation Front Door

This repo is a methodology workbench for kinematic classification studies. Start with the canonical story docs before diving into surveys or generated artifacts.

For the user-facing product map, start with [Four Products](story/three_products.md).
For the product-specific test boundaries and release gates, use the [Product Test Matrix](testing/TEST_MATRIX.md).
For the quick-turn feature/class/prior audit, use the [Static Admissibility Toolkit](story/static_admissibility_toolkit.md).

## Canonical Reading Order

Use the canonical reading order in [docs/story/02_reading_order.md](story/02_reading_order.md). This page stays as the brief front door and term glossary.

## Operational Workflows

- [Epic 1 showcase regeneration](workflows/epic1_showcase_regeneration.md): one command to rebuild the workbench evidence set, governed corpus-search lane, static-admissibility packets, and presentation export.
- [New study user guide](workflows/new_study_user_guide.md): how to create and run study configs.
- [Static audit bundle user guide](workflows/static_audit_bundle_user_guide.md): how to ingest feature/class/prior bundles.

## Core Terms

- Four products: Static Admissibility, the Classifier Evidence Ladder, RL Corpus Exploration, and Real-World Corpus & Validation.
- Study Candidate Evaluator: evaluates `s = (D, f, C, m, pi, b)`.
- Static Feature/Class/Prior Audit: checks whether `(f, C, pi)` is separable, informative, prior-robust, leak-free, and decisionable before corpus generation or classifier escalation.
- Corpus Explorer: generates, searches, validates, scores, and selects corpora.
- Algorithm Map: tracks promoted ladder rungs separately from benchmark, neural, calibration, and roadmap lanes.
- Classifier/Filter Ladder: builds comparable evidence providers.
- Evaluation/Promotion Layer: assigns promote, revise, reject, or defer.
- 1D Witness Suite: controlled proofs of methodology layers before 3D transition.
