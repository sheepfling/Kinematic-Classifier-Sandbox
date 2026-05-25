# PLN-024 Repo Story Coherence And Canonical Navigation

Title: Repo Story Coherence And Canonical Navigation
Plan ID: PLN-024
Status: done
Owner: @rick
Priority: P1
Last Updated: 2026-05-25

Objective:
Reshape the repository presentation so it tells one coherent methodology story: Generic Study Candidate Evaluator plus Corpus Explorer plus Classifier/Filter Ladder plus Evaluation/Promotion Layer plus 1D Witness Suite plus 3D Transition Plan.

Scope:
- Add a front-door story under `docs/story/`.
- Add canonical vocabulary and term aliases.
- Build a claim-to-evidence matrix that links claims to docs, artifacts, tests, limitations, and next work.
- Add witness-problem cards for the controlled 1D studies.
- Consolidate artifact navigation under `artifacts/repo_story/`.
- Refresh the showcase and team packet so they are organized by claims rather than folders.
- Generate and validate the repo-story proof layer from code so the front-door story does not drift from artifacts.

Out of Scope:
- Adding new classifiers, filters, trajectory generators, or optimization loops.
- Claiming the synthetic corpus is final.
- Claiming a single classifier family is globally best.
- Implementing IMM, PF, or RBPF before the current decision gates justify them.
- Reworking generated artifact internals except where needed for story navigation.

Implementation Steps:
1. Establish the canonical repo story.
2. Create the canonical vocabulary and alias rules.
3. Build the claim-to-evidence matrix.
4. Create the repo layer diagram and artifact dependency graph.
5. Write witness-problem cards.
6. Restructure the reader journey and document roles.
7. Add the Study Candidate Evaluator explainer.
8. Add the Corpus Explorer explainer.
9. Add the algorithm ladder page.
10. Add result-interpretation guidance.
11. Consolidate artifact manifests.
12. Refresh showcase and team packet indexes around claims.
13. Generate repo-story matrices, manifest, diagrams, proof gallery, story index, and team-packet front door from one source.
14. Add regression tests that verify claims, artifacts, witness cards, and generated outputs cannot silently drift.

Validation:
- All referenced artifact paths exist.
- Every witness card links to at least one plot and one table.
- Every major claim has at least one artifact, one doc, and one limitation.
- `README.md` links to the canonical read order.
- Older or duplicate docs are labeled as supporting or superseded where appropriate.
- Regression checks still pass.
- The team packet can be read without opening source code.
- `python3 scripts/render_repo_story.py` regenerates the repo-story bundle.
- `python -m kinematic_classifier_sandbox repo-story` regenerates the repo-story bundle through the package CLI.
- `tests/test_repo_story.py` verifies claim, witness, manifest, reference, and generated-output invariants.

Artifacts / Config:
- `docs/story/00_repo_story.md`
- `docs/story/01_methodology_map.md`
- `docs/story/02_reading_order.md`
- `docs/story/glossary.md`
- `docs/story/term_aliases.md`
- `docs/story/claim_evidence_matrix.md`
- `docs/story/artifact_graph.md`
- `docs/story/document_roles.md`
- `docs/story/study_candidate_evaluator.md`
- `docs/story/corpus_explorer.md`
- `docs/story/algorithm_ladder.md`
- `docs/story/how_to_interpret_results.md`
- `docs/witnesses/`
- `artifacts/repo_story/`
- `artifacts/showcase/proof_gallery.md`
- `artifacts/showcase/story_index.md`
- `artifacts/team_packet/index.md`
- `src/kinematic_classifier_sandbox/repo_story.py`
- `scripts/render_repo_story.py`
- `tests/test_repo_story.py`

Dependencies:
- `PLN-016` team-facing methodology showcase.
- `PLN-020` LaTeX documentation coverage audit and expansion.
- `PLN-021` objective-driven corpus explorer v1.
- `PLN-022` corpus explorer hardening.
- `PLN-023` math document hardening and equation traceability.
- Existing artifacts under `artifacts/common_1d_classifier_study/`, `artifacts/feature_analysis_*`, `artifacts/corpus_adequacy_audit_v1/`, `artifacts/generic_inference_contract/`, `artifacts/selected_generated_corpus/`, and `artifacts/showcase/`.

Milestones:
- `M43`: Canonical repo story.
- `M44`: Canonical vocabulary.
- `M45`: Claim-to-evidence matrix.
- `M46`: Repo layer diagram and artifact graph.
- `M47`: Witness problem cards.
- `M48`: Reader journey and document roles.
- `M49`: Study Candidate Evaluator explainer.
- `M50`: Corpus Explorer explainer.
- `M51`: Algorithm ladder page.
- `M52`: Result-interpretation checklist.
- `M53`: Consolidated artifact manifest.
- `M54`: Claim-oriented showcase refresh.
- `M55`: Generated repo-story proof layer.
  - Deliverables:
    - `src/kinematic_classifier_sandbox/repo_story.py`
    - `scripts/render_repo_story.py`
    - package CLI command `repo-story`
    - `tests/test_repo_story.py`
  - Exit criterion:
    - The claim matrix, artifact manifest, artifact graph, witness matrix, diagrams, proof gallery, story index, and team-packet front door are generated from a single canonical source and validated by regression tests.

Success Criteria:
- A team member can read `docs/story/00_repo_story.md` and explain the repo in five minutes.
- A reviewer can trace every headline claim to at least one doc, one artifact, and one limitation.
- The 1D examples are framed as witness problems, not final benchmark claims.
- The front door states that 3D transition is an adapter, feature, and dynamics lift rather than a rewrite of the evaluation stack.
- The repo-story proof layer is generated and validated, not hand-maintained.
