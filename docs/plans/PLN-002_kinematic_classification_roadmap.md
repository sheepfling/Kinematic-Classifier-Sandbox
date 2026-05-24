# PLN-002 Kinematic Classification Roadmap

Title: Kinematic Classification Lab Roadmap
Plan ID: PLN-002
Status: in_progress
Owner: @rick
Priority: P1
Objective: Turn the sandbox into a generic kinematic-classification and filtering methodology framework whose corpus, feature, evidence, posterior, filtering, analysis, and visualization layers remain reusable as the repo moves from 1D studies toward 3D studies.
Scope:
- Finish the current common experiment harness and feature/class-pair comparison milestones.
- Prove that classifiers and filters can be expressed through generic evidence and posterior contracts.
- Prove that feature metadata and feature sets are transferable abstractions rather than one-off 1D feature hacks.
- Prove that filtering backends can emit standardized state, evidence, and diagnostics artifacts.
- Audit scalar assumptions explicitly and define what is required before 3D work is credible.
- Keep IMM, particle filtering, and Rao-Blackwell particle filtering behind explicit decision gates.
Out of Scope:
- Full production 3D tracking and classification.
- Premature addition of IMM, particle filtering, or RBPF before the generic contracts exist.
- Production sensor fusion, geodesy, multi-target association, or operational deployment.
- Learned end-to-end models as the primary backlog focus.

Current State Assessment:
- The repo has already proven that it can build and test 1D classifiers, feature analyzers, corpus generators, Kalman filters, Monte Carlo packs, PCA reports, and rendered artifacts.
- The repo has not yet proven that those pieces are generic enough that moving to 3D mostly means new state representations, new feature extractors, new corpus data, and new dynamics/measurement models while reusing the same experiment, evidence, posterior, diagnostics, and reporting machinery.
- `M10` and `M11` are now best understood as the front of a broader generic-methodology proof phase, not just as more harness work.
- The core architectural question is now:
  - can state dimension, feature set, class set, and filter backend be swapped without rewriting the evaluation machinery?
- If the answer becomes yes, the repo is 3D-ready in methodology terms.
- If the answer remains no, the repo is still primarily a strong 1D demo and diagnostics lab.

Guiding Architecture:
- `TrajectoryCorpus`
  - observations, truth, metadata, class labels, scenario labels
- `FeatureExtractor`
  - feature table with provenance and feature taxonomy metadata
- `EvidenceProvider`
  - log likelihood by class from features, histories, innovations, or residuals
- `PosteriorUpdater`
  - posterior over classes through time
- `FilterBackend`
  - state summaries, covariances or particles, innovations, model likelihoods, diagnostics
- `ExperimentHarness`
  - runs methods on corpora, feature sets, class pairs, and priors
- `AnalysisLayer`
  - confusion, identifiability, PCA, leakage, calibration, prior sensitivity, coverage
- `VisualizationLayer`
  - plots and markdown reports from standardized artifacts

Key Design Principle:
- features and filters produce evidence
- posterior machinery consumes evidence
- analysis and visualization consume standardized artifacts

Implementation Steps:
1. Finish `M10` and `M11` as the entry to the generic methodology proof phase.
   - Complete the config-driven common experiment harness.
   - Keep feature sets and class pairs as first-class study dimensions.
   - Ensure these surfaces are stable enough to support the next contract layers.
2. Add the generic inference contract.
   - Define common schemas for classifier output, evidence output, posterior history, and filter output.
   - Require pointwise, windowed, Bayesian accumulator, and Kalman bank to emit the same logical artifact family.
3. Add the generic feature taxonomy.
   - Make every feature declare history behavior, sensitivity, geometry assumptions, and dimensional transferability.
   - Treat feature sets as tagged, queryable bundles rather than only named JSON lists.
4. Add the generic classification and evidence proof.
   - Express pointwise, windowed, empirical-feature, residual, and Kalman-innovation methods as `EvidenceProvider` variants.
   - Keep the posterior updater agnostic to the evidence source.
5. Add the generic filtering contract.
   - Standardize what a filter backend must emit for state summaries, evidence summaries, and diagnostics.
   - Keep Kalman as the first reference backend, and define where PF and RBPF would fit later.
6. Add the dimensional lift audit.
   - Identify scalar assumptions module by module.
   - Label code as dimension-agnostic, 1D-specific but adapter-compatible, or 1D-specific and requiring rewrite.
   - Run a fake vector-valued corpus through the generic harness far enough to emit standard artifacts.
7. Add the corpus adequacy and coverage framework.
   - Make corpus quality measurable independently of classifier choice.
   - Audit class balance, scenario balance, feature excitation, class-pair boundary coverage, covariate leakage, and sensor regime coverage.
8. Add switching and transition-model work only after the generic contracts exist.
   - Add class transitions and switching scenarios as a methodology exercise rather than a one-off algorithm addition.
9. Add advanced-filter decision reports before any advanced backend implementation.
   - Document what latent structure would justify IMM, PF, or RBPF.
   - Add particle or Rao-Blackwell particle filtering only when a failure case and metric justify them.

Validation:
- All implemented classifiers emit the same posterior and prediction schema.
- Posterior probabilities sum to one for every classifier and trajectory.
- Pointwise, windowed, Bayesian accumulator, and Kalman bank can be compared through the same metrics code.
- Two evidence providers with identical log-likelihood streams produce identical posterior histories.
- Every feature declares history behavior, sensitivity, and dimensional transfer status.
- Every cumulative feature is labeled cumulative.
- The dimensional lift audit marks each module with an explicit dimensional status.
- A fake vector-valued corpus can pass through the generic harness far enough to produce standard artifacts.
- PF and RBPF decision reports document:
  - what would be sampled
  - what would be marginalized analytically
  - what simpler method fails
  - what metric would justify implementation

Artifacts / Config:
- `artifacts/generic_inference_contract/`
- `artifacts/feature_taxonomy/`
- `artifacts/classification_evidence_proof/`
- `artifacts/filtering_contract/`
- `artifacts/dimensional_lift_audit/`
- `artifacts/corpus_adequacy_audit_v1/`
- `experiments/*.yaml`
- `feature_sets.json`, `class_pair_manifest.json`, `classifier_manifest.json`
- standardized predictions, posterior histories, likelihood histories, feature matrices, and run reports

Dependencies:
- Current `M10` common harness work in `PLN-007`
- Current `M11` feature-set and class-pair work in `PLN-009`
- Existing artifact contracts, feature analysis, PCA, adequacy, coverage, and generator modules
- Existing advanced-filter decision logic as a deferral baseline, not yet as a trigger for implementation

Recommended Milestone Order:
- `M10`: Common experiment harness
- `M11`: Feature-set and class-pair comparison
- `M12`: Generic inference contract
- `M13`: Generic feature taxonomy and feature-set proof
- `M14`: Generic classification and evidence-combination proof
- `M15`: Generic filtering contract
- `M16`: Dimensional lift audit
- `M17`: Corpus adequacy and coverage framework
- `M18`: Switching and transition models
- `M19`: Advanced filter decision report
- `M20`: Minimal particle-filter prototype, only if justified
- `M21`: Minimal Rao-Blackwell particle-filter prototype, only if justified

Decision Rule For Advanced Filters:
- Do not add IMM, particle filtering, or RBPF merely because they are available techniques.
- Add IMM only after simpler transition-aware methods fail on documented switching cases.
- Add PF only after nonlinear or non-Gaussian cases remain unresolved on documented failure cases.
- Add RBPF only after the repo can state clearly:
  - what latent variables are sampled
  - what conditional state is filtered analytically
  - why a Kalman bank or IMM is not enough
  - what benchmark proves Rao-Blackwellization helps

Last Updated: 2026-05-24
