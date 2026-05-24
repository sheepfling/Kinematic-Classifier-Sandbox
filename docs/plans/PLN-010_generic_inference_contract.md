# PLN-010 Generic Inference Contract

Title: M12 Generic Inference Contract
Plan ID: PLN-010
Status: done
Owner: @rick
Priority: P1
Objective: Define and prove a generic inference contract so pointwise, windowed, Bayesian accumulator, and Kalman-bank methods all emit compatible evidence, posterior, prediction, and optional filter artifacts.
Scope:
- Define common classifier, evidence, posterior-history, and filter-output schemas.
- Validate the current classifier ladder against one shared output contract.
- Emit one artifact family that proves the repo is using generic inference machinery rather than disconnected per-method writers.
Out of Scope:
- Implementing IMM, PF, or RBPF.
- Replacing current algorithms with a new modeling family.
- Full multidimensional corpus work beyond what the schema needs to admit.
Implementation Steps:
1. Define common interfaces for corpus adapters, evidence providers, posterior updaters, state estimators, experiment runners, and artifact writers.
2. Standardize required output columns for all methods.
3. Add filter-specific optional outputs for state means, covariances, innovations, ESS, resampling, and model probabilities.
4. Run pointwise, windowed, Bayesian accumulator, and Kalman bank through the same validation surface.
5. Emit the generic inference contract artifact family.
Validation:
- All current classifiers emit the same posterior and prediction schema.
- Posterior probabilities sum to one for every classifier and trajectory.
- Shared metrics code can consume all four method outputs without special casing.
Artifacts / Config:
- `artifacts/generic_inference_contract/contract_report.md`
- `artifacts/generic_inference_contract/classifier_output_schema.json`
- `artifacts/generic_inference_contract/evidence_provider_schema.json`
- `artifacts/generic_inference_contract/posterior_history_schema.json`
- `artifacts/generic_inference_contract/filter_output_schema.json`
- `artifacts/generic_inference_contract/validation_results.json`
Dependencies:
- `PLN-007`
- `PLN-009`
- `PLN-002`
Last Updated: 2026-05-24
