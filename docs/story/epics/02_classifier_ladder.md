# Classifier / Filter Evidence Ladder

Core question: how do we build an evidence ladder that starts simple, then deliberately excites advanced algorithms with named failure regimes that anticipate the 3D lift?

This epic treats classifiers and filters as evidence providers under a shared posterior contract. Pointwise, windowed, sequential Bayes, Kalman, transition, IMM, PF/GSF, and RBPF methods are not a winner ladder. They are a capability ladder: local evidence, local temporal evidence, history accumulation, dynamic residual evidence, switching logic, mode mixing, non-Gaussian posterior reasoning, and latent-structure inference.

The architectural claim is:

> Epic 2 proves both discipline and ambition: simple rungs establish the shared posterior-evidence contract, while advanced IMM/PF/RBPF witnesses demonstrate that the same architecture can escalate toward the nonlinear, switching, and latent-state problems expected in a 3D tracking lift.

The 1D witnesses prove the evaluation machinery, trace contracts, and promotion/defer logic. Richer 3D dynamics, sensor geometry, nonlinear measurements, occlusion, and latent maneuver states are where advanced algorithm usefulness should be fully excited.

The epic has three internal layers:

- Layer A: Baseline ladder. Pointwise through transition matrix prove the shared contract and give the sufficiency baseline.
- Layer B: Advanced algorithm showcase. IMM, PF, and RBPF each get a named witness that excites the assumptions the method was built for.
- Layer C: 3D lift bridge. The 1D witnesses keep evidence readable; 3D PVA, nonlinear geometry, sensor noise, occlusion, mode uncertainty, and latent maneuver structure make advanced inference operationally important.

Primary artifacts:

- `artifacts/packets/classifier_ladder_mvp/README.md`
- `artifacts/packets/classifier_ladder_mvp/decision_card.md`
- `artifacts/packets/classifier_ladder_mvp/evidence_capability_ladder.csv`
- `artifacts/packets/classifier_ladder_mvp/method_capability_matrix.csv`
- `artifacts/packets/classifier_ladder_mvp/filter_promotion_criteria.csv`
- `artifacts/packets/classifier_ladder_mvp/advanced_inference_architecture_map.csv`
- `artifacts/common_1d_classifier_study/unified_posterior_history.csv`
- `artifacts/rung_sufficiency/rung_promotion_matrix.csv`
- `artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv`

Main chart set:

- `06_posterior_timeline_witness`
- `06c_capability_ladder`
- `07_rung_sufficiency_map`
- `10_advanced_filter_gate_matrix`
- `10g_method_capability_matrix`
- `10h_advanced_inference_architecture_map`
- `10i_filter_promotion_criteria`
- `10e_advanced_filter_sweet_spot_matrix`
- `10f_simple_to_advanced_witness_bridge`

Advanced-filter boundary: advanced filters are not better everywhere. IMM is for switching dynamics with useful mode mixing. PF is for nonlinear or non-Gaussian posterior structure. RBPF is for sampled latent structure with conditional linear-Gaussian state estimation. The repo must include positive showcase witnesses for each one, while promotion remains witness-specific unless a declared study proves broader need.

Advanced showcase witnesses:

| Algorithm | Witness | Claim boundary |
| --- | --- | --- |
| IMM | `mode_switching_state_mixing` | `witness_supported` when mode probability, switch delay, innovation NLL, and state RMSE improve. |
| PF | `nonlinear_nongaussian_posterior` | `required_showcase` or candidate diagnostic until run-backed against robust baselines. |
| RBPF | `latent_event_timing` | `required_showcase` or candidate diagnostic until the sampled/marginalized split beats PF/IMM at fixed budget. |

Decision language:

- `architecturally_exercised`
- `simpler_rung_sufficient`
- `promote_rung`
- `revise_evidence_model`
- `defer_advanced_filter`
- `promote_advanced_filter_for_named_witness`

The audience should leave believing that the architecture can host IMM, PF/GSF, and RBPF once the scenario requires mode uncertainty, nonlinear posterior shape, or latent event inference.
