# Promotion Status Model

The repo now separates method tracking status from failure status.

Promotion in this model means methodological promotion on the evidence ladder,
not operational fielding. A method can be promoted as a rung, witness, or study
surface while still being inappropriate for deployment or fielded use.

## Status Ladder

| Status | Meaning |
| --- | --- |
| `researched` | Research note and intended failure mode are defined |
| `implemented` | Code exists inside the shared method-validation surface |
| `trace_validated` | Required traces and diagnostics are emitted and checked |
| `oracle_validated` | Oracle or negative-control checks exist for the method family |
| `witness_supported` | A named witness improves and the packet explains why |
| `study_justified` | Robustness and complexity gates pass for that witness family |
| `generalized` | Reserved for broader evidence across witnesses and corpora |

The `learning_evidence` lane uses the same ladder. Supervised tabular
baselines can be `witness_supported` when calibration and split discipline are
audited; compact sequence learners and unsupervised discovery stay at
`researched` until their evidence contracts are defined and validated.

## Epic 2 Gate Mapping

Epic 2 uses a simpler public gate model on top of this status ladder:

| Epic 2 gate | Practical meaning | Detailed status mapping |
| --- | --- | --- |
| `Implemented` | The method exists and basic fit/predict or equivalent execution works | at least `implemented` |
| `Integrated` | The method runs through the shared dataset, split, metric, and artifact pipeline | typically `trace_validated` or better |
| `Proven` | The family has comparison-backed evidence on a named 1D study surface | at least `witness_supported` |

For advanced or expensive methods, `Proven` is still not enough for broad
promotion. Broader methodological claims should generally wait for
`study_justified`, which means the witness survives robustness and complexity
gates.

That distinction matters for honesty:

- a method can be implemented without being integrated
- a method can be integrated without being proven
- a method can be proven on a named witness without being generalized
- fallback or proxy execution does not skip any of these gates

## Failure Ladder

| Failure status | Meaning |
| --- | --- |
| `missing` | No implementation yet |
| `blocked` | Prerequisite missing or lane deferred |
| `insufficient_evidence` | Evidence packet exists but cannot support a decision |
| `invalid_assumption` | Witness does not match method assumptions |
| `fails_oracle` | Oracle or negative-control check fails |
| `fails_robustness` | Improvement does not survive sweeps |
| `not_complexity_justified` | Improvement exists but does not justify extra complexity |

## Operating Rule

A method should not advance past `witness_supported` unless:

1. a simpler method fails on a named witness,
2. the failure matches the candidate method assumptions,
3. the candidate improves the relevant metrics,
4. the improvement survives robustness sweeps,
5. the traces explain why,
6. the complexity cost is accounted for.

That rule governs ladder promotion. Any decision about using the method in a
fielded system is a separate operational gate and can be stricter than the
methodology gate.

The generated status matrix lives in:

- `artifacts/method_validation_os_v1/algorithm_promotion_status_matrix.csv`
