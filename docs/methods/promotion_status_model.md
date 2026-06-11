# Promotion Status Model

The repo now separates method tracking status from failure status.

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

The generated status matrix lives in:

- `artifacts/method_validation_os_v1/algorithm_promotion_status_matrix.csv`
