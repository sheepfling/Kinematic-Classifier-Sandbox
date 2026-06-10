# Corpus Explorer

The Corpus Explorer is the corpus governance layer. It is broader than a data generator: it declares objectives, samples candidates, adapts backends, scores validity and difficulty, audits leakage, maintains archives, and selects corpora for study evaluation.

## Current Support Boundary

The current strongest supported corpus-evaluation path is the repo's default generated/common-study corpus evaluation flow.

- default generated/common-study corpus evaluation is the strongest supported path today
- selected generated corpus packet rerun is also strong because the selected packet is pushed back through adequacy, leakage, and regression checks
- the adapter contract for provided or external corpus sources exists, but it is not yet a full arbitrary-corpus adequacy pipeline
- this is therefore not yet a full generic corpus evaluation for any provided corpus

## Corpus Flow

```text
theta ~ q(theta | o, b)
tau ~ G_b(theta, xi)
D = {tau_i}_{i=1}^N
```

`o` is a corpus objective, `b` is a backend, `theta` is a candidate parameterization, `xi` is noise or randomization, `G_b` is the backend-specific generator, and `D` is the corpus.

## Corpus Score

```text
S_k = w_b B_k + w_c C_k + w_f F_k + w_d D_k
      - w_l L_k - w_t T_k - w_g G_k
```

Positive terms reward balance, class validity, feature excitation, and difficulty. Penalty terms reduce scores for leakage, triviality, and generator degeneracy.

## Responsibilities

| Question | Corpus Explorer responsibility |
| --- | --- |
| What should the corpus test? | Define a `CorpusObjective`. |
| How is data produced? | Choose a backend and candidate sampler. |
| Are labels meaningful? | Score class validity and relabel or reject weak candidates. |
| Are features exercised? | Measure feature excitation and coverage. |
| Is the corpus leaking answers? | Audit covariates, scenario shortcuts, and degeneracy. |
| Does search cover useful variety? | Maintain QD archive cells and selected elites. |
| What feeds evaluation? | Promote a `SelectedCorpus` with provenance and limitations. |

## Connection To Study Evaluator

The Corpus Explorer produces `D` for the study candidate `s = (D, f, C, m, pi, b)`. A weak corpus can invalidate downstream classifier claims even when the posterior updater and algorithm implementation are correct.

## External Backend Examples

The repo already has 1D examples of the external-backend interface shape that a future TAOS, TGx, or similar 3D trajectory generator should satisfy:

- `backend_adapter_proof.py` shows the adapter lifecycle: `prepare -> run -> normalize_output -> validate`.
- `corpus/exploration/external_backend_examples.py` packages that lifecycle into TAOS-like and TGx-like 1D examples with explicit input-deck, execution-log, raw-output, normalized-output, and validation-report artifacts.

That means the 3D lift does not need a new interface design. It needs a richer backend implementation behind the same contract.

That adapter contract for provided or external corpus sources is important, but it should be read as an interface and normalization proof, not yet a full arbitrary-corpus adequacy pipeline.
