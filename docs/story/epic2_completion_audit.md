# Epic 2 Completion Audit

This document is the current requirement-level read for Epic 2.

It is intentionally conservative. The question here is not whether code exists.
The question is whether the four public Epic 2 classifier families are
implemented, integrated into the shared pipeline, and honestly proven on the
current 1D evidence surface.

## Current Bottom Line

Epic 2 is not yet complete under the stricter family-scorecard standard.

Current family-level read from the method-validation operating system:

| Family | Implemented | Integrated | Proven | Current read |
| --- | --- | --- | --- | --- |
| Interpretable kinematic classifiers | yes | yes | yes | Proven on the current 1D witness set |
| Physics-aware inference classifiers | yes | yes | yes | Witness-backed on the current 1D core ladder, but the ceiling story is still bounded |
| Generic time-series benchmark classifiers | yes | partial | partial | Real bounded execution exists, but ceiling alignment is still weak |
| Learned sequence and embedding classifiers | yes | yes | partial | Witness-backed on bounded studies, but still weakly tied to the Epic 1 ceiling |

That means the repo has the family-evaluation framework and a first honest
scorecard layer, but not yet a strong enough cross-family ceiling-relative
story to declare Epic 2 finished.

## Requirement Read

### 1. Shared 1D evaluation framework

Current read: present.

Evidence:

- common method-validation status ladder
- shared witness registry
- common artifact-writing surfaces
- family maturity matrix in `artifacts/method_validation_os_v1/`

What is still open:

- broader robustness and study-justified expansion beyond the bounded Epic 2
  surface

### 2. Interpretable kinematic family

Current read: implemented, integrated, and proven on the current 1D witness
set.

What is present:

- pointwise
- windowed / robust-windowed
- shapelet / motif
- engineered-feature boosting

What is still open:

- broader robustness sweeps and study-justified comparison coverage

This family is the closest thing to complete inside Epic 2 today.

### 3. Physics-aware inference family

Current read: implemented, integrated, and witness-backed at the current 1D
family gate, but still bounded at the ceiling-relative layer.

What is present:

- HMM / transition matrix
- Kalman bank
- UKF
- Student-t / robust Kalman
- GSF
- IMM
- PF
- RBPF
- BOCPD
- HSMM
- bounded physics family promotion audit

What supports the public family gate:

- HMM / transition is now witness-supported on the named persistence/switching
  witness
- the Kalman bank is now witness-supported on the matched-endpoint dynamics
  witness
- IMM now has a narrow switching promotion audit and serves as the current
  study-justified switching state-mixing blocker on the audited witness family
- UKF now has a narrow nonlinear promotion audit and serves as the current
  study-justified nonlinear Gaussian blocker before mixture or particle
  escalation on the audited witness family
- GSF now has a narrow multimodal promotion audit and serves as the current
  study-justified least-complex blocker before PF on the audited multimodal
  witness family
- RBPF now clears the bounded compute-normalized latent frontier

Deferred from the public family gate:

- the switching-Kalman / SLDS lane remains researched-only, but it is now
  treated as a deferred extension rather than a core Epic 2 physics-family
  closure requirement

This family is now witness-backed at the public gate. What remains open is
broader robustness and a stronger Epic 1 ceiling comparison, especially for
the more advanced switching and particle methods.

### 4. Generic time-series benchmark family

Current read: implemented, partially integrated, and only partially proven at
the public family gate.

What is present:

- shared modern-TSC execution frontier
- named archive-versus-baseline witnesses
- bounded backend smoke packet
- bounded archive diagnosis packet
- bounded archive family promotion audit

What supports the public family gate:

- MiniRocket, dictionary-family, and HIVE-COTE now have real bounded
  promotion paths
- DrCIF now has a real integrated wrapper path and a narrow promotion audit,
  but it still remains below `witness_supported` because the current bounded
  evidence is parity-only rather than a positive witness win
- broader archive-family breadth, especially beyond the current bounded
  witness packets, remains open

This family now has real bounded execution and at least one honest promoted
member, but it still lacks a strong family-level ceiling-aligned comparison to
Epic 1. It should remain partial rather than being treated as finished.

### 5. Learned sequence and embedding family

Current read: implemented, integrated, and partially proven at the public
family gate.

What is present:

- trained `TCN` frontier
- trained `InceptionTime` frontier
- bounded neural-sequence robustness frontier
- `TS2Vec` embedding frontier
- bounded TS2Vec proxy-versus-external parity witness

What remains open:

- broader robustness and benchmark breadth for TCN / InceptionTime beyond the
  current bounded multi-seed packet
- broader benchmark breadth for TS2Vec beyond the bounded parity witness
- learned-hybrid methods like KalmanNet and differentiable PF remain deferred
  research and should stay outside the public Epic 2 family gate until they
  have their own mismatch witness

This means the public learned family is visible and boundedly useful, but it
still needs stronger ceiling-relative benchmarking before it counts as fully
proven for Epic 2 completion purposes.

### 6. Classifier family scorecard

Current read: present, and now the main completion gate.

What is present:

- classifier family atlas
- capability matrix
- first ceiling-efficiency table
- first classifier-efficiency-vs-ceiling chart

What the scorecard currently says:

- capability additions are now explicit by family
- expected win conditions are now explicit by family
- several temporal and dynamics-heavy families exceed the current static Epic 1
  proxy, which means the current proxy understates what those families can
  capture
- generic TSC and learned families still lack strong named family-level ceiling
  alignment

This is the key honest read: the missing centerpiece now exists, but it does
not yet support a strong “Epic 2 is done” claim.

## Honest Epic 2 Finish Criteria

Epic 2 should only be treated as complete when current evidence supports all of
the following:

1. All four public classifier families are implemented on the shared 1D
   pipeline.
2. All four families are integrated through common datasets, metrics, artifact
   outputs, and validation surfaces.
3. Each family has at least one honest, named, witness-supported path that
   proves why that family matters.
4. The classifier family scorecard must answer, for each family, what
   capability it adds and when it should win.
5. The scorecard must also show how close each family gets to the Epic 1
   admissibility ceiling, with aligned witnesses strong enough that “proxy
   exceeded” and “no family ceiling alignment” are exceptions rather than
   dominant caveats.
6. The generic-TSC lane must keep at least one real family member promoted
   against current baselines and then decide whether the remaining archive
   families deserve broader closure work or explicit deferral.
7. The learned-sequence/embedding lane must move beyond bounded witness support
   to a stronger ceiling-relative comparison.
8. The remaining physics-aware methods either broaden their promotion support
   beyond the current bounded witnesses or stay explicitly bounded without
   inflating the public family claim.

Current read against these criteria: not yet satisfied.

## Highest-Signal Remaining Moves

1. Strengthen the classifier family scorecard so each family has a named,
   witness-aligned ceiling comparison rather than proxy-only or missing
   alignment.
2. Broaden the generic-TSC lane beyond the current bounded archive witness
   packets while keeping DrCIF explicitly partial until it earns a named win.
3. Broaden the learned-sequence and TS2Vec witnesses beyond compact parity and
   single-frontier coverage.
4. Keep the physics-aware family honest by broadening robustness and
   study-justified coverage without reopening already-cleared foundation-rung
   promotion questions.

Those moves are still blockers for a strong Epic 2 completion claim.
