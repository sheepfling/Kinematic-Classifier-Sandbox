# Three Products

The repository is easiest to understand as three connected products. Each
product answers a different question and produces evidence for the next one.

```text
study intent
    |
    v
1. Static Admissibility
    |  Is the study meaningful enough to pursue?
    v
2. Classifier Evidence Ladder
    |  What evidence builder is sufficient?
    v
3. RL Corpus Exploration
       Which valid, difficult cases reveal the next failure mode?
       ^
       +---------- feedback into the classifier ladder
```

The products share the same study candidate, evidence/posterior contract, and
decision language. They are not three unrelated applications, and the
presentation or showcase packets are exports of this system rather than a
fourth product.

## Product 1: Static Admissibility

### Core question

Can the proposed feature set, class set, and prior regime support a meaningful
study before we spend effort on corpus generation or classifier escalation?

### What it does

Static Admissibility inspects the study candidate before dynamic inference. It
checks whether the classes are distinguishable enough to study, whether the
features are relevant and sufficiently excited, whether priors create a
pathology, and whether leakage, provenance, coverage, or observability issues
make the evidence invalid.

### Product output

The product produces a decision packet, not just a collection of metrics. Its
routes are:

- `promote_to_corpus_explorer`
- `promote_with_warnings`
- `revise_feature_set`
- `revise_class_set`
- `revise_prior`
- `reject`

A promotion means “this study is worth routing forward.” It does not mean that
the eventual classifier will perform well or that the study is ready for 3D
deployment.

The generic bundle is dimension-agnostic: the same audit accepts a 1D scalar,
2D/3D vector-derived features, or a higher-dimensional feature vector. The
dimension is declared metadata; the diagnostics operate over the supplied
feature columns rather than assuming a coordinate count.

### Main inputs and artifacts

- feature schema, class schema, prior regime, and controlled samples
- static decision card and validation report
- confusability, relevance, redundancy, synergy, prior-pathology, coverage,
  and leakage tables
- reusable bundle workflow in
  [Static Audit Bundle User Guide](../workflows/static_audit_bundle_user_guide.md)

The detailed product specification is
[Static Admissibility](epics/01_static_admissibility.md).
The quick-turn command and diagnostic index are in the
[Static Admissibility Toolkit](static_admissibility_toolkit.md).

## Product 2: Classifier Evidence Ladder

### Core question

Given an admissible study, what kind of evidence is sufficient to distinguish
the classes, and what additional capability is justified by each failure mode?

The classifier product is an evidence system rather than a leaderboard. Every
method consumes the same study surface and emits comparable evidence and
posterior histories. A more complicated method is useful only when a simpler
method has a diagnosed limitation that the added capability can address.

### Fidelity tiers

Here, “fidelity” means the richness of the evidence model and its assumptions.
It does not mean that a higher tier is automatically more accurate, more
realistic, or more deployment-ready.

| Fidelity tier | Evidence capability | Representative methods |
| --- | --- | --- |
| 1. Direct and engineered evidence | Pointwise, windowed, robust-windowed, motif, and engineered feature evidence | pointwise/windowed classifiers, shapelets, feature boosting |
| 2. Temporal evidence | Accumulate evidence over time and represent class-transition structure | sequential Bayes, HMM, transition-matrix methods |
| 3. State-space evidence | Use dynamics, residuals, uncertainty, and latent state estimates | Kalman bank, UKF, robust Kalman, GSF |
| 4. Switching and nonlinear evidence | Represent mode switches, nonlinear dynamics, and latent uncertainty | IMM, PF, RBPF |

Generic time-series benchmarks and learned sequence/embedding methods run as
parallel family lanes on the same evaluation surface. They can provide a
performance ceiling or a different representation, but they are not promoted
solely because they are more complex.

The practical movement rule is:

```text
diagnosed failure -> add one evidence capability -> compare on the same study
                  -> keep the simpler rung when it is sufficient
```

The ladder can therefore move upward when the study requires it and back down
when a simpler rung explains the evidence adequately.

### Product output

The product produces comparable posterior histories, calibration and
confusion diagnostics, runtime and robustness evidence, witness-to-method
coverage, and a promotion decision such as:

- `implemented`
- `integrated`
- `proven`
- `simpler_rung_sufficient`
- `witness_supported`
- `study_justified`
- `insufficient_evidence`
- `not_complexity_justified`

The classifier product also contains four public family lanes:

1. interpretable kinematic classifiers
2. physics-aware inference classifiers
3. generic time-series benchmark classifiers
4. learned sequence and embedding classifiers

The existing A–D study tiers—sanity/easy, timing, local motif/shape, and
robustness/realism—are evaluation scenarios inside this product. They are not
the same thing as the fidelity tiers above.

The detailed product specification is
[Classifier Ladder](epics/02_classifier_ladder.md), with the method map in
[Algorithm Ladder](algorithm_ladder.md).

## Product 3: RL Corpus Exploration

### Core question

Can the workbench discover valid, difficult, decision-useful cases that expose
where the current study or classifier is weak?

This product is the Corpus Explorer with reinforcement-learning search as one
of its most important experimental capabilities. It is not a promise that PPO
or any other search method produces truth by itself.

### What it does

RL Corpus Exploration turns a study need into a governed corpus objective. It
can search for boundary cases, coverage gaps, prior-sensitive cases, feature
excitation, class confusion, and downstream classifier stress. Candidate cases
must remain valid, adequately covered, non-leaky, and useful for a decision.

The search layer may compare multiple backends, including:

- CEM and other baseline optimizers
- PPO and other sequential-control policies
- quality-diversity archives
- trajectory generators and backend registries

RL is therefore a corpus-search backend, not a replacement for corpus
adequacy, classifier evaluation, or promotion review.

### Product output

The product produces candidate frontiers, selected corpus manifests, adequacy
and leakage audits, backend comparisons, hard-case packets, and routes such as:

- `selected_corpus_supported`
- `revise_corpus_policy`
- `route_hard_pair_to_ladder`
- `trigger_advanced_filter_candidate`
- `reject_invalid_hard_case`

PPO needs baseline comparison, ablation, seed stability, and downstream
diagnostic yield before the repository makes a strong claim about it.

The detailed product specification is
[Corpus Evaluation and Advanced Exploration](epics/03_corpus_exploration.md),
with the operational lane in [Corpus Explorer](corpus_explorer.md).

## How the products work together

| Product | Owns | Feeds | Receives feedback from |
| --- | --- | --- | --- |
| Static Admissibility | Study validity before expensive search or inference | corpus objectives and admissible study candidates | corpus and classifier diagnostics |
| Classifier Evidence Ladder | Comparable evidence, posterior histories, and fidelity escalation | failure modes and evidence gaps | static study definition and explored hard cases |
| RL Corpus Exploration | Search for valid hard cases and coverage/stress objectives | new corpora and targeted witness cases | static warnings and classifier failures |

The loop is deliberate:

1. Static Admissibility blocks invalid or under-specified studies early.
2. The Classifier Evidence Ladder establishes the simplest sufficient evidence
   and records where it fails.
3. RL Corpus Exploration searches for valid cases around those boundaries.
4. The resulting cases return to the classifier product as witnesses or stress
   tests.
5. Promotion decisions remain tied to explicit artifacts and claim boundaries.

## Recommended reading path

Start with this page, then go deeper through the product that matches the
question:

1. [Static Admissibility](epics/01_static_admissibility.md)
2. [Classifier Ladder](epics/02_classifier_ladder.md)
3. [Corpus Evaluation and Advanced Exploration](epics/03_corpus_exploration.md)
4. [Methodology Map](01_methodology_map.md)
5. [Claim Evidence Matrix](claim_evidence_matrix.md)

The full repository story remains the best guide to the shared architecture:
[Repo Story](00_repo_story.md).
