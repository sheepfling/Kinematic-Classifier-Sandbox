# Four Products

The repository is easiest to understand as four connected products. Each product answers a different question and produces governed evidence that can feed the others.

The file name is retained for link compatibility; this page is now the canonical four-product map.

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
    |  Which valid, difficult cases reveal the next failure mode?
    |
    +----------------------+----------------------+
                           |                      |
                           v                      |
4. Real-World Corpus & Validation <--------------+
   Do the claims survive independently collected physical observations?
                           |
                           +---- feedback to Products 1-3
```

The products share the same study candidate, evidence/posterior contract, decision language, and claim-governance rules. They are not unrelated applications. Presentation and showcase packets remain export profiles over governed artifacts rather than another product.

## Product 1: Static Admissibility

### Core question

Can the proposed feature set, class set, and prior regime support a meaningful study before we spend effort on corpus generation or classifier escalation?

### What it does

Static Admissibility inspects the study candidate before dynamic inference. It checks whether the classes are distinguishable enough to study, whether the features are relevant and sufficiently excited, whether priors create a pathology, and whether leakage, provenance, coverage, or observability issues make the evidence invalid.

### Product output

The product produces a decision packet, not just a collection of metrics. Its routes include:

- `promote_to_corpus_explorer`
- `promote_with_warnings`
- `revise_feature_set`
- `revise_class_set`
- `revise_prior`
- `reject`

A promotion means the study is worth routing forward. It does not mean that the eventual classifier will perform well or that the study is deployment-ready.

The detailed product specification is [Static Admissibility](epics/01_static_admissibility.md).

## Product 2: Classifier Evidence Ladder

### Core question

Given an admissible study, what kind of evidence is sufficient to distinguish the classes, and what additional capability is justified by each diagnosed failure mode?

The classifier product is an evidence system rather than a leaderboard. Every method consumes the same study surface and emits comparable evidence and posterior histories. Added complexity is justified only when a simpler method has a diagnosed limitation the added capability can address.

### Fidelity tiers

| Fidelity tier | Evidence capability | Representative methods |
| --- | --- | --- |
| 1. Direct and engineered evidence | Pointwise, windowed, robust-windowed, motif, and engineered feature evidence | pointwise/windowed classifiers, shapelets, feature boosting |
| 2. Temporal evidence | Accumulate evidence over time and represent class-transition structure | sequential Bayes, HMM, transition-matrix methods |
| 3. State-space evidence | Use dynamics, residuals, uncertainty, and latent state estimates | Kalman bank, UKF, robust Kalman, GSF |
| 4. Switching and nonlinear evidence | Represent mode switches, nonlinear dynamics, and latent uncertainty | IMM, PF, RBPF |

The practical movement rule is:

```text
diagnosed failure -> add one evidence capability -> compare on the same study
                  -> keep the simpler rung when it is sufficient
```

### Product output

The product produces comparable posterior histories, calibration and confusion diagnostics, runtime and robustness evidence, witness-to-method coverage, and promotion decisions such as `proven`, `simpler_rung_sufficient`, `witness_supported`, `study_justified`, `insufficient_evidence`, and `not_complexity_justified`.

The detailed product specification is [Classifier Ladder](epics/02_classifier_ladder.md), with the method map in [Algorithm Ladder](algorithm_ladder.md).

## Product 3: RL Corpus Exploration

### Core question

Can the workbench discover valid, difficult, decision-useful cases that expose where the current study or classifier is weak?

### What it does

RL Corpus Exploration turns a study need into a governed search objective. It can search for boundary cases, coverage gaps, prior-sensitive cases, feature excitation, class confusion, and downstream classifier stress. Candidate cases must remain valid, adequately covered, non-leaky, and useful for a decision.

Search backends may include CEM and other optimizers, PPO and other sequential-control policies, quality-diversity archives, and trajectory generators. RL is a corpus-search backend, not a replacement for corpus adequacy, classifier evaluation, or real-world validation.

### Product output

The product produces candidate frontiers, selected corpus manifests, adequacy and leakage audits, backend comparisons, hard-case packets, and routes such as:

- `selected_corpus_supported`
- `revise_corpus_policy`
- `route_hard_pair_to_ladder`
- `trigger_advanced_filter_candidate`
- `reject_invalid_hard_case`

The detailed product specification is [Corpus Evaluation and Advanced Exploration](epics/03_corpus_exploration.md), with the operational lane in [Corpus Explorer](corpus_explorer.md).

## Product 4: Real-World Corpus & Validation

### Core question

Do the methodology and classifier claims survive contact with independently collected real-world trajectories across the active physical-domain corpus lanes?

### What it does

Real-World Corpus & Validation is the grounding layer for the workbench. It acquires or references real source artifacts, preserves source-native semantics, normalizes trajectory state through a common contract, records provenance and quality, creates leakage-safe groupings and immutable snapshots, and constructs classifier-facing study views without inserting source identity or audit-only fields into the kinematic measurement sequence.

Product 4 currently targets six active corpus lanes:

1. `land_surface`
2. `sea_surface`
3. `sea_subsurface`
4. `air_atmospheric`
5. `space_near`
6. `space_orbital`

The initial LAND implementation on TGSIM is the reference vertical slice, not a special-case architecture. Other adapters must converge on the common corpus contract while retaining domain-specific time, frame, vertical, observation-modality, state-role, label-evidence, and lineage semantics.

### Product boundary

Product 4 owns:

- source registry, access metadata, citation, license, and version identity
- acquisition recipes and source-specific adapters
- the persistent real-world trajectory corpus contract
- source-native and normalized state views
- episode, mission, object, recording, deployment, and split-group identity
- label assertions with evidence strength and proxy status
- quality findings and processing lineage
- immutable prepared snapshots and hashes
- leakage-safe study selection and classifier projection
- real-world validation cohorts and claim decisions

Product 4 does not own synthetic hard-case generation; that remains Product 3. It also does not make classifier-complexity decisions; that remains Product 2.

### Persistent versus classifier-facing objects

The persistent aggregate should be an episode-level manifest rather than a classifier window:

```text
TrajectoryEpisodeManifest
    + TrajectoryStateView[]
    + segments
    + LabelAssertion[]
    + GroupingKey[]
    + quality
    + lineage
    + domain_extension

              | study policy
              v

ClassifierTrajectoryView
              |
              v
ClassifierWindow[]
```

`NormalizedTrack` remains useful as a normalized trajectory/state representation, but it is not the root cross-domain corpus object. This distinction allows AIS with no observed altitude, underwater dead reckoning, reconstructed near-space trajectories, and propagated orbital states to coexist without flattening their semantics.

### Product output

Product 4 produces governed source cards, adapter reports, prepared trajectory episodes, immutable corpus snapshots, coverage and balance reports, leakage audits, study manifests, classifier views, and real-world evidence packets. Decision routes include:

- `real_world_evidence_supported`
- `supported_with_limits`
- `independent_validation_required`
- `revise_source_portfolio`
- `revise_grouping_policy`
- `revise_label_claim`
- `insufficient_real_world_evidence`
- `reject_invalid_source`

The detailed product specification and implementation backlog are in [Real-World Corpus & Validation](epics/04_real_world_corpus_validation.md).

## How the products work together

| Product | Owns | Feeds | Receives feedback from |
| --- | --- | --- | --- |
| Static Admissibility | Study validity before expensive search or inference | admissible study candidates and corpus objectives | classifier, corpus-search, and real-world diagnostics |
| Classifier Evidence Ladder | Comparable evidence, posterior histories, and fidelity escalation | diagnosed failure modes and evidence gaps | static study definition, explored hard cases, real-world holdouts |
| RL Corpus Exploration | Search/generation of valid hard cases and coverage/stress objectives | synthetic or generated witnesses and boundary cases | static warnings, classifier failures, real-world failures |
| Real-World Corpus & Validation | Governed physical observations, provenance, snapshots, source-shift cohorts | real-world studies and independent validation evidence | source gaps, classifier failures, search hypotheses, claim requirements |

The loop is deliberate:

1. Static Admissibility blocks invalid or under-specified studies early.
2. The Classifier Evidence Ladder establishes the simplest sufficient evidence and records where it fails.
3. RL Corpus Exploration searches around diagnosed boundaries and coverage gaps.
4. Real-World Corpus & Validation tests the methodology against independently collected physical observations and source shifts.
5. Real-world failures can route back to Products 1-3 without weakening the distinction between synthetic search and observed reality.
6. Promotion decisions remain tied to explicit artifacts, grouping rules, provenance, and claim boundaries.

## Recommended reading path

1. [Static Admissibility](epics/01_static_admissibility.md)
2. [Classifier Ladder](epics/02_classifier_ladder.md)
3. [Corpus Evaluation and Advanced Exploration](epics/03_corpus_exploration.md)
4. [Real-World Corpus & Validation](epics/04_real_world_corpus_validation.md)
5. [Methodology Map](01_methodology_map.md)
6. [Claim Evidence Matrix](claim_evidence_matrix.md)

The full repository story remains the best guide to the shared architecture: [Repo Story](00_repo_story.md).
