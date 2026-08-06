# Product 1 Pitch Brief: Static Admissibility

## The executive framing

Static Admissibility is the front door for a proposed feature/class/prior
study. It answers a practical question before corpus search, classifier
training, or reinforcement-learning exploration:

> What can we eliminate, repair, or target before the expensive problem begins?

The tool accepts a portable study bundle, runs the same declared evidence
surface through feature, class, prior, coverage, and leakage checks, and emits
both machine-readable findings and a decision route.

## What it can eliminate upfront

| surface | examples of what it catches | resulting decision |
| --- | --- | --- |
| Feature space | label leakage, unavailable features, weak features, exact aliases, high redundancy, unconfirmed synergy | remove/recompute features, reduce dimensions, or send a pair to ablation |
| Class space | hard confusable pairs, exact/near feature collisions, unobserved future classes, non-decisionable boundaries | merge, split, redefine, or prune classes |
| Prior space | prior odds outside the observed evidence range, classes selected rarely or never under the proxy | rebalance priors, add witness evidence, or remove the class from the active set |
| Corpus-search space | low-count class-feature cells and incomplete future-class coverage | create targeted Corpus Explorer objectives instead of searching blindly |

The benefit is not merely a report. It is a smaller and better-defined problem
handed to the next product.

## Five-minute demonstration

Regenerate the complete evidence atlas:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit-suite \
  experiments/static_admissibility/epic1_exemplar_suite.yaml \
  --output-dir artifacts/validation_packets/01_static_admissibility
```

Then present the generated [executive brief](../../artifacts/validation_packets/01_static_admissibility/executive_brief.md):

1. Start with `figures/02a_static_exemplar_suite_routing_matrix.png` to show
   one automated route across all static gates.
2. Show the class-overlap card: a hard pair is routed to
   `revise_class_set` before the classifier is blamed.
3. Show the prior-domination card: the rare class has zero own-surface
   selection under the declared prior/evidence proxy, producing
   `PRIOR_DOMINATION` and `PRIOR_SELECTION_SKEW`.
4. Show the future-class card: an unobserved future class has an expected exact
   signature collision, so it is pruned or redefined before corpus search.
5. Close with `figures/02b_static_audit_decision_card.png`, which summarizes
   promotion, revision, and rejection routes.

## Evidence outputs to open live

- `source_artifacts/exemplar_suite_manifest.csv`: generated route, issue codes,
  and first recommendation for every exemplar.
- `source_artifacts/exemplar_route_matrix.csv`: feature, class, prior,
  selection, coverage, and leakage gate statuses.
- `source_artifacts/*/prior_selection_balance.csv`: class-level prior-weighted
  selection balance and own-class selection rate.
- `source_artifacts/*/static_resolution_plan.csv`: severity, evidence,
  recommended action, verification, and route.
- `figures/02m_static_exemplar_fingerprint_strip.png`: compact comparison of
  the seven systemic cases.

## The story in one sentence

Static Admissibility turns “should we explore this study?” into a repeatable
file-backed gate that can shrink feature and class space, expose prior skew,
and hand the next team a specific repair or search objective.

## Claim boundary

This is an early admissibility and routing tool. It can expose holes, reduce
candidate space, and prescribe follow-up checks. It does not prove deployed
classifier performance, causal feature importance, or operational coverage.

See the [Static Admissibility Toolkit](static_admissibility_toolkit.md) for the
full input/output contract and the [Epic 1 Exemplar Suite](epics/01_static_admissibility_exemplars.md)
for the source families.
