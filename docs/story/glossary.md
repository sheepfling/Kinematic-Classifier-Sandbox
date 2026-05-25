# Glossary

## StudyCandidate

A proposed study unit `s = (D, f, C, m, pi, b)` combining corpus, feature set, class set or class pair, classifier/filter family, prior regime, and optional backend or dynamics family.

## CorpusObjective

A declarative target for a corpus: what classes, regimes, stressors, feature excitation, backend capabilities, and leakage constraints the corpus should exercise.

## CorpusCandidate

A generated or selected trajectory bundle before final adequacy and selection gates.

## SelectedCorpus

A corpus that has passed enough objective, validity, leakage, and adequacy checks to feed a study candidate.

## FeatureSet

A named set of computed features used as evidence inputs or separability diagnostics.

## ClassSet

The full set of labels a study is allowed to distinguish.

## ClassPair

A two-class slice used for pairwise separability, AUC, overlap, and prior-sensitivity analysis.

## EvidenceProvider

Any pointwise, windowed, sequential, filter, or transition model that converts observations, features, or residuals into class evidence.

## PosteriorUpdater

The shared machinery that combines priors and evidence into normalized posterior histories.

## ClassifierFamily

A family of evidence providers with a common construction, such as pointwise likelihoods, windowed features, sequential Bayes accumulation, or transition-aware accumulation.

## FilterBackend

A dynamics-aware model or backend that can produce residuals, innovations, state estimates, or likelihoods for classification.

## PriorRegime

The assumed class prior configuration and the sweep or stress used to test whether decisions are fragile to prior choices.

## ValidationLadder

The ordered set of checks that turns a study candidate into a promotion decision.

## WitnessProblem

A small controlled problem used to prove one methodology layer or failure mode before lifting it into more complex 3D settings.

## PromotionDecision

The final disposition assigned to a study candidate: `promote`, `revise`, `reject`, or `defer`.
