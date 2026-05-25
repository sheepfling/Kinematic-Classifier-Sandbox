# Witness: generated_corpus_stress

Purpose: Show that the Corpus Explorer can generate, score, and select hard or fragile examples rather than relying only on fixed examples.

Class set: selected generated corpus class definitions.

Feature set: generated-corpus feature matrix and classifier score features.

Classifier/filter family: selected corpus classifier scoring and common posterior contract.

Prior regime: explicit downstream study priors and fragility checks.

Corpus objective: stress, adequacy, leakage, class validity, feature excitation, and archive coverage.

What it proves: generated candidates can be scored, audited, selected, and connected to study evaluation artifacts.

What it does not prove: the corpus is final or that closed-loop QD search is complete.

Key equations: corpus sampling `theta ~ q(theta | o,b)`, generation `tau ~ G_b(theta,xi)`, and score `S_k`.

Key plots:
- `artifacts/generic_corpus_exploration/selected_trajectory_gallery.png`
- `artifacts/selected_generated_corpus/selected_corpus_summary_dashboard.png`

Key tables:
- `artifacts/generic_corpus_exploration/candidate_scores.csv`
- `artifacts/selected_generated_corpus/classifier_scores.csv`

Key artifacts:
- `artifacts/generic_corpus_exploration/selected_corpus_manifest.json`
- `artifacts/selected_generated_corpus/corpus_manifest.json`

Promotion status: v1 complete with hardening still open.

Next extension toward 3D: add 3D backend adapters and QD archive dimensions for geometry, dynamics, and sensor regimes.
