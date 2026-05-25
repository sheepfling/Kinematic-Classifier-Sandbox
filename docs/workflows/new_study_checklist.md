# New Study Checklist

- Declare the study candidate in YAML.
- Confirm the target class pair exists in the current study manifests.
- Confirm the selected feature sets exist in `experiments/common_1d_classifier_study/feature_sets.json`.
- Confirm the selected classifier IDs exist in `experiments/common_1d_classifier_study/classifier_manifest.json`.
- Run feature/class geometry analysis before generating more corpus data.
- Check pairwise overlap, AUC, redundancy, prior fragility, and oracle preview.
- Generate and compare corpus candidates.
- Inspect candidate scores, selected corpus manifest, and feature-excitation outputs.
- Audit class validity, leakage, and adequacy before trusting ladder results.
- Run the classifier ladder only after the selected corpus passes basic quality gates.
- Inspect sufficiency and insufficiency tables, not just endpoint accuracy.
- Assign `promote`, `revise`, `reject`, or `defer`.
- Package the result with a report, decision card, and visual gallery.

