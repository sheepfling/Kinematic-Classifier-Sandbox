# New Study User Guide

This repo is a study framework for kinematic classification. The canonical user workflow is:

`Declare -> Analyze -> Generate -> Audit -> Evaluate -> Decide -> Package`

## 1. Declare The Study

Start by declaring a concrete study candidate, not by choosing a classifier first.

The default template is [study_candidate.yaml](templates/study_candidate.yaml).

The declaration should identify:

- the class set or class pairs
- the feature sets under consideration
- the candidate classifier or filter families
- the prior regimes you want to test
- the corpus objective and difficulty intent
- the backend or generator constraints, if any

The canonical demo study lives at [new_study_workflow_demo.yaml](experiments/new_study_workflow_demo/new_study_workflow_demo.yaml).

## 2. Analyze Feature And Class Geometry

Before generating more data, inspect whether the declared features and classes are meaningful.

Run:

```bash
python3 scripts/workflows/analyze_feature_class_geometry.py \
  --study experiments/new_study_workflow_demo/new_study_workflow_demo.yaml \
  --output-dir artifacts
```

This phase packages:

- pairwise AUC and overlap
- feature separation evidence
- feature redundancy
- prior fragility preview
- oracle separability preview
- class-hierarchy notes

## 3. Generate And Explore Corpora

Once the feature/class geometry is plausible, generate and compare corpus candidates.

Run:

```bash
python3 scripts/workflows/generate_and_explore_corpus.py \
  --study experiments/new_study_workflow_demo/new_study_workflow_demo.yaml \
  --output-dir artifacts
```

This phase packages:

- generated candidate rows
- candidate corpus scores
- selected corpus manifest
- feature-excitation preview
- corpus-policy exploration summaries

## 4. Audit The Selected Corpus

Do not trust downstream classifier results until class validity, leakage, and coverage have been checked.

Run:

```bash
python3 scripts/workflows/audit_selected_corpus.py \
  --study experiments/new_study_workflow_demo/new_study_workflow_demo.yaml \
  --output-dir artifacts
```

This phase packages:

- class validity scores
- label-status distribution plot
- covariate leakage audit
- corpus adequacy report
- a compact corpus decision gate

## 5. Run The Classifier Ladder

Evaluate the declared study using the simplest plausible methods first.

Run:

```bash
python3 scripts/workflows/run_classifier_ladder.py \
  --study experiments/new_study_workflow_demo/new_study_workflow_demo.yaml \
  --output-dir artifacts
```

This phase packages:

- posterior history by method
- filtered method metrics for the study
- confusion by method
- prior sensitivity by method
- sufficiency and insufficiency tables

## 6. Decide And Package

The final step is to assign a decision and write a report bundle.

Run:

```bash
python3 scripts/workflows/evaluate_and_package.py \
  --study experiments/new_study_workflow_demo/new_study_workflow_demo.yaml \
  --output-dir artifacts
```

This phase writes:

- `05_report/study_report.md`
- `05_report/decision_card.md`
- `05_report/visual_gallery.md`
- a top-level workflow `index.md`

## 7. Decision Interpretation

Every study should end in one of:

- `promote`: current evidence and rung are sufficient
- `revise`: a localized corpus, feature, prior, or method issue remains
- `reject`: the declared study claim is not supported
- `defer`: the study depends on unimplemented or out-of-scope capabilities

## 8. Canonical Output Layout

The workflow writes:

```text
artifacts/<study_id>/
  00_study_declaration/
  01_feature_class_analysis/
  02_corpus_generation/
  03_corpus_audit/
  04_ladder_evaluation/
  05_report/
  index.md
```

The canonical demo uses `study_id = new_study_workflow_demo`, so its bundle lands under `artifacts/new_study_workflow_demo/`.

