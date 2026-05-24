# PLN-016 Team Facing Methodology Showcase

Title: Team-Facing Methodology Showcase And Evidence Packet
Plan ID: PLN-016
Status: done
Owner: @rick
Priority: P1
Objective: Package the repo as a team-facing methodology and evidence suite so another engineer can understand the problem framing, implemented algorithm ladder, study suite, artifact evidence, generic 1D-to-3D transition path, and reproducible rerun flow without reading the codebase first.
Scope:
- Create a top-level showcase documentation and artifact structure for team consumption.
- Generate executive, methodology, algorithm, feature, identifiability, corpus, filtering-decision, and 3D-transition reports from repo artifacts.
- Add a visualization gallery, artifact manifest, run cards, and one-command team-packet export.
- Make the showcase emphasize what is proven, what is still experimental, and how 3D readiness depends on generic methodology contracts rather than only more algorithms.
Out of Scope:
- Implementing IMM, PF, or RBPF beyond decision reports.
- Claiming 3D readiness without the dimensional-lift audit and generic methodology proof artifacts.
- Replacing the current repo README with only showcase content.
- Using raw CSVs as the primary team-facing deliverable without reports, gallery, or narrative context.

Implementation Steps:
1. Stabilize the packet entrypoints.
   - Record current regression state and known gaps.
   - Ensure the common study runner, abstract inspection bundle, and artifact exporters are stable enough to feed the packet.
2. Add showcase structure.
   - Create `docs/showcase/` narrative sources.
   - Create `artifacts/showcase/` generated outputs with `reports/`, `plots/`, `tables/`, `run_cards/`, and manifest files.
3. Generate the executive and methodology reports.
   - Explain the problem, what exists now, what has been validated, what is still unproven, and how the architecture generalizes from 1D to 3D.
   - Make the corpus -> features -> evidence -> posterior -> metrics -> reports flow explicit.
4. Generate the algorithm-comparison report.
   - Present the algorithm ladder from pointwise through Kalman bank and transition-model plans.
   - Explain what each method assumes, what artifacts it emits, where it succeeds, where it fails, and when it should or should not be used.
5. Generate the feature and class-pair study reports.
   - Surface feature taxonomy, feature-set studies, pairwise confusability, oracle separability, duration/noise sensitivity, and prior sensitivity.
   - Make clear which confusions are feature/data problems versus classifier problems.
6. Generate the corpus adequacy and leakage report.
   - Show class balance, scenario balance, feature excitation, class-pair coverage, and covariate leakage.
   - Treat this as a credibility layer for the rest of the packet.
7. Generate the filtering and advanced-method decision report.
   - Show Kalman status, transition-matrix plans, and explicit go/no-go decisions for IMM, PF, and RBPF.
   - State what would be sampled, what would be marginalized analytically, what simpler method must fail first, and what metric would justify implementation.
8. Generate the dimensional-lift and 3D-transition report.
   - Show what stays generic, what becomes 3D-specific, what adapters are required, and which modules are dimension-agnostic versus 1D-specific.
9. Add a visualization gallery and run cards.
   - Group plots by question answered.
   - Require captions, source artifacts, interpretation notes, and limitation notes.
   - Generate one-page run cards for each major study.
10. Add team-packet export.
   - Add one command that generates `artifacts/team_packet/` and an optional zip.
   - Make the exported packet browsable without having to run the repo.

Validation:
- Full regression still passes after showcase tooling and docs additions.
- `artifact_manifest.json` lists every generated report, table, and plot in the showcase packet.
- All required reports exist and are nonempty.
- Core metrics tables exist and are nonempty.
- The plot gallery references existing files only.
- Every plot has a caption and interpretation note.
- Every feature has taxonomy metadata.
- Every class pair has at least one identifiability row in the packet.
- Advanced-method reports explicitly state go/no-go status and justification.
- The 3D transition report identifies dimension-agnostic versus 1D-specific modules.

Artifacts / Config:
- `docs/showcase/00_executive_summary.md`
- `docs/showcase/01_problem_framing.md`
- `docs/showcase/02_methodology_overview.md`
- `docs/showcase/03_algorithm_ladder.md`
- `docs/showcase/04_feature_taxonomy.md`
- `docs/showcase/05_filtering_taxonomy.md`
- `docs/showcase/06_study_suite.md`
- `docs/showcase/07_visualization_gallery.md`
- `docs/showcase/08_results_summary.md`
- `docs/showcase/09_3d_transition_plan.md`
- `docs/showcase/10_open_risks_and_next_steps.md`
- `artifacts/showcase/index.md`
- `artifacts/showcase/artifact_manifest.json`
- `artifacts/showcase/summary_metrics.json`
- `artifacts/showcase/reports/`
- `artifacts/showcase/plots/`
- `artifacts/showcase/tables/`
- `artifacts/showcase/run_cards/`
- `artifacts/team_packet/`
- optional `artifacts/kinematic_classifier_team_packet.zip`
- helper scripts such as:
  - `scripts/build_showcase.py`
  - `scripts/build_gallery.py`
  - `scripts/export_team_packet.py`
  - `scripts/validate_artifacts.py`
  - `scripts/audit_corpus.py`
  - `scripts/audit_dimensions.py`

Dependencies:
- `PLN-007` common experiment harness
- `PLN-009` feature-set and class-pair comparison
- `PLN-010` generic inference contract
- `PLN-011` generic feature taxonomy
- `PLN-012` classification evidence proof
- `PLN-013` generic filtering contract
- `PLN-014` dimensional lift audit
- `PLN-015` corpus coverage framework
- existing abstract inspection bundle, corpus adequacy artifacts, coverage reports, and advanced-filter decision surfaces

Recommended Work Phases:
1. `P0`: Repo hygiene and known gaps capture
2. `P1`: Artifact contract and manifest
3. `P2`: Common study runner finish
4. `P3`: Report generator
5. `P4`: Visualization gallery
6. `P5`: Corpus adequacy and leakage audit
7. `P6`: Feature taxonomy and transferability
8. `P7`: Class-pair and oracle separability
9. `P8`: Filtering and advanced-method decision
10. `P9`: Dimensional lift and 3D readiness
11. `P10`: Team packet export

Success Criteria:
- A team member can open `artifacts/showcase/index.md` and understand:
  - what problem the repo is solving
  - what algorithms are implemented
  - what studies were run
  - what the most important plots mean
  - what has been proven
  - what remains unproven
  - what would justify IMM, PF, or RBPF
  - how the repo transitions from 1D to 3D
  - how to rerun the work
- A team member can browse `artifacts/team_packet/` without needing repo-internal context.

Last Updated: 2026-05-24
