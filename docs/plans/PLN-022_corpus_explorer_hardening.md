# PLN-022 Corpus Explorer Hardening

Title: Corpus Explorer Hardening
Plan ID: PLN-022
Status: proposed
Owner: @rick
Priority: P1
Last Updated: 2026-05-25

Objective:
Harden the PLN-021 objective-driven corpus explorer from a functional v1 proof into a more trustworthy automated corpus-development engine. The goal is to improve objective enforcement, leakage control, relabel-aware selection, closed-loop archive behavior, corpus partitioning, and package-level reproducibility so generated corpora can support stronger evaluation claims without silently mixing stress artifacts, out-of-pair relabels, and loosely satisfied objectives.

Scope:
- Add explicit objective-satisfaction scoring for generated candidates and selected corpora.
- Enforce leakage constraints during corpus selection rather than only declaring them in objective metadata.
- Split relabeled and invalid trajectories into distinct corpus partitions instead of silently allowing them into normal pairwise examples.
- Expand selected corpus products into separate demo, stress, and balanced evaluation corpora.
- Upgrade the archive from iterative fill over generated candidates toward archive-driven mutation and replacement.
- Improve package reproducibility and evaluation-packet clarity so review bundles are runnable or explicitly labeled as deltas.
- Preserve the existing CorpusGym-centered generation path and the common harness consumability proof from PLN-021.

Out of Scope:
- Deep RL policy training or policy optimization.
- Real 3D, 3DOF, 6DOF, or external high-fidelity backend integration.
- Claiming domain realism beyond the current 1D scope.
- Replacing the common study harness or the existing feature/classifier stacks.

Implementation Steps:
1. Add objective-satisfaction scoring.
   - Compute candidate-level and corpus-level satisfaction against:
     - feature excitation targets
     - posterior entropy targets
     - environment regime targets
     - backend constraints
     - runtime budget usage
     - leakage constraint limits
   - Make objective satisfaction a first-class surface in corpus selection and archive replacement.
2. Enforce leakage constraints during selection.
   - Measure objective-scoped leakage violations explicitly.
   - Prevent non-stress corpora from being selected when declared leakage limits are exceeded.
   - Preserve explicit stress exemptions only when the objective type or corpus partition allows them.
3. Add relabel-aware corpus partitioning.
   - Split generated examples into:
     - valid pair examples
     - ambiguous boundary examples
     - out-of-pair relabel examples
     - invalid rejected examples
   - Prevent out-of-pair relabels from silently entering pairwise accuracy metrics.
4. Produce multiple selected corpus products.
   - Materialize at least:
     - demo corpus
     - stress corpus
     - balanced evaluation corpus
   - Add minimum class and class-pair counts for the balanced evaluation corpus.
5. Upgrade the archive to a more closed-loop QD process.
   - Use archive elites to propose new mutations.
   - Track parent elite, mutation operator, child candidate, replacement decision, and objective-satisfaction gain.
   - Preserve separate successful and failed archive channels.
6. Improve reproducibility and packet execution.
   - Make evaluation packets runnable by themselves or clearly mark them as delta packets.
   - Include package-level exports, dependency context, and a reproducibility note in the packet.

Validation:
- Objective-satisfaction scoring exists and is non-trivial for selected and rejected candidates.
- Leakage constraint violations are measured and can block selection for non-stress corpora.
- At least one relabeled out-of-pair example is diverted away from normal pairwise evaluation.
- Balanced evaluation corpus meets declared minimum class or class-pair counts.
- Archive replacement is driven by previous elites, not only by a pre-generated candidate list.
- Archive coverage still increases over iterations after the closed-loop upgrade.
- Evaluation packet is either standalone runnable or contains an explicit delta/runbook statement.
- Targeted regression for the hardening plan passes.

Artifacts / Config:
- `docs/plans/PLN-022_corpus_explorer_hardening.md`
- `artifacts/objective_satisfaction/objective_satisfaction_scores.csv`
- `artifacts/objective_satisfaction/objective_satisfaction_report.md`
- `artifacts/objective_satisfaction/objective_constraint_coverage.csv`
- `artifacts/leakage_enforcement/leakage_constraint_violations.csv`
- `artifacts/leakage_enforcement/leakage_enforcement_report.md`
- `artifacts/relabel_partitions/valid_pair_examples.csv`
- `artifacts/relabel_partitions/ambiguous_boundary_examples.csv`
- `artifacts/relabel_partitions/out_of_pair_relabel_examples.csv`
- `artifacts/relabel_partitions/invalid_rejected_examples.csv`
- `artifacts/relabel_partitions/relabel_partition_report.md`
- `artifacts/selected_demo_corpus/`
- `artifacts/selected_stress_corpus/`
- `artifacts/selected_balanced_eval_corpus/`
- `artifacts/closed_loop_qd/archive_cells.csv`
- `artifacts/closed_loop_qd/archive_elites.csv`
- `artifacts/closed_loop_qd/archive_coverage_by_iteration.csv`
- `artifacts/closed_loop_qd/archive_mutation_lineage.csv`
- `artifacts/closed_loop_qd/archive_replacement_decisions.csv`
- `artifacts/closed_loop_qd/closed_loop_qd_report.md`
- `artifacts/evaluation_packet_reproducibility/reproducibility_manifest.json`
- `artifacts/evaluation_packet_reproducibility/packet_runbook.md`

Dependencies:
- `PLN-018` agentic corpus synthesis
- `PLN-019` multi-backend trajectory exploration
- `PLN-021` objective-driven corpus explorer v1
- existing modules:
  - `corpus_gym.py`
  - `objective_corpus_gym_runner.py`
  - `corpus_objectives.py`
  - `candidate_generation.py`
  - `class_validity.py`
  - `generated_corpus_features.py`
  - `corpus_classifier_scoring.py`
  - `objective_driven_qd_archive.py`
  - `selected_generated_corpus.py`
  - `common_experiment_harness.py`

Milestones:
- `M43`: Objective Satisfaction Scoring
  - Goal:
    - Turn declared objective fields into measured satisfaction scores.
  - Deliverables:
    - `artifacts/objective_satisfaction/objective_satisfaction_scores.csv`
    - `artifacts/objective_satisfaction/objective_constraint_coverage.csv`
    - `artifacts/objective_satisfaction/objective_satisfaction_report.md`
  - Visualizations:
    - objective satisfaction heatmap
    - objective satisfaction vs selected status plot
    - constraint coverage matrix
  - Exit criterion:
    - Selected and rejected candidates have explicit measured objective-satisfaction breakdowns, not only declared objective metadata.

- `M44`: Leakage Enforcement
  - Goal:
    - Enforce leakage constraints in corpus selection.
  - Deliverables:
    - `artifacts/leakage_enforcement/leakage_constraint_violations.csv`
    - `artifacts/leakage_enforcement/leakage_enforcement_report.md`
  - Visualizations:
    - leakage violation severity chart
    - selected vs rejected leakage comparison
  - Exit criterion:
    - Non-stress selections can be blocked by objective leakage violations.

- `M45`: Relabel-Aware Corpus Partitioning
  - Goal:
    - Keep relabeled and invalid trajectories from silently contaminating pairwise evaluation corpora.
  - Deliverables:
    - `artifacts/relabel_partitions/valid_pair_examples.csv`
    - `artifacts/relabel_partitions/ambiguous_boundary_examples.csv`
    - `artifacts/relabel_partitions/out_of_pair_relabel_examples.csv`
    - `artifacts/relabel_partitions/invalid_rejected_examples.csv`
    - `artifacts/relabel_partitions/relabel_partition_report.md`
  - Visualizations:
    - relabel partition breakdown
    - class-pair consistency matrix
  - Exit criterion:
    - Out-of-pair relabels are no longer counted as normal pairwise examples.

- `M46`: Multiple Selected Corpus Products
  - Goal:
    - Materialize separate corpus products for different uses.
  - Deliverables:
    - `artifacts/selected_demo_corpus/`
    - `artifacts/selected_stress_corpus/`
    - `artifacts/selected_balanced_eval_corpus/`
    - each corpus includes normalized trajectory, feature, class-validity, classifier-score, and posterior-history tables
  - Visualizations:
    - corpus product comparison dashboard
    - class balance plot
    - pair coverage plot
  - Exit criterion:
    - Balanced evaluation corpus satisfies declared minimum counts and is distinct from demo and stress corpora.

- `M47`: Closed-Loop QD Archive
  - Goal:
    - Make archive contents drive future candidate generation.
  - Deliverables:
    - `artifacts/closed_loop_qd/archive_cells.csv`
    - `artifacts/closed_loop_qd/archive_elites.csv`
    - `artifacts/closed_loop_qd/archive_coverage_by_iteration.csv`
    - `artifacts/closed_loop_qd/archive_mutation_lineage.csv`
    - `artifacts/closed_loop_qd/archive_replacement_decisions.csv`
    - `artifacts/closed_loop_qd/closed_loop_qd_report.md`
  - Visualizations:
    - archive coverage by iteration
    - parent-to-child mutation graph
    - replacement decision Sankey or transition chart
  - Exit criterion:
    - New candidate proposals are driven by current archive elites and mutation operators, not only by one-shot sampler output.

- `M48`: Evaluation Packet Reproducibility
  - Goal:
    - Make corpus evaluation packets reproducible and audit-friendly.
  - Deliverables:
    - `artifacts/evaluation_packet_reproducibility/reproducibility_manifest.json`
    - `artifacts/evaluation_packet_reproducibility/packet_runbook.md`
    - optionally a standalone runnable packet bundle
  - Visualizations:
    - packet contents overview
    - dependency graph
  - Exit criterion:
    - The packet is either runnable by itself or explicitly and unambiguously labeled as a delta with a runbook.

Recommended Execution Order:
1. `M43` objective satisfaction scoring
2. `M44` leakage enforcement
3. `M45` relabel-aware partitioning
4. `M46` multiple selected corpus products
5. `M47` closed-loop QD archive
6. `M48` evaluation packet reproducibility

Design Notes:
- Treat PLN-021 as the functional v1 explorer, not the final hardening pass.
- Keep `CorpusGym` as the single execution lane for generated corpus candidates wherever practical.
- Strengthen trustworthiness before adding more simulator backends or RL.
- Preserve explicit separation between stress-useful examples and clean evaluation examples.
- Prefer measured objective satisfaction and leakage enforcement over hand-tuned selection heuristics.

Success Criteria:
- Generated candidates are scored against objective satisfaction, not just generic utility.
- Leakage constraints affect selection decisions in a traceable way.
- Relabeled out-of-pair trajectories no longer silently distort pairwise evaluation.
- Closed-loop archive mutation is operating over current elites.
- Multiple selected corpus products exist for different use cases.
- Evaluation packets are reproducible enough for external review without ambiguity.
