# PLN-018 Agentic Corpus Synthesis

Title: Agentic Corpus Synthesis, Quality-Diversity Search, And Adversarial Trajectory Discovery
Plan ID: PLN-018
Status: proposed
Owner: @rick
Priority: P1
Last Updated: 2026-05-24

Objective:
Add a generic corpus-synthesis layer on top of the existing motion generators so the repo can search for useful trajectories and useful corpora rather than only hand-configuring them. The new layer should support class-targeted generation, feature-targeted generation, quality-diversity archive filling, and adversarial/stress trajectory discovery, with reinforcement learning treated as an optional backend only after simpler search methods are proven.

Scope:
- Define a generic trajectory-search environment contract aligned to the existing generator, feature, classifier, and corpus-audit stack.
- Support target-conditioned corpus synthesis goals:
  - target class
  - target class pair
  - target feature cell
  - target difficulty
  - target failure mode
  - target prior sensitivity
  - target switching pattern
- Add non-RL search baselines first:
  - random search
  - grid / design-of-experiments search
  - rejection / scoring search
- Add quality-diversity archive logic to fill under-covered feature / difficulty / failure cells.
- Add adaptive stress generation for classifier failures, ambiguity, prior fragility, and filter mismatch cases.
- Add a comparison layer across manual generation, search baselines, quality-diversity, and stress search.
- Keep the design dimension-aware so it can later support 3D/vector trajectories without rethinking the whole selection loop.

Out of Scope:
- Jumping directly to deep RL before simpler search baselines exist.
- Replacing the existing trajectory generator; this plan adds a search layer on top of it.
- Claiming realistic operational data generation from adversarial search alone.
- Mixing nominal, boundary, adversarial, stress, and realistic corpus tiers into one unlabeled pool.
- Implementing a full 3D motion generator here; this plan should be 3D-ready in contract, not 3D-complete in physics.

Implementation Steps:
1. Define CorpusGym environment and scoring contracts.
   - Represent a search environment with:
     - `reset(target)`
     - `step(action)`
     - `simulate(parameterization)`
     - `trajectory()`
     - `score(trajectory)`
     - `render_diagnostics()`
   - Define target descriptors for:
     - class-conditioned generation
     - class-pair boundary generation
     - feature-cell filling
     - failure-mode search
     - prior-sensitive search
     - switching-pattern search
   - Define decomposed reward / score components:
     - class validity
     - feature excitation
     - coverage gain
     - boundary closeness
     - classifier stress
     - prior sensitivity
     - leakage penalty
     - physical invalidity penalty
2. Add the trajectory-parameter search baseline.
   - Use the current parameterized generator and corpus auditor.
   - Implement:
     - random search
     - DOE / Latin hypercube / Sobol-style sampling
     - rejection search on reward components
   - Preserve candidate trajectories and rejected candidates with their full score decompositions.
3. Add a quality-diversity archive.
   - Define archive cell axes over:
     - class or class pair
     - duration bucket
     - sample-count bucket
     - speed-range bucket
     - acceleration-range bucket
     - monotonicity bucket
     - posterior-entropy bucket
     - prior-sensitivity bucket
     - classifier error type
   - Maintain high-quality elites per cell rather than one global optimum.
   - Make archive coverage and novelty explicit first-class metrics.
4. Add an adaptive stress corpus layer.
   - Search for trajectories that expose:
     - wrong classification
     - high posterior entropy
     - small prior perturbation flip
     - raw-extrema failure
     - irregular-window failure
     - Kalman mismatch
     - switching delay
   - Require all stress cases to remain class-valid or be explicitly labeled invalid / rejected.
5. Add an RL backend decision gate.
   - Do not implement deep RL by default.
   - Create a decision report that states:
     - state space
     - action space
     - reward
     - episode definition
     - baseline to beat
     - success metric required to justify RL
     - current go / no-go decision
6. Add a corpus-synthesis comparison layer.
   - Compare:
     - manual generator
     - random search
     - DOE search
     - quality-diversity
     - adaptive stress search
     - optional RL backend
   - Score them on:
     - class balance
     - feature excitation
     - archive coverage
     - boundary coverage
     - leakage penalty
     - invalid trajectory rate
     - classifier error discovery rate
     - prior-sensitive case discovery rate
     - diversity / novelty

Validation:
- A CorpusGym-style environment contract exists and can reproduce a trajectory from seed and action / parameter sequence.
- Search-generated trajectories preserve class validity or are explicitly rejected / marked invalid.
- The search baseline produces candidate trajectories with higher score than a random-average baseline on at least one target objective.
- The quality-diversity archive increases coverage over iterations.
- The archive contains non-duplicate boundary and adversarial cells.
- Stress search finds trajectories with higher error or higher uncertainty than the random baseline.
- Leakage penalties prevent the selected corpus from drifting into class-linked duration / sample-count / noise artifacts.
- The RL backend decision report exists and explicitly states go / no-go.
- Corpus-synthesis comparison artifacts show which generator family produced the best:
  - corpus-quality result
  - archive-coverage result
  - stress-case result

Artifacts / Config:
- `docs/plans/PLN-018_agentic_corpus_synthesis.md`
- `artifacts/corpus_gym/environment_contract.json`
- `artifacts/corpus_gym/example_targets.json`
- `artifacts/corpus_gym/corpus_gym_report.md`
- `artifacts/corpus_search_baseline/search_config.yaml`
- `artifacts/corpus_search_baseline/generated_candidates.csv`
- `artifacts/corpus_search_baseline/candidate_scores.csv`
- `artifacts/corpus_search_baseline/selected_candidates.csv`
- `artifacts/corpus_search_baseline/search_baseline_report.md`
- `artifacts/quality_diversity_corpus/qd_config.yaml`
- `artifacts/quality_diversity_corpus/archive_cells.csv`
- `artifacts/quality_diversity_corpus/archive_elites.csv`
- `artifacts/quality_diversity_corpus/archive_coverage.csv`
- `artifacts/quality_diversity_corpus/qd_corpus_manifest.json`
- `artifacts/quality_diversity_corpus/qd_corpus_report.md`
- `artifacts/adaptive_stress_corpus/stress_search_config.yaml`
- `artifacts/adaptive_stress_corpus/stress_cases.csv`
- `artifacts/adaptive_stress_corpus/stress_case_scores.csv`
- `artifacts/adaptive_stress_corpus/stress_case_report.md`
- `artifacts/rl_corpus_agent/rl_backend_decision_report.md`
- `artifacts/corpus_synthesis_comparison/generator_comparison.csv`
- `artifacts/corpus_synthesis_comparison/corpus_quality_by_generator.csv`
- `artifacts/corpus_synthesis_comparison/feature_excitation_by_generator.csv`
- `artifacts/corpus_synthesis_comparison/classifier_stress_by_generator.csv`
- `artifacts/corpus_synthesis_comparison/corpus_synthesis_comparison_report.md`

Dependencies:
- `PLN-002` kinematic classification roadmap
- `PLN-003` corpus adequacy audit
- `PLN-007` common experiment harness
- `PLN-009` feature-set and class-pair comparison
- `PLN-011` generic feature taxonomy
- `PLN-014` dimensional lift audit
- `PLN-015` corpus coverage framework
- `PLN-017` automated methodology proof and LaTeX exposition
- existing modules:
  - `trajectory_generator.py`
  - `feature_analysis.py`
  - `corpus_adequacy_audit.py`
  - `corpus_autodevelopment.py`
  - `study_candidate_generation.py`
  - `validation_ladder.py`

Milestones:
- `M25`: CorpusGym environment contract
  - Goal: make trajectory-search targets and reward components explicit and reproducible.
  - Required outputs:
    - `artifacts/corpus_gym/environment_contract.json`
    - `artifacts/corpus_gym/example_targets.json`
    - `artifacts/corpus_gym/corpus_gym_report.md`
  - Exit criterion: a target can be reset, simulated, scored, and reproduced from seed plus action / parameter sequence.
- `M26`: Search baseline
  - Goal: prove simple search can generate better corpus candidates than unguided random generation.
  - Required outputs:
    - `generated_candidates.csv`
    - `candidate_scores.csv`
    - `selected_candidates.csv`
    - `search_baseline_report.md`
  - Exit criterion: selected candidates beat the random-average baseline on at least one target objective.
- `M27`: Quality-diversity archive
  - Goal: fill under-covered feature/difficulty/failure cells with diverse valid trajectories.
  - Required outputs:
    - `archive_cells.csv`
    - `archive_elites.csv`
    - `archive_coverage.csv`
    - `qd_corpus_manifest.json`
    - `qd_corpus_report.md`
  - Exit criterion: archive coverage increases and the archive contains valid non-duplicate elites.
- `M28`: Adaptive stress corpus
  - Goal: generate useful hard cases for classifier and filter analysis.
  - Required outputs:
    - `stress_cases.csv`
    - `stress_case_scores.csv`
    - `stress_case_report.md`
  - Exit criterion: stress search finds higher-error or higher-uncertainty cases than random search.
- `M29`: RL backend decision
  - Goal: decide whether RL is actually justified after the non-RL baselines exist.
  - Required outputs:
    - `artifacts/rl_corpus_agent/rl_backend_decision_report.md`
  - Exit criterion: the repo states a concrete RL go / no-go decision with explicit success criteria.
- `M30`: Corpus synthesis comparison
  - Goal: compare generator/search families on corpus utility rather than novelty alone.
  - Required outputs:
    - comparison CSVs and report
  - Exit criterion: the repo can identify the best current corpus-synthesis method for coverage, stress discovery, and validity.

Guardrails:
- Prevent reward hacking by keeping reward decomposed and inspectable.
- Penalize physically invalid or smoothness-violating trajectories explicitly.
- Keep nominal, boundary, adversarial, stress, and realistic tiers separated in manifests and reports.
- Preserve rejected trajectories and invalid cases for audit instead of silently dropping them.
- Use diversity / novelty controls so the generator does not repeat one exploitative trajectory family.
- Treat class validity as a first-class post-generation check, not an assumption.

Recommended First Execution Order:
1. `M25` CorpusGym contract
2. `M26` search baseline
3. `M27` quality-diversity archive
4. `M28` adaptive stress corpus
5. `M29` RL backend decision
6. `M30` corpus synthesis comparison

Success Criteria:
- The repo can search for useful trajectories and corpora instead of only hand-defining them.
- Feature-space coverage and corpus adequacy can improve through explicit search pressure.
- The repo can deliberately find informative boundary, adversarial, and prior-sensitive cases.
- RL remains a disciplined backend decision rather than a default starting point.
