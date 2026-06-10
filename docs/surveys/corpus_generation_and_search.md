# Corpus Generation, Evaluation, and Exploration

This note is the corpus-side pillar of the methodology stack. It is not just a
description of how trajectories are synthesized. It is meant to answer:

- what variables define a corpus candidate
- what objective function is being optimized
- how corpus quality is measured before classifiers are evaluated
- how adequacy pressure is turned into a scalar score and Pareto surface
- how corpus candidates become promoted studies
- which artifacts demonstrate those claims numerically

In the canonical repo story, this note is the `Corpus Explorer` pillar. It is
the part of the repo that turns an objective and backend into a selected corpus
that the Study Candidate Evaluator can trust.

## Current Support Status

The current strongest supported corpus-evaluation path is the repo's default generated/common-study corpora.

- default generated/common-study corpora are the strongest supported evaluation path
- selected generated corpus packet rerun is also strong and is the main closed-loop adequacy path
- the adapter/contract layer for provided or external corpus sources exists
- that adapter/contract layer is not yet a full arbitrary-corpus adequacy pipeline
- the repo should therefore not be described as offering full generic corpus evaluation for any provided corpus

## Scope and Relation to Other Documents

This document owns the corpus lifecycle:

- corpus objectives and candidate generation
- corpus adequacy and leakage evaluation
- archive-based exploration and selection
- CorpusGym-style execution and reward surfaces
- backend-aware planning for corpus search

It is intentionally narrower than the other two core documents:

- the methodology evaluation framework explains how to judge studies
- the classifier ladder explains how evidence providers generate posteriors
- this document explains how the corpus is produced, measured, and selected
  before those classifiers are asked to interpret it

## Corpus Explorer Contract

The generic corpus-side contract is:

```tex
(o, b, q, G_b) \mapsto D^\star,
```

where:

- `o` is the corpus objective
- `b` is the backend family
- `q` is the candidate proposal distribution
- `G_b` is the backend-specific generator
- `D^*` is the selected corpus after adequacy, leakage, and validity audits

In study terms, the explorer is upstream of the study candidate

```tex
s = (D^\star, f, C, m, \pi, b).
```

That is why this document is not merely about synthetic generation. It is about
corpus governance: which candidate corpora exist, which are rejected, which are
selected, and which study claims they can support.

## Corpus Evaluation Criteria

Before any study is promoted, the corpus itself must be judged on its own
merits. The corpus-side evaluation question is not whether the classifier is
already good; it is whether the generated corpus is broad, valid, auditable,
and hard in the intended way.

The most important corpus-level criteria are:

- class balance and class-pair balance
- boundary coverage and ambiguity pressure
- feature excitation over the active feature set
- difficulty diversity across tiers and regimes
- leakage control from duration, noise, sampling, or environment
- degeneracy control so the corpus does not collapse to trivial repeats
- provenance completeness so the selected corpus is reproducible

These checks happen before classifier conclusions are trusted. A strong
classifier result on a leaky or trivial corpus is not a strong study.

A useful corpus-evaluation summary vector is:

```tex
\mathbf{m}_k
=
\big[
    B_k,\,
    C_k,\,
    F_k,\,
    D_k,\,
    1-L_k,\,
    1-T_k,\,
    1-G_k,\,
    P_k
\big]
```

where `P_k` is provenance completeness. The corpus score in
`corpus_autodevelopment.py` is one concrete scalarization of that vector,
while the archive and selected-corpus artifacts preserve the non-scalarized
tradeoffs.

## 1. Problem Statement

The corpus layer exists to solve a methodological problem, not only a data
generation problem:

```tex
\text{How do we generate and select datasets that are informative enough to test
classification methods without making the task trivial or biased?}
```

That means corpus generation must be tied to explicit objectives rather than
only to “more synthetic trajectories.”

## 2. Global Notation

The main objects are:

- `tau`: a generated trajectory
- `D_k`: corpus candidate `k`
- `D^*`: selected corpus after audit and selection
- `theta_class`: class-specific motion parameters
- `theta_tier`: difficulty-tier controls
- `theta_noise`: corruption parameters
- `theta_sampling`: timing and sample-count parameters
- `S_k`: scalar score for corpus candidate `k`
- `o_k`: Pareto objective vector for candidate `k`

The front-door corpus story is:

```tex
\theta_k \sim q(\theta \mid o, b), \qquad
\tau_i \sim G_b(\theta_k, \xi_i), \qquad
D_k = \{\tau_i\}_{i=1}^{N_k}, \qquad
D^\star = \operatorname*{select}_k(D_k \mid S_k, o_k, \text{adequacy gates}).
```

The selected corpus is therefore not merely the most recently generated one. It
is the candidate that survives the declared score, Pareto, and gate logic.

## 3. Trajectory Parameterization and Witness Problems

### 3.1 Problem

`trajectory_generator.py` and `corpus_objectives.py` define the base witness
problems used by the rest of the repo.

### 3.2 Assumptions

The core assumptions are:

- class semantics can be represented by parameterized families of trajectories
- difficulty tiers can be represented by controlled changes in noise,
  irregularity, outliers, and step counts
- the induced synthetic geometry is meaningful enough to support feature,
  posterior, and adequacy studies

### 3.3 Implementation Mapping

- `trajectory_generator.py`
- `corpus_objectives.py`
- objective YAML in
  `experiments/corpus_objectives/common_1d_corpus_objectives.yaml`

### 3.4 Why This Matters

Every later artifact inherits the geometry defined here. If the synthetic class
definitions are weak or biased, no downstream classifier result is trustworthy.

## 4. Corpus-Shaping Layers

The repo has several modules that perturb or search the corpus distribution:

- `adaptive_stress_corpus.py`
- `environment_aware_corpus.py`
- `quality_diversity_corpus.py`
- `objective_driven_qd_archive.py`

At a high level, they move from one corpus distribution to another:

```tex
\mathcal{D}
\rightarrow
\mathcal{D}'(\text{noise}, \text{irregularity}, \text{outliers}, \text{stress}, \text{diversity}).
```

These are not separate data silos. They are search directions over corpus
properties that may reveal different classifier or feature failures.

## 5. Corpus Autodevelopment

### 5.1 Problem

`corpus_autodevelopment.py` asks:

```tex
\text{Can the repo score and select between multiple candidate corpora using declared adequacy goals?}
```

### 5.2 Score Construction

For candidate `k`, the implemented scalar score is:

```tex
S_k
= B_k + C_k + F_k + D_k - L_k - T_k - G_k,
```

where:

- `B_k`: class-balance score
- `C_k`: class-pair boundary coverage score
- `F_k`: feature-excitation score
- `D_k`: difficulty-diversity score
- `L_k`: leakage penalty
- `T_k`: triviality penalty
- `G_k`: degeneracy penalty

This is not merely conceptual. These terms are computed explicitly by:

- `_balance_score(...)`
- `_boundary_coverage_score(...)`
- `_feature_excitation_score(...)`
- `_difficulty_diversity_score(...)`
- `_leakage_penalty(...)`
- `_triviality_penalty(...)`
- `_degeneracy_penalty(...)`

### 5.3 Pareto Surface

The same module also defines a vector objective:

```tex
\mathbf{o}_k =
\big[
  B_k,\,
  C_k,\,
  F_k,\,
  D_k,\,
  -L_k,\,
  -T_k,\,
  -G_k
\big].
```

A candidate is dominated when another candidate is no worse in every coordinate
and strictly better in at least one. This is the actual meaning of the Pareto
front in the code.

### 5.4 Assumptions

The score assumes:

- the positive terms should be maximized
- the penalties should be minimized
- a single scalar score is useful for selection
- but the Pareto front should still be preserved so non-dominated tradeoffs are
  not erased

### 5.5 Worked Example

The numeric artifact
[corpus_autodevelopment_numeric_walkthrough.md](artifacts/corpus_autodevelopment_v1/corpus_autodevelopment_numeric_walkthrough.md)
is the current concrete proof for this section. It shows one real selected
candidate and:

- substitutes the actual score-term values into `S_k`
- expands the difficulty-diversity subscore against the configured target
  fractions
- shows leakage-threshold rows
- compares the selected candidate against the highest-scoring rejected one
- explains why selection and Pareto non-dominance are not the same claim

That artifact is the proper bridge from symbolic objective to implementation.

## 6. Corpus Search and Baseline Ranking

The broader search layer is implemented through:

- `corpus_search_baseline.py`
- `corpus_synthesis_comparison.py`
- `generic_corpus_exploration.py`
- `selected_generated_corpus.py`

The methodological statement here is:

```tex
\text{corpus search}
\neq
\text{generate more random trajectories}.
```

Instead, the repo is moving toward objective-driven exploration over corpus
properties that affect identifiability, calibration, leakage, and robustness.

### 6.1 Generic Corpus Explorer

`generic_corpus_exploration.py` is the clearest implementation of the repo’s
Corpus Explorer idea. It starts from a heterogeneous candidate pool of
backend-specific trajectory specifications and scores each executed run by a
normalized utility rather than by one raw classification metric.

For one executed run, the implemented explorer utility is:

```tex
U_{\text{explore}}
= 0.22 \cdot \text{validity}
+ 0.18 \cdot \text{coverage novelty}
+ 0.18 \cdot \text{boundary score}
+ 0.18 \cdot \text{classifier stress}
+ 0.12 \cdot \text{environment score}
+ 0.12 \cdot \text{provenance completeness}.
```

This score is intentionally mixed. It rewards:

- validity of the executed run
- novelty of the candidate’s coverage cell
- pressure on known class boundaries
- stress placed on current classifier families
- usefulness of environment-regime structure
- preservation of provenance metadata for later audit

The utility is easier to interpret if written as a short symbol map:

```tex
U_{\text{explore}} = 0.22 V + 0.18 N + 0.18 B + 0.18 S + 0.12 E + 0.12 P.
```

That is the form used by the numeric walkthrough artifact, which substitutes
the concrete values for `V`, `N`, `B`, `S`, `E`, and `P` before comparing the
selected row with a random baseline.

So the explorer is not only asking “which trajectory is hardest?” It is asking
“which executed trajectories make the corpus more useful as a study object?”

### 6.2 Explorer Archive and Selection Logic

The explorer also constructs archive cells over:

- backend
- scenario family
- target class
- difficulty tier

For each cell, the elite is the run with maximal `total_utility`. The selected
corpus is then compared against a same-size random baseline by coverage:

```tex
\Delta_{\text{coverage}}
=
\#\{\text{selected archive cells}\}
-
\#\{\text{random-baseline archive cells}\}.
```

If `h(tau)` is the archive-cell map for a trajectory `tau`, then the selected
elite is

```tex
A[h(\tau)]
\leftarrow
\arg\max_{\tau' : h(\tau') = h(\tau)} U_{\text{explore}}(\tau').
```

That gives the explorer a clear audit question: does the selected corpus cover
more useful behavioral cells than a naïve random sample of equal size?

### 6.3 Worked Example

The numeric artifact
[generic_corpus_explorer_numeric_walkthrough.md](artifacts/generic_corpus_exploration/generic_corpus_explorer_numeric_walkthrough.md)
now expands one selected corpus row into:

- its `U_explore` utility decomposition
- its archive-cell role
- its elite interpretation
- its contribution to the selected-versus-random coverage comparison

That artifact is the Explorer-side proof that corpus selection is not only a
visual dashboard. It is a numeric utility and coverage argument on one concrete
selected row.

## 7. Corpus Gym

### 7.1 Problem

`corpus_gym.py` reframes corpus generation as a targeted environment rather than
just a batch sampler. The question is:

```tex
\text{Can we specify desired failure pressure or feature geometry and reward matching trajectories?}
```

### 7.2 Variables

The main objects are:

- `target`: a desired class, class pair, feature cell, failure mode, or prior-sensitive regime
- `action`: a parameterized perturbation of the base tier
- `reward`: a structured utility decomposition
- `episode`: `(target, action, trajectory, diagnostics, reward)`

### 7.3 Reward Construction

The reward in `CorpusGymReward` is multi-term rather than monolithic:

- `class_validity`
- `feature_excitation`
- `coverage_gain`
- `boundary_closeness`
- `classifier_stress`
- `prior_sensitivity`
- `leakage_penalty`
- `physical_invalidity_penalty`

The actual implemented utility is:

```tex
U_{\text{gym}}
= 0.22 \cdot V
+ 0.14 \cdot E
+ 0.14 \cdot G
+ 0.14 \cdot B
+ 0.14 \cdot S
+ 0.12 \cdot P
- 0.10 \cdot L
- 0.14 \cdot I,
```

where:

- `V`: class validity
- `E`: feature excitation match
- `G`: coverage gain
- `B`: boundary closeness
- `S`: classifier-stress pressure
- `P`: prior-sensitivity pressure
- `L`: leakage penalty
- `I`: physical invalidity penalty

The current code computes the component terms explicitly through:

- `_class_validity_score(...)`
- `_feature_excitation_score(...)`
- `_coverage_gain_score(...)`
- `_boundary_closeness_score(...)`
- `_classifier_stress_score(...)`
- `_prior_sensitivity_score(...)`
- `_leakage_penalty(...)`
- `_physical_invalidity_penalty(...)`

So the Gym is already a concrete objective function, not just an idea for one.

### 7.4 Coverage, Leakage, and Invalidity Terms

Three subterms are especially important because they prevent the Gym from
rewarding obviously biased or pathological cases.

The coverage-gain term is:

```tex
G
= 0.40 \cdot \text{tier match}
+ 0.30 \cdot \text{class match}
+ 0.30 \cdot \text{novelty}(a),
```

where the novelty term is the average of capped measurement, irregularity,
outlier, and step scales induced by the action.

The leakage penalty is:

```tex
L
= 0.30 \cdot \text{duration risk}
+ 0.30 \cdot \text{sample-count risk}
+ 0.40 \cdot \text{noise risk}.
```

This is how the Gym avoids rewarding trajectories merely because duration,
sample count, or noise level become class-identifying shortcuts.

The physical-invalidity penalty guards against:

- non-increasing time grids
- implausibly large absolute accelerations

So the Gym can search aggressively without silently drifting into invalid
trajectory geometry.

### 7.5 Environment Contract

`CorpusGymEnvironment` makes the corpus-search loop explicit. It provides:

- `reset(target)`
- `simulate(action)`
- `step(action)`
- `trajectory()`
- `score(trajectory)`
- `render_diagnostics(trajectory)`

Methodologically, the Gym can therefore be written as:

```tex
\text{target}
\xrightarrow{\text{reset}}
\text{state}
\xrightarrow{\text{action}}
\text{trajectory, reward, diagnostics}.
```

That matters because it makes targeted corpus search a reusable interface rather
than one hard-coded artifact generator.

### 7.6 Objective-to-Gym Execution Bridge

`objective_corpus_gym_runner.py` ties the declarative corpus-objective layer to
the Gym layer. It defines the maps:

```tex
\Psi_{\text{target}}(\text{objective}) \rightarrow \text{CorpusGymTarget},
\qquad
\Psi_{\text{action}}(\text{candidate}) \rightarrow \text{CorpusGymAction}.
```

Then the actual execution chain becomes:

```tex
\text{objective}
\rightarrow
\text{candidate}
\rightarrow
\text{target, action}
\rightarrow
\text{episode}
\rightarrow
\text{validated trajectory run}.
```

This is the critical bridge from declarative study design to executed explorer
or gym records. Without it, the Gym would be an isolated search environment.

### 7.7 Why This Matters

This turns “find me a hard example” into a measurable optimization target. The
corpus gym is therefore the bridge from declarative corpus goals to targeted
trajectory proposals.

### 7.8 Worked Example

The numeric artifact
[corpus_gym_numeric_walkthrough.md](artifacts/corpus_gym/corpus_gym_numeric_walkthrough.md)
now works through one real Gym episode. It shows:

- the selected target
- the actual action scales
- the resulting trajectory diagnostics
- the implemented reward equation
- the numeric substitution for each reward component

That artifact is the Gym-side proof that the search reward is not just
qualitative prose. It is a concrete score decomposition on one executed
trajectory.

## 8. Objective-Driven Quality-Diversity Archive

### 8.1 Problem

`objective_driven_qd_archive.py` asks a different question from the scalar
autodevelopment score:

```tex
\text{How do we preserve diverse elites across multiple difficulty and evidence cells instead of selecting only one best corpus?}
```

### 8.2 Archive Utility

For one successful archive cell, the current elite utility is assembled from:

```tex
U_{\text{archive}}
= 0.30 \cdot \text{validity}
+ 0.25 \cdot \text{acceleration-range pressure}
+ 0.25 \cdot \text{classifier stress}
+ 0.20 \cdot (1 - \text{mean margin}).
```

The cell definition itself also depends on discretized buckets over:

- assigned class
- difficulty tier
- backend
- duration
- acceleration range
- entropy
- prior-flip threshold

So the archive is not only ranking trajectories. It is preserving structured
coverage over behaviorally distinct cells.

### 8.3 Implementation Mapping

- `objective_driven_qd_archive.py`
- `generated_corpus_features.py`
- `corpus_classifier_scoring.py`

### 8.4 Methodological Use

This layer matters because it separates:

- “best current elite in a cell”
- “which cells have been covered at all”
- “which mutation lineages produce useful diversity”

That is a stronger corpus-search story than one scalar winner alone.

The implementation also makes the archive utility operational:

```tex
U_{\text{archive}}
=
0.30 \cdot \text{validity score}
+ 0.25 \cdot \min\!\left(\frac{\text{accel range}}{0.40}, 1\right)
+ 0.25 \cdot \text{max classifier stress}
+ 0.20 \cdot (1 - \text{mean posterior margin}).
```

So the archive preserves high-validity, high-stress, low-margin witnesses in
distinct cells rather than preserving diversity in the abstract.

Successful and failed archive cells are tracked separately. If `tau_t` is the
trajectory processed at iteration `t`, then

```tex
A_t^{\text{succ}}[h(\tau_t)]
\leftarrow
\arg\max_{\tau' : h(\tau') = h(\tau_t)}
U_{\text{archive}}(\tau')
```

only when the run succeeds and the label status is `valid_target_class`;
otherwise the failed-cell counter is updated in `A_t^{fail}`. The emitted
coverage curves are

```tex
C_t^{\text{succ}}
=
\frac{|A_t^{\text{succ}}|}{|A_T^{\text{succ}}|},
\qquad
C_t^{\text{fail}}
=
\frac{|A_t^{\text{fail}}|}{|A_T^{\text{fail}}|}.
```

That is why invalid or failed runs do not inflate successful coverage.

### 8.5 Quality-Diversity Corpus Layer

`quality_diversity_corpus.py` implements a lighter-weight archive on top of
CorpusGym episodes. Its cell key is

```tex
h_{\text{qd}}(\tau)
=
\big(
    c(\tau),
    \mathrm{tier}(\tau),
    b_{\mathrm{dur}}(\tau),
    b_{\mathrm{acc}}(\tau),
    b_{\mathrm{turn}}(\tau)
\big),
```

where the last three coordinates are duration, acceleration-range, and
direction-change buckets. The elite replacement rule is the same `argmax` rule
as above, but the utility is the episode reward `U_gym` rather than
`U_archive`. The current coverage fraction is

```tex
\mathrm{coverage\_fraction}_t
=
\frac{\#\{\text{filled QD cells at iteration } t\}}{81},
```

because the current 1D archive discretizes `3 x 3 x 3 x 3` regime
combinations across the non-class axes.

## 8A. Corpus Hyperparameter Policy And Tuning

The corpus-search layer now has an explicit hyperparameter surface in
`corpus_policy.py` and `corpus_policy_sweep.py`. A policy is

```tex
p
=
\big(
    w^{+},\,
    w^{-},\,
    w^{\text{explore}},\,
    w^{\text{gym}},\,
    w^{\text{archive}},\,
    n,\,
    g
\big),
```

where `w+` are positive corpus weights, `w-` penalty weights, `w^explore`
generic-explorer weights, `w^gym` CorpusGym weights, `w^archive` archive
weights, `n` sampler budgets, and `g` adequacy gates.

Whenever a weight group is normalized, the implementation applies

```tex
\bar{w}_r
=
\frac{w_r}{\sum_u w_u},
```

so the scalar objectives remain comparable under reweighting. This is the role
of `normalize_corpus_policy_spec()`.

For policy `p`, the generic-explorer utility becomes

```tex
U_{\text{explore}}^{(p)}
=
\sum_{r \in \mathcal{R}_{\text{explore}}}
\bar{w}^{(p)}_r u_r,
```

the archive utility becomes

```tex
U_{\text{archive}}^{(p)}
=
\sum_{r \in \mathcal{R}_{\text{archive}}}
\bar{w}^{(p)}_r a_r,
```

and the sampler mixture becomes

```tex
\pi_s^{(p)}
=
\frac{n_s^{(p)}}{\sum_{s'} n_{s'}^{(p)}}.
```

So a policy changes both how candidates are scored and how search effort is
allocated.

The current tuning sweep evaluates a policy by a downstream adequacy proxy:

```tex
A_{\text{policy}}(p)
=
0.25 \cdot \text{validity}
+ 0.20 \cdot \text{boundary coverage}
+ 0.20 \cdot \min\!\left(\frac{\text{feature excitation}}{1.5}, 1\right)
+ 0.15 \cdot \text{classifier stress}
+ 0.20 \cdot \text{provenance completeness}
- 0.20 \cdot \text{leakage},
```

followed by the bounded policy score

```tex
J_{\text{policy}}(p)
=
\operatorname{clip}
\big(
    A_{\text{policy}}(p) + 0.10 \cdot \text{classifier stress},
    0,
    1
\big).
```

This is the quantity emitted as `policy_score` in the sweep results.

There is now a numeric walkthrough artifact for one real recommended policy
row:

- [corpus_policy_numeric_walkthrough.md](artifacts/corpus_hyperparameter_tuning_v1/corpus_policy_numeric_walkthrough.md)

That artifact expands the adequacy proxy, stress bonus, selected-set Jaccard,
rank stability, and dev-vs-holdout comparison numerically for the recommended
policy.

The sweep also checks whether a better score is only a re-ranking accident. For
selected sets `S(p_a)` and `S(p_b)`, the stability metric is

```tex
J_{\text{set}}(p_a, p_b)
=
\frac{|S(p_a) \cap S(p_b)|}{|S(p_a) \cup S(p_b)|}.
```

Rank stability is then reported through Spearman and Kendall correlations of
the ranked candidate lists. The policy question is therefore not only “which
weights maximize one scalar?” but also “which weights preserve a stable,
scientifically sensible selected set?”

## 9. Study Candidate Generation

### 9.1 Problem

Once corpus candidates exist, the next question is not “which dataset is best
in the abstract?” but “which study should the team run next?”

The study-candidate layer is implemented across:

- `candidate_generation.py`
- `capability_aware_search.py`
- `study_candidate_generation.py`
- `study_candidate_protocol.py`

### 9.2 Variables

For a candidate study built from:

- class pair `p`
- classifier `m`
- feature set `f`
- prior regime `r`

the current study-selection logic defines a static screening score
`Q_static(p,m,f,r)` before any Monte Carlo acceptance score is considered:

```tex
Q_{\text{static}}(p,m,f,r)
= 0.18 \cdot \text{feature-class compatibility}
+ 0.18 \cdot \text{expected separability}
+ 0.14 \cdot \text{classifier fit}
+ 0.14 \cdot \text{corpus coverage}
+ 0.12 \cdot \text{dimensional transfer}
+ 0.12 \cdot \text{implementation readiness}
+ 0.12 \cdot (1 - \text{dependency risk})
- 0.10 \cdot \text{cumulative-history risk}
- 0.10 \cdot \text{prior-sensitivity risk}.
```

In the current code, that high-level score is instantiated through named
subterms such as:

- `feature_class_compatibility_score`
- `expected_separability_score`
- `classifier_assumption_fit`
- `corpus_coverage_score`
- `dimensional_transfer_score`
- `implementation_readiness_score`
- `feature_dependency_risk`
- `cumulative_double_counting_risk`
- `prior_sensitivity_risk`

The important point is that these are not placeholders. They are the actual
named columns written into the `static_candidate_scores.csv` artifact by
`study_candidate_generation.py`.

### 9.3 Assumptions

The current static score assumes:

- feature/class compatibility and oracle separability are the strongest early
  evidence for whether a study is worth running
- corpus coverage and classifier-family fit should matter, but not dominate
- 3D transferability and implementation readiness should influence prioritization
  before the repo is fully 3D-ready
- dependency growth, cumulative-history reuse, and prior fragility are real
  methodological risks and should subtract from the screening score

It also assumes a two-stage process:

```tex
\text{static screening} \rightarrow \text{Monte Carlo confirmation} \rightarrow \text{promotion decision}.
```

That is a stronger claim than “rank everything once.” It means the repo is
already distinguishing between proposal quality before execution and evidence
quality after execution.

### 9.4 Monte Carlo Confirmation Layer

After static screening, the module uses cross-method metrics from
`analyze_common_experiment(...)` to compute a second score:

```tex
Q_{\text{mc}}
= 0.60 \cdot \text{accuracy}
+ 0.25 \cdot (1 - \text{prior flip fraction})
+ 0.15 \cdot (1 - \max(0, \text{oracle gap})).
```

This is the acceptance surface that the current code actually uses to decide
whether a study is promoted, revised, or rejected once benchmark evidence
exists.

The current promote gate is approximately:

```tex
\text{compatible}
\land
Q_{\text{static}} \ge 0.45
\land
Q_{\text{mc}} \ge 0.90
\land
\text{accuracy} \ge 0.83
\land
\text{prior flip fraction} \le 0.12.
```

If the feature set is compatible and the accuracy is merely decent, the
decision is usually `revise`; otherwise the decision falls to `reject` or
`defer`. This is the point where the repo stops being a proposal generator and
starts acting like an evidence-gated study selector.

### 9.5 Implementation Mapping

- `candidate_generation.py`
  - sampler families: random, grid, LHS, boundary mutation, archive mutation,
    stress mutation
- `capability_aware_search.py`
  - backend-aware search-method planning and runtime-budget logic
- `study_candidate_generation.py`
  - static score assembly, Monte Carlo lookup, and decision bucketing
- `study_candidate_protocol.py`
  - schema and validation-ladder contract for promoted studies

The lower-level sampler lives in `candidate_generation.py`; the promotion logic
lives in the higher-level study-candidate modules.

The current generated tables for this layer are not generic placeholders. They
include:

- `generated_study_candidates.json`
- `static_candidate_scores.csv`
- `monte_carlo_candidate_scores.csv`
- `promoted_candidates.csv`
- `rejected_candidates.csv`

That means the score decomposition and the decision vocabulary can already be
audited without reading source code.

### 9.6 Why This Matters

This is the point where corpus logic, feature logic, classifier logic, and
readiness logic start to interact. That is why this layer belongs in the
methodology docs, not only in artifact indexes.

## 10. Candidate Generation as a Sampler Family

`candidate_generation.py` does not emit one candidate per objective. For an
objective `o` and backend `b`, it effectively builds a candidate population

```tex
\mathcal{C}(o,b)
=
\mathcal{C}_{\text{random}}
\cup
\mathcal{C}_{\text{grid}}
\cup
\mathcal{C}_{\text{lhs}}
\cup
\mathcal{C}_{\text{boundary mutation}}
\cup
\mathcal{C}_{\text{archive mutation}}
\cup
\mathcal{C}_{\text{stress mutation}}.
```

That means candidate generation is already a search policy:

- `random` provides broad stochastic perturbation
- `grid` provides deterministic baseline coverage
- `lhs` provides space-filling parameter coverage
- `boundary_mutation` pushes toward harder near-boundary proposals
- `archive_mutation` performs local search around previously promising cells
- `stress_mutation` deliberately increases corruption or compression pressure

In probabilistic language, the module is not using one proposal distribution. It
is using a mixture over search heuristics:

```tex
q(c \mid o,b)
= \sum_s \pi_s \, q_s(c \mid o,b),
```

where `s` ranges over sampler families and the mixture weights are implemented
implicitly through per-family candidate budgets rather than estimated online.

This matters because the search surface is already hybrid:

- broad coverage samplers explore
- mutation samplers exploit
- stress samplers deliberately push toward failure regimes

## 11. Capability-Aware Search Planning

`capability_aware_search.py` determines which search methods should be used for
which backend family. The planner conditions on backend capability attributes
such as:

- runtime class
- dimensionality
- environment support
- sequential-control support
- stochastic versus deterministic execution

At the methodology level, that planner is a map

```tex
\Pi(\text{backend capabilities})
\rightarrow
\{\text{recommended search methods}, \text{budget class}, \text{planner rationale}\}.
```

This is important because the repo is no longer assuming that all backends
should be searched the same way.

The implemented rule set is not abstract. It contains explicit branches for:

- runtime class: `cheap`, `medium`, `expensive`
- environment support
- sequential-control support
- stochastic versus deterministic execution

Operationally, the current rule families behave like

```tex
M_{\text{rt}}(\kappa)
=
\begin{cases}
    \{\text{random},\text{lhs},\text{sobol},\text{qd}\}, & \kappa_{\text{runtime}}=\text{cheap}, \\
    \{\text{lhs},\text{sobol},\text{qd}\}, & \kappa_{\text{runtime}}=\text{medium}, \\
    \{\text{small DOE},\text{surrogate},\text{active learning}\}, & \kappa_{\text{runtime}}=\text{expensive},
\end{cases}
```

and

```tex
M_{\text{ctl}}(\kappa)
=
\begin{cases}
    \{\text{adaptive stress},\text{cross entropy}\}, & \kappa_{\text{seq}}=1, \\
    \varnothing, & \kappa_{\text{seq}}=0.
\end{cases}
```

with analogous environment- and stochasticity-dependent additions. So search
planning is already encoded as a capability map, not a loose recommendation
paragraph.

So the actual planner is closer to:

```tex
\Pi(\kappa)
=
\big(
M_{\text{runtime}}(\kappa),
M_{\text{environment}}(\kappa),
M_{\text{control}}(\kappa),
M_{\text{stochastic}}(\kappa)
\big),
```

where `kappa` is the backend capability vector and the resulting method set is
deduplicated into one backend plan row. Cheap stochastic backends receive broad
search budgets; expensive deterministic backends are pushed toward smaller DOE,
surrogate assistance, and cache-priority execution.

## 12. Study Candidate Protocol and Validation Ladder

`study_candidate_protocol.py` defines the contract that the generated-candidate
layer must satisfy. It specifies:

- the `StudyCandidate` schema
- the `ValidationLadder` schema
- the required terminal decision vocabulary: `promote`, `revise`, `reject`, `defer`

The promotion story is therefore not just:

```tex
\text{good score} \Rightarrow \text{promote}.
```

It is:

```tex
\text{candidate specification}
\rightarrow
\text{validation ladder evidence}
\rightarrow
\text{terminal decision in a constrained vocabulary}.
```

That is what makes the study layer auditable rather than ad hoc.

The protocol is also stronger than a single schema file. It defines two linked
objects:

- `StudyCandidate`
- `ValidationLadder`

The validation ladder itself has ordered levels, including:

- static compatibility
- corpus adequacy
- feature separability
- oracle separability
- classifier performance
- posterior and calibration quality
- prior sensitivity
- stress and adversarial robustness
- dimensional transfer assessment
- promotion decision

So the real promotion contract is:

```tex
\text{proposal}
\rightarrow
\{\ell_1, \ell_2, \dots, \ell_{10}\}
\rightarrow
d,
\qquad
d \in \{\text{promote}, \text{revise}, \text{reject}, \text{defer}\}.
```

This matters because it prevents a high static score from bypassing evidence,
and it prevents a visually interesting candidate from being promoted without an
explicit decision trail.

## 13. Generated Class/Feature Exploration

### 13.1 Generated Corpus Features

`generated_corpus_features.py` routes objective-driven generated trajectories
back through the real feature pipeline. This is important because generated
candidates are not only evaluated by proxy score columns. They are turned into
real `TrajectoryArtifact` rows, relabeled through class-validity logic, grouped
into tier datasets, and fed to `analyze_feature_datasets(...)`.

Methodologically, the feature pipeline therefore becomes:

```tex
\text{generated candidate}
\rightarrow
\text{executed trajectory}
\rightarrow
\text{validity-adjusted label}
\rightarrow
\text{feature row}
\rightarrow
\text{excitation and separability analysis}.
```

### 13.2 Corpus-Conditioned Classifier Scoring

`corpus_classifier_scoring.py` then asks how the current classifier ladder
behaves on those generated and relabeled trajectories. It rebuilds pointwise,
accumulator, windowed, and Kalman-family scoring surfaces and tracks:

- posterior entropy
- top-two posterior margin
- confident errors
- time to confidence
- measured classifier stress
- prior-flip sensitivity

The classifier-stress proxy can be read as:

```tex
\text{stress}
\approx
0.5 \cdot (1 - \text{margin})
+ 0.5 \cdot \text{entropy}
+ 0.35 \cdot \mathbf{1}\{\text{final prediction wrong}\}.
```

That makes this layer the bridge from corpus search to actual classifier
pressure. A generated corpus candidate is valuable only if it changes the
downstream evidence and decision landscape in an interpretable way.

## 14. End-to-End Corpus Methodology Flow

The current intended flow is:

```tex
\text{trajectory generator}
\rightarrow
\text{corpus candidate}
\rightarrow
\text{adequacy-scored corpus}
\rightarrow
\text{backend-aware candidate population}
\rightarrow
\text{study candidate}
\rightarrow
\text{validation ladder}
\rightarrow
\text{promotion / revise / reject / defer}.
```

This is the most important interpretation point in the file: synthetic witness
problems are not the final product. They are inputs to a reusable study-design
loop.

## 15. Failure Modes

The corpus-side methodology can still fail in several ways:

- the synthetic class families may be too stylized
- the difficulty tiers may not match the claimed hard boundaries
- the scalar score may hide a meaningful Pareto tradeoff
- study promotion may overvalue convenience and undervalue scientific pressure

That is why the worked example, the adequacy audit, and the search artifacts
must be read together.

## 16. What This Document Proves

This note is complete only if it supports the following claims:

- corpus generation is tied to explicit variables and objectives
- corpus autodevelopment has a formal score and Pareto definition
- the score terms have code-level implementations
- at least one real candidate score has been decomposed numerically
- study promotion is treated as a methodology problem, not only as a report
  listing
