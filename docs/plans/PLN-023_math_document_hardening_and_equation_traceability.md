# PLN-023 Math Document Hardening And Equation Traceability

Title: Math Document Hardening And Equation Traceability
Plan ID: PLN-023
Status: proposed
Owner: @rick
Priority: P1
Last Updated: 2026-05-25

Objective:
Convert the current methodology PDFs from architecture and synthesis documents into standalone algorithm and math references. The goal is to make the classification, filtering, corpus-search, and evaluation math reproducible without opening the code, while still preserving explicit traceability from equations to implementation, tests, and generated artifacts.

Scope:
- Add a shared symbol glossary and equation registry for the methodology document stack.
- Harden the current methodology and corpus PDFs with fuller definitions, assumptions, normalization rules, and worked numeric substitutions.
- Add a code-to-equation crosswalk so every implemented equation points to code, tests, and artifacts.
- Split stable math content into dedicated documents for foundations, classifier ladder math, corpus-search math, and numeric walkthroughs.
- Upgrade figure captions, tables, and notation so the PDFs are readable as standalone technical references rather than only roadmap-style papers.
- Make equation coverage auditable through generated registry and crosswalk artifacts.

Out of Scope:
- Rewriting the underlying classifier, filter, feature, or corpus-generation implementations unless a documentation gap reveals a real implementation inconsistency.
- Claiming new algorithmic capability beyond what the code already implements.
- Replacing the survey-document stack created in `PLN-020`; this plan hardens and reorganizes it.
- Adding IMM, PF, or RBPF implementations before the current decision gates justify them.

Implementation Steps:
1. Add a shared math foundation layer.
   - Create `docs/math/symbol_glossary.tex`.
   - Create `docs/math/equation_registry.yaml`.
   - Create `docs/math/code_equation_crosswalk.md`.
   - Define a consistent notation split for:
     - raw observations `z_k`
     - latent state `x_k`
     - feature vectors `\phi_k`
     - generic evidence input `y_k`
     - log evidence `\ell_k(c)`
     - posterior `p_k(c)`
     - candidate scores `Q_{\text{static}}`, `Q_{\text{mc}}`
     - corpus scores and utility terms
   - Require every symbol used in a displayed equation to appear in the glossary or be explicitly marked local to that section.
2. Harden the top-level methodology PDF.
   - Expand the notation section into a real glossary-style table.
   - Add an explicit trajectory, observation, and feature-extraction model.
   - Add a reusable evidence-provider contract:
     - generic evidence term
     - log-sum-exp normalization
     - two-class log-odds recursion
   - Add the transition-matrix posterior update.
   - Add the Kalman-bank innovation likelihood and class-conditioned state-space model.
   - Add explicit warnings for cumulative and windowed feature dependence and double-counting risk.
   - Add pairwise separability metrics and calibration metrics.
   - Add a code-to-equation crosswalk section.
3. Harden the corpus-generation and search PDF.
   - Expand trajectory generation into a distributional model over objectives, backends, candidate parameters, and noise realizations.
   - Define each corpus score subterm operationally:
     - range
     - normalization
     - interpretation
     - failure meaning
     - implementation path
   - Add formal Pareto-dominance and Pareto-front definitions.
   - Define CorpusGym as an MDP-like interface, while stating that current usage is score-and-execution rather than learned control.
   - Expand CorpusGym reward components into a formal table with numeric substitution.
   - Add QD archive cell mapping and elite-selection equations.
   - Add explicit sampler-mixture weights for candidate generation families.
   - Add study-candidate promotion and validation-gate definitions.
4. Add stable standalone math documents.
   - Create `docs/latex/math_foundations.tex`.
   - Create `docs/latex/classifier_ladder_math.tex`.
   - Create `docs/latex/corpus_search_math.tex`.
   - Create `docs/latex/numeric_walkthroughs.tex`.
   - Keep each document stable in purpose:
     - `math_foundations.tex`: shared notation, observation model, evidence model, priors, calibration, pairwise metrics
     - `classifier_ladder_math.tex`: pointwise, windowed, Bayes accumulator, transition-matrix, Kalman bank, advanced-filter gate placement
     - `corpus_search_math.tex`: corpus distribution, adequacy score, Pareto reasoning, CorpusGym reward, archive selection, sampler mixture, validation ladder
     - `numeric_walkthroughs.tex`: worked arithmetic examples only
5. Add worked numeric substitutions directly into the PDFs.
   - Promote existing walkthrough artifacts into inline document content where possible:
     - Bayes one-step update
     - transition-matrix switching update
     - prior sweep and flip threshold
     - Kalman innovation likelihood
     - corpus autodevelopment score decomposition
     - CorpusGym reward decomposition
     - corpus explorer utility decomposition
     - advanced-filter decision-gate walkthrough
   - Ensure each walkthrough states:
     - inputs
     - intermediate terms
     - normalization step
     - final value
     - matching artifact path
6. Build an equation discovery and traceability workflow.
   - Search score functions, weights, likelihood terms, and artifact columns across `src/`, `tests/`, `scripts/`, and generated artifacts.
   - Populate the equation registry with:
     - equation ID
     - LaTeX form
     - symbol definitions
     - implementation module/function
     - test path
     - artifact path
     - status: `implemented` or `conceptual`
   - Count equations traced to code and report unresolved items.
7. Improve PDF readability and figure interpretability.
   - Replace overly wide prose-heavy tables with symbol-first versions.
   - Fix clipped or cramped columns in methodology walkthrough tables.
   - Expand figure captions so they explain:
     - what question the plot answers
     - how to read axes/colors
     - what a good or bad result looks like
     - any asymmetry or diagonal conventions
   - Reduce overfull tables and overfull displayed equations in the hardened math docs.

Validation:
- Every symbol used in a displayed equation appears in the glossary or is explicitly marked section-local.
- Every implemented equation has an implementation reference in `equation_registry.yaml`.
- Every score term has:
  - range
  - positive or penalty direction
  - interpretation
  - source path
- Every explicit weight vector has a source path and is reflected consistently in docs and walkthroughs.
- Every numeric walkthrough embedded in a PDF matches a generated artifact or code-produced value.
- The top-level methodology PDF includes:
  - evidence-provider contract
  - transition update
  - Kalman innovation likelihood
  - cumulative-feature warning
  - pairwise separability metrics
  - calibration metrics
- The corpus-search PDF includes:
  - corpus distribution model
  - operational score definitions
  - Pareto definition
  - CorpusGym formalism
  - reward component table
  - QD archive mapping
  - sampler-mixture weights
  - study-candidate promotion gates
- All new PDFs build successfully without clipped columns or overfull tables in the main algorithm/math displays.
- Figure captions in hardened PDFs explain how to read the plots.

Artifacts / Config:
- `docs/plans/PLN-023_math_document_hardening_and_equation_traceability.md`
- `docs/math/symbol_glossary.tex`
- `docs/math/equation_registry.yaml`
- `docs/math/code_equation_crosswalk.md`
- `docs/latex/math_foundations.tex`
- `docs/latex/classifier_ladder_math.tex`
- `docs/latex/corpus_search_math.tex`
- `docs/latex/numeric_walkthroughs.tex`
- updated `docs/latex/kinematic_classifier_methodology.tex`
- updated `docs/surveys/corpus_generation_and_search.tex`
- updated `docs/surveys/classifier_ladder_and_contracts.tex`
- updated `docs/surveys/methodology_evaluation_framework.tex`
- updated `docs/surveys/dimensional_lift_and_advanced_filter_gates.tex`
- generated PDFs under `artifacts/latex/`
- generated walkthrough tables and supporting LaTeX tables under `docs/latex/tables/` or `artifacts/latex/`
- generated equation traceability exports if needed from the registry

Dependencies:
- `PLN-017` automated methodology proof and LaTeX exposition
- `PLN-020` LaTeX documentation coverage audit and expansion
- existing survey docs in `docs/surveys/`
- existing methodology synthesis generator in `src/kinematic_classifier_sandbox/methodology_latex.py`
- existing worked-example artifacts:
  - transition-matrix numeric walkthrough
  - corpus autodevelopment numeric walkthrough
  - CorpusGym numeric walkthrough
  - corpus explorer numeric walkthrough
  - advanced-filter decision numeric walkthrough

Milestones:
- `M49`: Symbol Glossary And Equation Registry
  - Deliverables:
    - `docs/math/symbol_glossary.tex`
    - `docs/math/equation_registry.yaml`
    - `docs/math/code_equation_crosswalk.md`
  - Exit criterion:
    - All core symbols and implemented equations are catalogued with source links.
- `M50`: Methodology PDF Math Hardening
  - Deliverables:
    - hardened `docs/latex/kinematic_classifier_methodology.tex`
    - refreshed `artifacts/latex/kinematic_classifier_methodology.pdf`
  - Exit criterion:
    - The methodology PDF stands on its own for Bayes updates, transition updates, Kalman likelihoods, pairwise metrics, and calibration.
- `M51`: Corpus PDF Math Hardening
  - Deliverables:
    - hardened `docs/surveys/corpus_generation_and_search.tex`
    - refreshed `artifacts/corpus_generation_and_search.pdf`
  - Exit criterion:
    - The corpus PDF defines every major score operationally and includes compact numeric substitutions.
- `M52`: Standalone Math Document Set
  - Deliverables:
    - `docs/latex/math_foundations.tex`
    - `docs/latex/classifier_ladder_math.tex`
    - `docs/latex/corpus_search_math.tex`
    - `docs/latex/numeric_walkthroughs.tex`
  - Exit criterion:
    - The repo has stable standalone math references in addition to synthesis notes.
- `M53`: Equation Traceability And Formatting Cleanup
  - Deliverables:
    - completed crosswalks
    - improved figure captions
    - fixed wide tables and clipped columns
  - Exit criterion:
    - Equation-to-code traceability is explicit and the PDFs are readable without layout failures.

Recommended Execution Order:
1. `M49` symbol glossary and equation registry
2. `M50` methodology PDF hardening
3. `M51` corpus PDF hardening
4. `M52` standalone math document set
5. `M53` traceability cleanup and formatting cleanup

Success Criteria:
- A reader can reproduce the main posterior and corpus-scoring arithmetic from the PDFs without opening the code.
- The documents clearly separate raw observations, features, evidence terms, posterior updates, and corpus/objective scores.
- The implemented classifier ladder reads as a family of evidence providers under one posterior contract.
- The corpus-search stack reads as a formal objective-and-selection pipeline rather than a collection of heuristics.
- Every important equation in the hardened docs can be traced to code, tests, and artifacts or is explicitly marked conceptual.
- The document set proves not only that the methodology exists, but that it is mathematically inspectable.

Open Questions To Resolve During Implementation:
- Which equations should remain conceptual because they summarize policy or decision logic rather than literal code?
- Whether the stable math documents should be generated directly from the survey sources or maintained as independent LaTeX entry points.
- How much duplication is acceptable between the synthesis PDF and the standalone math references before a shared include system is warranted.
