# Term Aliases

Use the canonical terms in new documentation. Older names remain searchable, but the canonical term should appear first.

| Prefer | Instead of | Reason |
| --- | --- | --- |
| 1D witness problem | toy problem | Frames the examples as controlled proof cases, not trivial demos. |
| Study Candidate Evaluator | feature + class + classifier thing | Names the generic evaluation engine. |
| Corpus Explorer | data generator | The system generates, searches, validates, scores, and selects. |
| CorpusObjective | generation goal | Keeps corpus intent explicit and auditable. |
| SelectedCorpus | generated dataset | Separates candidate generation from validated selection. |
| EvidenceProvider | classifier output code | Emphasizes the shared evidence contract. |
| PosteriorUpdater | Bayes helper | Names the shared posterior machinery. |
| ClassifierFamily | method | Groups comparable evidence providers. |
| FilterBackend | model backend | Reserves the term for dynamics or residual-producing machinery. |
| PriorRegime | prior | Includes sweeps, stress, and fragility analysis. |
| PromotionDecision | result | Forces decisions to distinguish promote, revise, reject, and defer. |

## Superseded Language

`toy_1d` remains in module and artifact names for compatibility. Narrative docs should call these studies `1D witness problems`.

`data generator` is acceptable only when referring to a narrow implementation module. For the methodology layer, use `Corpus Explorer`.
