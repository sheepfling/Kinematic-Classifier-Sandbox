# How To Interpret Results

Do not interpret top-line accuracy first. Interpret results in this order.

1. Did the corpus pass adequacy?
   - If no, do not trust leaderboard claims yet.
2. Are the class labels valid?
   - If no, fix class definitions or relabel.
3. Are features excited?
   - If no, generate a better corpus or remove the unsupported feature claim.
4. Are classes separable by oracle or static analysis?
   - If no, this is a feature, corpus, or class-definition issue.
5. Does the classifier underperform the oracle?
   - If yes, inspect algorithm, evidence construction, posterior update, and calibration.
6. Are decisions prior-sensitive?
   - If yes, report fragility and avoid overclaiming.
7. Is confusion localized by class pair, time phase, or sensor regime?
   - If yes, target the localized failure mode.
8. Does a more advanced method address a demonstrated failure?
   - If yes, justify escalation to the next ladder rung.

## Team Rule

A poor confusion matrix is not automatically a bad classifier. It may mean the corpus is biased, features are not excited, classes overlap under the chosen representation, priors dominate the evidence, or the classifier is using the wrong evidence contract.
