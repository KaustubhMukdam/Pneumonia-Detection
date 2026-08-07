# Learnings

## 2026-07-19 — Project reset

- The old project is useful as a baseline, but its accuracy-only reporting is insufficient for a medical-imaging study.
- Model complexity is a hypothesis, not a guarantee of better performance.
- Documentation must distinguish historical claims from results reproduced under the v2 protocol.

## 2026-08-07 - Baseline CNN

- A high ROC-AUC does not guarantee a useful fixed decision threshold. The custom CNN produced ROC-AUC 0.8257 on test data but had only 15.81% specificity at threshold 0.50.
- Sensitivity and overfitting are different concepts. The model's high sensitivity came from the threshold and class distribution; overfitting appeared as rising validation loss after epoch 6 while training performance kept improving.
- Thresholds must be selected with validation data, not by inspecting test performance. A validation-derived 0.65 threshold retained 95.71% sensitivity while improving validation specificity to 71.64%.
- Applying that threshold to test data improved specificity from 15.81% to 35.04%, but it did not reach the validation specificity. A good validation operating point may still generalize poorly when the test distribution differs.

## Entry template

- Observation:
- Evidence:
- Interpretation:
- How this changes the next experiment:
