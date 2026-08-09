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

## 2026-08-10 - Frozen EfficientNetB0

- Freezing a pretrained backbone does not mean the model cannot improve: the small classification head learned rapidly from ImageNet features while only 1,281 parameters were trainable.
- No overfitting signal appeared in this run because validation loss decreased through epoch 20 and stayed close to training loss. The lower validation loss is compatible with training-time augmentation and dropout.
- A validation ROC-AUC of 0.9853 is promising but not final evidence. The validation split is derived from the same source dataset, and the official test set must remain unused until transfer-learning candidates are selected.

## Entry template

- Observation:
- Evidence:
- Interpretation:
- How this changes the next experiment:
