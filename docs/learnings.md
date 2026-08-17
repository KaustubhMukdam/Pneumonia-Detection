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

## 2026-08-11 - Frozen DenseNet121

- Frozen DenseNet121 reached validation ROC-AUC 0.9858, closely matching EfficientNetB0 at 0.9853. The small AUC difference is not enough to decide the model by itself.
- At the policy-selected threshold of 0.67, DenseNet121 retained 95.03% sensitivity and improved validation specificity to 95.52%, with 29 false negatives and 9 false positives.
- Threshold 0.50 produced fewer false negatives (18 versus 29) but more false positives (18 versus 9). Threshold choice is therefore a documented trade-off, not a purely mathematical “best” value.
- A first metric printout claimed perfect validation performance despite a confusion matrix containing 36 errors. Direct recomputation resolved the issue. Metric outputs must always be checked against TN, FP, FN, and TP before being recorded.
- DenseNet121 does not show clear overfitting in its learning curves: training and validation loss both decreased through the best epoch. High validation scores still require caution because patient-level separation has not been established.
- The next experiment should be model comparison and one-time official-test evaluation, not immediate fine-tuning.

## 2026-08-13 - Frozen DenseNet121 official test evaluation

- Selecting a threshold on validation data and applying it once to test preserved the evaluation protocol: threshold 0.67 achieved 96.15% sensitivity and 68.38% specificity on test.
- Validation performance overstated normal-class discrimination: specificity fell from 95.52% on validation to 68.38% on test, while sensitivity remained high. This is a generalization/calibration warning, not sufficient evidence by itself to diagnose overfitting.
- Confusion matrices are more informative than accuracy alone: the test model missed 15 pneumonia images but produced 74 false alarms among normal images.
- DenseNet121 is a stronger current candidate than the custom CNN on the recorded test metrics, but the comparison must mention that the baseline test result was not fully blinded during threshold analysis.
- Fine-tuning should not be started merely because the test score is imperfect. First inspect false positives, false negatives, and Grad-CAM visualizations to define a defensible hypothesis.

## 2026-08-17 - Phase 5 error analysis and Grad-CAM

- At the pre-selected threshold of 0.67, false positives are the dominant test error: 74 NORMAL images were predicted as PNEUMONIA, compared with 15 PNEUMONIA images predicted as NORMAL. This describes the error distribution; it does not establish why the errors occurred.
- Grad-CAM should use the pre-sigmoid logit rather than the post-sigmoid probability. The corrected implementation produces attributions from DenseNet121's `conv5_block16_2_conv` and avoids saturated sigmoid gradients.
- Grad-CAM maps that extend to borders, laterality markers, or other non-lung context are a shortcut-learning warning. They do not prove that a specific artifact drove a prediction, and uniformly blue maps are low- or zero-positive-attribution visualizations rather than meaningful negative evidence.
- The observed specificity gap supports investigating a validation-only hypothesis about dataset-image characteristics or calibration, but it does not authorize tuning against the already-viewed official test set.

## Entry template

- Observation:
- Evidence:
- Interpretation:
- How this changes the next experiment:
