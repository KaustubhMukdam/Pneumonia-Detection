# Evaluation Plan

## Primary evaluation unit

The image is the current unit of evaluation. The official test split remains untouched during model selection. The supplied `val/` folder has only 16 images and is not used for early stopping, threshold selection, or model selection; a deterministic, stratified validation split is created from train data instead.

## Metrics

- Accuracy: overall correctness, reported for context only.
- Precision: how often predicted pneumonia is pneumonia.
- Recall/sensitivity: how many pneumonia cases are detected.
- Specificity: how many normal cases are correctly rejected.
- F1: balance of precision and recall.
- ROC-AUC: ranking performance across thresholds, with limitations noted.
- Confusion matrix: explicit false-positive and false-negative counts.

The positive class is `PNEUMONIA`, and every result must state its decision threshold.

## Operating-threshold policy

For the custom-CNN baseline, choose the highest validation-derived threshold that maintains sensitivity of at least 95%. The selected threshold is 0.65, based on the validation set only:

- Sensitivity: 95.71%
- Specificity: 71.64%
- Precision: 90.73%
- False negatives: 25
- False positives: 57

This threshold is a portfolio-study operating point, not a clinical recommendation. The next test evaluation applies it once without further adjustment.

For DenseNet121, the verified validation operating points are:

- Threshold 0.50: sensitivity 96.91%, specificity 91.04%, precision 96.91%, F1 96.91%, FN 18, FP 18.
- Threshold 0.67: sensitivity 95.03%, specificity 95.52%, precision 98.40%, F1 96.68%, FN 29, FP 9.

Under the current policy, threshold 0.67 is the policy-selected DenseNet121 threshold because it is the highest tested threshold retaining sensitivity of at least 95%. Threshold 0.50 remains an explicitly reported alternative with fewer false negatives. The final model and threshold were selected before the official test was opened.

## Selection policy

The best model is not automatically the one with the highest accuracy. We will consider recall, specificity, calibration/threshold behavior, resource cost, and error patterns together. Final results must note that six exact duplicate pairs remain in the official test split. No model will be described as clinically validated.

## Candidate validation summary

| Candidate | Validation ROC-AUC | Validation operating point | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| Custom CNN | 0.9412 | threshold 0.65 | 95.71% | 71.64% | 93.16% |
| EfficientNetB0 frozen | 0.9853 | threshold 0.57 | 95.03% | 91.54% | 96.01% |
| DenseNet121 frozen | 0.9858 | threshold 0.67 | 95.03% | 95.52% | 96.68% |

This table is for validation-based selection only. It does not establish which model will generalize best to the untouched official test split.

## Selected-model test result

Frozen DenseNet121 was selected and evaluated once on the official test split at the pre-selected threshold of 0.67:

- Accuracy: 85.74%
- Precision: 83.52%
- Sensitivity: 96.15%
- Specificity: 68.38%
- F1-score: 89.39%
- ROC-AUC: 95.22%
- Confusion matrix: TN 160, FP 74, FN 15, TP 375

The model retained high pneumonia sensitivity but showed a substantial specificity decline from validation (95.52%) to test (68.38%). The result is reported as an educational benchmark and not as evidence of clinical readiness.

## Phase 5 error analysis and Grad-CAM

The test prediction table at the pre-selected threshold of 0.67 was reviewed. It contains 160 true negatives, 74 false positives, 15 false negatives, and 375 true positives. The largest error mode is therefore false-positive pneumonia predictions on NORMAL images (74 of 234 NORMAL images); this is consistent with the observed test-specificity decline. The false negatives are fewer (15 of 390 PNEUMONIA images), but remain consequential for the study's sensitivity-oriented operating policy.

Representative false-positive, false-negative, and correct-prediction examples were visualized with Grad-CAM from DenseNet121 layer `conv5_block16_2_conv`. The implementation targets the pre-sigmoid pneumonia logit, avoiding the saturated sigmoid-gradient failure that produced uniformly blue maps in the earlier version. Some reviewed examples contain image borders, laterality markers, or acquisition context, and some maps extend beyond the lung fields. These are shortcut-learning warnings, not evidence that any one artifact caused a prediction. A fully blue map denotes low or zero positive attribution after Grad-CAM's ReLU/normalization step and is uninformative; it must not be interpreted as clinical evidence.

Grad-CAM is a qualitative localization aid only. It does not prove causality, establish clinical reasoning, or validate the model for use on patients.
