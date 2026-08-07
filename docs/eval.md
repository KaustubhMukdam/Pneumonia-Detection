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

## Selection policy

The best model is not automatically the one with the highest accuracy. We will consider recall, specificity, calibration/threshold behavior, resource cost, and error patterns together. Final results must note that six exact duplicate pairs remain in the official test split. No model will be described as clinically validated.
