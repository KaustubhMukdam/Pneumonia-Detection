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

The positive class must be explicitly defined as `PNEUMONIA`, and the decision threshold must be recorded.

## Selection policy

The best model is not automatically the one with the highest accuracy. We will consider recall, specificity, calibration/threshold behavior, resource cost, and error patterns together. Final results must note that six exact duplicate pairs remain in the official test split. No model will be described as clinically validated.
