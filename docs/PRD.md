# Product Requirements Document

## Problem

Build a research-grade prototype that classifies chest X-ray images as `NORMAL` or `PNEUMONIA` and helps a learner understand the trade-offs between a compact CNN and transfer-learning models.

## Users

- Primary: the project author, for learning and portfolio demonstration.
- Secondary: reviewers who need to reproduce the experiments and understand their limitations.

## Requirements

1. Load and inspect the dataset without silently changing labels or splits.
2. Establish a simple CNN baseline before pretrained models.
3. Compare models using accuracy, precision, recall, F1, ROC-AUC, confusion matrix, sensitivity, and specificity where computable.
4. Track each material experiment and its configuration.
5. Inspect false positives and false negatives.
6. Generate Grad-CAM or an equivalent visual explanation for representative cases.
7. Produce a model card and a README that do not overstate medical performance.

## Quality bar

Every reported number must identify the split, positive class, threshold, and model checkpoint. Results must be reproducible from documented code and configuration.

## Out of scope for the first iteration

Clinical deployment, external validation, patient-level demographic analysis, and production monitoring.
