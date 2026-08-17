# Project Context

## Working title

Pneumonia Detection v2

## Context

This repository contains an earlier chest-X-ray binary-classification project. The historical implementation compared a simple CNN, a dropout CNN, and frozen VGG16 transfer learning. The reported test accuracies were 82.37%, 80.77%, and 85.90%, respectively. Those results are a starting point, not a final benchmark: the original workflow reports limited metrics, trains for only five epochs, and does not yet include explainability or systematic error analysis.

## Why v2

The objective is to rebuild the project as a reproducible medical-imaging study that measures clinically relevant trade-offs, compares simple and pretrained models fairly, and documents limitations honestly.

## Non-goals

- This is not a clinical diagnostic device.
- Accuracy will not be treated as sufficient evidence of safety or usefulness.
- We will not claim clinical readiness from the public dataset alone.

## Success conditions

The project should have reproducible data handling, a defensible validation protocol, multiple evaluation metrics, documented experiments, error analysis, and visual explanations for selected predictions.

## Current focus

Phase 5 analysis is complete. Frozen DenseNet121 was selected from validation results and evaluated once on the official test split at threshold 0.67. Review found 74 false positives and 15 false negatives, and qualitative Grad-CAM examples were generated with explicit shortcut-learning and non-causality caveats. Fine-tuning is not automatic and, if pursued, must be justified and selected using training/validation data only.
