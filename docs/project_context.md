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

Close Phase 3 by adding the cleaned Kaggle baseline notebook and recording one fixed-threshold (0.65) test evaluation. Model improvements and transfer learning remain out of scope until the baseline is documented.
