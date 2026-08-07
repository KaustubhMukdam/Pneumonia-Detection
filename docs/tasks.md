# Project Tasks

## Phase 1 - Repository preparation

- [x] Create and switch to `modernization-v2`.
- [x] Add initial documentation.
- [x] Commit and push the documentation checkpoint.

## Phase 2 - Dataset audit

- [x] Add a reproducible, non-mutating audit utility and run instructions.
- [x] Confirm dataset source, license, directory layout, and class labels.
- [x] Count images by split and class.
- [x] Check image dimensions, channels, unreadable files, and duplicates if feasible.
- [x] Decide loader and augmentation policy.
- [x] Verify that exact duplicate groups do not cross splits or labels.

## Phase 3 - Baseline

- [x] Implement and train a clean simple CNN on Kaggle GPU.
- [x] Create a deterministic, duplicate-safe, stratified validation split from train data.
- [x] Define the initial evaluation protocol and validation-only threshold policy.
- [x] Record training curves, validation findings, and the exploratory test result at threshold 0.50.
- [x] Add the cleaned Kaggle notebook as `notebooks/02_baseline_cnn.ipynb`.
- [x] Evaluate the selected threshold (0.65) on the official test set and record those metrics.
- [x] Complete the Phase 3 baseline summary in the experiment log and model card.

## Phase 4 - Transfer learning

- [ ] Train candidate pretrained model(s) with frozen backbone.
- [ ] Compare against the baseline.
- [ ] Fine-tune only with a documented hypothesis.

## Phase 5 - Analysis

- [ ] Produce confusion matrices and threshold-aware metrics.
- [ ] Review false positives and false negatives.
- [ ] Add Grad-CAM examples.

## Phase 6 - Finalization

- [ ] Complete model card and README.
- [ ] Export reproducible results.
- [ ] Review claims for medical overstatement.
