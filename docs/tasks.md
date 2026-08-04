# Project Tasks

## Phase 1 — Repository preparation

- [x] Create and switch to `modernization-v2`.
- [x] Add initial documentation.
- [x] Commit and push the documentation checkpoint.

## Phase 2 — Dataset audit

- [x] Add a reproducible, non-mutating audit utility and run instructions.
- [x] Confirm dataset source, license, directory layout, and class labels.
- [x] Count images by split and class.
- [x] Check image dimensions, channels, unreadable files, and duplicates if feasible.
- [x] Decide loader and augmentation policy.
- [x] Verify that exact duplicate groups do not cross splits or labels.

## Phase 3 — Baseline

- [ ] Implement a clean simple CNN.
- [ ] Define a fixed evaluation protocol.
- [ ] Record metrics and learning curves.

## Phase 4 — Transfer learning

- [ ] Train candidate pretrained model(s) with frozen backbone.
- [ ] Compare against the baseline.
- [ ] Fine-tune only with a documented hypothesis.

## Phase 5 — Analysis

- [ ] Produce confusion matrices and threshold-aware metrics.
- [ ] Review false positives and false negatives.
- [ ] Add Grad-CAM examples.

## Phase 6 — Finalization

- [ ] Complete model card and README.
- [ ] Export reproducible results.
- [ ] Review claims for medical overstatement.
