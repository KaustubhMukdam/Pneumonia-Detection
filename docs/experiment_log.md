# Experiment Log

Append one entry per material run. Do not overwrite old results.

## Run 000 - Historical reference

- Source: original `legacy/x_ray.ipynb` and report.
- Models: simple CNN, dropout CNN, frozen VGG16.
- Reported test accuracy: 82.37%, 80.77%, and 85.90% respectively.
- Limitation: trained for five epochs; exact reproducibility settings and complete medical metrics were not verified.

## Run 001 - Dataset audit (2026-07-28)

- Dataset: Kaggle Chest X-Ray Images (Pneumonia), inner `chest_xray/` directory.
- Readable files: 5,856; unreadable: 0.
- Split counts: train 5,216, val 16, test 624.
- Class counts: train 1,341 NORMAL / 3,875 PNEUMONIA; test 234 NORMAL / 390 PNEUMONIA.
- Modes: 5,573 `L`, 283 RGB; dimensions vary.
- Exact duplicates: 30 groups containing 62 files. Zero groups cross splits or labels; therefore, no exact train/test leakage or label conflict was found.
- Decision: do not use the 16-image validation folder for model selection. Create one deterministic, stratified validation split from train. Preserve the raw official test set and disclose its six duplicate pairs in final reporting.

## Run 002 - Custom CNN baseline (2026-08-07)

- Hypothesis: a compact CNN trained from scratch can establish a reproducible performance floor before transfer learning.
- Data: duplicate-safe, stratified split from the original training directory, seed 42. Training contained 4,432 images and validation contained 784 images (201 NORMAL / 583 PNEUMONIA). The official 624-image test set was not used for training or checkpoint selection.
- Input: 224 x 224 RGB. Images were scaled to [0, 1].
- Augmentation: training-only small rotation, translation, and zoom; no flips or intensity transformations.
- Architecture: three convolutional blocks (32, 64, 128 filters), batch normalization, max pooling, dropout, global average pooling, 64-unit dense head, sigmoid output. Total parameters: 296,673.
- Training: Adam learning rate 1e-3, batch size 32, up to 25 epochs, binary cross-entropy, no class weighting. Model checkpoint and early stopping monitored validation loss.
- Best checkpoint: epoch 6 of 11 completed epochs, validation loss 0.3282, validation accuracy 0.8546, validation ROC-AUC 0.9412.
- Training observation: training accuracy reached 0.9549 while later validation loss rose to 4.8670. This is overfitting/instability after the best checkpoint, so the epoch-6 checkpoint was restored.
- Exploratory test result at threshold 0.50: accuracy 0.6795, precision 0.6627, sensitivity 0.9923, specificity 0.1581, F1 0.7947, ROC-AUC 0.8257; TN 37, FP 197, FN 3, TP 387.
- Threshold analysis: using validation only, threshold 0.65 was selected as the highest tested threshold that maintained sensitivity at or above 95%. At 0.65, validation sensitivity was 0.9571, specificity 0.7164, precision 0.9073, F1 0.9316, FN 25, and FP 57.
- Fixed-threshold test result at 0.65: accuracy 0.7308, precision 0.7110, sensitivity 0.9590, specificity 0.3504, F1 0.8166, ROC-AUC 0.8257; TN 82, FP 152, FN 16, TP 374.
- Interpretation: compared with threshold 0.50, the selected threshold improved test accuracy (+5.13 percentage points) and specificity (+19.23 points) while sensitivity fell by 3.33 points. The validation-to-test specificity gap (71.64% vs. 35.04%) remains substantial, indicating limited generalization/calibration stability.
- Limitation: the test set was inspected at threshold 0.50 before the threshold policy was finalized. The threshold-0.65 test result is therefore non-blinded and must not be used to tune future experiments.
- Next decision: add the cleaned notebook to the repository and close Phase 3. Future model selection uses validation only; transfer-learning candidates will use the same data protocol.

## Run template

### Run ID / date

- Hypothesis:
- Dataset version and split:
- Model and pretrained weights:
- Input size/channels:
- Augmentation:
- Seed:
- Batch size / epochs:
- Optimizer / learning rate:
- Best-checkpoint rule:
- Metrics:
- Interpretation:
- Next decision:
