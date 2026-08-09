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

## Run 003 - Frozen EfficientNetB0 (2026-08-10)

- Hypothesis: ImageNet-pretrained EfficientNetB0 features can improve discriminative performance and reduce the baseline CNN's false-positive tendency without fine-tuning the backbone.
- Data: the same duplicate-safe, stratified train/validation split as Run 002; seed 42; 4,432 training images and 784 validation images (201 NORMAL / 583 PNEUMONIA).
- Input: 224 x 224 RGB. Pixels remained in the [0, 255] range because Keras EfficientNetB0 contains its own rescaling layer.
- Augmentation: training-only small rotation, translation, and zoom; no flips or intensity transformations.
- Architecture: ImageNet-pretrained EfficientNetB0 backbone with global average pooling, frozen throughout training; dropout 0.30 and a sigmoid binary-classification head. Total parameters: 4,050,852; trainable parameters: 1,281.
- Training: Adam learning rate 3e-4, batch size 32, up to 20 epochs, binary cross-entropy, no class weighting. Validation-loss checkpointing and early stopping were enabled.
- Best checkpoint: epoch 20. Validation loss 0.1385, validation accuracy 0.9413, validation ROC-AUC 0.9853.
- Default validation threshold (0.50): accuracy 0.9413, precision 0.9621, sensitivity 0.9588, specificity 0.8905, F1 0.9605; TN 179, FP 22, FN 24, TP 559.
- Threshold analysis: threshold 0.57 was selected by the policy of highest threshold retaining sensitivity of at least 95%. It gave validation sensitivity 0.9503, specificity 0.9154, precision 0.9702, F1 0.9601; TN 184, FP 17, FN 29, TP 554.
- Interpretation: no observable overfitting in this frozen-stage run. Validation loss declined throughout training, and training/validation accuracy remained close. The validation result substantially exceeds the custom CNN baseline, but it is internal validation only.
- Limitation: no official test evaluation was run. Do not tune or evaluate this candidate on test until comparison with DenseNet121 and any justified fine-tuning is complete. Random image-level splitting may also remain optimistic if multiple images from a patient occur in both train and validation.
- Next decision: train frozen DenseNet121 with the same protocol and compare candidates using validation data only.

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
