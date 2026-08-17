# Model Card - Initial Draft

## Model status

The custom-CNN baseline has been trained and evaluated. Frozen EfficientNetB0 and frozen DenseNet121 were trained under the same validation protocol. Frozen DenseNet121 was selected using validation results and evaluated once on the official test split at threshold 0.67. It is the current v2 study candidate, not a clinically validated model.

## Intended use

Educational research and portfolio demonstration of image-classification experimentation.

## Not intended for

Diagnosis, triage, treatment decisions, or unsupervised clinical use.

## Data

The project uses a public Kaggle chest-X-ray dataset with `NORMAL` and `PNEUMONIA` labels. The v2 audit found 5,856 readable images, class imbalance toward pneumonia, variable dimensions, mixed grayscale/RGB encoding, and no exact duplicate groups crossing data splits or labels.

## Baseline configuration

- Architecture: custom CNN with three convolutional blocks and 296,673 parameters.
- Input: 224 x 224 RGB, scaled to [0, 1].
- Training: seed 42, Adam at 1e-3, batch size 32, early stopping and checkpointing by validation loss, no class weights.
- Selected threshold: 0.65, chosen on validation data to retain sensitivity of at least 95%.

## Frozen EfficientNetB0 candidate

- Backbone: ImageNet-pretrained EfficientNetB0, frozen throughout training.
- Head: dropout 0.30 and a single sigmoid output; 1,281 trainable parameters out of 4,050,852 total.
- Input: 224 x 224 RGB in the [0, 255] range; EfficientNetB0 performs its own rescaling.
- Training: seed 42, Adam at 3e-4, batch size 32, early stopping and checkpointing by validation loss, no class weights.
- Best checkpoint: epoch 20, validation loss 0.1385, validation accuracy 94.13%, validation ROC-AUC 98.53%.
- Selected validation threshold: 0.57, retaining 95.03% sensitivity with 91.54% specificity.

## Frozen DenseNet121 candidate

- Backbone: ImageNet-pretrained DenseNet121, frozen throughout training.
- Head: global average pooling, dropout 0.30, and a single sigmoid output; 1,025 trainable parameters out of 7,038,529 total.
- Input: 224 x 224 RGB with the Keras DenseNet preprocessing convention.
- Training: seed 42, Adam at 3e-4, batch size 32, early stopping and checkpointing by validation loss, no class weights.
- Best checkpoint: epoch 20, validation loss 0.1311, validation accuracy 95.41%, validation ROC-AUC 98.58%.
- Validation threshold 0.50: sensitivity 96.91%, specificity 91.04%, precision 96.91%, F1-score 96.91%; TN 183, FP 18, FN 18, TP 565.
- Policy-selected validation threshold 0.67: sensitivity 95.03%, specificity 95.52%, precision 98.40%, F1-score 96.68%; TN 192, FP 9, FN 29, TP 554.
- Threshold 0.67 follows the existing rule of selecting the highest tested threshold that retains sensitivity of at least 95%. Threshold 0.50 remains a documented alternative with fewer false negatives.

## Selected DenseNet121 official test result

- Test split: official 624-image test split; `PNEUMONIA` is the positive class.
- Decision threshold: 0.67, selected before test evaluation from validation data.
- Accuracy: 85.74%
- Precision: 83.52%
- Sensitivity: 96.15%
- Specificity: 68.38%
- F1-score: 89.39%
- ROC-AUC: 95.22%
- Confusion matrix counts: TN 160, FP 74, FN 15, TP 375.

The result indicates strong pneumonia sensitivity but weaker normal-class specificity than validation suggested. The model classified 74 NORMAL images as PNEUMONIA and missed 15 PNEUMONIA images.

## Known limitations

The custom CNN overfit after its best validation epoch. Frozen DenseNet121 did not show clear curve-based overfitting, but its validation performance was optimistic relative to test specificity. The project has not verified patient-level separation. The official test set includes six exact duplicate pairs. No external validation exists.

## Error analysis and visual explanations

At the pre-selected test threshold, 74 NORMAL images were false positives and 15 PNEUMONIA images were false negatives. False positives are the dominant observed error mode and align with the test-specificity decline; this finding does not identify their cause.

Grad-CAM examples were generated from `conv5_block16_2_conv` using the pre-sigmoid pneumonia logit. The maps are qualitative and may include attention outside the lung fields, including borders or laterality markers visible in some images. Such patterns are a reason for caution and further study, not proof of shortcut learning or causal model reasoning. Uniformly blue maps represent low or zero positive attribution after processing and are not interpretable evidence.

## Baseline test metrics

Official test split, `PNEUMONIA` positive class, threshold 0.65:

- Accuracy: 73.08%
- Precision: 71.10%
- Sensitivity: 95.90%
- Specificity: 35.04%
- F1-score: 81.66%
- ROC-AUC: 82.57%
- Confusion matrix counts: TN 82, FP 152, FN 16, TP 374

The threshold came from validation data, but the test split had already been inspected at threshold 0.50. This result is non-blinded and must not be used to tune later models.

## Transfer-learning evaluation status

DenseNet121 has completed its single pre-specified official-test evaluation. EfficientNetB0 has no official test evaluation because it was not selected. Validation results for both candidates remain useful for comparison, but must not be presented as final performance.

The initial DenseNet report contained impossible perfect metrics despite a confusion matrix with 36 errors. Those values were discarded and recomputed directly from validation probabilities and labels.
