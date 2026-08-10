# Model Card - Initial Draft

## Model status

The custom-CNN baseline has been trained and evaluated. Frozen EfficientNetB0 and frozen DenseNet121 have been trained under the same validation protocol. DenseNet121 is currently the leading validation candidate, but no transfer-learning model has been evaluated on the official test split and no final v2 model has been selected.

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

## Known limitations

The custom CNN overfit after its best validation epoch. Frozen EfficientNetB0 did not show the same pattern, but its high validation scores may still be optimistic because the project has not verified patient-level separation. The official test set includes six exact duplicate pairs. Grad-CAM is not yet implemented and no external validation exists.

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

No official test metrics are recorded for EfficientNetB0 or DenseNet121. Their validation results are for candidate comparison only and must not be presented as final performance.

The initial DenseNet report contained impossible perfect metrics despite a confusion matrix with 36 errors. Those values were discarded and recomputed directly from validation probabilities and labels.
