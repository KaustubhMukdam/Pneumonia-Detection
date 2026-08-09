# Model Card - Initial Draft

## Model status

The custom-CNN baseline has been trained and evaluated. Frozen EfficientNetB0 is the leading validation candidate, but it is not the final selected v2 model and has not been evaluated on the official test split.

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

## EfficientNetB0 evaluation status

No official test metrics are recorded for EfficientNetB0. Its validation results are for candidate comparison only and must not be presented as final performance.
