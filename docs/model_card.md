# Model Card - Initial Draft

## Model status

The custom-CNN baseline has been trained and evaluated, but it is not the final selected v2 model. Its selected validation checkpoint is from epoch 6, and its operating threshold is 0.65.

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

## Known limitations

The model overfit after its best validation epoch. An exploratory test evaluation at threshold 0.50 had sensitivity 99.23% but specificity 15.81%, indicating a strong false-positive tendency at that threshold and a validation-to-test generalization gap. The official test set includes six exact duplicate pairs. Grad-CAM is not yet implemented and no external validation exists.

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
