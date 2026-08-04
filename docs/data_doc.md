# Data Documentation

## Dataset

The project uses the [Kaggle Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney.

- Kaggle path: `/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray/chest_xray`
- Directory layout: the outer `chest_xray/` contains another `chest_xray/`; the inner directory contains `train/`, `val/`, and `test/`, each with `NORMAL/` and `PNEUMONIA/`.
- License: the Kaggle page does not state one. Attribute Paul Mooney and the original Guangzhou Women and Children's Medical Center source; do not assume commercial or clinical-use rights.

## Verified audit (2026-07-28)

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| train | 1,341 | 3,875 | 5,216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

- Positive class: `PNEUMONIA`.
- Class imbalance: approximately 74% of train images are `PNEUMONIA`.
- Readable images: 5,856; unreadable files: 0.
- Modes: 5,573 grayscale (`L`) and 283 RGB images.
- Dimensions: variable; no single native image size.

## Exact duplicates

The byte-level audit found 30 duplicate groups containing 62 files (32 extra copies).

- 24 groups are entirely inside `train/`; 6 groups are entirely inside `test/`.
- Every group remains within a single label.
- No group crosses a split or label, so there is no exact train/test leakage or label conflict from these duplicates.
- The effective number of unique byte-level files is 5,190 in train and 618 in test.

Raw data will remain unmodified. The official test set will not be de-duplicated because it is retained as the published benchmark split; final results must disclose that it includes six duplicate pairs. Duplicate images in train may slightly overweight some cases, so any de-duplicated-training experiment must be separately logged and compared with the raw-training baseline.

## Split and loader policy

- Preserve existing `train/` and `test/` directories. Never tune, reshuffle, or alter the test set.
- Do not use the provided `val/` directory for early stopping, model selection, or threshold tuning: it has only 16 images.
- Create one deterministic, stratified validation split from `train/`; document its fraction, random seed, and class counts.
- Resize to `224 x 224` and convert every image to RGB for a common interface.
- A custom CNN uses `[0, 1]` scaling. Each pretrained model uses its own TensorFlow/Keras preprocessing function; no generic ImageNet normalization is applied across models.
- Start with conservative training-only augmentation: small rotation (up to 7 degrees), small translation, and modest zoom. Do not use flips or intensity transformations initially.
- Begin with an unweighted baseline. Treat class weighting as a logged follow-up experiment and assess recall/specificity trade-offs.

## Limitations

The dataset is imbalanced, has mixed channels and variable image dimensions, and lacks confirmed patient-level metadata in this project. It does not establish performance across patient populations, hardware, or clinical settings.
