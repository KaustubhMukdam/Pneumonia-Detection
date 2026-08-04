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
