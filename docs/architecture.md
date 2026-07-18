# Proposed Architecture

```text
Kaggle chest_xray dataset
        |
        v
Dataset audit -> image/label inspection -> train/validation/test loaders
        |
        v
Baseline CNN -> transfer-learning candidates -> optional fine-tuning
        |
        v
Fixed test evaluation -> error analysis -> Grad-CAM
        |
        v
Reports, model card, and documented conclusions
```

## Design principles

- Keep the test set isolated until final evaluation.
- Use the same evaluation protocol across candidates.
- Record preprocessing, augmentation, seed, batch size, epochs, optimizer, and threshold.
- Treat class imbalance and false negatives as explicit design concerns.
- Let measured results decide whether the simple model or a pretrained model is preferred.

## Planned model progression

1. Historical models as reference only.
2. Clean simple CNN baseline.
3. One or more pretrained candidates, initially MobileNetV2 and/or EfficientNetB0.
4. Fine-tuning of the strongest candidate only if justified by validation results.
