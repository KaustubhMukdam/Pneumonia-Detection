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
3. Frozen ImageNet-pretrained EfficientNetB0 and DenseNet121 candidates, evaluated with the baseline's same data protocol.
4. Select one candidate using validation results and a validation-derived operating threshold.
5. Fine-tune the selected candidate only if validation evidence justifies it.

## Frozen transfer-learning findings

EfficientNetB0 and DenseNet121 have now been trained with frozen ImageNet backbones under the same split, augmentation, and threshold-analysis protocol. DenseNet121 currently has the strongest validation operating point: at threshold 0.67 it retained 95.03% sensitivity and achieved 95.52% specificity, compared with EfficientNetB0's 95.03% sensitivity and 91.54% specificity at threshold 0.57. Their ROC-AUC values are nearly tied (0.9858 versus 0.9853), so the comparison should not be reduced to ROC-AUC alone.

The next architectural decision is to compare all candidates using validation results, choose one model and threshold, and then evaluate that frozen candidate once on the official test split. Fine-tuning remains conditional; it should be started only if the official test result or error analysis identifies a clear, testable limitation.
