# Model Card — Initial Draft

## Model status

No v2 model has been selected yet. The historical VGG16 result is a reference baseline, not a validated release.

## Intended use

Educational research and portfolio demonstration of image-classification experimentation.

## Not intended for

Diagnosis, triage, treatment decisions, or unsupervised clinical use.

## Data

The project references a public Kaggle chest-X-ray dataset with `NORMAL` and `PNEUMONIA` labels. Dataset composition, licensing, and audit findings will be recorded before final training.

## Limitations

Potential class imbalance, unknown population and acquisition bias, label noise, limited external validation, and possible dataset leakage or shortcut learning. Grad-CAM is an explanation aid, not proof that the model used clinically meaningful evidence.

## Final metrics

To be completed after the v2 evaluation protocol is run. Metrics will include split, threshold, positive class, and uncertainty where feasible.
