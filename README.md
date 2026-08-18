# Pneumonia Detection v2

An educational computer-vision portfolio project that compares a custom CNN with frozen ImageNet-pretrained transfer-learning models for binary chest-X-ray classification. It is **not** a clinical diagnostic, triage, or treatment system.

## What this study does

- Audits the public Kaggle Chest X-Ray Images (Pneumonia) dataset before modelling.
- Creates a deterministic, duplicate-safe, stratified validation split from the training data.
- Compares a custom CNN baseline, frozen EfficientNetB0, and frozen DenseNet121 using validation data.
- Selects a decision threshold before one official-test evaluation.
- Reports threshold-aware metrics, error counts, and qualitative Grad-CAM examples alongside limitations.

## Dataset and protocol

The project uses the [Kaggle Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney. The verified audit contains 5,856 readable images: 5,216 train, 16 supplied validation, and 624 official-test images. `PNEUMONIA` is the positive class.

The supplied validation split is too small for model selection, so the project creates a seed-42, duplicate-safe stratified split from training data: 4,432 training images and 784 validation images. The official test set was not used to select the model or threshold. Patient-level separation has not been verified, and the official test set contains six exact duplicate pairs.

See [data documentation](docs/data_doc.md) and the [evaluation plan](docs/eval.md) for the complete protocol.

## Current v2 result

Frozen DenseNet121 was selected using validation results and evaluated once on the official test split at the pre-selected threshold of 0.67.

| Metric | Official-test result |
|---|---:|
| Accuracy | 85.74% |
| Precision | 83.52% |
| Sensitivity / recall | 96.15% |
| Specificity | 68.38% |
| F1-score | 89.39% |
| ROC-AUC | 95.22% |
| Confusion matrix | TN 160, FP 74, FN 15, TP 375 |

The model retained high pneumonia sensitivity, but specificity fell substantially from 95.52% on validation to 68.38% on the official test split. The principal observed test error was false-positive pneumonia prediction: 74 NORMAL images were classified as PNEUMONIA, while 15 PNEUMONIA images were classified as NORMAL. This is a dataset-specific educational result, not evidence of clinical readiness.

## Notebooks

| Notebook | Purpose |
|---|---|
| [01_dataset_audit.md](notebooks/01_dataset_audit.md) | Dataset structure, image, and duplicate audit instructions |
| [02_baseline_cnn.ipynb](notebooks/02_baseline_cnn.ipynb) | Custom-CNN baseline |
| [03_efficientnetb0_frozen.ipynb](notebooks/03_efficientnetb0_frozen.ipynb) | Frozen EfficientNetB0 candidate |
| [04-densenet121-frozen.ipynb](notebooks/04-densenet121-frozen.ipynb) | Frozen DenseNet121 training and validation selection |
| [04-densenet121-evaluation.ipynb](notebooks/04-densenet121-evaluation.ipynb) | One-time official-test evaluation |
| [05-error-analysis.ipynb](notebooks/05-error-analysis.ipynb) | Test-error review and Grad-CAM |

The legacy implementation and its historical accuracy-only results are retained in [`legacy/`](legacy/) for context. They are not the v2 benchmark.

## Reproducing the study

1. Create a Kaggle Notebook with GPU enabled and attach the Kaggle dataset above.
2. Run the notebooks in numerical order. The transfer-learning notebooks use TensorFlow/Keras, NumPy, pandas, Matplotlib, seaborn, and scikit-learn.
3. Do not change the official-test threshold after viewing test results. For the selected DenseNet121 candidate, use the already selected threshold of 0.67 only for the recorded test evaluation.
4. Treat any later model change, including fine-tuning, as a new validation-only experiment. Do not use the already-viewed official test set to select it.

## Limits and responsible use

- No external validation, patient-level split verification, demographic analysis, or clinical evaluation has been performed.
- Dataset labels and acquisition context may not generalize to other sites, hardware, populations, or clinical workflows.
- Grad-CAM is qualitative. Attention outside lung fields, image borders, or laterality markers is a shortcut-learning warning, not proof of why the model predicted a class. A fully blue Grad-CAM map is low or zero positive attribution after processing, not clinical evidence.
- The project must not be used for diagnosis, triage, treatment decisions, or unsupervised clinical use.

For detailed caveats and experiment history, read the [model card](docs/model_card.md), [experiment log](docs/experiment_log.md), and [learnings](docs/learnings.md).

## License

This repository is licensed under the [MIT License](LICENSE). Dataset rights are separate; consult the dataset page and original-source attribution before any reuse.
