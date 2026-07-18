# Folder Structure

```text
Pneumonia-Detection/
├── docs/                 # decisions, plans, experiment and evaluation records
├── notebooks/            # v2 notebooks, added after documentation phase
├── src/                  # reusable data, training, evaluation, and Grad-CAM code
├── models/               # locally ignored model artifacts or small metadata files
├── reports/              # generated plots and comparison tables
├── legacy/               # preserved v1 notebook/report when reorganization begins
├── README.md
├── LICENSE
└── .gitignore
```

The dataset and large model files must remain outside Git unless an explicit, size-appropriate artifact strategy is chosen.
