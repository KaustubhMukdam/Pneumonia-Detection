# Dataset Audit — Run Instructions

This is the Phase 2 audit entry point. Run it in Kaggle after attaching the Chest X-Ray Images (Pneumonia) dataset, or locally after downloading the data. Do not train a model in this phase.

```python
from pathlib import Path
import sys

sys.path.append("/kaggle/working/Pneumonia-Detection")
from src.dataset_audit import audit_dataset, print_report

dataset_root = Path("/kaggle/input/chest-xray-pneumonia/chest_xray")
result = audit_dataset(dataset_root)
print_report(result)
```

## Required observations to copy into `docs/data_doc.md`

- Dataset path and source/version
- License information and attribution
- Split names and image counts
- Counts for every split/class combination
- Image dimensions and modes
- Unreadable file count and paths
- Exact duplicate groups, if any
- Any unexpected labels or directory names

## Provisional loader policy

Until the audit is complete, use the dataset's existing train/validation/test directories and do not reshuffle the test set. Resize images to one documented input size, convert consistently to the channel format required by the selected model, normalize using the model-specific preprocessing function, and apply augmentation only to training data. Class weighting will be considered after observing the audit counts.
