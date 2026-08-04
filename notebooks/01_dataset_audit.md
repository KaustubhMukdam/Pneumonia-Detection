# Dataset Audit - Run Instructions

Run this in Kaggle after attaching the Chest X-Ray Images (Pneumonia) dataset. Do not train a model in this phase.

```python
from pathlib import Path
import sys

sys.path.append("/kaggle/working/Pneumonia-Detection")
from src.dataset_audit import audit_dataset, print_report

dataset_root = Path(
    "/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/"
    "chest_xray/chest_xray"
)
result = audit_dataset(dataset_root)
print_report(result)
```

The report must show zero duplicate groups crossing splits and labels before Phase 3 begins. To inspect every group:

```python
for index, group in enumerate(result["duplicate_summary"]["groups"], start=1):
    print(f"Group {index}: splits={group['splits']}, labels={group['labels']}")
    print(*group["paths"], sep="\n  ")
```

Record the source, path, counts, image properties, unreadable files, and duplicate placement in `docs/data_doc.md`. The supplied `val/` directory contains only 16 images; create a deterministic, stratified validation split from `train/` in Phase 3 rather than using it for model selection.
