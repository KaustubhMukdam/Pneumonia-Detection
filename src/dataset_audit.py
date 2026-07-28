"""Dataset audit utilities for Pneumonia Detection v2.

Run this against a local/Kaggle copy of the dataset before model training.
The script intentionally does not alter images or labels.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def audit_dataset(root: Path) -> dict:
    rows = []
    unreadable = []
    hashes: dict[str, list[str]] = defaultdict(list)

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        split = parts[0] if len(parts) >= 1 else "unknown"
        label = parts[1] if len(parts) >= 2 else "unknown"
        try:
            with Image.open(path) as image:
                image.load()
                record = {
                    "path": str(relative),
                    "split": split,
                    "label": label,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                }
                rows.append(record)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes[digest].append(str(relative))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            unreadable.append({"path": str(relative), "error": str(exc)})

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "total_readable": len(rows),
        "by_split": Counter(row["split"] for row in rows),
        "by_split_label": Counter((row["split"], row["label"]) for row in rows),
        "dimensions": Counter((row["width"], row["height"]) for row in rows),
        "modes": Counter(row["mode"] for row in rows),
        "unreadable": unreadable,
        "duplicate_groups": duplicate_groups,
    }


def print_report(result: dict) -> None:
    print(f"Readable images: {result['total_readable']}")
    print("\nImages by split:")
    for key, value in sorted(result["by_split"].items()):
        print(f"  {key}: {value}")
    print("\nImages by split and label:")
    for (split, label), value in sorted(result["by_split_label"].items()):
        print(f"  {split}/{label}: {value}")
    print("\nImage modes:")
    for key, value in sorted(result["modes"].items()):
        print(f"  {key}: {value}")
    print("\nMost common dimensions:")
    for key, value in result["dimensions"].most_common(10):
        print(f"  {key[0]}x{key[1]}: {value}")
    print(f"\nUnreadable files: {len(result['unreadable'])}")
    print(f"Exact duplicate groups: {len(result['duplicate_groups'])}")
    if result["unreadable"]:
        print("Unreadable details:")
        for item in result["unreadable"]:
            print(f"  {item['path']}: {item['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Path containing train/, test/, and val/")
    args = parser.parse_args()
    print_report(audit_dataset(args.root))
