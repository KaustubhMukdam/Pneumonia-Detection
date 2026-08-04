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


def _duplicate_summary(duplicate_groups: list[list[str]]) -> dict:
    """Summarize duplicate placement without changing the source dataset."""
    groups = []
    for paths in duplicate_groups:
        locations = [Path(path).parts for path in paths]
        splits = sorted({parts[0] for parts in locations})
        labels = sorted({parts[1] for parts in locations if len(parts) > 1})
        groups.append(
            {
                "paths": paths,
                "file_count": len(paths),
                "splits": splits,
                "labels": labels,
                "cross_split": len(splits) > 1,
                "cross_label": len(labels) > 1,
            }
        )

    return {
        "groups": groups,
        "group_count": len(groups),
        "files_in_groups": sum(group["file_count"] for group in groups),
        "extra_duplicate_files": sum(group["file_count"] - 1 for group in groups),
        "cross_split_groups": sum(group["cross_split"] for group in groups),
        "cross_label_groups": sum(group["cross_label"] for group in groups),
    }


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
        "duplicate_summary": _duplicate_summary(duplicate_groups),
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
    duplicate_summary = result["duplicate_summary"]
    print(f"Exact duplicate groups: {duplicate_summary['group_count']}")
    print(f"Files in duplicate groups: {duplicate_summary['files_in_groups']}")
    print(f"Extra duplicate files: {duplicate_summary['extra_duplicate_files']}")
    print(f"Duplicate groups crossing splits: {duplicate_summary['cross_split_groups']}")
    print(f"Duplicate groups crossing labels: {duplicate_summary['cross_label_groups']}")
    if result["unreadable"]:
        print("Unreadable details:")
        for item in result["unreadable"]:
            print(f"  {item['path']}: {item['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Path containing train/, test/, and val/")
    parser.add_argument(
        "--show-duplicates",
        action="store_true",
        help="Print paths, splits, and labels for every exact-duplicate group.",
    )
    args = parser.parse_args()
    result = audit_dataset(args.root)
    print_report(result)
    if args.show_duplicates:
        for index, group in enumerate(result["duplicate_summary"]["groups"], start=1):
            print(f"\nDuplicate group {index}: {group['splits']} / {group['labels']}")
            for path in group["paths"]:
                print(f"  {path}")
