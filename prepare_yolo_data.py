"""
Converts LabelMe JSON annotations into YOLO pose format, and automatically
splits your images into train/ and val/ folders (80/20 split by default).

This matters at real scale: using the same photos for training AND testing
(what we did with 4 practice photos) gives misleadingly perfect numbers.
A held-out val set gives you an honest accuracy check.

Usage:
    python prepare_yolo_data.py
"""

import json
import random
import shutil
from pathlib import Path
from PIL import Image

LABELME_DIR = Path("../data/labelme_json")   # where your LabelMe .json files are
SOURCE_IMAGES_DIR = Path("../data/images")    # put ALL your raw photos directly here first
                                                # (not in train/ or val/ subfolders yet)

TRAIN_IMAGES_DIR = Path("../data/images/train")
VAL_IMAGES_DIR = Path("../data/images/val")
TRAIN_LABELS_DIR = Path("../data/labels/train")
VAL_LABELS_DIR = Path("../data/labels/val")

VAL_SPLIT = 0.2  # 20% held out for validation (e.g. 10 of 50 photos)

# The 8 real points, in a fixed order
POINT_ORDER = [
    "EarApex", "Endocrine", "ShenMen", "SuperiorTriangularFossa",
    "Heart", "Pancreas", "Sympathetic", "Kidney",
]


def labelme_to_yolo_line(json_path, image_width, image_height):
    with open(json_path, "r") as f:
        data = json.load(f)

    points_found = {}
    for shape in data.get("shapes", []):
        if shape["shape_type"] == "point":
            points_found[shape["label"]] = shape["points"][0]

    row = [0, 0.5, 0.5, 1.0, 1.0]  # class 0 "ear", full-image box (simple first pass)

    for point_name in POINT_ORDER:
        if point_name in points_found:
            x, y = points_found[point_name]
            row.append(round(x / image_width, 6))
            row.append(round(y / image_height, 6))
            row.append(2)  # visible
        else:
            row.extend([0.0, 0.0, 0])  # not labeled in this image

    return " ".join(str(v) for v in row)


def main():
    for d in [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    json_files = list(LABELME_DIR.glob("*.json"))
    if not json_files:
        print(f"No labeled files found in {LABELME_DIR}. Label some images with LabelMe first.")
        return

    print(f"Found {len(json_files)} labeled images.")

    # Shuffle and split into train/val
    random.seed(42)  # fixed seed so the split is reproducible each time you rerun this
    random.shuffle(json_files)
    n_val = max(1, int(len(json_files) * VAL_SPLIT))
    val_files = set(json_files[:n_val])
    train_files = json_files[n_val:]

    print(f"Splitting: {len(train_files)} train, {len(val_files)} val\n")

    for json_path in json_files:
        image_name = json_path.stem
        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = SOURCE_IMAGES_DIR / f"{image_name}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"  Skipping {json_path.name} - no matching image in {SOURCE_IMAGES_DIR}")
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        yolo_line = labelme_to_yolo_line(json_path, width, height)

        is_val = json_path in val_files
        img_dest = (VAL_IMAGES_DIR if is_val else TRAIN_IMAGES_DIR) / image_path.name
        label_dest = (VAL_LABELS_DIR if is_val else TRAIN_LABELS_DIR) / f"{image_name}.txt"

        shutil.copy(image_path, img_dest)
        with open(label_dest, "w") as f:
            f.write(yolo_line + "\n")

        split_name = "val" if is_val else "train"
        print(f"  [{split_name}] {json_path.name} -> {image_path.name} + {label_dest.name}")

    print(f"\nDone. {len(train_files)} training images, {len(val_files)} validation images.")
    print("Next: run train_yolo.py")


if __name__ == "__main__":
    main()
