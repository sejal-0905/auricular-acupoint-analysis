"""
Extracts colour/texture features at each labelled acupoint for the three
study groups (HE = healthy, HY = hypertension, DB = diabetes).
Images without one of these prefixes (e.g. 01_L) are skipped.

Uses the ground-truth LabelMe points (data/labelme_json) on the images in data/images.

Usage (from the project root):
    python scripts/extract_features.py
"""

import json
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "images"
JSON_DIR = ROOT / "data" / "labelme_json"
OUT_DIR = ROOT / "results"
OVERLAY_DIR = OUT_DIR / "overlays"

PATCH_RADIUS = 7          # px; images are ~241x433, so keep this small
SKIP_POINTS = []          # e.g. ["EarApex"]
PREFIX_TO_GROUP = {"HE": "healthy", "HY": "hypertension", "DB": "diabetes"}


def patch_features(img, x, y, r):
    h, w = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    x1, x2 = max(0, x - r), min(w, x + r + 1)
    y1, y2 = max(0, y - r), min(h, y + r + 1)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return {
        "R": rgb[..., 0].mean(), "G": rgb[..., 1].mean(), "B": rgb[..., 2].mean(),
        "L": lab[..., 0].mean(), "a": lab[..., 1].mean(), "b": lab[..., 2].mean(),
        "saturation": hsv[..., 1].mean(), "brightness": hsv[..., 2].mean(),
        "texture": gray.std(),
        "patch_pixels": patch.shape[0] * patch.shape[1],
    }


def main():
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {JSON_DIR}")
        return

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped_general = 0

    for jpath in json_files:
        with open(jpath) as f:
            d = json.load(f)

        img_name = Path(d["imagePath"].replace("\\", "/")).name
        image_id = Path(img_name).stem                      # e.g. "HE_01_L"

        prefix = image_id.split("_")[0].upper()
        if prefix not in PREFIX_TO_GROUP:                   # general images (01_L etc.)
            skipped_general += 1
            continue
        group = PREFIX_TO_GROUP[prefix]

        parts = image_id.split("_")
        if len(parts) != 3:
            print(f"Unexpected name '{image_id}' (expected e.g. HE_01_L) - skipping")
            continue
        subject, side = f"{parts[0]}_{parts[1]}", parts[2]  # "HE_01", "L"

        img = cv2.imread(str(IMG_DIR / img_name))
        if img is None:
            print("Could not load", img_name)
            continue

        vis = img.copy()
        for s in d["shapes"]:
            name = s["label"]
            if name in SKIP_POINTS:
                continue
            x, y = s["points"][0]
            feats = patch_features(img, x, y, PATCH_RADIUS)
            if feats is None:
                continue
            rows.append({"group": group, "subject": subject, "side": side,
                         "image": img_name, "point": name, **feats})
            cv2.rectangle(vis, (int(x) - PATCH_RADIUS, int(y) - PATCH_RADIUS),
                          (int(x) + PATCH_RADIUS, int(y) + PATCH_RADIUS), (0, 255, 0), 1)
            cv2.putText(vis, name[:5], (int(x) + 8, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        cv2.imwrite(str(OVERLAY_DIR / img_name), vis)

    if not rows:
        print("No HE_/HY_/DB_ images processed. Check file names and folders.")
        return

    df = pd.DataFrame(rows)

    # lighting-normalised versions: point value minus that image's mean across points
    for col in ["a", "L", "R", "G", "B", "saturation", "brightness"]:
        df[col + "_rel"] = df[col] - df.groupby("image")[col].transform("mean")

    df.to_csv(OUT_DIR / "features.csv", index=False)

    print(f"Skipped {skipped_general} general (unlabelled-diagnosis) images")
    print("Table shape:", df.shape)
    print("\nImages per group:")
    print(df.groupby("group")["image"].nunique())
    print("\nSubjects per group:")
    print(df.groupby("group")["subject"].nunique())
    print(f"\nSaved {OUT_DIR / 'features.csv'} and overlays in {OVERLAY_DIR}")


if __name__ == "__main__":
    main()