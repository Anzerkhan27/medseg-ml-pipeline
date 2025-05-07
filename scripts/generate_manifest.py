#scripts/generate_manifest.py

import os
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import argparse

def extract_slice_number(path):
    """Extract slice number from filename like TCGA_CS_4941_19960909_46.tif"""
    stem = Path(path).stem
    if "_mask" in stem:
        stem = stem.replace("_mask", "")
    return int(stem.split("_")[-1])

def has_tumor(mask_path):
    """Returns 1 if mask has non-zero pixels, else 0"""
    try:
        with Image.open(mask_path) as img:
            return int(np.max(np.array(img)) > 0)
    except:
        return None

def build_manifest(dataset_root):
    dataset_root = Path(dataset_root)
    all_tifs = list(dataset_root.rglob("*.tif"))

    orig_paths = [p for p in all_tifs if "_mask" not in p.name]
    mask_paths = [p for p in all_tifs if "_mask" in p.name]

    print(f"Found {len(orig_paths)} original images, {len(mask_paths)} masks")

    # Create lookup dict for masks
    mask_dict = {
        (Path(p).parent.name, extract_slice_number(p)): p
        for p in mask_paths
    }

    records = []

    for orig in orig_paths:
        patient_id = Path(orig).parent.name
        slice_num = extract_slice_number(orig)
        mask = mask_dict.get((patient_id, slice_num))

        try:
            with Image.open(orig) as im:
                width, height = im.size
        except:
            width, height = None, None

        # Calculate binary label based on mask content
        label = has_tumor(mask) if mask else None

        records.append({
            "patient_id": patient_id,
            "slice_number": slice_num,
            "image_path": str(orig),
            "mask_path": str(mask) if mask else None,
            "width": width,
            "height": height,
            "label": label
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to kaggle_3m directory")
    parser.add_argument("--output", type=str, default="data/manifest.csv", help="Where to save manifest")
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)
    df = build_manifest(args.input)
    df.to_csv(args.output, index=False)
    print(f"✅ Manifest with labels saved to {args.output}")
