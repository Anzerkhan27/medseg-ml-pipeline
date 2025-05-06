import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import argparse

def preprocess_image(path, size=(256, 256)):
    with Image.open(path) as img:
        img = img.convert("L")  # ensure grayscale
        img = img.resize(size)
        arr = np.array(img).astype("float32") / 255.0  # normalize to [0, 1]
    return arr

def preprocess_mask(path, size=(256, 256)):
    if path is None or pd.isna(path):
        return None
    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize(size)
        return np.array(img).astype("float32") / 255.0  # keep consistent dtype

def run_standardization(manifest_path, output_dir):
    df = pd.read_csv(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, row in df.iterrows():
        image_path = row["image_path"]
        mask_path = row["mask_path"]
        label = row["label"]
        patient_id = row["patient_id"]
        slice_number = row["slice_number"]

        try:
            image = preprocess_image(image_path)
            mask = preprocess_mask(mask_path)

            out_path = output_dir / f"{patient_id}_{slice_number}.npz"
            np.savez_compressed(out_path,
                                image=image,
                                mask=mask,
                                label=label,
                                patient_id=patient_id,
                                slice_number=slice_number)
            
            if i % 500 == 0:
                print(f"Processed {i}/{len(df)}")

        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")

    print(f"\n✅ All data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/manifest.csv", help="Path to manifest.csv")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory for .npz files")
    args = parser.parse_args()

    run_standardization(args.manifest, args.output)
