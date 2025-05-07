import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm  # Add this at the top
import os

def run_batch_inference(model_path, data_dir, dataset_name, output_dir):
    model = tf.keras.models.load_model(model_path)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_csv = output_dir / f"{dataset_name}_{timestamp}.csv"
    results = []

    for file in tqdm(sorted(data_dir.glob("*.npz")), desc="🔍 Predicting"):
        data = np.load(file)
        image = data["image"]
        label = int(data["label"])
        image_input = np.expand_dims(image, axis=(0, -1)).astype(np.float32)

        prob = float(model.predict(image_input, verbose=0).squeeze())
        pred_label = int(prob > 0.5)

        results.append({
            "file_name": file.name,
            "label_true": label,
            "label_pred": pred_label,
            "probability": round(prob, 4)
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Batch inference saved to {output_csv}")

# -------------------------
# CLI Entry
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model (.h5 or .keras)")
    parser.add_argument("--data", required=True, help="Path to directory with .npz files")
    parser.add_argument("--name", default="inference", help="Dataset name prefix")
    parser.add_argument("--out", default="outputs/predictions", help="Output directory for predictions CSV")
    args = parser.parse_args()

    run_batch_inference(args.model, args.data, args.name, args.out)
