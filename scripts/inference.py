import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse

# ----------------------
# Inference Function
# ----------------------
def predict_sample(model_path, sample_path):
    # Load the model (supports .h5 or .keras)
    model = tf.keras.models.load_model(model_path)

    # Load the .npz sample
    data = np.load(sample_path)
    image = data['image']  # shape: (256, 256)
    label = data['label']  # ground truth label: 0 or 1

    # Preprocess: add batch and channel dimensions → (1, 256, 256, 1)
    image_input = np.expand_dims(image, axis=(0, -1)).astype(np.float32)

    # Predict
    pred = model.predict(image_input).squeeze()
    pred_label = int(pred > 0.5)

    print(f"\n📂 Sample: {Path(sample_path).name}")
    print(f"✅ Ground Truth: {int(label)}")
    print(f"🔮 Predicted Label: {pred_label}")
    print(f"📊 Raw Output (Sigmoid): {pred:.4f}")

# ----------------------
# CLI Entry Point
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.h5 or .keras)")
    parser.add_argument("--sample", type=str, required=True, help="Path to .npz sample")
    args = parser.parse_args()

    predict_sample(args.model, args.sample)
