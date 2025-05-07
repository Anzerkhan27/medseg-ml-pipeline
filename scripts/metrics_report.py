import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pathlib import Path
from data_loader import build_dataset, tf_wrapper

# ---------------------
# Configuration
# ---------------------
MODEL_PATH = "outputs/models/classifier_model.h5"
DATA_DIR = "data/processed"
BATCH_SIZE = 64

# ---------------------
# Load Data
# ---------------------
print("🔄 Loading validation data...")
ds, total = build_dataset(DATA_DIR, task="classification", shuffle=False)
val_count = max(1, int(0.1 * total))
val_ds = ds.take(val_count).map(tf_wrapper("classification"))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---------------------
# Load Model
# ---------------------
print("📦 Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------
# Predict and Collect
# ---------------------
print("🔍 Predicting on validation set...")
y_true, y_pred = [], []

for images, labels in val_ds:
    preds = model.predict(images).squeeze()
    preds_binary = (preds > 0.5).astype(int)

    y_true.extend(labels.numpy().astype(int))
    y_pred.extend(preds_binary)

# ---------------------
# Metrics
# ---------------------
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

try:
    auc = roc_auc_score(y_true, y_pred)
    print(f"🔵 ROC AUC Score: {auc:.4f}")
except Exception as e:
    print(f"⚠️ Could not compute ROC AUC: {e}")
