"""
metrics_report.py

Usage (CLI):
    python scripts/metrics_report.py --pred outputs/predictions/myset_20250507_1530.csv
"""

from pathlib import Path
import argparse

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ────────────────────────────────────────────────────────────────────
def generate_metrics_report(
    predictions_file: str,
    model_output: str | None = None,
    digits: int = 4,
) -> None:
    """
    Read a CSV created by `run_batch_inference` and print classification metrics.

    Parameters
    ----------
    predictions_file : str
        Path to the CSV that contains file_name,label_true,label_pred,probability
    model_output : str | None
        (Optional) path to a saved model – not used here but convenient
        if you later want to load the model for extra analyses.
    digits : int
        Number of decimals for classification_report.
    """
    predictions_file = Path(predictions_file)
    if not predictions_file.exists():
        raise FileNotFoundError(predictions_file)

    df = pd.read_csv(predictions_file)

    if df.empty:
        print("❌ Predictions CSV is empty. Nothing to evaluate.")
        return

    y_true = df["label_true"].astype(int).tolist()
    y_pred = df["label_pred"].astype(int).tolist()

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=digits))

    print("\n📉 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    try:
        auc = roc_auc_score(y_true, df["probability"])
        print(f"\n🔵 ROC AUC Score: {auc:.4f}")
    except Exception as e:
        print(f"\n⚠️  Could not compute ROC AUC: {e}")

# ────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to predictions CSV")
    parser.add_argument("--model", help="(Optional) path to model")
    parser.add_argument("--digits", type=int, default=4)
    args = parser.parse_args()

    generate_metrics_report(args.pred, args.model, args.digits)
