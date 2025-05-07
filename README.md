# 🧠 MedSeg ML Pipeline

An end-to-end machine learning pipeline for binary brain tumor classification using the LGG MRI segmentation dataset. Built with TensorFlow, Prefect, and modular design principles for production-readiness.

---

## 🎓 Project Background

This pipeline is a production-grade refactor of my **BSc Final Year Project**, which tackled:

- **Tumor classification**: Predicting presence of tumor (binary) from MRI slices using CNN
- **Tumor segmentation**: Pixel-level prediction of tumor regions using a ResUNet (coming soon)

---

## 📁 Project Structure

```
medseg-ml-pipeline/
├── data/                      # Input and processed data
│   ├── manifest.csv           # Auto-generated file linking images to labels
│   └── processed/             # Standardized `.npz` files
├── outputs/                  # Model weights, predictions, and logs
│   ├── models/                # Trained models (.h5 and .keras)
│   └── predictions/           # Inference CSV outputs
├── scripts/                  # Modular Python scripts
│   ├── generate_manifest.py
│   ├── standardize_data.py
│   ├── train_classifier.py
│   ├── infer_batch.py
│   ├── metrics_report.py
│   └── model_export.py
├── orchestration/
│   └── prefect_flow.py       # End-to-end orchestrated workflow
├── README.md
```

---

## 🚀 Features

* **Automated Manifest Creation**: Links MRI images with masks and labels.
* **Standardization**: Normalizes and resizes images into `.npz` format.
* **Binary Classifier**: Trains a CNN with validation monitoring.
* **Batch Inference**: Predicts across entire processed datasets.
* **Evaluation Metrics**: Outputs precision, recall, F1, ROC AUC, and confusion matrix.
* **Prefect Orchestration**: One-click execution of the full pipeline locally.
* **Plug-and-Play Design**: Supports future datasets with minimal code change.

---

## 🔄 Plug-and-Play Pipeline

Thanks to Prefect, you can:

* Run the pipeline with a single command (`python orchestration/prefect_flow.py`)
* Avoid duplicate downloads (checks if dataset exists)
* Standardize new data automatically
* Re-train and evaluate models seamlessly
* Get serialized predictions and reports

No manual wiring or editing across multiple scripts is needed.

---

## 🧪 Example Run

```bash
conda activate tf215gpu
python orchestration/prefect_flow.py
```

Prefect will:

1. ✅ Download the dataset (if not already present)
2. 📝 Generate the manifest
3. 🧼 Standardize the data
4. 🧠 Train the classifier
5. 🔮 Perform batch inference
6. 📊 Evaluate the model

---

## 📊 Sample Evaluation Output

```
📊 Classification Report:
              precision    recall  f1-score   support
           0     0.8472    0.7833    0.8140      2556
           1     0.6462    0.7371    0.6887      1373

accuracy                         0.7671      3929
macro avg     0.7467    0.7602    0.7513      3929
weighted avg  0.7770    0.7671    0.7702      3929

📉 Confusion Matrix:
[[2002  554]
 [ 361 1012]]
🔵 ROC AUC Score: 0.8578
```

---

## 📦 Requirements

Install requirements in a Conda environment:

```bash
conda activate tf215gpu
pip install -r requirements.txt
```

---
