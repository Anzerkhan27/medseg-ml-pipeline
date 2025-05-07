# 🧠 MedSeg ML Pipeline

A production-style machine learning pipeline for binary classification of brain MRI slices from the LGG MRI segmentation dataset. Built with modularity and orchestration readiness in mind, this project automates the end-to-end workflow from raw image preprocessing to model training, inference, and evaluation.

---

## 🎓 Project Background

This pipeline is a production-grade refactor of my **BSc Final Year Project**, which tackled:

- **Tumor classification**: Predicting presence of tumor (binary) from MRI slices using CNN
- **Tumor segmentation**: Pixel-level prediction of tumor regions using a ResUNet (coming soon)

---

## 🚀 Features

* 📅 Downloads & preprocesses LGG MRI dataset from KaggleHub
* 🗾 Generates image-mask manifest with tumor presence labels
* 🩼 Standardizes image-mask pairs into `.npz` format
* 🧠 Trains a CNN classifier on tumor detection
* 📊 Generates classification metrics and batch predictions
* ⚙️ Orchestrated via [Prefect](https://docs.prefect.io/) (optional but included)

---

## 🔧 Requirements

Create and activate a virtual environment, then install:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
tensorflow>=2.10
numpy
pandas
scikit-learn
Pillow
matplotlib
tqdm
kagglehub
prefect>=2.0
```

---

## 🌀 How to Run the Pipeline (Prefect)

Run the full pipeline using Prefect:

```bash
python orchestration/prefect_flow.py
```

This flow performs:

1. ✅ Dataset check & download (via `kagglehub`)
2. 🗾 Manifest generation
3. 🩼 Standardization to `.npz`
4. 🧠 CNN training
5. 🔮 Batch inference over validation set
6. 📊 Metrics report (accuracy, precision, recall, F1, AUC)

---

## 📁 Project Structure

```
.
├── scripts/                # Modular script files
├── outputs/                # Model + predictions
│   ├── models/
│   └── predictions/
├── data/                   # Standardized and raw data
├── orchestration/          # Prefect flow definition
├── requirements.txt
└── README.md
```

---

## 🏥 Real-World Use Case: Healthcare AI Integration

This pipeline demonstrates how a real-world medical ML system might be structured:

* Hospitals could schedule weekly/automated retraining using incoming radiology data.
* Model performance can be monitored, evaluated, and version-controlled.
* Orchestration (via Prefect) allows QA engineers or ML Ops to manage deployments cleanly.
* Batch inference reports can help prioritize high-risk patients for review.

⚙️ Tools like **Prefect** make this setup scalable and production-ready.

---

## 🧪 Run Individual Components

You can run each module separately if you prefer:

```bash
# Generate manifest
python scripts/generate_manifest.py --input /path/to/raw --output data/manifest.csv

# Standardize
python scripts/standardize_data.py

# Train
python scripts/train_classifier.py

# Inference
python scripts/infer_batch.py --model outputs/models/classifier_model.keras --data data/processed --name processed --out outputs/predictions

# Evaluate
python scripts/metrics_report.py
```

---

## 📬 Contact

Built by [Anzer Khan](https://github.com/Anzerkhan27) | MSc Artificial Intelligence | 2025
