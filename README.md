# 🧠 MedSeg ML Pipeline

A clean, production-style machine learning pipeline for brain tumor segmentation and classification using MRI scans.  
Inspired by scientific ML engineering workflows (e.g. Met Office, research labs), this project emphasizes **reproducibility**, **data curation**, and **scalable training**.

---

## 🎓 Project Background

This pipeline is a production-grade refactor of my **BSc Final Year Project**, which tackled:

- **Tumor classification**: Predicting presence of tumor (binary) from MRI slices using CNN
- **Tumor segmentation**: Pixel-level prediction of tumor regions using a ResUNet (coming soon)

---

## 📦 Current Features

### ✅ Standardized Dataset Pipeline

- Loads MRI image-mask pairs from the LGG dataset  
- Preprocesses them (grayscale, resize to 256×256, normalize to [0, 1])  
- Stores them in `.npz` format for fast, reproducible access  
- Metadata includes: patient ID, slice number, label (tumor present/absent)

### ✅ Manifest Generator

- Maps all image-mask pairs
- Extracts patient/slice metadata
- Outputs `manifest.csv`

### ✅ Image Classification Model (CNN)

- Model trained using TensorFlow/Keras
- Uses a modular CNN for binary classification (tumor vs. no tumor)
- Enhanced with:
  - Custom `tf.data.Dataset` loader
  - Strong data augmentations (flip, brightness, contrast, rotation)
  - Balanced random train/val splitting
  - Model checkpointing and early stopping
- Best model weights saved to `outputs/models/classifier_model.h5`

---

## 🚀 How to Run

```bash
# Step 1: Download dataset
python scripts/download_lgg_dataset.py

# Step 2: Generate manifest
python scripts/generate_manifest.py --input "path/to/kaggle_3m" --output data/manifest.csv

# Step 3: Standardize images into .npz format
python scripts/standardize_data.py --manifest data/manifest.csv --output data/processed

# Step 4: Train classifier (CNN)
python scripts/train_classifier.py
````

---

## 🧪 Future Work

### 🔜 In Progress

* [ ] Visualize training curves (accuracy/loss)
* [ ] Evaluate on test set with confusion matrix, AUC
* [ ] Export predictions from `.h5` model
* [ ] Add CLI to train/evaluate easily

### 🧠 Coming Soon

* [ ] ResUNet-based segmentation training pipeline
* [ ] Evaluation metrics: Dice, IoU, FP/FN overlays
* [ ] Convert outputs to HuggingFace Dataset or NetCDF format
* [ ] Integration with orchestration (e.g., Prefect or Makefile)
* [ ] GPU/Colab version for large-scale testing
* [ ] Scientific logging with `mlflow` or `wandb`

---

## 🗂 Folder Structure

```
medseg-ml-pipeline/
├── data/
│   ├── manifest.csv
│   ├── processed/                ← standardized .npz files
├── outputs/
│   └── models/                   ← saved .h5 weights
├── scripts/
│   ├── download_lgg_dataset.py
│   ├── generate_manifest.py
│   ├── standardize_data.py
│   ├── data_loader.py
│   └── train_classifier.py
├── notebooks/
│   └── Cnn_Classifier.ipynb      ← original baseline notebook
├── README.md
└── requirements.txt
```

---

## 👨‍💻 Author

Made with care by [Anzer Khan](https://github.com/Anzerkhan27)
Feel free to star ⭐, fork 🍴, or contribute 🤝

