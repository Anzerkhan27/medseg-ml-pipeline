# 🧠 MedSeg ML Pipeline

A clean, production-style machine learning pipeline for brain tumor segmentation using MRI scans.  
Built from the ground up to handle real-world medical imaging workflows — from raw `.tif` images to structured manifests and model-ready data.

---


### 🎓 Original BSc Project

This pipeline is a refactored and productionized version of my **BSc Final Year Project**, which focused on:

* **Brain tumor detection** (classification) using CNNs and ResNet
* **Tumor segmentation** from MRI scans using a fine-tuned **ResUNet**
* Model training and evaluation on annotated MRI datasets with metrics like **Dice coefficient** and **IoU**

The goal was to detect and segment tumor regions from brain MRI slices — a critical task in assisting clinical diagnosis and treatment planning.

---


## 📦 What’s Inside

This repository contains everything needed to take raw brain MRI slices and convert them into a machine-learning-ready format:

- 🔍 **Manifest Generator**  
  Automatically maps original MRI images to their tumor segmentation masks  
  ➤ Extracts metadata like slice number, dimensions, patient ID  
  ➤ Outputs a clean `manifest.csv` for reproducible training workflows

- 🧪 **Future Modules (coming soon)**  
  - Data standardization (resizing, normalization, npz conversion)  
  - Training pipeline for classification & segmentation (CNN, ResUNet)  
  - Validation with Dice, IoU, and prediction overlays  
  - CLI-friendly modules and reproducible workflow

---

## 🧠 Dataset

This project uses the publicly available [LGG Brain MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) hosted on Kaggle.

MRI slices and segmentation masks are automatically downloaded and paired using `kagglehub`.

---

## 🚀 How to Run

```bash
# Download the dataset (only once)
python scripts/download_lgg_dataset.py

# Generate the manifest (image ↔ mask mapping)
python scripts/generate_manifest.py --input "path/to/kaggle_3m" --output data/manifest.csv
````

> Make sure to activate your virtual environment and install dependencies from `requirements.txt`.

---

## 🛠 Tech Stack

* Python 3.11
* pandas, Pillow
* kagglehub
* pathlib, argparse
* Virtualenv for environment management

---

## 📁 Folder Structure

```
medseg-ml-pipeline/
├── data/
│   └── manifest.csv            ← Output manifest (image ↔ mask metadata)
├── scripts/
│   ├── download_lgg_dataset.py
│   └── generate_manifest.py
├── .gitignore
├── README.md
└── .venv/
```

---

## 👨‍💻 Author

Made with care by [Anzer Khan](https://github.com/Anzerkhan27)
Feel free to star 🌟, fork 🍴, or contribute 🤝

---

```

---


