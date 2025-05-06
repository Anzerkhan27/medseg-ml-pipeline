import kagglehub
from pathlib import Path

# Download dataset
print("📥 Downloading LGG MRI segmentation dataset...")
dataset_path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

print("✅ Dataset downloaded to:", dataset_path)

# Inspect contents
dataset_path = Path(dataset_path)
for item in dataset_path.iterdir():
    print(f"📁 {item.name}")
    if item.is_dir():
        for sub in list(item.iterdir())[:3]:
            print(f"   └── {sub.name}")





