import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def inspect_npz(npz_path):
    data = np.load(npz_path)

    image = data["image"]
    mask = data["mask"] if "mask" in data.files and np.max(data["mask"]) > 0 else None
    label = data["label"]
    patient_id = str(data["patient_id"])
    slice_number = int(data["slice_number"])

    print(f"\nFile: {npz_path.name}")
    print(f"Patient: {patient_id} | Slice: {slice_number} | Label: {int(label)}")
    print(f"Image shape: {image.shape} | Mask present: {'Yes' if mask is not None else 'No'}")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("MRI Slice")
    plt.axis("off")

    if mask is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(image, cmap="gray")
        plt.imshow(mask, cmap="Reds", alpha=0.4)
        plt.title("Mask Overlay")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to .npz file")
    args = parser.parse_args()

    inspect_npz(Path(args.file))
