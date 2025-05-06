import tensorflow as tf
import numpy as np
import os
from pathlib import Path

def load_npz(path):
    """Load image, label, and optionally mask from .npz file"""
    with np.load(path.numpy().decode("utf-8")) as data:
        image = data["image"].astype(np.float32)
        label = data["label"].astype(np.float32)
        mask = data["mask"].astype(np.float32) if "mask" in data.files else None
    return image, label, mask

def tf_wrapper(path, task="classification"):
    """TensorFlow wrapper for loading .npz files"""
    def wrapped_fn(p):
        image, label, mask = tf.py_function(
            load_npz,
            [p],
            [tf.float32, tf.float32, tf.float32 if task == "segmentation" else tf.float32]
        )
        image.set_shape([256, 256])
        if task == "classification":
            label.set_shape([])
            return tf.expand_dims(image, axis=-1), label  # shape: (256,256,1), scalar label
        else:
            mask.set_shape([256, 256])
            return (tf.expand_dims(image, axis=-1), tf.expand_dims(mask, axis=-1))  # both shape: (256,256,1)
    return wrapped_fn


def build_dataset(data_dir, task="classification", shuffle=True):
    data_dir = Path(data_dir)
    npz_files = sorted([str(p) for p in data_dir.glob("*.npz")])
    ds = tf.data.Dataset.from_tensor_slices(npz_files)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(npz_files))
    return ds, len(npz_files)

