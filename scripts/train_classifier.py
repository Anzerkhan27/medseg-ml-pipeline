import tensorflow as tf
import os
from pathlib import Path
from data_loader import build_dataset, tf_wrapper

# ----------------------
# Configuration
# ----------------------
BATCH_SIZE = 32
EPOCHS = 30
INPUT_SHAPE = (256, 256, 1)
DATA_DIR = "data/processed"
MODEL_DIR = "outputs/models"
MODEL_PATH = os.path.join(MODEL_DIR, "classifier_model.h5")

# ----------------------
# Model Definition
# ----------------------
def build_cnn(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# ----------------------
# Data Augmentation
# ----------------------
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

# ----------------------
# Main Training Loop
# ----------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("🔄 Loading dataset...")
    ds, total = build_dataset(DATA_DIR, task="classification", shuffle=False)

    print("🔀 Shuffling and splitting...")
    ds = ds.shuffle(buffer_size=total, seed=42)

    val_count = max(1, int(0.1 * total))
    val_ds = ds.take(val_count).map(tf_wrapper("classification")).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_ds = ds.skip(val_count).map(tf_wrapper("classification"))
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("🏗 Building model...")
    model = build_cnn(INPUT_SHAPE)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)
    ]

    print("🚀 Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"✅ Best model weights saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
