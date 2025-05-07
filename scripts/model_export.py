from tensorflow.keras.models import load_model # type: ignore

# Load the existing .h5 model
model = load_model("outputs/models/classifier_model.h5")

# Save it in .keras format
model.save("outputs/models/classifier_model.keras")

print("✅ Converted .h5 → .keras successfully")
