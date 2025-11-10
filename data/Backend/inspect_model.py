from tensorflow.keras.models import load_model

MODEL_PATH = "../../models/skin-disease-transfer.h5"

model = load_model(MODEL_PATH)
print("\nModel loaded successfully!\n")

print(" Model summary:")
model.summary()

print("\n Model layers:")
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} — {layer.__class__.__name__}")
