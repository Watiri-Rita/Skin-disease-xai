import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# -------------------------
# Paths
# -------------------------
model_path = "../models/skin_disease_model.h5"
test_dir = "../data/raw/skin disease/SkinDisease/test"

# -------------------------
# Load test dataset
# -------------------------
test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode="int"   # integer labels (works for transfer models too)
)

class_names = test_ds.class_names
print(f"✅ Classes found: {class_names}")

# -------------------------
# Load model
# -------------------------
model = keras.models.load_model(model_path)

# -------------------------
# Compile with correct loss
# -------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # since labels are int
    metrics=["accuracy"]
)

# -------------------------
# Evaluate
# -------------------------
loss, acc = model.evaluate(test_ds)
print(f"\n✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test Loss: {loss:.4f}")

# -------------------------
# Predictions for classification report
# -------------------------
y_true = np.concatenate([y for _, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
