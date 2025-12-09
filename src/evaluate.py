import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# -------------------------
# Paths
# -------------------------
model_path = "../../models/skin-disease-model.h5"
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
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# =====================================================
# 1. LOAD YOUR BEST TRAINED MODEL
# =====================================================
model = tf.keras.models.load_model("best_model.h5")  
print("Model loaded successfully!")

# =====================================================
# 2. SET UP TEST DATA
# =====================================================
test_dir = "data/test"   # change to your test folder

IMG_SIZE = 224
BATCH = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False
)

# =====================================================
# 3. PREDICT ON TEST DATA
# =====================================================
print("Predicting on test data...")
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# True labels
y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

# =====================================================
# 4. GENERATE METRICS
# =====================================================
print("\n=== CLASSIFICATION REPORT ===")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Save report as text file
with open("classification_report.txt", "w") as f:
    f.write(report)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

# =====================================================
# 5. CONFUSION MATRIX
# =====================================================
print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Save confusion matrix as text file
np.savetxt("confusion_matrix.txt", cm, fmt="%d")

print("\nEvaluation complete! Reports saved:")
print(" - classification_report.txt")
print(" - confusion_matrix.txt")
