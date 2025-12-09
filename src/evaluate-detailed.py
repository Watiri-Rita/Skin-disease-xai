import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

# Load model
model = keras.models.load_model("../models/skin-disease-transfer.h5")

# Load test dataset
img_height, img_width = 224, 224
test_dir = r"../data/raw/skin disease/SkinDisease/test"

test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=32,
    shuffle=False
).map(lambda x, y: (x/255.0, y))  # Normalize images if model was trained with scaling

# Define class names
class_names = ["Acne", "FU-ringworm", "eczema", "normal"]
print("Classes:", class_names)

# Predict
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Overall accuracy
accuracy = np.sum(y_true == y_pred) / len(y_true)
print(f"\nOverall Accuracy: {accuracy:.2f}")
