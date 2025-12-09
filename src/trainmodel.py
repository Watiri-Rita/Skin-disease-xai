# train_transfer.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import json
import pickle

# -------------------------------
# 1. Paths
# -------------------------------
train_dir = "../data/raw/skin disease/SkinDisease/train"
val_dir   = "../data/raw/skin disease/SkinDisease/val"
model_dir = "../models"

os.makedirs(model_dir, exist_ok=True)

# -------------------------------
# 2. Data Augmentation & Loading
# -------------------------------
img_size = (224, 224)
batch_size = 32

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

print("\ Data Loaded Successfully!")
print("Class indices:", train_generator.class_indices)

# Save class indices
with open(os.path.join(model_dir, "class_indices.json"), "w") as f:
    json.dump(train_generator.class_indices, f)

#  Compute Class Weights (Handle Imbalance)
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# 4. Build Transfer Learning Model
base_model = keras.applications.MobileNetV2(
    weights="imagenet", 
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze pretrained layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(train_generator.num_classes, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 5. Train - Phase 1 (Frozen Base)
# -------------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

print("\nStarting Phase 1 Training (Frozen Base Layers)...")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# Save first model version
model.save(os.path.join(model_dir, "skin-disease-transfer.h5"))
print("\nModel saved after Phase 1: skin-disease-transfer.h5")

# Save training history
with open(os.path.join(model_dir, "training_history_phase1.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# -------------------------------
# 6. Fine-Tuning Phase 2
# -------------------------------
print("\nStarting Phase 2: Fine-tuning last layers...")

# Unfreeze last 30 layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# Save fine-tuned model
model.save(os.path.join(model_dir, "skin-disease-transfer-finetuned.h5"))
print("\n Fine-tuned model saved as: skin-disease-transfer-finetuned.h5")

# Save fine-tuning history
with open(os.path.join(model_dir, "training_history_finetune.pkl"), "wb") as f:
    pickle.dump(fine_tune_history.history, f)

print("\nTraining Complete! You can now test or deploy your model.")
