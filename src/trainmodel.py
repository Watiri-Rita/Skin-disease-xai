# train_transfer.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# -------------------------------
# 1. Paths
# -------------------------------
train_dir = "../data/raw/skin disease/SkinDisease/train"
val_dir   = "../data/raw/skin disease/SkinDisease/val"

# -------------------------------
# 2. Data Augmentation & Loading
# -------------------------------
img_size = (224, 224)
batch_size = 32

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
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

# -------------------------------
# 3. Compute Class Weights (to handle imbalance)
# -------------------------------
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# -------------------------------
# 4. Build Transfer Learning Model
# -------------------------------
base_model = keras.applications.MobileNetV2(
    weights="imagenet", 
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # freeze pretrained layers

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
# 5. Train
# -------------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# -------------------------------
# 6. Save model
# -------------------------------
os.makedirs("../models", exist_ok=True)
model.save("../models/skin-disease-transfer.h5")
print("✅ Model saved as skin-disease-transfer.h5")
