# test_cnn_architectures.py
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os

# -------------------------------
# 1. Paths
# -------------------------------
train_dir = "../data/raw/skin disease/SkinDisease/train"
val_dir   = "../data/raw/skin disease/SkinDisease/val"

img_size = (224, 224)
batch_size = 32

# -------------------------------
# 2. Data Generators
# -------------------------------
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)

# -------------------------------
# 3. Model Builder
# -------------------------------
def build_model(base_arch, input_shape=(224,224,3), num_classes=4):
    base_model = base_arch(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------------
# 4. Architectures to Test
# -------------------------------
architectures = {
    "MobileNetV2": applications.MobileNetV2,
    "ResNet50": applications.ResNet50,
    "EfficientNetB0": applications.EfficientNetB0
}

results = {}

# -------------------------------
# 5. Train Each Architecture
# -------------------------------
for name, arch in architectures.items():
    print(f"\n=== Training {name} ===\n")
    model = build_model(arch, num_classes=train_gen.num_classes)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,  # quick comparison; increase for full training
        verbose=1
    )
    
    best_val_acc = max(history.history['val_accuracy'])
    results[name] = best_val_acc
    print(f"{name} best validation accuracy: {best_val_acc:.4f}")

# -------------------------------
# 6. Show Best Architecture
# -------------------------------
best_arch = max(results, key=results.get)
print("\n=== Summary ===")
for arch_name, acc in results.items():
    print(f"{arch_name}: {acc:.4f}")
print(f"\nBest architecture: {best_arch} with validation accuracy {results[best_arch]:.4f}")
