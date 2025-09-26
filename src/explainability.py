import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
model_path = "../models/skin-disease-model.h5"
test_folder = "../data/raw/skin disease/SkinDisease/test/Acne"   # change to any class folder
output_folder = "../gradcam_results"
img_size = (224, 224)
last_conv_layer_name = "conv2d_2"   # update if your last conv layer has a different name
# -----------------------

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load model
model = keras.models.load_model(model_path)

# Build graph by calling once
dummy_input = np.zeros((1, img_size[0], img_size[1], 3))
_ = model(dummy_input)

# Grad-CAM function
def generate_gradcam(img_path, model, last_conv_layer_name, save_path):
    # Load and preprocess image
    img = keras.utils.load_img(img_path, target_size=img_size)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    preds = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(preds[0])

    # Grad-CAM
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = keras.models.Model([model.input], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize
    heatmap = np.maximum(heatmap[0], 0)
    heatmap /= np.max(heatmap)

    # Convert to color
    heatmap = cv2.resize(heatmap, img_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    orig_img = cv2.imread(img_path)
    orig_img = cv2.resize(orig_img, img_size)
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

    # Save
    cv2.imwrite(save_path, superimposed_img)
    return predicted_class, preds[0]

# -----------------------
# Run Grad-CAM for all images in folder
# -----------------------
test_images = list(Path(test_folder).glob("*.jpeg"))  # change extension if needed

for img_path in test_images:
    img_name = Path(img_path).stem
    save_path = os.path.join(output_folder, f"{img_name}_gradcam.png")
    pred_class, probs = generate_gradcam(str(img_path), model, last_conv_layer_name, save_path)
    print(f"[✓] Processed {img_name} → class {pred_class}, saved at {save_path}")

print(f"\nAll Grad-CAM results saved in: {os.path.abspath(output_folder)}")
