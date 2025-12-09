
import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import cv2
from gradcam_utils import generate_gradcam_overlay, find_last_conv_layer

MODEL_PATH = "../../models/skin-disease-transfer.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Acne", "Eczema", "Ringworm", "Normal"] 
CONF_THRESHOLD = 0.60

# -------------------
# APP
# -------------------
app = Flask(__name__, template_folder="templates")

print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# autodetect last conv layer
try:
    LAST_CONV_LAYER_NAME = find_last_conv_layer(model)
    print("Detected last conv layer:", LAST_CONV_LAYER_NAME)
except:
    LAST_CONV_LAYER_NAME = None
    print("Could not detect last conv layer. Grad-CAM may fail.")

# UTILITIES
def preprocess_image(image_pil, target_size):
    img = image_pil.convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def image_to_dataurl(img_pil, fmt="PNG"):
    buffer = io.BytesIO()
    img_pil.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"
# ROUTES
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # validate upload
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    try:
        img_pil = Image.open(file.stream)
    except:
        return jsonify({"error": "invalid image"}), 400

    # preprocess
    img_array = preprocess_image(img_pil, IMG_SIZE)

    # predict
    preds = model.predict(img_array)[0]
    preds = np.array(preds, dtype=float)
    preds = np.exp(preds) / np.sum(np.exp(preds))  # softmax

    # determine class names
    num_classes = len(preds)
    class_names = CLASS_NAMES if len(CLASS_NAMES) == num_classes else [f"class_{i}" for i in range(num_classes)]

    confidences = {class_names[i]: float(preds[i]) for i in range(num_classes)}
    top_idx = int(np.argmax(preds))
    top_conf = float(preds[top_idx])
    top_label = class_names[top_idx]  # updated from final_label

    # encode original image
    disp = img_pil.copy()
    disp.thumbnail((700, 700))
    original_base64 = image_to_dataurl(disp)

    # GRAD-CAM
    gradcam_base64 = None
    if LAST_CONV_LAYER_NAME is not None:
        overlay = generate_gradcam_overlay(model, img_array, top_idx)
        if overlay is not None:
            #RGB for PIL
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlay_pil = Image.fromarray(overlay_rgb)
            gradcam_base64 = image_to_dataurl(overlay_pil)

    # RETURN JSON

    result = {
        "prediction": top_label,
        "confidence": top_conf,
        "confidences": confidences,
        "original_base64": original_base64,
        "gradcam_base64": gradcam_base64
    }

    return jsonify(result)

# RUN

if __name__ == "__main__":
    app.run(port=5000, debug=True)

