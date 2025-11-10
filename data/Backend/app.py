import os
import json
import numpy as np
import base64
import cv2
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import io
from gradcam_utils import generate_gradcam_overlay

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/skin-disease-transfer.h5")
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), "../../models/class_indices.json")
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)
idx_to_class = {v:k for k,v in class_indices.items()}

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    return np.expand_dims(img_array, axis=0)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error":"No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error":"Empty filename"}), 400

        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)

        preds = model.predict(img_array)
        pred_idx = int(np.argmax(preds))
        confidence = float(np.max(preds)*100)
        pred_label = idx_to_class.get(pred_idx,"Unknown")

        # Save uploaded file (optional)
        saved_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(saved_path,"wb") as f:
            f.write(img_bytes)

        # Convert original image to Base64
        original_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        original_img = original_img.resize((224,224))
        original_array = np.array(original_img)
        _, buffer = cv2.imencode('.png', original_array)
        original_base64 = "data:image/png;base64," + base64.b64encode(buffer).decode()

        # Generate Grad-CAM overlay
        overlay_img = generate_gradcam_overlay(model, img_array, pred_idx)
        overlay_base64 = None
        if overlay_img is not None:
            _, buffer = cv2.imencode('.png', overlay_img)
            overlay_base64 = "data:image/png;base64," + base64.b64encode(buffer).decode()

        return jsonify({
            "prediction": pred_label,
            "confidence": f"{confidence:.2f}%",
            "original_base64": original_base64,
            "gradcam_base64": overlay_base64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__=="__main__":
    app.run(debug=True)
