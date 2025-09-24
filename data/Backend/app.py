from flask import Flask, request, jsonify
from flask_cors import CORS
from model import SkinDiseaseModel
from explain import grad_cam
from PIL import Image
import numpy as np
import cv2
import io
import base64

app = Flask(__name__)
CORS(app)  # allow frontend requests

model_handler = SkinDiseaseModel("../models/skin_disease_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")

    # Prediction
    pred_class, confidence, preds = model_handler.predict(image)

    # Grad-CAM
    img_array = model_handler.preprocess(image)
    heatmap = grad_cam(model_handler.model, img_array)

    # Overlay heatmap
    img = np.array(image.resize((224,224)))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Encode heatmap
    _, buffer = cv2.imencode(".png", overlay)
    heatmap_b64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "prediction": pred_class,
        "confidence": confidence,
        "heatmap": heatmap_b64
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
