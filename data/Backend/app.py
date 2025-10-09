import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# --------------------------------------
# ✅ Paths
# --------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/skin-disease-transfer.h5")
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), "../../models/class_indices.json")

print(f"Looking for model at: {MODEL_PATH}")
print(f"Looking for class indices at: {CLASS_INDICES_PATH}")

# --------------------------------------
# ✅ Load Model
# --------------------------------------
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")

# --------------------------------------
# ✅ Load Class Indices
# --------------------------------------
try:
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    print("Class indices loaded successfully:", class_indices)
except Exception as e:
    print(f" Failed to load class_indices.json: {e}")
    class_indices = {}

# Reverse mapping (index → class label)
idx_to_class = {v: k for k, v in class_indices.items()}

# --------------------------------------
# ✅ Preprocessing Function
# --------------------------------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# --------------------------------------
# ✅ Routes
# --------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            print("❌ No file uploaded")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            print("❌ Empty filename")
            return jsonify({"error": "Empty filename"}), 400

        print(f"📂 Received file: {file.filename}")
        img_bytes = file.read()

        img_array = preprocess_image(img_bytes)
        print("✅ Preprocessing done. Shape:", img_array.shape)

        preds = model.predict(img_array)
        print("✅ Model prediction:", preds)

        pred_idx = int(np.argmax(preds))
        confidence = float(np.max(preds) * 100)
        pred_label = idx_to_class.get(pred_idx, "Unknown")

        print(f"✅ Prediction: {pred_label}, Confidence: {confidence:.2f}%")

        return jsonify({
            "prediction": pred_label,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    # 'debug=True' helps show detailed error logs in your terminal
    app.run(debug=True)