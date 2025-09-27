from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # allow requests from your frontend

# Load model
MODEL_PATH = "../../models/skin-disease-model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (update if needed)
CLASS_NAMES = ["Acne", "Eczema", "Ringworm", "Normal"]

def preprocess_image(image_bytes):
    """Preprocess uploaded image for model input"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # match your training size
    img_array = np.array(img) / 255.0  # normalize
    return np.expand_dims(img_array, axis=0)  # add batch dimension

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Preprocess and predict
        img_array = preprocess_image(file.read())
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        return jsonify({
            "class": CLASS_NAMES[predicted_idx],
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
