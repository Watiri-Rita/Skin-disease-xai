from flask import Flask, request, jsonify
from model import SkinDiseaseModel
import os

# Initialize Flask
app = Flask(__name__)

# Load model
MODEL_PATH = "../models/skin-disease-model.h5"

CLASS_NAMES = ["Acne", "Eczema", "FU-ringworm"]   # <-- update this if you have exact class names

model_handler = SkinDiseaseModel(MODEL_PATH)

# Upload & predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temporarily
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Run prediction
    predicted_label, confidence = model_handler.predict(file_path, CLASS_NAMES)

    # Remove temp file
    os.remove(file_path)

    return jsonify({
        "predicted_class": predicted_label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
