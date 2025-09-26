# model.py - Dummy model for testing

import numpy as np

class SkinDiseaseModel:
    def __init__(self):
        print("⚠️ Using Dummy Model (no real .h5 file loaded).")

    def predict(self, image):
        # Fake prediction for testing
        diseases = ["acne", "eczema", "ringworm", "normal"]
        prediction = np.random.choice(diseases)
        return {"prediction": prediction, "confidence": round(np.random.uniform(0.7, 0.99), 2)}
