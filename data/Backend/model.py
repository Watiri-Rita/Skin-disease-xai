from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

class SkinDiseaseModel:
    def __init__(self, model_path):
        # Load trained model
        self.model = load_model("../models/skin_disease_model.h5"
)

    def preprocess_image(self, img_path, target_size=(224, 224)):
        """Prepares an image for prediction"""
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def predict(self, img_path, class_names):
        """Predicts the class of the given image"""
        processed_img = self.preprocess_image(img_path)
        preds = self.model.predict(processed_img)
        predicted_index = np.argmax(preds, axis=1)[0]
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(preds))
        return predicted_label, confidence
