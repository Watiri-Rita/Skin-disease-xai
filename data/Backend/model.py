import tensorflow as tf
import numpy as np
from PIL import Image

class SkinDiseaseModel:
    def __init__(self, model_path="../models/skin_disease_model.h5"):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ["Acne", "Eczema", "Ringworm", "Normal"]

    def preprocess(self, image: Image.Image):
        image = image.resize((224,224))
        img_array = np.expand_dims(np.array(image)/255.0, axis=0)
        return img_array

    def predict(self, image: Image.Image):
        img_array = self.preprocess(image)
        preds = self.model.predict(img_array)
        pred_class = self.class_names[np.argmax(preds)]
        confidence = float(np.max(preds))
        return pred_class, confidence, preds.tolist()
