import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("skin-disease-transfer.h5")

# Convert to TF-Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save it
with open("skin-disease-transfer.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion complete! Saved as skin-disease-transfer.tflite")
