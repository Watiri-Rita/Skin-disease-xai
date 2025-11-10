import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# 🔹 1. Set your model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/skin-disease-transfer.h5")

# 🔹 2. Load your model
print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# 🔹 3. Recursive function to find last convolutional layer
def find_last_conv_layer(model):
    """
    Recursively find the last Conv2D or DepthwiseConv2D layer.
    Works even for nested models (like MobileNetV2 base).
    """
    last_conv = None
    for layer in model.layers:
        # If it's a nested model (like mobilenetv2_1.00_224)
        if isinstance(layer, tf.keras.Model):
            nested_conv = find_last_conv_layer(layer)
            if nested_conv:
                last_conv = nested_conv
        elif isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            last_conv = layer
    return last_conv

# 🔹 4. Find and print the layer
last_conv_layer = find_last_conv_layer(model)

if last_conv_layer:
    print(f"✅ Last Conv layer found: {last_conv_layer.name}")
    print(f"📐 Output shape: {last_conv_layer.output.shape}")
else:
    print("⚠️ No Conv2D or DepthwiseConv2D layer found!")
