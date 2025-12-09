import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        elif hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return f"{layer.name}/{sublayer.name}"
    return None

def generate_gradcam_overlay(model, img_array, class_index, intensity=0.4):
    try:
        inner_model = model.layers[0] if hasattr(model.layers[0], "input") else model
        last_conv_layer_name = find_last_conv_layer(inner_model)
        if last_conv_layer_name is None:
            print("No Conv2D layer found.")
            return None

        last_conv_layer = inner_model.get_layer(last_conv_layer_name)
        grad_model = Model(inputs=inner_model.input,
                           outputs=[last_conv_layer.output, inner_model.output])

        if not isinstance(img_array, tf.Tensor):
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        if tf.reduce_max(heatmap) != 0:
            heatmap /= tf.reduce_max(heatmap)

        img = img_array[0].numpy() if isinstance(img_array[0], tf.Tensor) else img_array[0]
        img = (img*255).astype(np.uint8)
        if img.shape[-1] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255*heatmap)
        jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img, 1-intensity, jet, intensity, 0)
        return overlay
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return None



