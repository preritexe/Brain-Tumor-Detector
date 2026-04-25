import tensorflow as tf
import numpy as np
import cv2

def get_gradcam_heatmap(model, img, last_conv_layer_name):
    
    # Create a model that outputs:
    # 1. Last conv layer output
    # 2. Final prediction
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record operations for gradient calculation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, 0]  # tumor class

    # Compute gradients of loss wrt conv layer
    grads = tape.gradient(loss, conv_outputs)

    # Average gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Weight feature maps by gradients
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = np.power(heatmap, 0.5)

    return heatmap

def overlay_heatmap(img,heatmap):
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # Normalize
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    # 🔥 Keep top 10% only
    threshold = np.percentile(heatmap, 70)
    heatmap[heatmap < threshold] = 0

    # 🔥 Convert to binary mask
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    # Resize
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 🔥 Create red mask
    colored = np.zeros_like(img)
    colored = cv2.applyColorMap(
    np.uint8(255 * heatmap), 
    cv2.COLORMAP_JET
)

    # Overlay
    superimposed_img = cv2.addWeighted(img, 0.7, colored, 0.3, 0)

    return superimposed_img