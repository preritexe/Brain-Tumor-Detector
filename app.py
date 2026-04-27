import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import requests
from preprocess import preprocess_img
from gradcam import get_gradcam_heatmap, overlay_heatmap

MODEL_PATH = "brain_tumor_model.h5"

url = "https://github.com/preritexe/Brain-Tumor-Detector/releases/download/v2.0/brain_tumor_model.h5"

if not os.path.exists(MODEL_PATH):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

print("Model size:", os.path.getsize(MODEL_PATH))

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

st.title("🧠 Brain Tumor Detector")
st.warning("⚠️ Please upload only brain MRI images. Other images may give incorrect results.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

def is_mri_image(img):

    if len(img.shape) != 3:
        return False

    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]

    color_diff = np.mean(np.abs(b - g)) + np.mean(np.abs(b - r))

    # Only detect clearly non-MRI (color images)
    return color_diff < 50

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    if not is_mri_image(img):
        st.error("❌ This is not a valid MRI image. Please upload a brain MRI.")
    
    else:
        # ⚠️ Soft warning for grayscale (since we can't be sure)
        original_img = img.copy()
        img = preprocess_img(img)

        prob = model.predict(img)[0][0]
        heatmap = get_gradcam_heatmap(model, img, "conv2d_3")
        result_img = overlay_heatmap(original_img, heatmap)

        if prob > 0.20:
            st.error(f"⚠️ Tumor Detected (Confidence: {prob*100:.2f})")
        else:
            st.success(f"✅ No Tumor Detected (Confidence: {(1-prob)*100:.2f})")
        st.warning("⚠️ Note: If the uploaded image is not MRI please ignore the results above")   
        st.subheader("Grad-CAM Visualization")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Original MRI", use_container_width=True)

        with col2:
            st.image(result_img, caption="Model Focus (Grad-CAM)", use_container_width=True)
        st.info("🔴 Red areas indicate regions the model focused on while making the prediction. This does not always mean tumor, but shows important regions.")    
