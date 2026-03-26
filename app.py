import streamlit as st
import torch
from PIL import Image
import os
import time

import config
import model_arch
import data_utils

# -----------------------------
# Arduino Setup (FINAL FIXED)
# -----------------------------
USE_ARDUINO = True
arduino = None

if USE_ARDUINO:
    try:
        import serial
        arduino = serial.Serial('COM6', 9600, timeout=1)
        time.sleep(3)  # 🔴 IMPORTANT: wait for Arduino reset
    except Exception as e:
        print("Arduino connection failed:", e)
        arduino = None
        USE_ARDUINO = False

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="TB Detection",
    page_icon="🫁",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = model_arch.get_pytorch_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False
    )

    if not os.path.exists(config.FINAL_MODEL_SAVE_PATH):
        return None

    model.load_state_dict(
        torch.load(config.FINAL_MODEL_SAVE_PATH, map_location=config.DEVICE)
    )

    model = model.to(config.DEVICE)
    model.eval()
    return model

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image, model):
    transform = data_utils.get_data_transforms()['test']

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        return "Tuberculosis", prob
    else:
        return "Normal", 1 - prob

# -----------------------------
# UI
# -----------------------------
st.title("🫁 Tuberculosis Detection System")
st.markdown("Upload a **Chest X-ray image** to detect Tuberculosis.")

model = load_model()

if model is None:
    st.error("❌ Model not found. Train your model first.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded X-ray", width=400)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            label, confidence = predict_image(image, model)

        st.subheader("Result")

        print("Sending:", label)  # 🔍 DEBUG

        if label == "Tuberculosis":
            st.error(f"🦠 Prediction: {label}")
            st.markdown("🔴 **RED LED (TB DETECTED)**")

            if USE_ARDUINO and arduino:
                arduino.write(b'1')
                arduino.flush()
                time.sleep(0.1)

        else:
            st.success(f"✅ Prediction: {label}")
            st.markdown("🟢 **GREEN LED (NORMAL)**")

            if USE_ARDUINO and arduino:
                arduino.write(b'0')
                arduino.flush()
                time.sleep(0.1)

        st.write(f"Confidence: **{confidence*100:.2f}%**")
        st.info("⚠️ This is an AI prediction. Consult a doctor.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with PyTorch + Streamlit + Arduino")
