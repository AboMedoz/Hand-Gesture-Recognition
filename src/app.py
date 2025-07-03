import json
import os

import numpy as np
from PIL import Image
import streamlit as st

from main import predict_frame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
CLASS_NAMES_PATH = os.path.join(ROOT, 'models', 'class_names.json')

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

st.write(
    """
    # Hand Gesture Recognition
    """
)

st.subheader("Take a Photo with your Webcam to Start")

frame = st.camera_input('Frame to Predict')

st.subheader("Prediction")
st.write("### Gestures")
st.write(class_names)
if frame:
    frame = Image.open(frame)
    frame = np.array(frame)
    prediction = predict_frame(frame)
    st.write(f"Predicted Class: {class_names[prediction]}")