import os

import numpy as np
from tensorflow.keras.models import load_model

from image_preprocessing import preprocess_img

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT, 'models', 'hand_gesture_model.keras')


def predict_frame(frame):
    model = load_model(MODEL_PATH)

    frame = preprocess_img(frame, 0)
    prediction = model.predict(frame)
    gesture = np.argmax(prediction)
    return gesture


