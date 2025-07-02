import json
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from Hand_Gesture_Recognition.src.image_preprocessing import preprocess_img

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT, 'models', 'hand_gesture_model.keras')
CLASS_NAMES_PATH = os.path.join(ROOT, 'models', 'class_names.json')

cap = cv2.VideoCapture(0)
model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

while True:
    ret, frame = cap.read()

    preprocessed_frame = preprocess_img(frame, 0)
    prediction = model.predict(preprocessed_frame)
    gesture = np.argmax(prediction)
    cv2.putText(frame, f"Gesture: {class_names[gesture]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0))

    cv2.imshow('Hand Gesutre Recognition', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()