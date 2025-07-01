import cv2
import numpy as np


def preprocess_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = image / 255.00
    image = np.expand_dims(image, -1)
    return image
