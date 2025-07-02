import json
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from image_preprocessing import preprocess_img

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT, 'data', 'dataset', 'leapGestRecog')
MODEL_DIR = os.path.join(ROOT, 'models')
subfolders = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]

imgs = []
labels = []

for i in range(0, len(subfolders)):
    sub = os.path.join(DATA_DIR, subfolders[i])
    for cat in os.listdir(sub):
        category_folder = os.path.join(sub, cat)
        for img in os.listdir(category_folder):
            img_path = os.path.join(category_folder, img)
            image = cv2.imread(img_path)
            proccesd_image = preprocess_img(image, -1)
            imgs.append(proccesd_image)
            labels.append(cat)
        # print(f"Finished Folder {category_folder}")

class_names = sorted(set(labels))
class_to_index = {name: idx for idx, name in enumerate(class_names)}
int_labels = [class_to_index[label] for label in labels]

x = np.array(imgs, dtype=np.float32)
y = to_categorical(np.array(int_labels), num_classes=10)
x, y = shuffle(x, y, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

_, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

model.save(os.path.join(MODEL_DIR, 'hand_gesture_model.keras'))

with open(os.path.join(MODEL_DIR, 'class_names.json'), 'w') as f:
    json.dump(class_names, f)
