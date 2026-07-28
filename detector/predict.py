import os
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "pcos_model.keras")

model = None


def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    return model


def predict_pcos(image_path):
    print("Reading image...")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    model = get_model()

    print("Running prediction...")

    prediction = model.predict(image, verbose=0)

    prediction = prediction[0][0]

    if prediction >= 0.5:
        result = "PCOS"
        confidence = prediction * 100
    else:
        result = "Non-PCOS"
        confidence = (1 - prediction) * 100

    return result, round(float(confidence), 2)