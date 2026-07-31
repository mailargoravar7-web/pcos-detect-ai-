import os
import traceback
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PCOS_MODEL_PATH = os.path.join(BASE_DIR, "pcos_model.keras")

print("=" * 60)
print("BASE_DIR :", BASE_DIR)
print("PCOS MODEL :", PCOS_MODEL_PATH)
print("=" * 60)

pcos_model = None


# =====================================================
# LOAD PCOS MODEL
# =====================================================

def get_pcos_model():

    global pcos_model

    if pcos_model is None:

        print("Loading PCOS Model...", flush=True)

        try:
            pcos_model = load_model(PCOS_MODEL_PATH)
            print("PCOS Model Loaded Successfully.")

        except Exception:
            traceback.print_exc()
            raise

    return pcos_model


# =====================================================
# PCOS PREDICTION
# =====================================================

def predict_pcos(image_path):

    try:

        print("=" * 60)
        print("Prediction Started")
        print("=" * 60)

        # ------------------------------------------------
        # STEP 1 : Read Image
        # ------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Unable to read uploaded image.")

        # ------------------------------------------------
        # STEP 2 : Preprocess Image
        # ------------------------------------------------

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        # ------------------------------------------------
        # STEP 3 : Load Model
        # ------------------------------------------------

        model = get_pcos_model()

        prediction = model.predict(image, verbose=0)[0][0]

        print("=" * 60)
        print(f"PCOS Model Score : {prediction:.4f}")
        print("=" * 60)

        # ------------------------------------------------
        # STEP 4 : Prediction
        # ------------------------------------------------

        if prediction >= 0.5:
            result = "PCOS"
            confidence = prediction * 100
        else:
            result = "Non-PCOS"
            confidence = (1 - prediction) * 100

        confidence = round(float(confidence), 2)

        print("=" * 60)
        print("Prediction :", result)
        print("Confidence :", confidence)
        print("=" * 60)

        return result, confidence

    except Exception:
        traceback.print_exc()
        raise