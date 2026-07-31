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
ULTRASOUND_MODEL_PATH = os.path.join(BASE_DIR, "ultrasound_model.keras")

print("=" * 60)
print("BASE_DIR :", BASE_DIR)
print("PCOS MODEL :", PCOS_MODEL_PATH)
print("ULTRASOUND MODEL :", ULTRASOUND_MODEL_PATH)
print("=" * 60)

pcos_model = None
ultrasound_model = None


# =====================================================
# LOAD ULTRASOUND MODEL
# =====================================================

def get_ultrasound_model():

    global ultrasound_model

    if ultrasound_model is None:

        print("Loading Ultrasound Model...", flush=True)

        try:

            ultrasound_model = load_model(ULTRASOUND_MODEL_PATH)

            print("Ultrasound Model Loaded Successfully.")

        except Exception:

            traceback.print_exc()
            raise

    return ultrasound_model


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
# VALIDATE ULTRASOUND IMAGE
# =====================================================

def validate_ultrasound(image_path):

    try:

        image = cv2.imread(image_path)

        if image is None:
            return False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        model = get_ultrasound_model()

        prediction = model.predict(image, verbose=0)[0][0]

        print("=" * 60)
        print(f"Ultrasound Validator Score : {prediction:.4f}")
        print("=" * 60)

        # Class mapping:
        # 0 = Not_Ultrasound
        # 1 = Ultrasound

        if prediction >= 0.5:
            return True

        return False

    except Exception:

        traceback.print_exc()
        return False


# =====================================================
# PCOS PREDICTION
# =====================================================

def predict_pcos(image_path):

    try:

        print("=" * 60)
        print("Prediction Started")
        print("=" * 60)

        # ------------------------------------------------
        # STEP 1 : Validate Ultrasound
        # ------------------------------------------------

        print("Checking whether image is an ultrasound...")

        is_ultrasound = validate_ultrasound(image_path)

        if not is_ultrasound:

            raise ValueError(
                "Invalid Image Uploaded. Please upload an ovarian ultrasound image."
            )

        print("Ultrasound image verified.")

        # ------------------------------------------------
        # STEP 2 : Read Image
        # ------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Unable to read uploaded image.")

        # ------------------------------------------------
        # STEP 3 : Preprocess
        # ------------------------------------------------

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        # ------------------------------------------------
        # STEP 4 : Load PCOS Model
        # ------------------------------------------------

        model = get_pcos_model()

        prediction = model.predict(image, verbose=0)[0][0]

        print("=" * 60)
        print(f"PCOS Model Score : {prediction:.4f}")
        print("=" * 60)

        # ------------------------------------------------
        # STEP 5 : Final Prediction
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