import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "pcos_model.keras")
TEST_DIR = os.path.join(BASE_DIR, "Test_Images")

CLASS_NAMES = ["Non-PCOS", "PCOS"]

print("=" * 60)
print("Loading Model...")
model = load_model(MODEL_PATH)
print("Model Loaded Successfully!")
print("=" * 60)

for folder in CLASS_NAMES:

    folder_path = os.path.join(TEST_DIR, folder)

    correct = 0
    wrong = 0

    print("\n==============================")
    print("Testing:", folder)
    print("==============================")

    for file in os.listdir(folder_path):

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder_path, file)

        img = image.load_img(img_path, target_size=(224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        score = float(model.predict(img, verbose=0)[0][0])

        prediction = "PCOS" if score >= 0.5 else "Non-PCOS"

        if prediction == folder:
            correct += 1
        else:
            wrong += 1

            print(f"Wrong: {file}")
            print(f"Score: {score:.4f}")
            print("-"*40)

    print("\nRESULT")
    print("Correct :", correct)
    print("Wrong   :", wrong)