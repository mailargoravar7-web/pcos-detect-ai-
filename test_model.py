import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "pcos_model.keras")

TEST_DIR = os.path.join(BASE_DIR, "Test_Images")

CLASS_NAMES = ["Non-PCOS", "PCOS"]

print("=" * 60)
print("Loading Model...")
model = load_model(MODEL_PATH)
print("Model Loaded Successfully!")
print("=" * 60)

y_true = []
y_pred = []

for label, folder in enumerate(CLASS_NAMES):

    folder_path = os.path.join(TEST_DIR, folder)

    print(f"\nTesting {folder}...")

    for file in os.listdir(folder_path):

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder_path, file)

        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        score = float(model.predict(img, verbose=0)[0][0])

        prediction = 1 if score >= 0.5 else 0

        y_true.append(label)
        y_pred.append(prediction)

accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 60)
print(f"Overall Accuracy : {accuracy*100:.2f}%")
print("=" * 60)

print("\nConfusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report")
print(classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
))