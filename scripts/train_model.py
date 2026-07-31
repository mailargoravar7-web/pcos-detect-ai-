# ==========================================================
# Evaluate PCOS Model on Independent Test Dataset
# ==========================================================

import os
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
TEST_DIR = os.path.join(BASE_DIR, "Test_Images")

# ==========================================================
# Parameters
# ==========================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================================
# Load Model
# ==========================================================

print("\n===================================")
print("LOADING MODEL")
print("===================================\n")

model = load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ==========================================================
# Load Test Dataset
# ==========================================================

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\n===================================")
print("TEST DATASET")
print("===================================\n")

print("Images :", test_data.samples)
print("Classes:", test_data.class_indices)

# ==========================================================
# Prediction
# ==========================================================

print("\n===================================")
print("GENERATING PREDICTIONS")
print("===================================\n")

predictions = model.predict(test_data, verbose=1)

predicted_classes = (predictions >= 0.5).astype(int).flatten()

true_classes = test_data.classes

# ==========================================================
# Accuracy
# ==========================================================

accuracy = accuracy_score(true_classes, predicted_classes)

print("\n===================================")
print("RESULTS")
print("===================================\n")

print(f"Overall Accuracy : {accuracy*100:.2f}%")

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(true_classes, predicted_classes)

print("\nConfusion Matrix\n")
print(cm)

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=list(test_data.class_indices.keys()),
        digits=4,
        zero_division=0
    )
)

print("\n===================================")
print("SUMMARY")
print("===================================\n")

print("Total Images :", len(true_classes))
print("Correct Predictions :", np.sum(predicted_classes == true_classes))
print("Wrong Predictions :", np.sum(predicted_classes != true_classes))

print("\nEvaluation Completed Successfully!")