import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "ultrasound_model.keras")

TEST_DIR = os.path.join(BASE_DIR, "Test_Images")

PCOS_DIR = os.path.join(TEST_DIR, "PCOS")
NON_PCOS_DIR = os.path.join(TEST_DIR, "Non-PCOS")

# =====================================================
# LOAD MODEL
# =====================================================

print("=" * 60)
print("Loading Ultrasound Validator...")
print("=" * 60)

model = load_model(MODEL_PATH)

print("Model Loaded Successfully!")
print("=" * 60)


# =====================================================
# PREDICT FUNCTION
# =====================================================

def predict(image_path):

    try:

        print(f"Reading : {os.path.basename(image_path)}", flush=True)

        image = cv2.imread(image_path)

        if image is None:
            print("Cannot read image.")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        image = image.astype(np.float32) / 255.0

        image = np.expand_dims(image, axis=0)

        print("Predicting...", flush=True)

        score = model.predict(image, verbose=0)[0][0]

        print(f"Score : {score:.4f}")

        return score

    except Exception as e:

        print(f"ERROR : {e}")

        return None


# =====================================================
# TESTING
# =====================================================

total = 0
correct = 0

pcos_rejected = []
healthy_rejected = []

# =====================================================
# PCOS
# =====================================================

print("\n")
print("=" * 60)
print("TESTING PCOS IMAGES")
print("=" * 60)

for file in sorted(os.listdir(PCOS_DIR)):

    if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    print(f"\nTesting : {file}")

    path = os.path.join(PCOS_DIR, file)

    score = predict(path)

    if score is None:
        continue

    total += 1

    if score >= 0.5:
        correct += 1
    else:
        pcos_rejected.append((file, score))

# =====================================================
# NON-PCOS
# =====================================================

print("\n")
print("=" * 60)
print("TESTING NON-PCOS IMAGES")
print("=" * 60)

for file in sorted(os.listdir(NON_PCOS_DIR)):

    if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    print(f"\nTesting : {file}")

    path = os.path.join(NON_PCOS_DIR, file)

    score = predict(path)

    if score is None:
        continue

    total += 1

    if score >= 0.5:
        correct += 1
    else:
        healthy_rejected.append((file, score))

# =====================================================
# RESULTS
# =====================================================

print("\n")
print("=" * 60)
print("ULTRASOUND VALIDATOR REPORT")
print("=" * 60)

print(f"Total Images Tested      : {total}")
print(f"Accepted Ultrasounds     : {correct}")
print(f"Rejected Ultrasounds     : {total - correct}")

accuracy = (correct / total) * 100 if total > 0 else 0

print(f"\nOverall Accuracy : {accuracy:.2f}%")

# =====================================================
# REJECTED PCOS
# =====================================================

print("\n")
print("=" * 60)
print(f"Rejected PCOS Images : {len(pcos_rejected)}")
print("=" * 60)

for file, score in pcos_rejected:
    print(f"{file} ---> {score:.4f}")

# =====================================================
# REJECTED HEALTHY
# =====================================================

print("\n")
print("=" * 60)
print(f"Rejected Non-PCOS Images : {len(healthy_rejected)}")
print("=" * 60)

for file, score in healthy_rejected:
    print(f"{file} ---> {score:.4f}")

print("\n")
print("=" * 60)
print("TEST COMPLETED")
print("=" * 60)

# =====================================================
# SAVE REJECTED IMAGES
# =====================================================

with open("rejected_ultrasound_images.txt", "w") as f:

    for file, score in pcos_rejected:
        f.write(f"PCOS,{file},{score:.4f}\n")

    for file, score in healthy_rejected:
        f.write(f"Non-PCOS,{file},{score:.4f}\n")

print("\nRejected image list saved as rejected_ultrasound_images.txt")