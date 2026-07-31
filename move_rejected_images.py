import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REPORT = os.path.join(BASE_DIR, "rejected_ultrasound_images.txt")

SOURCE_PCOS = os.path.join(BASE_DIR, "Test_Images", "PCOS")
SOURCE_NONPCOS = os.path.join(BASE_DIR, "Test_Images", "Non-PCOS")

DESTINATION = os.path.join(
    BASE_DIR,
    "ultrasound_dataset_ready",
    "train",
    "Ultrasound"
)

count = 0

with open(REPORT, "r") as f:

    for line in f:

        cls, filename, score = line.strip().split(",")

        if cls == "PCOS":
            src = os.path.join(SOURCE_PCOS, filename)
        else:
            src = os.path.join(SOURCE_NONPCOS, filename)

        dst = os.path.join(DESTINATION, filename)

        if os.path.exists(src):

            shutil.copy2(src, dst)
            count += 1

print("="*50)
print(f"Copied {count} rejected images.")
print("="*50)