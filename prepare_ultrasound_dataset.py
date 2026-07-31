import os
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

random.seed(42)

# ===========================
# PATHS
# ===========================

PROJECT_DIR = Path.cwd()

BUSI_DIR = PROJECT_DIR / "ultrasound_dataset" / "Dataset_BUSI_with_GT"

INTEL_DIR = PROJECT_DIR / "ultrasound_dataset" / "intel-image-classification" / "seg_train"

OUTPUT_DIR = PROJECT_DIR / "ultrasound_dataset_ready"

# ===========================
# CREATE OUTPUT FOLDERS
# ===========================

for split in ["train", "validation", "test"]:
    for cls in ["Ultrasound", "Not_Ultrasound"]:
        os.makedirs(OUTPUT_DIR / split / cls, exist_ok=True)

# ===========================
# COLLECT ULTRASOUND IMAGES
# ===========================

ultrasound_images = []

for folder in ["benign", "malignant", "normal"]:
    current = BUSI_DIR / folder

    for file in current.glob("*"):
        if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            ultrasound_images.append(file)

print(f"Ultrasound Images : {len(ultrasound_images)}")

# ===========================
# COLLECT NON ULTRASOUND
# ===========================

categories = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

non_ultrasound = []

IMAGES_PER_CATEGORY = 300

for cat in categories:

    files = list((INTEL_DIR / cat).glob("*"))

    files = [
        f for f in files
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    random.shuffle(files)

    non_ultrasound.extend(files[:IMAGES_PER_CATEGORY])

print(f"Non Ultrasound Images : {len(non_ultrasound)}")

# ===========================
# SPLIT FUNCTION
# ===========================

def split_dataset(images):

    train, temp = train_test_split(
        images,
        test_size=0.20,
        random_state=42
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42
    )

    return train, val, test

# ===========================
# SPLIT DATA
# ===========================

u_train, u_val, u_test = split_dataset(ultrasound_images)

n_train, n_val, n_test = split_dataset(non_ultrasound)

# ===========================
# COPY FUNCTION
# ===========================

def copy_images(images, split, cls):

    destination = OUTPUT_DIR / split / cls

    for img in images:
        shutil.copy(img, destination / img.name)

copy_images(u_train, "train", "Ultrasound")
copy_images(u_val, "validation", "Ultrasound")
copy_images(u_test, "test", "Ultrasound")

copy_images(n_train, "train", "Not_Ultrasound")
copy_images(n_val, "validation", "Not_Ultrasound")
copy_images(n_test, "test", "Not_Ultrasound")

print("\nDataset Prepared Successfully!\n")

print(OUTPUT_DIR)