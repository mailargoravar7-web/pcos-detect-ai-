import os
import shutil
from sklearn.model_selection import train_test_split

# ==========================================
# Paths
# ==========================================

input_root = "processed_dataset"
output_root = "cnn_dataset"

classes = ["PCOS", "Non-PCOS"]

# ==========================================
# Create Output Folders
# ==========================================

for dataset in ["train", "validation", "test"]:
    for cls in classes:
        os.makedirs(
            os.path.join(output_root, dataset, cls),
            exist_ok=True
        )

# ==========================================
# Split Dataset
# ==========================================

for cls in classes:

    print(f"\nProcessing {cls}...")

    folder = os.path.join(input_root, cls)

    images = os.listdir(folder)

    # 70% Train
    train, temp = train_test_split(
        images,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    # Remaining 30%
    # 15% Validation
    # 15% Test
    validation, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42,
        shuffle=True
    )

    print(f"Train      : {len(train)}")
    print(f"Validation : {len(validation)}")
    print(f"Test       : {len(test)}")

    # ------------------------------
    # Copy Train
    # ------------------------------
    for img in train:

        shutil.copy(
            os.path.join(folder, img),
            os.path.join(output_root, "train", cls, img)
        )

    # ------------------------------
    # Copy Validation
    # ------------------------------
    for img in validation:

        shutil.copy(
            os.path.join(folder, img),
            os.path.join(output_root, "validation", cls, img)
        )

    # ------------------------------
    # Copy Test
    # ------------------------------
    for img in test:

        shutil.copy(
            os.path.join(folder, img),
            os.path.join(output_root, "test", cls, img)
        )

print("\n=================================")
print("Dataset Split Completed")
print("=================================")