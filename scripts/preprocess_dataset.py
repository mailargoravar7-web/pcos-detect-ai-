import cv2
import numpy as np
import os

# ==========================================
# Input and Output Directories
# ==========================================

input_root = "dataset/archive/data/train"
output_root = "processed_dataset"

classes = ["PCOS", "Non-PCOS"]

# Create output folders
for cls in classes:
    os.makedirs(os.path.join(output_root, cls), exist_ok=True)

# ==========================================
# Process Images
# ==========================================

for cls in classes:

    input_folder = os.path.join(input_root, cls)
    output_folder = os.path.join(output_root, cls)

    print(f"\nProcessing {cls} images...")

    count = 0

    for filename in os.listdir(input_folder):

        image_path = os.path.join(input_folder, filename)

        image = cv2.imread(image_path)

        if image is None:
            continue

        # -------------------------
        # Resize
        # -------------------------
        resized = cv2.resize(image, (224, 224))

        # -------------------------
        # Median Filter
        # -------------------------
        median = cv2.medianBlur(resized, 5)

        # -------------------------
        # Grayscale
        # -------------------------
        gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

        # -------------------------
        # CLAHE
        # -------------------------
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8,8)
        )

        clahe_image = clahe.apply(gray)

        # -------------------------
        # Thresholding
        # -------------------------
        _, threshold = cv2.threshold(
            clahe_image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # -------------------------
        # Morphological Opening
        # -------------------------
        kernel = np.ones((3,3), np.uint8)

        opening = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            kernel
        )

        # -------------------------
        # Segmentation
        # -------------------------
        segmented = cv2.bitwise_and(
            clahe_image,
            clahe_image,
            mask=opening
        )

        # -------------------------
        # Normalization
        # -------------------------
        normalized = segmented.astype(np.float32) / 255.0

        # Save image
        save_image = (normalized * 255).astype(np.uint8)

        save_path = os.path.join(output_folder, filename)

        cv2.imwrite(save_path, save_image)

        count += 1

        if count % 100 == 0:
            print(f"{count} images processed...")

    print(f"Completed {cls}: {count} images")

print("\n===================================")
print("Dataset Preprocessing Completed")
print("===================================")