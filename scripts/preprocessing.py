import cv2
import numpy as np
import os

# ==========================================
# PCOS Image Preprocessing Pipeline
# ==========================================

# Input Image
image_path = r"dataset/archive/data/train/PCOS/img1.jpg"

# Output Folder
output_folder = "processed_images/sample"
os.makedirs(output_folder, exist_ok=True)

# Read Image
image = cv2.imread(image_path)

if image is None:
    print("❌ Error: Image not found!")

else:

    print("✅ Image Loaded")
    print("Original Shape:", image.shape)

    # ==========================================
    # STEP 1 : Resize
    # ==========================================
    resized = cv2.resize(image, (224, 224))
    cv2.imwrite(os.path.join(output_folder, "01_resized.jpg"), resized)
    print("✅ Resized Image Saved")

    # ==========================================
    # STEP 2 : Median Filter
    # ==========================================
    median = cv2.medianBlur(resized, 5)
    cv2.imwrite(os.path.join(output_folder, "02_median_filter.jpg"), median)
    print("✅ Median Filter Saved")

    # ==========================================
    # STEP 3 : Convert to Grayscale
    # ==========================================
    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, "03_grayscale.jpg"), gray)
    print("✅ Grayscale Image Saved")

    # ==========================================
    # STEP 4 : CLAHE
    # ==========================================
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_image = clahe.apply(gray)

    cv2.imwrite(
        os.path.join(output_folder, "04_clahe.jpg"),
        clahe_image
    )

    print("✅ CLAHE Saved")

    # ==========================================
    # STEP 5 : Thresholding
    # ==========================================
    _, threshold = cv2.threshold(
        clahe_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cv2.imwrite(
        os.path.join(output_folder, "05_threshold.jpg"),
        threshold
    )

    print("✅ Threshold Saved")

    # ==========================================
    # STEP 6 : Morphological Opening
    # ==========================================
    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    cv2.imwrite(
        os.path.join(output_folder, "06_opening.jpg"),
        opening
    )

    print("✅ Morphological Opening Saved")

    # ==========================================
    # STEP 7 : Segmentation
    # ==========================================
    segmented = cv2.bitwise_and(
        clahe_image,
        clahe_image,
        mask=opening
    )

    cv2.imwrite(
        os.path.join(output_folder, "07_segmented.jpg"),
        segmented
    )

    print("✅ Segmentation Saved")

    # ==========================================
    # STEP 8 : Edge Detection
    # ==========================================
    edges = cv2.Canny(
        segmented,
        50,
        150
    )

    cv2.imwrite(
        os.path.join(output_folder, "08_edges.jpg"),
        edges
    )

    print("✅ Edge Detection Saved")

    # ==========================================
    # STEP 9 : Normalization
    # ==========================================
    normalized = segmented.astype(np.float32) / 255.0

    normalized_image = (normalized * 255).astype(np.uint8)

    cv2.imwrite(
        os.path.join(output_folder, "09_normalized.jpg"),
        normalized_image
    )

    print("✅ Normalization Saved")

    print()
    print("===============================")
    print(" ALL PREPROCESSING COMPLETED ")
    print("===============================")