import os

# ===========================
# Dataset Paths
# ===========================

train_path = "dataset/archive/data/train"
test_path = "dataset/archive/data/test"

# ===========================
# Counters
# ===========================

train_total = 0
test_total = 0

# ===========================
# Training Dataset
# ===========================

print("=" * 60)
print("TRAINING DATASET")
print("=" * 60)

for category in sorted(os.listdir(train_path)):

    category_path = os.path.join(train_path, category)

    if os.path.isdir(category_path):

        images = [
            file for file in os.listdir(category_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"\nClass : {category}")
        print(f"Number of Images : {len(images)}")

        print("First 5 Images:")

        for image in images[:5]:
            print("   ", image)

        train_total += len(images)

# ===========================
# Testing Dataset
# ===========================

print("\n")
print("=" * 60)
print("TESTING DATASET")
print("=" * 60)

for category in sorted(os.listdir(test_path)):

    category_path = os.path.join(test_path, category)

    if os.path.isdir(category_path):

        images = [
            file for file in os.listdir(category_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"\nClass : {category}")
        print(f"Number of Images : {len(images)}")

        print("First 5 Images:")

        for image in images[:5]:
            print("   ", image)

        test_total += len(images)

# ===========================
# Summary
# ===========================

print("\n")
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Total Training Images : {train_total}")
print(f"Total Testing Images  : {test_total}")
print(f"Grand Total Images    : {train_total + test_total}")

print("=" * 60)