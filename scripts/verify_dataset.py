import os
import hashlib

train_path = "dataset/archive/data/train"
test_path = "dataset/archive/data/test"

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(4096)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

train_hashes = {}

print("Reading training images...")

for category in os.listdir(train_path):
    category_path = os.path.join(train_path, category)

    if os.path.isdir(category_path):
        for file in os.listdir(category_path):

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                path = os.path.join(category_path, file)
                train_hashes[get_file_hash(path)] = path

duplicates = []

print("Checking testing images...")

for category in os.listdir(test_path):
    category_path = os.path.join(test_path, category)

    if os.path.isdir(category_path):
        for file in os.listdir(category_path):

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                path = os.path.join(category_path, file)
                file_hash = get_file_hash(path)

                if file_hash in train_hashes:
                    duplicates.append((train_hashes[file_hash], path))

print("\n==============================")

if duplicates:
    print(f"Duplicate images found: {len(duplicates)}")
    print("\nFirst 10 duplicates:\n")

    for train_file, test_file in duplicates[:10]:
        print("TRAIN:", train_file)
        print("TEST :", test_file)
        print("-" * 40)

else:
    print("No duplicate images found.")

print("==============================")