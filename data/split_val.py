import os
import shutil
import random

# Paths
dataset_dir = r"C:/Users/PC/skin-disease-xai/data/raw/skin disease/SkinDisease/train"
val_dir = r"C:/Users/PC/skin-disease-xai/data/raw/skin disease/SkinDisease/val"

print("Looking inside:", dataset_dir)

# Create val directory if it doesn’t exist
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# Loop through each class folder
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    print("Checking folder:", class_path)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        print(f"  Found {len(images)} images in {class_name}")

        if len(images) == 0:
            continue

        random.shuffle(images)

        # Take 20% for validation
        val_size = max(1, int(0.2 * len(images)))
        val_images = images[:val_size]

        # Create class folder inside val/
        class_val_path = os.path.join(val_dir, class_name)
        if not os.path.exists(class_val_path):
            os.makedirs(class_val_path)

        # Copy files instead of moving
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(class_val_path, img)
            shutil.copy(src, dst)

        print(f"✅ Copied {len(val_images)} images from {class_name} to {class_val_path}")
