import os
import random
import shutil

# Base dataset path
base_dir = r"C:\Users\PC\skin-disease-xai\data\raw\skin disease\SkinDisease"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Percentage of train images to move into val if val is empty
VAL_RATIO = 0.2  

for class_name in os.listdir(train_dir):
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)

    if not os.path.isdir(train_class_dir):
        continue

    os.makedirs(val_class_dir, exist_ok=True)

    # Count images in val
    val_images = [f for f in os.listdir(val_class_dir) if os.path.isfile(os.path.join(val_class_dir, f))]

    if len(val_images) == 0:
        # Get train images
        train_images = [f for f in os.listdir(train_class_dir) if os.path.isfile(os.path.join(train_class_dir, f))]
        
        # Pick some to move
        num_to_move = max(1, int(VAL_RATIO * len(train_images)))
        selected = random.sample(train_images, num_to_move)

        for f in selected:
            src = os.path.join(train_class_dir, f)
            dst = os.path.join(val_class_dir, f)
            shutil.move(src, dst)

        print(f"📦 Moved {len(selected)} {class_name} images from train → val")

print("\n✅ Validation set rebalanced.")
