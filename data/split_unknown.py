import os
import shutil
import random

# Dataset root
base_dir = r"C:\Users\PC\skin-disease-xai\data\raw\skin disease\SkinDisease"

train_unknown = os.path.join(base_dir, "train", "unknown")
val_unknown   = os.path.join(base_dir, "val", "unknown")
test_unknown  = os.path.join(base_dir, "test", "unknown")

# Create target folders if missing
os.makedirs(train_unknown, exist_ok=True)
os.makedirs(val_unknown, exist_ok=True)

# How many to move
MOVE_TO_TRAIN_PERCENT = 0.30
MOVE_TO_VAL_PERCENT   = 0.20

# Get only images inside test/unknown
images = [
    f for f in os.listdir(test_unknown)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

if len(images) == 0:
    print("⚠️ No images found in test/unknown.")
    exit()

random.shuffle(images)

num_train = int(len(images) * MOVE_TO_TRAIN_PERCENT)
num_val   = int(len(images) * MOVE_TO_VAL_PERCENT)

train_images = images[:num_train]
val_images   = images[num_train:num_train + num_val]

# Move images
for img in train_images:
    shutil.move(os.path.join(test_unknown, img),
                os.path.join(train_unknown, img))

for img in val_images:
    shutil.move(os.path.join(test_unknown, img),
                os.path.join(val_unknown, img))

print(f"Moved {len(train_images)} images → train/unknown")
print(f"Moved {len(val_images)} images → val/unknown")
print("Done.")
