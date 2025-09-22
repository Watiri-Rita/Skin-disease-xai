import os
import random
import shutil

# Paths
base_dir = r"C:\Users\PC\skin-disease-xai\data\raw\skin disease\SkinDisease"

eczema_dir = os.path.join(base_dir, "test", "eczema")

train_dir = os.path.join(base_dir, "train", "eczema")
val_dir = os.path.join(base_dir, "val", "eczema")
test_dir = os.path.join(base_dir, "test", "eczema")

# Make sure folders exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all eczema images
images = [f for f in os.listdir(eczema_dir) if os.path.isfile(os.path.join(eczema_dir, f))]
random.shuffle(images)

# Split sizes
train_size = int(0.7 * len(images))
val_size = int(0.2 * len(images))

train_images = images[:train_size]
val_images = images[train_size:train_size + val_size]
test_images = images[train_size + val_size:]

# Function to move files
def move_files(file_list, src, dst):
    for f in file_list:
        shutil.move(os.path.join(src, f), os.path.join(dst, f))

# Move files
move_files(train_images, eczema_dir, train_dir)
move_files(val_images, eczema_dir, val_dir)
move_files(test_images, eczema_dir, test_dir)

print(f"✅ Done! Split {len(images)} eczema images into:")
print(f"   {len(train_images)} → train/eczema")
print(f"   {len(val_images)} → val/eczema")
print(f"   {len(test_images)} → test/eczema")
