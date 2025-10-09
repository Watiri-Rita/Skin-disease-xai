import os
import shutil
import random

# --- Paths ---
base_dir = "../data/raw/skin disease/SkinDisease"
source_dir = os.path.join(base_dir, "normal")  # your current Normal folder

train_dir = os.path.join(base_dir, "train", "normal")
val_dir = os.path.join(base_dir, "val", "normal")
test_dir = os.path.join(base_dir, "test", "normal")

# --- Create folders if missing ---
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# --- Get all images ---
images = [f for f in os.listdir(source_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(images)

# --- Split ratios ---
train_split = 0.7
val_split = 0.2
test_split = 0.1

n_total = len(images)
n_train = int(n_total * train_split)
n_val = int(n_total * val_split)

train_files = images[:n_train]
val_files = images[n_train:n_train + n_val]
test_files = images[n_train + n_val:]

# --- Copy files ---
for f in train_files:
    shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))

for f in val_files:
    shutil.copy(os.path.join(source_dir, f), os.path.join(val_dir, f))

for f in test_files:
    shutil.copy(os.path.join(source_dir, f), os.path.join(test_dir, f))

print(f"✅ Done splitting {len(images)} Normal images:")
print(f" - Train: {len(train_files)}")
print(f" - Val:   {len(val_files)}")
print(f" - Test:  {len(test_files)}")
