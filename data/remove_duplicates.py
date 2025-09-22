import os
import hashlib
import shutil

# Base dataset path
base_dir = r"C:\Users\PC\skin-disease-xai\data\raw\skin disease\SkinDisease"
splits = ["train", "val", "test"]

def file_hash(file_path):
    """Return MD5 hash of file contents."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Track hashes across ALL splits
seen_hashes = {}

for split in splits:
    split_dir = os.path.join(base_dir, split)
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if os.path.isdir(class_dir):
            for f in os.listdir(class_dir):
                path = os.path.join(class_dir, f)
                try:
                    h = file_hash(path)
                    if h in seen_hashes:
                        # Duplicate found → remove
                        os.remove(path)
                        print(f"🗑️ Removed duplicate from {split}/{class_name}: {f}")
                    else:
                        seen_hashes[h] = path
                except Exception as e:
                    print(f"⚠️ Skipped {path}: {e}")

print("\n✅ Done! All duplicate images across train/val/test removed.")
