import os
import random

# -------------------------------
# Paths
# -------------------------------
dataset_base = "../data/raw/skin disease/SkinDisease"
splits = ["train", "val", "test"]
target_unknown_count = 200  # exactly 200 Unknown images per split

# -------------------------------
# Reduce Unknown images to exactly 200
# -------------------------------
for split in splits:
    unknown_dir = os.path.join(dataset_base, split, "Unknown")
    if not os.path.exists(unknown_dir):
        print(f"⚠️ Unknown folder not found in {split}")
        continue

    files = os.listdir(unknown_dir)
    current_count = len(files)
    
    if current_count > target_unknown_count:
        # randomly select files to delete
        files_to_delete = random.sample(files, current_count - target_unknown_count)
        for f in files_to_delete:
            os.remove(os.path.join(unknown_dir, f))
        print(f"✅ Reduced 'Unknown' images in {split} from {current_count} to {target_unknown_count}")
    elif current_count < target_unknown_count:
        print(f"ℹ️ {split} has only {current_count} 'Unknown' images (less than 200), nothing deleted")
    else:
        print(f"ℹ️ {split} already has exactly 200 'Unknown' images")

print("🎉 Done! Each split now has exactly 200 'Unknown' images (or less if there weren’t enough).")
