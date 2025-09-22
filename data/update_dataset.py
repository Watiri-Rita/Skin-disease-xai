import os
import shutil
import hashlib
from collections import defaultdict

# Paths
base_dir = "C:/Users/PC/skin-disease-xai/data/raw/skin disease\SkinDisease"
splits = ["train", "val", "test"]

# ---------- 1. Remove duplicates ----------
def file_hash(file_path):
    """Return MD5 hash of a file"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def remove_duplicates():
    seen = {}
    removed = 0

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img)
                h = file_hash(img_path)
                if h in seen:
                    os.remove(img_path)
                    removed += 1
                else:
                    seen[h] = img_path
    print(f"[Duplicates Removed] {removed} images")

# ---------- 2. Rebalance validation ----------
def rebalance_val():
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        return

    for cls in os.listdir(train_dir):
        train_cls = os.path.join(train_dir, cls)
        val_cls = os.path.join(val_dir, cls)

        if not os.path.exists(val_cls):
            os.makedirs(val_cls)

        # If val class empty, move 10% of train images
        if len(os.listdir(val_cls)) == 0 and len(os.listdir(train_cls)) > 0:
            images = os.listdir(train_cls)
            move_count = max(1, len(images) // 10)
            for img in images[:move_count]:
                src = os.path.join(train_cls, img)
                dst = os.path.join(val_cls, img)
                shutil.move(src, dst)
            print(f"[Rebalanced] Moved {move_count} {cls} images to val")

# ---------- 3. Count dataset ----------
def count_dataset():
    summary = defaultdict(dict)
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                summary[split][cls] = len(os.listdir(cls_dir))

    print("\n[Dataset Summary]")
    for split in splits:
        print(f"\n{split.upper()}:")
        for cls, count in summary[split].items():
            print(f"  {cls}: {count} images")

if __name__ == "__main__":
    remove_duplicates()
    rebalance_val()
    count_dataset()
