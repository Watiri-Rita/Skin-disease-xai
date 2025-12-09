import os
import random

base_dir = r"C:\Users\PC\skin-disease-xai\data\raw\skin disease\SkinDisease"

splits = ["train", "val", "test"]
TARGET_COUNT = 200  # desired images per class

for split in splits:
    split_dir = os.path.join(base_dir, split)
    
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]
        excess = len(images) - TARGET_COUNT
        
        if excess > 0:
            print(f"{split}/{class_name}: {len(images)} images, removing {excess}")
            to_delete = random.sample(images, excess)
            for img in to_delete:
                os.remove(os.path.join(class_dir, img))
        else:
            print(f"{split}/{class_name}: {len(images)} images, keeping all")
