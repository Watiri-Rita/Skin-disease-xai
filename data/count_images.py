import os

# Path to your dataset root
base_dir = r"C:\Users\PC\skin-disease-xai\data\raw\skin disease\SkinDisease"

splits = ["train", "val", "test"]

for split in splits:
    print(f"\n📂 {split.upper()} SET")
    split_dir = os.path.join(base_dir, split)
    total = 0

    # Go through each class (acne, eczema, ringworm, etc.)
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if os.path.isdir(class_dir):
            count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
            print(f"  {class_name}: {count}")
            total += count

    print(f"➡ Total images in {split}: {total}")
