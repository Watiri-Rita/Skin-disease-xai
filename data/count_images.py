import os

# Path to your dataset root
base_dir = r"C:\Users\PC\skin-disease-xai\data\raw\skin disease\SkinDisease"

splits = ["train", "val", "test"]
classes = sorted([d for d in os.listdir(os.path.join(base_dir, "train")) if os.path.isdir(os.path.join(base_dir, "train", d))])

# Initialize total counters
grand_total_per_class = {cls: 0 for cls in classes}
grand_total_all = 0

for split in splits:
    print(f"{split.upper()} SET")
    split_dir = os.path.join(base_dir, split)
    total_split = 0

    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        if os.path.isdir(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
        else:
            count = 0
        print(f"  {cls}: {count}")
        total_split += count
        grand_total_per_class[cls] += count

    print(f"Total images in {split}: {total_split}")
    grand_total_all += total_split

print("\n GRAND TOTAL ACROSS ALL SPLITS")
for cls, count in grand_total_per_class.items():
    print(f"  {cls}: {count}")
print(f"Total images in dataset: {grand_total_all}")
