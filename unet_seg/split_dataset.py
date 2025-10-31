import os
import shutil
import random
from tqdm import tqdm

# ---------------- CONFIG ----------------
main_folder = "/home/htic/fetalai_data/preprocessed/mediscan-seg/FL"  # Change this
output_folder = "/home/htic/fetalai_data/vasanth/unet_dataset/FL"  # Change this


train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# ---------------- SETUP ----------------
image_dir = os.path.join(main_folder, "images")
mask_dir = os.path.join(main_folder, "masks")

# Collect files (ignore hidden/system files)
image_files = [f for f in os.listdir(image_dir) if not f.startswith('.')]
mask_files = [f for f in os.listdir(mask_dir) if not f.startswith('.')]

# Build filename sets without extensions
image_names = {os.path.splitext(f)[0]: f for f in image_files}
mask_names = {os.path.splitext(f)[0]: f for f in mask_files}

# Find matching filenames
common_names = sorted(set(image_names.keys()) & set(mask_names.keys()))

if not common_names:
    raise ValueError("No matching filenames found between images and masks!")

print(f" Found {len(common_names)} matching image–mask pairs.")

# Pair up matching files
paired_files = [(image_names[name], mask_names[name]) for name in common_names]

# Shuffle for randomness
random.shuffle(paired_files)

# ---------------- SPLIT DATA ----------------
total = len(paired_files)
train_end = int(total * train_ratio)
val_end = int(total * (train_ratio + val_ratio))

train_set = paired_files[:train_end]
val_set = paired_files[train_end:val_end]
test_set = paired_files[val_end:]

splits = {
    "train": train_set,
    "validation": val_set,
    "test": test_set
}

# ---------------- COPY FILES ----------------
for split_name, data in splits.items():
    img_out_dir = os.path.join(output_folder, split_name, "images")
    mask_out_dir = os.path.join(output_folder, split_name, "masks")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    print(f"\n Copying {split_name} set ({len(data)} samples)...")

    for img_name, mask_name in tqdm(data, desc=f"{split_name}", total=len(data)):
        src_img = os.path.join(image_dir, img_name)
        src_mask = os.path.join(mask_dir, mask_name)
        dst_img = os.path.join(img_out_dir, img_name)
        dst_mask = os.path.join(mask_out_dir, mask_name)
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_mask, dst_mask)

print("\n Dataset successfully split and saved!")

