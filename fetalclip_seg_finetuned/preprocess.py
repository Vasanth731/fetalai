import os
import json
import shutil

import cv2
import numpy as np
import albumentations as A

from tqdm import tqdm
from PIL import Image
from pathlib import Path

with open('config.json', 'r') as file:
    config = json.load(file)

DIR_DATA = config['paths']['dir_data']
PATH_TRAIN_VAL_SPLIT = config['paths']['path_train_val_split']
PATH_TEST_SPLIT = config['paths']['path_test_split']
SAVE_PREPROCESSED_DIR = config['paths']['dir_preprocessed']

SAVE_IMAGE_SIZE = 224
N_AUGMENTATIONS = 5

# MULTICLASS SEGMENTATION: Categories mapped to class labels
# Background = 0, BPD = 1, AC = 2, FL = 3
CATEGORIES = {'BPD': 1, 'AC': 2, 'FL': 3}
NUM_CLASSES = 4  # Background (0) + BPD (1) + AC (2) + FL (3)

with open(PATH_TRAIN_VAL_SPLIT, 'r') as file:
    dict_list_pid = json.load(file)

with open(PATH_TEST_SPLIT, 'r') as file:
    list_test_pid = json.load(file)

list_test_pid = list(list_test_pid.keys())

AUGMENTATION = A.Compose([
    A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.5),
    A.CLAHE(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.2,
        scale_limit=0.0,
        rotate_limit=20,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT, value=0, p=1.
    ),
])

preprocessing = A.Compose([
    A.Resize(
        SAVE_IMAGE_SIZE, SAVE_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC,
        mask_interpolation=0, always_apply=True
    ),
])

if os.path.exists(SAVE_PREPROCESSED_DIR):
    shutil.rmtree(SAVE_PREPROCESSED_DIR)
os.makedirs(os.path.join(SAVE_PREPROCESSED_DIR, 'train'))
os.makedirs(os.path.join(SAVE_PREPROCESSED_DIR, 'val'))
os.makedirs(os.path.join(SAVE_PREPROCESSED_DIR, 'test'))

def make_image_square_with_zero_padding(image):
    width, height = image.size

    # Determine the size of the square
    max_side = max(width, height)

    # Create a new square image with black padding (0 for black in RGB or L modes)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    if image.mode == "RGB":
        padding_color = (0, 0, 0)  # Black for RGB images
    elif image.mode == "L":
        padding_color = 0  # Black for grayscale images

    # Create a new square image
    new_image = Image.new(image.mode, (max_side, max_side), padding_color)

    # Calculate padding
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2

    # Paste the original image in the center of the new square image
    new_image.paste(image, (padding_left, padding_top))

    return new_image

def fill_contour(img):
        assert len(img.shape) == 2
        assert len(np.unique(img)) == 2
        assert img.dtype == np.uint8

        out = img.copy()
        contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)

        return out
list_pid_train = []
list_pid_val = []
for _, v in dict_list_pid.items():
    list_pid_train.extend(v[0])
    list_pid_val.extend(v[1])
list_pid_train = list(set(list_pid_train))
list_pid_val   = list(set(list_pid_val))

len(list_pid_train), len(list_pid_val)

# Normalize patient IDs: try to convert to int for numeric IDs, keep as string otherwise
def normalize_pid(pid):
    try:
        return str(int(pid))
    except ValueError:
        return str(pid)

list_pid_train = [normalize_pid(l) for l in list_pid_train]
list_pid_val   = [normalize_pid(l) for l in list_pid_val]
list_test_pid  = [normalize_pid(l) for l in list_test_pid]

# Process data from category-based folder structure
# fetalclip_dataset/BPD/train/images, fetalclip_dataset/BPD/train/masks, etc.
base_path = Path(DIR_DATA)

print("Processing TRAIN data...")
for category, class_label in tqdm(CATEGORIES.items(), desc="Categories"):
    image_dir = base_path / category / 'train' / 'images'
    mask_dir = base_path / category / 'train' / 'masks'

    if not image_dir.exists() or not mask_dir.exists():
        print(f"Warning: {category} train directories not found, skipping...")
        continue

    # Get all image files
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')) +
                        list(image_dir.glob('*.jpeg')))

    for img_file in tqdm(image_files, desc=f"Processing {category} train", leave=False):
        # Extract patient ID from filename (assuming format like <pid>_<addendum>.png or just <pid>.png)
        filename_stem = img_file.stem
        # Try to extract patient ID (first part before underscore, or entire stem if no underscore)
        pid = filename_stem.split('_')[0] if '_' in filename_stem else filename_stem

        # Check if this patient is in training set
        if pid not in list_pid_train:
            continue

        # Load image and corresponding mask
        mask_file = mask_dir / img_file.name
        if not mask_file.exists():
            print(f"Warning: Mask not found for {img_file.name}, skipping...")
            continue

        img = Image.open(img_file)
        ann = Image.open(mask_file).convert("L")

        assert np.array(ann).shape[:2] == np.array(img).shape[:2]

        img = make_image_square_with_zero_padding(img)
        ann = make_image_square_with_zero_padding(ann)

        img = np.array(img)
        ann = np.array(ann)

        # MULTICLASS: Relabel mask to use class_label for this category
        # Any non-zero pixel in the mask becomes the category's class label
        relabeled_ann = np.zeros_like(ann, dtype=np.uint8)
        relabeled_ann[ann > 0] = class_label
        ann = relabeled_ann

        assert ann.shape[:2] == img.shape[:2]

        data = {'img': img.copy(), 'label': ann.copy()}

        # Create augmented versions
        for i in range(N_AUGMENTATIONS):
            img = data['img'].copy()
            ann = data['label'].copy()

            if AUGMENTATION:
                augmented = AUGMENTATION(image=img, mask=ann)
                img = augmented['image']
                ann = augmented['mask']

            img_ann = preprocessing(image=img, mask=ann)
            img = img_ann['image']
            ann = img_ann['mask']

            # Save with category and original filename
            save_name = f"{pid}_{category}_{img_file.stem}_{i}.npz"
            np.savez(
                os.path.join(SAVE_PREPROCESSED_DIR, "train", save_name),
                img=img, ann=ann,
            )

print("\nProcessing VALIDATION data...")
for category, class_label in tqdm(CATEGORIES.items(), desc="Categories"):
    image_dir = base_path / category / 'train' / 'images'
    mask_dir = base_path / category / 'train' / 'masks'

    if not image_dir.exists() or not mask_dir.exists():
        print(f"Warning: {category} train directories not found, skipping...")
        continue

    # Get all image files
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')) +
                        list(image_dir.glob('*.jpeg')))

    for img_file in tqdm(image_files, desc=f"Processing {category} val", leave=False):
        filename_stem = img_file.stem
        pid = filename_stem.split('_')[0] if '_' in filename_stem else filename_stem

        # Check if this patient is in validation set
        if pid not in list_pid_val:
            continue

        # Load image and corresponding mask
        mask_file = mask_dir / img_file.name
        if not mask_file.exists():
            print(f"Warning: Mask not found for {img_file.name}, skipping...")
            continue

        img = Image.open(img_file)
        ann = Image.open(mask_file).convert("L")

        assert np.array(ann).shape[:2] == np.array(img).shape[:2]

        img = make_image_square_with_zero_padding(img)
        ann = make_image_square_with_zero_padding(ann)

        img = np.array(img)
        ann = np.array(ann)

        # MULTICLASS: Relabel mask to use class_label for this category
        relabeled_ann = np.zeros_like(ann, dtype=np.uint8)
        relabeled_ann[ann > 0] = class_label
        ann = relabeled_ann

        assert ann.shape[:2] == img.shape[:2]

        img_ann = preprocessing(image=img, mask=ann)
        img = img_ann['image']
        ann = img_ann['mask']

        # Save without augmentation for validation
        save_name = f"{pid}_{category}_{img_file.stem}.npz"
        np.savez(
            os.path.join(SAVE_PREPROCESSED_DIR, "val", save_name),
            img=img, ann=ann,
        )

print("\nProcessing TEST data...")
for category, class_label in tqdm(CATEGORIES.items(), desc="Categories"):
    image_dir = base_path / category / 'train' / 'images'
    mask_dir = base_path / category / 'train' / 'masks'

    if not image_dir.exists() or not mask_dir.exists():
        print(f"Warning: {category} train directories not found, skipping...")
        continue

    # Get all image files
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')) +
                        list(image_dir.glob('*.jpeg')))

    for img_file in tqdm(image_files, desc=f"Processing {category} test", leave=False):
        filename_stem = img_file.stem
        pid = filename_stem.split('_')[0] if '_' in filename_stem else filename_stem

        # Check if this patient is in test set
        if pid not in list_test_pid:
            continue

        # Load image and corresponding mask
        mask_file = mask_dir / img_file.name
        if not mask_file.exists():
            print(f"Warning: Mask not found for {img_file.name}, skipping...")
            continue

        img = Image.open(img_file)
        ann = Image.open(mask_file).convert("L")

        assert np.array(ann).shape[:2] == np.array(img).shape[:2]

        img = make_image_square_with_zero_padding(img)
        ann = make_image_square_with_zero_padding(ann)

        img = np.array(img)
        ann = np.array(ann)

        # MULTICLASS: Relabel mask to use class_label for this category
        relabeled_ann = np.zeros_like(ann, dtype=np.uint8)
        relabeled_ann[ann > 0] = class_label
        ann = relabeled_ann

        assert ann.shape[:2] == img.shape[:2]

        img_ann = preprocessing(image=img, mask=ann)
        img = img_ann['image']
        ann = img_ann['mask']

        # Save without augmentation for test
        save_name = f"{pid}_{category}_{img_file.stem}.npz"
        np.savez(
            os.path.join(SAVE_PREPROCESSED_DIR, "test", save_name),
            img=img, ann=ann,
        )

print(f"\nPreprocessing complete!")
print(f"Train samples: {len(os.listdir(os.path.join(SAVE_PREPROCESSED_DIR, 'train')))}")
print(f"Val samples: {len(os.listdir(os.path.join(SAVE_PREPROCESSED_DIR, 'val')))}")
print(f"Test samples: {len(os.listdir(os.path.join(SAVE_PREPROCESSED_DIR, 'test')))}")
 
