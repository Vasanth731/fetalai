"""
Scan fetalclip_dataset folder structure and generate train_val_split.json and test_id.json.
Expects folder structure: fetalclip_dataset/BPD/train/images, fetalclip_dataset/AC/train/images, etc.
"""
import json
import os
import random
from pathlib import Path

# Configuration
BASE_PATH = "/home/htic/fetalai_data/vasanth/datasets_to_work_with/retro/fetalclip_dataset"  # Root directory containing BPD, AC, FL folders
CATEGORIES = ['BPD', 'AC', 'FL']
TRAIN_VAL_RATIO = 0.9  # 90% for train+val, 10% for test
VAL_RATIO = 0.1  # 10% of train+val for validation
NUM_FOLDS = 5  # Number of cross-validation folds
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def extract_patient_ids_from_folder(base_path, categories):
    """
    Extract all unique patient IDs from all categories.

    Args:
        base_path: Path to the root directory (e.g., 'fetalclip_dataset')
        categories: List of category names (e.g., ['BPD', 'AC', 'FL'])

    Returns:
        Sorted list of unique patient IDs
    """
    base_path = Path(base_path)
    all_patient_ids = set()

    for category in categories:
        image_dir = base_path / category / 'train' / 'images'

        if not image_dir.exists():
            print(f"Warning: Directory not found: {image_dir}")
            continue

        # Get all image files
        image_files = list(image_dir.glob('*.png')) + \
                     list(image_dir.glob('*.jpg')) + \
                     list(image_dir.glob('*.jpeg'))

        print(f"{category}: Found {len(image_files)} images")

        # Extract patient IDs from filenames
        for img_file in image_files:
            # Assuming filename format: <pid>_<addendum>.png or <pid>.png
            filename_stem = img_file.stem
            pid = filename_stem.split('_')[0] if '_' in filename_stem else filename_stem

            # Convert to string and handle numeric IDs
            try:
                pid = str(int(pid))  # Normalize numeric IDs
            except ValueError:
                pid = str(pid)  # Keep as string if not numeric

            all_patient_ids.add(pid)

    return sorted(list(all_patient_ids))

def create_splits(patient_ids, train_val_ratio=0.9, val_ratio=0.1, num_folds=5):
    """
    Create train/val/test splits with cross-validation folds.

    Args:
        patient_ids: List of all patient IDs
        train_val_ratio: Ratio of data for train+val (remaining goes to test)
        val_ratio: Ratio of train+val data for validation
        num_folds: Number of cross-validation folds

    Returns:
        train_val_dict: Dictionary with fold splits
        test_dict: Dictionary with test IDs
    """
    random.shuffle(patient_ids)

    # Split into train+val and test
    n = len(patient_ids)
    train_val_end = int(n * train_val_ratio)

    train_val_ids = patient_ids[:train_val_end]
    test_ids = patient_ids[train_val_end:]

    print(f"\nTotal patient IDs: {n}")
    print(f"Train+Val IDs: {len(train_val_ids)}")
    print(f"Test IDs: {len(test_ids)}")

    # Create cross-validation folds
    def split_into_chunks(lst, num_chunks):
        k, m = divmod(len(lst), num_chunks)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_chunks)]

    train_val_chunks = split_into_chunks(train_val_ids, num_folds)

    train_val_dict = {}
    for i, chunk in enumerate(train_val_chunks):
        # Split each chunk into train and val
        chunk_train_end = int(len(chunk) * (1 - val_ratio))
        train_chunk = chunk[:chunk_train_end]
        val_chunk = chunk[chunk_train_end:]

        train_val_dict[str(i)] = [train_chunk, val_chunk]
        print(f"Fold {i}: Train={len(train_chunk)}, Val={len(val_chunk)}")

    # Create test dictionary
    test_dict = {pid: "test" for pid in test_ids}

    return train_val_dict, test_dict

if __name__ == "__main__":
    print("="*60)
    print("GENERATING TRAIN/VAL/TEST SPLITS")
    print("="*60)
    print(f"Base path: {BASE_PATH}")
    print(f"Categories: {CATEGORIES}")
    print(f"Train+Val ratio: {TRAIN_VAL_RATIO}")
    print(f"Validation ratio: {VAL_RATIO}")
    print(f"Number of folds: {NUM_FOLDS}")
    print(f"Random seed: {RANDOM_SEED}")
    print("="*60)

    # Extract patient IDs from all categories
    print("\nExtracting patient IDs from categories...")
    patient_ids = extract_patient_ids_from_folder(BASE_PATH, CATEGORIES)

    if not patient_ids:
        print("ERROR: No patient IDs found! Please check your data directory structure.")
        exit(1)

    print(f"\nFound {len(patient_ids)} unique patient IDs")
    print(f"Sample IDs: {patient_ids[:10]}")

    # Create splits
    print("\nCreating splits...")
    train_val_dict, test_dict = create_splits(
        patient_ids,
        train_val_ratio=TRAIN_VAL_RATIO,
        val_ratio=VAL_RATIO,
        num_folds=NUM_FOLDS
    )

    # Save to JSON files
    print("\nSaving splits to JSON files...")

    with open("train_val_split.json", "w") as f:
        json.dump(train_val_dict, f, indent=4)
    print("✓ Saved train_val_split.json")

    with open("test_id.json", "w") as f:
        json.dump(test_dict, f, indent=4)
    print("✓ Saved test_id.json")

    print("\n" + "="*60)
    print("SPLITS GENERATION COMPLETE!")
    print("="*60)
    print("\nYou can now run preprocess.py to preprocess the data.") 
