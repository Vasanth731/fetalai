"""
Script to relabel mask pixel values based on anatomical structure folder.
BPD -> 1, AC -> 2, FL -> 3
Copies relabeled masks to a new output directory.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil


def relabel_masks(base_path, output_path, categories=None, mask_folder='masks',
                  image_folder='images', copy_images=True):
    """
    Relabel mask pixel values based on the category folder and save to output directory.

    Args:
        base_path: Base directory path (e.g., 'unet_dataset_precise_split')
        output_path: Output directory path for relabeled masks
        categories: Dict mapping category names to pixel values
                   Default: {'BPD': 1, 'AC': 2, 'FL': 3}
        mask_folder: Name of the folder containing masks (default: 'masks')
        image_folder: Name of the folder containing images (default: 'images')
        copy_images: Whether to also copy the corresponding images (default: True)
    """
    if categories is None:
        categories = {'BPD': 1, 'AC': 2, 'FL': 3}

    base_path = Path(base_path)
    output_path = Path(output_path)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each category
    for category, pixel_value in categories.items():
        # Path to masks folder for this category
        mask_path = base_path / category / 'train' / mask_folder
        image_path = base_path / category / 'train' / image_folder

        # Create output directories for this category
        output_mask_path = output_path / category / 'train' / mask_folder
        output_image_path = output_path / category / 'train' / image_folder

        output_mask_path.mkdir(parents=True, exist_ok=True)
        if copy_images:
            output_image_path.mkdir(parents=True, exist_ok=True)

        if not mask_path.exists():
            print(f"Warning: Path not found - {mask_path}")
            continue

        # Get all image files in the masks folder
        mask_files = list(mask_path.glob('*.png')) + \
                     list(mask_path.glob('*.jpg')) + \
                     list(mask_path.glob('*.jpeg')) + \
                     list(mask_path.glob('*.bmp')) + \
                     list(mask_path.glob('*.tif'))

        if not mask_files:
            print(f"Warning: No mask files found in {mask_path}")
            continue

        print(f"\nProcessing {category} masks (pixel value: {pixel_value})")
        print(f"Found {len(mask_files)} masks in {mask_path}")
        print(f"Output directory: {output_mask_path}")

        # Process each mask
        for mask_file in tqdm(mask_files, desc=f"Relabeling {category}"):
            try:
                # Load mask
                mask = np.array(Image.open(mask_file))

                # Create new mask with specified pixel value
                # Assuming binary masks where non-zero pixels should be relabeled
                new_mask = np.zeros_like(mask, dtype=np.uint8)
                new_mask[mask > 0] = pixel_value

                # Save the relabeled mask to output directory
                output_file = output_mask_path / mask_file.name
                Image.fromarray(new_mask).save(output_file)

                # Copy corresponding image if requested
                if copy_images and image_path.exists():
                    # Find matching image file (may have different extension)
                    mask_stem = mask_file.stem
                    image_files = list(image_path.glob(f"{mask_stem}.*"))

                    if image_files:
                        for img_file in image_files:
                            output_img_file = output_image_path / img_file.name
                            shutil.copy2(img_file, output_img_file)

            except Exception as e:
                print(f"Error processing {mask_file}: {e}")

    print(f"\nRelabeling complete! Output saved to: {output_path}")


def verify_relabeling(output_path, categories=None, mask_folder='masks'):
    """
    Verify that masks have been correctly relabeled.

    Args:
        output_path: Output directory path where relabeled masks are stored
        categories: Dict mapping category names to expected pixel values
        mask_folder: Name of the folder containing masks
    """
    if categories is None:
        categories = {'BPD': 1, 'AC': 2, 'FL': 3}

    output_path = Path(output_path)

    print("\n" + "="*50)
    print("VERIFICATION")
    print("="*50)

    for category, expected_value in categories.items():
        mask_path = output_path / category / 'train' / mask_folder

        if not mask_path.exists():
            continue

        mask_files = list(mask_path.glob('*.png')) + \
                     list(mask_path.glob('*.jpg')) + \
                     list(mask_path.glob('*.jpeg')) + \
                     list(mask_path.glob('*.bmp')) + \
                     list(mask_path.glob('*.tif'))

        if not mask_files:
            continue

        # Check a few random masks
        sample_size = min(5, len(mask_files))
        import random
        sample_files = random.sample(mask_files, sample_size)

        print(f"\n{category} category (expected value: {expected_value}):")
        for mask_file in sample_files:
            mask = np.array(Image.open(mask_file))
            unique_values = np.unique(mask)
            print(f"  {mask_file.name}: unique values = {unique_values}")


if __name__ == "__main__":
    # Set your paths here
    BASE_PATH = "unet_dataset_precise_split"
    OUTPUT_PATH = "unet_dataset_precise_split_relabeled"

    # Define the mapping (category_name -> pixel_value)
    CATEGORIES = {
        'BPD': 1,
        'AC': 2,
        'FL': 3
    }

    # If your masks/images are in different folder names, change these
    MASK_FOLDER = 'masks'  # or 'labels', 'ground_truth', etc.
    IMAGE_FOLDER = 'images'  # or 'imgs', 'data', etc.
    COPY_IMAGES = True  # Set to False if you only want to copy masks

    print("="*50)
    print("MASK RELABELING SCRIPT")
    print("="*50)
    print(f"Input path: {BASE_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Categories: {CATEGORIES}")
    print(f"Mask folder: {MASK_FOLDER}")
    print(f"Image folder: {IMAGE_FOLDER}")
    print(f"Copy images: {COPY_IMAGES}")
    print("="*50)

    # Ask for confirmation
    response = input("\nProceed with relabeling? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        # Relabel masks and copy to output directory
        relabel_masks(
            BASE_PATH,
            OUTPUT_PATH,
            CATEGORIES,
            MASK_FOLDER,
            IMAGE_FOLDER,
            COPY_IMAGES
        )

        # Verify the relabeling
        verify_relabeling(OUTPUT_PATH, CATEGORIES, MASK_FOLDER)
    else:
        print("Operation cancelled.")
