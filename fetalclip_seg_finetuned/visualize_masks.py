"""
Script to verify and visualize relabeled masks.
The masks have pixel values 0, 1, 2, 3 which appear black in normal viewers.
This script helps visualize them with proper colors.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def check_mask_values(mask_path):
    """
    Check the unique pixel values in a mask file.

    Args:
        mask_path: Path to the mask file

    Returns:
        numpy array of unique values
    """
    mask = np.array(Image.open(mask_path))
    unique_values = np.unique(mask)
    return unique_values, mask


def visualize_mask(mask_path, save_path=None, show=True):
    """
    Visualize a segmentation mask with proper colors.

    Args:
        mask_path: Path to the mask file
        save_path: Optional path to save the visualization
        show: Whether to display the plot
    """
    mask = np.array(Image.open(mask_path))

    # Define colors for each class
    # 0: black (background)
    # 1: red (BPD)
    # 2: green (AC)
    # 3: blue (FL)
    colors = ['black', 'red', 'green', 'blue']
    n_classes = 4
    cmap = mcolors.ListedColormap(colors[:n_classes])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Get unique values and statistics
    unique_values = np.unique(mask)
    non_zero_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    percentage = (non_zero_pixels / total_pixels) * 100

    # Create figure with more info
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Show original (will appear mostly black)
    ax1.imshow(mask, cmap='gray', vmin=0, vmax=255)
    ax1.set_title(f'Original View (8-bit grayscale)\nUnique values: {unique_values}')
    ax1.axis('off')

    # Show with proper colors
    im = ax2.imshow(mask, cmap=cmap, norm=norm)

    # Build detailed title
    if len(unique_values) == 1 and unique_values[0] == 0:
        title2 = f'⚠️ EMPTY MASK - ALL ZEROS\n(Black=0/background only)'
    else:
        class_names = {0: 'Background', 1: 'BPD', 2: 'AC', 3: 'FL'}
        present_classes = [class_names.get(v, f'Unknown({v})') for v in unique_values]
        title2 = f'Colored Visualization\nClasses present: {", ".join(present_classes)}\n({percentage:.2f}% non-zero pixels)'

    ax2.set_title(title2)
    ax2.axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, ticks=[0, 1, 2, 3])
    cbar.set_label('Class ID')
    cbar.ax.set_yticklabels(['0-BG', '1-BPD', '2-AC', '3-FL'])

    plt.suptitle(f'Mask: {Path(mask_path).name}\nShape: {mask.shape}', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    if show:
        plt.show()

    plt.close()


def verify_dataset(output_path, categories=None, mask_folder='masks', num_samples=3):
    """
    Verify and visualize masks from the relabeled dataset.

    Args:
        output_path: Path to the output directory with relabeled masks
        categories: Dict mapping category names to expected pixel values
        mask_folder: Name of the folder containing masks
        num_samples: Number of sample masks to check per category
    """
    if categories is None:
        categories = {'BPD': 1, 'AC': 2, 'FL': 3}

    output_path = Path(output_path)

    print("="*60)
    print("MASK VERIFICATION")
    print("="*60)

    for category, expected_value in categories.items():
        mask_path = output_path / category / 'train' / mask_folder

        if not mask_path.exists():
            print(f"\n{category}: Path not found - {mask_path}")
            continue

        # Get all mask files
        mask_files = list(mask_path.glob('*.png')) + \
                     list(mask_path.glob('*.jpg')) + \
                     list(mask_path.glob('*.jpeg'))

        if not mask_files:
            print(f"\n{category}: No mask files found")
            continue

        print(f"\n{category} Category (Expected value: {expected_value})")
        print(f"Found {len(mask_files)} masks in {mask_path}")

        # Check sample masks
        sample_size = min(num_samples, len(mask_files))
        sample_files = mask_files[:sample_size]

        for i, mask_file in enumerate(sample_files, 1):
            unique_values, mask = check_mask_values(mask_file)
            non_zero_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            percentage = (non_zero_pixels / total_pixels) * 100

            print(f"\n  Sample {i}: {mask_file.name}")
            print(f"    Unique values: {unique_values}")
            print(f"    Mask region: {non_zero_pixels}/{total_pixels} pixels ({percentage:.2f}%)")

            # Check if values are correct
            if len(unique_values) > 2:
                print(f"    ⚠️  WARNING: More than 2 unique values found!")
            elif len(unique_values) == 2 and expected_value in unique_values and 0 in unique_values:
                print(f"    ✓ CORRECT: Background (0) and {category} ({expected_value})")
            elif len(unique_values) == 1 and unique_values[0] == 0:
                print(f"    ⚠️  WARNING: Only background pixels (all zeros)")
            else:
                print(f"    ⚠️  WARNING: Unexpected values: {unique_values}")


def visualize_samples(output_path, categories=None, mask_folder='masks',
                     num_samples=1, output_viz_folder='mask_visualizations'):
    """
    Create visualizations for sample masks from each category.

    Args:
        output_path: Path to the output directory with relabeled masks
        categories: Dict mapping category names to expected pixel values
        mask_folder: Name of the folder containing masks
        num_samples: Number of samples to visualize per category
        output_viz_folder: Folder name to save visualizations
    """
    if categories is None:
        categories = {'BPD': 1, 'AC': 2, 'FL': 3}

    output_path = Path(output_path)
    viz_path = output_path / output_viz_folder
    viz_path.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    for category, expected_value in categories.items():
        mask_path = output_path / category / 'train' / mask_folder

        if not mask_path.exists():
            continue

        mask_files = list(mask_path.glob('*.png')) + \
                     list(mask_path.glob('*.jpg')) + \
                     list(mask_path.glob('*.jpeg'))

        if not mask_files:
            continue

        print(f"\n{category} Category:")

        sample_size = min(num_samples, len(mask_files))
        sample_files = mask_files[:sample_size]

        for i, mask_file in enumerate(sample_files, 1):
            save_path = viz_path / f"{category}_{mask_file.stem}_visualization.png"
            visualize_mask(mask_file, save_path=save_path, show=False)
            print(f"  Created visualization {i}/{sample_size}: {save_path.name}")

    print(f"\n✓ All visualizations saved to: {viz_path}")


def diagnose_original_masks(base_path, categories=None, mask_folder='masks', num_samples=3):
    """
    Check the original masks to see if they have non-zero pixels.

    Args:
        base_path: Path to the original dataset
        categories: Dict mapping category names to expected pixel values
        mask_folder: Name of the folder containing masks
        num_samples: Number of samples to check per category
    """
    if categories is None:
        categories = {'BPD': 1, 'AC': 2, 'FL': 3}

    base_path = Path(base_path)

    print("\n" + "="*60)
    print("CHECKING ORIGINAL MASKS (BEFORE RELABELING)")
    print("="*60)

    for category in categories.keys():
        mask_path = base_path / category / 'train' / mask_folder

        if not mask_path.exists():
            print(f"\n{category}: ❌ Path not found - {mask_path}")
            continue

        # Get all mask files
        mask_files = list(mask_path.glob('*.png')) + \
                     list(mask_path.glob('*.jpg')) + \
                     list(mask_path.glob('*.jpeg')) + \
                     list(mask_path.glob('*.bmp'))

        if not mask_files:
            print(f"\n{category}: ❌ No mask files found")
            continue

        print(f"\n{category} Category (ORIGINAL)")
        print(f"Found {len(mask_files)} masks in {mask_path}")

        # Check sample masks
        sample_size = min(num_samples, len(mask_files))
        sample_files = mask_files[:sample_size]

        all_empty = True

        for i, mask_file in enumerate(sample_files, 1):
            unique_values, mask = check_mask_values(mask_file)
            non_zero_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            percentage = (non_zero_pixels / total_pixels) * 100

            print(f"\n  Sample {i}: {mask_file.name}")
            print(f"    Unique values: {unique_values}")
            print(f"    Shape: {mask.shape}")
            print(f"    Mask region: {non_zero_pixels}/{total_pixels} pixels ({percentage:.2f}%)")

            if non_zero_pixels > 0:
                all_empty = False
                print(f"    Min: {mask.min()}, Max: {mask.max()}, Mean: {mask.mean():.2f}")
                print(f"    ✓ GOOD: Has mask pixels")
            else:
                print(f"    ⚠️  WARNING: This original mask is empty!")

        if all_empty:
            print(f"\n  ⚠️⚠️⚠️  ALL SAMPLED {category} ORIGINAL MASKS ARE EMPTY!")
            print(f"  This explains why relabeled masks are also empty.")


if __name__ == "__main__":
    # Set your paths
    BASE_PATH = "/home/htic/fetalai_data/vasanth/datasets_to_work_with/retro/unet_dataset_precise_split"
    OUTPUT_PATH = "/home/htic/fetalai_data/vasanth/datasets_to_work_with/retro/fetalclip_dataset"

    CATEGORIES = {
        'BPD': 1,
        'AC': 2,
        'FL': 3
    }

    MASK_FOLDER = 'masks'
    NUM_SAMPLES = 3  # Number of samples to check/visualize per category

    # First, check the original masks to diagnose the issue
    print("="*60)
    print("STEP 1: DIAGNOSING ORIGINAL MASKS")
    print("="*60)
    diagnose_original_masks(BASE_PATH, CATEGORIES, MASK_FOLDER, NUM_SAMPLES)

    # Then verify the relabeled masks
    print("\n" + "="*60)
    print("STEP 2: VERIFYING RELABELED MASKS")
    print("="*60)
    verify_dataset(OUTPUT_PATH, CATEGORIES, MASK_FOLDER, NUM_SAMPLES)

    # Create visualizations
    print("\n" + "="*60)
    print("STEP 3: CREATING VISUALIZATIONS")
    print("="*60)
    visualize_samples(OUTPUT_PATH, CATEGORIES, MASK_FOLDER, NUM_SAMPLES)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nNote: The masks appear black in normal image viewers because")
    print("the pixel values (1, 2, 3) are very close to 0 (black).")
    print("\nIf the visualization shows 'EMPTY MASK', it means:")
    print("1. The original mask was already empty (all zeros), OR")
    print("2. The relabeling process failed for that category")
    print("\nCheck the diagnostic output above to identify the root cause.")
