import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path


class FetalBiometryDataset(Dataset):
    """
    Dataset for fetal biometry with BPD, AC, and FL classes.

    Expected folder structure:
    root_folder/
        BPD/
            train/
                images/
                masks/
            validation/
                images/
                masks/
        AC/
            train/
                images/
                masks/
            validation/
                images/
                masks/
        FL/
            train/
                images/
                masks/
            validation/
                images/
                masks/
    """

    def __init__(self, root_folder, split='train', target_size=(224, 224)):
        """
        Args:
            root_folder: Path to the root folder containing BPD, AC, FL subfolders
            split: 'train' or 'validation'
            target_size: (height, width) tuple for resizing images
        """
        self.root_folder = Path(root_folder)
        self.split = split
        self.target_size = target_size

        # Class mapping: 0=background (not used in finetuning), 1=BPD, 2=AC, 3=FL
        self.class_names = ['BPD', 'AC', 'FL']
        self.class_to_idx = {'BPD': 1, 'AC': 2, 'FL': 3}

        # Collect all image paths and their corresponding masks and labels
        self.samples = []
        self._load_dataset()

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_dataset(self):
        """Load all image paths, mask paths, and labels from the folder structure."""
        for class_name in self.class_names:
            class_folder = self.root_folder / class_name / self.split
            images_folder = class_folder / 'images'
            masks_folder = class_folder / 'masks'

            if not images_folder.exists():
                print(f"Warning: {images_folder} does not exist, skipping...")
                continue

            if not masks_folder.exists():
                print(f"Warning: {masks_folder} does not exist, skipping...")
                continue

            # Get all image files
            image_files = sorted([f for f in os.listdir(images_folder)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

            for img_file in image_files:
                img_path = images_folder / img_file

                # Try to find corresponding mask with same name
                mask_path = masks_folder / img_file

                # If exact match doesn't exist, try common mask naming patterns
                if not mask_path.exists():
                    # Try with _mask suffix
                    name, ext = os.path.splitext(img_file)
                    mask_path = masks_folder / f"{name}_mask{ext}"

                if not mask_path.exists():
                    print(f"Warning: Mask not found for {img_path}, skipping...")
                    continue

                label = self.class_to_idx[class_name]
                self.samples.append((str(img_path), str(mask_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape [1, 1, H, W] (temporal_dim=1, channels=1, H, W)
            mask: Tensor of shape [1, H, W] (single channel binary mask)
            label: Integer label (1, 2, or 3)
        """
        img_path, mask_path, label = self.samples[idx]

        # Load and preprocess image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        image = cv2.resize(image, self.target_size)
        image = image / 255.0  # Normalize to [0, 1]

        # Load and preprocess mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        mask = cv2.resize(mask, self.target_size)
        mask = mask / 255.0  # Normalize to [0, 1]

        # Convert to tensors
        # Image shape: [1, 1, H, W] for temporal dimension
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

        # Mask shape: [1, H, W]
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask, label


def get_dataloaders(root_folder, batch_size=4, num_workers=4, target_size=(224, 224)):
    """
    Create train and validation dataloaders.

    Args:
        root_folder: Path to the root folder containing BPD, AC, FL subfolders
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        target_size: (height, width) tuple for resizing images

    Returns:
        train_loader, val_loader
    """
    train_dataset = FetalBiometryDataset(
        root_folder=root_folder,
        split='train',
        target_size=target_size
    )

    val_dataset = FetalBiometryDataset(
        root_folder=root_folder,
        split='validation',
        target_size=target_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataloader
    root_folder = "/home/htic/fetalai_data/vasanth/datasets_to_work_with/retro/unet_dataset"
    

    train_loader, val_loader = get_dataloaders(
        root_folder=root_folder,
        batch_size=2,
        num_workers=0
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Test loading one batch
    for images, masks, labels in train_loader:
        print(f"Image batch shape: {images.shape}")  # Should be [batch, 1, 1, 224, 224]
        print(f"Mask batch shape: {masks.shape}")    # Should be [batch, 1, 224, 224]
        print(f"Labels: {labels}")                   # Should be [batch] with values 1, 2, or 3
        break
