import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index]) #.replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


if __name__ == "__main__":
    # Replace with your actual paths
    image_folder = "data/train/images"   # e.g., path to folder with .jpg images
    mask_folder = "data/train/masks"     # e.g., path to folder with _mask.gif masks

    # Instantiate the dataset
    dataset = CarvanaDataset(image_dir=image_folder, mask_dir=mask_folder)

    # Load a sample and print shapes
    image, mask = dataset[0]  # First sample
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)