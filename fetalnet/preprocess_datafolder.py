import os
import json
import cv2
import numpy as np
from glob import glob

def create_masks_from_json(input_folder, output_folder='/home/htic/fetalai_data/vasanth/datasets_to_work_with/retro/unet_dataset_precise/BPD'):
    # Define output folders
    image_output_dir = os.path.join(output_folder, 'images')
    mask_output_dir = os.path.join(output_folder, 'masks')
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # Get all JSON files
    json_files = glob(os.path.join(input_folder, '*.json'))

    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Get corresponding image
        image_name = data.get('imagePath')
        if not image_name:
            print(f"⚠️ Skipping {json_path} — no imagePath found")
            continue

        image_path = os.path.join(input_folder, image_name)
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found for {json_path}")
            continue

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Could not read image {image_path}")
            continue

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw polygons
        for shape in data['shapes']:
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)

        # Define filenames (save both as .jpg)
        base_name = os.path.splitext(image_name)[0]
        image_save_path = os.path.join(image_output_dir, base_name + '.jpg')
        mask_save_path = os.path.join(mask_output_dir, base_name + '.jpg')

        # Save image and mask as .jpg
        cv2.imwrite(image_save_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(mask_save_path, mask, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"✅ Saved: {base_name}.jpg and mask")


if __name__ == "__main__":
    input_folder = "/home/htic/vasanth/gdrive_downloads/bpd_annotated_6digit"  # Change this path
    create_masks_from_json(input_folder)
