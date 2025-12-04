import os
import re
import csv
import cv2
from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Path to the folder containing images
image_folder = '/home/htic/Pictures/3_class/FL'  # ← Replace this with your folder path
output_csv = 'extract_mm/extracted_measurements_FL.csv'

# Regex pattern to find "1 D" and "2 D" measurements
pattern_1d = re.compile(r"1\s*D[:\-]?\s*([\d.]+)\s*cm", re.IGNORECASE)
pattern_2d = re.compile(r"2\s*D[:\-]?\s*([\d.]+)\s*cm", re.IGNORECASE)

# Prepare output data list
results = [["filename", "1D", "2D"]]

# Process each image
for filename in sorted(os.listdir(image_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        ocr_result = ocr.ocr(image, cls=True)
        all_text = " ".join([line[1][0] for res in ocr_result for line in res])

        # Search for 1 D and 2 D values
        value_1d = pattern_1d.search(all_text)
        value_2d = pattern_2d.search(all_text)

        value_1d = value_1d.group(1) if value_1d else ""
        value_2d = value_2d.group(1) if value_2d else ""

        results.append([filename, value_1d, value_2d])
        print(f"Processed {filename}: 1 D = {value_1d}, 2 D = {value_2d}")

        # Save to CSV
        with open(output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)

print(f"CSV saved to: {output_csv}")
