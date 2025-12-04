import os
import re
import cv2
from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# ==== CHANGE THIS ====
image_path = "/home/htic/vasanth/fetalai_project/fetalai/pipeline/ANID_1000.jpg"  # ← Replace with your image path
# ======================

# Regex patterns to extract measurements
pattern_1d = re.compile(r"1\s*D[:\-]?\s*([\d.]+)\s*cm", re.IGNORECASE)
pattern_2d = re.compile(r"2\s*D[:\-]?\s*([\d.]+)\s*cm", re.IGNORECASE)

# Load image
image = cv2.imread(image_path)
if image is None:
    print("Error: Cannot read image. Check the path.")
    exit()

# Run OCR
ocr_result = ocr.ocr(image, cls=True)

# Extract all recognized text into one string
all_text = " ".join([line[1][0] for res in ocr_result for line in res])

# Search for values
value_1d_match = pattern_1d.search(all_text)
value_2d_match = pattern_2d.search(all_text)

value_1d = value_1d_match.group(1) if value_1d_match else "Not found"
value_2d = value_2d_match.group(1) if value_2d_match else "Not found"

# Print extracted information
print("\n===== OCR EXTRACTED MEASUREMENTS =====")
print(f"Image: {os.path.basename(image_path)}")
print(f"Detected Text:\n{all_text}\n")
print(f"1D Measurement: {value_1d}")
print(f"2D Measurement: {value_2d}")
print("=======================================\n")

