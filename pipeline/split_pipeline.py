import cv2
import numpy as np
import os
from typing import List, Tuple

# ==========================================
# 1. CONFIGURATION & COORDINATES
# ==========================================

OUTPUT_DIR = "split_output_images_s"

# --- MINDRAY SETTINGS ---
# ROI for Blue Text detection [x_start, y_start, x_end, y_end]
MINDRAY_TEXT_ROI = [873, 83, 1092, 174]
# Cyan Color Range (HSV)
MINDRAY_CYAN_LOWER = np.array([80, 100, 100])
MINDRAY_CYAN_UPPER = np.array([100, 255, 255])
MINDRAY_MIN_RATIO = 0.001
# *** USER TO UPDATE: Area to crop before splitting ***
# CROP_X = 330, CROP_Y = 116, CROP_W = 1102, CROP_H = 764
MINDRAY_TRUNCATE_ROI = [330, 116, 1450, 880]


# --- VOLUSON SETTINGS ---
# Geometric Detection Settings
VOL_SEARCH_STRIP = 6
VOL_DARK_THRESH = 65
VOL_MIN_RATIO = 0.71
# Frame coordinates for Geometric Center Calculation [x_start, y_start, x_end, y_end]
# (You previously gave: 390, 88, 1528, 849)
VOL_FRAME_ROI = [390, 88, 1528, 849] 
# *** USER TO UPDATE: Area to crop before splitting (likely same as Frame ROI) ***
VOL_TRUNCATE_ROI = [390, 88, 1528, 849]

# --- SONOSCAPE SETTINGS ---
# ROI for Blue Logo detection [x_start, y_start, x_end, y_end]
SONO_LOGO_ROI = [1024, 112, 1144, 192] 
# Blue Color Range (HSV)
SONO_BLUE_LOWER = np.array([90, 50, 50])
SONO_BLUE_UPPER = np.array([120, 255, 255])
SONO_MIN_RATIO = 0.005
# *** USER TO UPDATE: Area to crop before splitting (The Active Ultrasound View) ***
# [x_start, y_start, x_end, y_end]
# CROP_X = 294, CROP_Y = 113, CROP_W = 1501, CROP_H = 855
SONO_TRUNCATE_ROI = [294, 113, 1795, 968] 



# ==========================================
# 2. DETECTION FUNCTIONS
# ==========================================

def is_mindray(img: np.ndarray) -> bool:
    """Checks for Mindray Split Screen (Cyan Text)."""
    x1, y1, x2, y2 = MINDRAY_TEXT_ROI
    # Bounds check
    if y2 > img.shape[0] or x2 > img.shape[1]: return False
    
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return False
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, MINDRAY_CYAN_LOWER, MINDRAY_CYAN_UPPER)
    ratio = cv2.countNonZero(mask) / mask.size
    return ratio >= MINDRAY_MIN_RATIO

def is_voluson(img: np.ndarray) -> bool:
    """Checks for Voluson Split Screen (Geometric Center Line)."""
    fx1, _, fx2, _ = VOL_FRAME_ROI
    # Bounds check
    if fx2 > img.shape[1]: return False
    
    frame_w = fx2 - fx1
    split_x = fx1 + int(frame_w * 0.50)
    
    x_start = max(0, split_x - (VOL_SEARCH_STRIP // 2))
    x_end = min(img.shape[1], split_x + (VOL_SEARCH_STRIP // 2))
    
    strip = img[:, x_start:x_end]
    if strip.size == 0: return False
    
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    dark_count = np.sum(gray < VOL_DARK_THRESH)
    ratio = dark_count / gray.size
    return ratio >= VOL_MIN_RATIO

def is_sonoscape(img: np.ndarray) -> bool:
    """Checks for Sonoscape Split Screen (Blue Logo)."""
    x1, y1, x2, y2 = SONO_LOGO_ROI
    # Bounds check
    if y2 > img.shape[0] or x2 > img.shape[1]: return False

    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return False
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Using STRICT blue mask (no white) to avoid bone false positives
    mask = cv2.inRange(hsv, SONO_BLUE_LOWER, SONO_BLUE_UPPER)
    ratio = cv2.countNonZero(mask) / mask.size
    return ratio >= SONO_MIN_RATIO

# ==========================================
# 3. PROCESSING PIPELINE
# ==========================================

def process_image(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: Not found - {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None: return

    filename = os.path.splitext(os.path.basename(image_path))[0]
    detected_machine = None
    crop_coords = None

    print(f"Analyzing: {filename}...")

    # --- PRIORITY CHECK SEQUENCE ---
    # 1. Check Mindray FIRST (Specific Text)
    if is_mindray(img):
        detected_machine = "Mindray"
        crop_coords = MINDRAY_TRUNCATE_ROI
    
    # 2. Check Voluson SECOND (Geometric line is very distinct)
    elif is_voluson(img):
        detected_machine = "Voluson"
        crop_coords = VOL_TRUNCATE_ROI
        
    # 3. Check Sonoscape LAST (Logo can be confused with other blue UI)
    elif is_sonoscape(img):
        detected_machine = "Sonoscape"
        crop_coords = SONO_TRUNCATE_ROI
    
    if not detected_machine:
        print("-> Result: No Split Screen Detected (or Unknown Machine).")
        return

    print(f"-> Detected: {detected_machine} Split Screen!")

    # --- CROPPING & SPLITTING ---
    
    # 1. Validate Truncate Coordinates
    if crop_coords == [0, 0, 0, 0]:
        print(f"CRITICAL WARNING: {detected_machine} TRUNCATE_ROI coordinates are not set in the script!")
        print("   -> Cannot crop/split. Please update the 'CONFIGURATION' section.")
        return

    tx1, ty1, tx2, ty2 = crop_coords
    
    # Safety check for crop bounds
    h_img, w_img = img.shape[:2]
    if tx2 > w_img or ty2 > h_img:
        print(f"Error: Crop coordinates {crop_coords} are larger than image size {w_img}x{h_img}.")
        return

    truncated_img = img[ty1:ty2, tx1:tx2]
    
    if truncated_img.size == 0:
        print("Error: Truncated image is empty.")
        return

    # 2. Split into Halves
    h_crop, w_crop = truncated_img.shape[:2]
    mid_x = w_crop // 2
    
    left_half = truncated_img[:, 0:mid_x]
    right_half = truncated_img[:, mid_x:w_crop]

    # --- SAVE ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    left_name = f"{filename}_{detected_machine}_Left.jpg"
    right_name = f"{filename}_{detected_machine}_Right.jpg"
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, left_name), left_half)
    cv2.imwrite(os.path.join(OUTPUT_DIR, right_name), right_half)
    
    print(f"-> Saved: {left_name} & {right_name}")


# ==========================================
# 4. EXECUTION
# ==========================================

if __name__ == '__main__':

    TEST_IMAGE = "/home/htic/fetalai_data/vasanth/datasets_to_work_with/parsed/new_parsed_dataset/02_BPD_HC_Transthalamic/INDTNCHE0100195_visit_1_Head_1_Transthalamic plane_1.png" 
    
    process_image(TEST_IMAGE)