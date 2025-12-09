import os
import cv2

MINDRAY_TRUNCATE_ROI = [330, 116, 1450, 880]   # x1, y1, x2, y2
# SONO_TRUNCATE_ROI = [294, 113, 1795, 968] 

def crop_folder_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        if not (filename.lower().endswith(".png") or filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg")):
            continue

        img = cv2.imread(file_path)
        if img is None:
            print(f"Skipping unreadable file: {filename}")
            continue
        
        x1, y1, x2, y2 = MINDRAY_TRUNCATE_ROI
        cropped = img[y1:y2, x1:x2]

        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cropped)
        print(f"Cropped & saved: {save_path}")


# Example usage:
# input_folder = "/home/htic/fetalai_data/vasanth/datasets_to_work_with/parsed/new_parsed_dataset/01_Abdominal_Circumference"
input_folder = "/home/htic/fetalai_data/vasanth/datasets_to_work_with/parsed/new_parsed_dataset/02_BPD_HC_Transthalamic"
output_folder = "/home/htic/fetalai_data/vasanth/datasets_to_work_with/parsed/old_parsed_biometry/cropped/bpd" # "cropped_images_sonoscape_fl"
crop_folder_images(input_folder, output_folder)
