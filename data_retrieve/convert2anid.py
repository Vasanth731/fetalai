import os
import csv

# -------- CONFIG --------
source_dir = "orig"          # directory containing folders like 245677, 45673, ...
csv_path = "output_mapping_original.csv"

# Get folder names sorted (optional)
folders = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])

mapping = []

for idx, folder_name in enumerate(folders, start=3000):
    new_name = f"ANID_{idx}"
    
    src_path = os.path.join(source_dir, folder_name)
    dst_path = os.path.join(source_dir, new_name)

    # Rename the folder
    os.rename(src_path, dst_path)

    # Save mapping
    mapping.append([folder_name, new_name])

# Write CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["old_name", "new_name"])
    writer.writerows(mapping)

print("Renaming Completed!")
print("CSV saved at:", csv_path)
