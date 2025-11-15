import os
import shutil
import pandas as pd

# Base directory where your folders are located
base_dir = "orig"
# Directory where the images will be copied and organized
output_dir = "classwise_dataset"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Dictionary to keep track of image counts for each unique image name
image_counts = {}

# Traverse the folder structure to collect unique image names
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.jpg'):  # Check if the file is an image
            # Extract the ANID and original image name
            anid = os.path.basename(os.path.dirname(root))
            image_name = file

            # Update the count of the image name
            if image_name not in image_counts:
                image_counts[image_name] = 0
            image_counts[image_name] += 1

            # Create a folder for the image name if it doesn't exist
            image_folder = os.path.join(output_dir, image_name.split('.')[0])
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)

            # Rename the image to include the ANID number
            new_image_name = f"{anid}.jpeg"

            # Copy the image to the respective folder with the new name
            shutil.copy(os.path.join(root, file), os.path.join(image_folder, new_image_name))

# Convert the dictionary to a DataFrame
df = pd.DataFrame(list(image_counts.items()), columns=['Image Name', 'Count'])

# Save the DataFrame to an Excel file
df.to_csv(os.path.join(output_dir, 'classwise_count.csv'), index=False)

print("Images copied and Excel file created successfully in the 'classwise_dataset' directory.")
